#!/usr/bin/env python3

# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command for scoring MT systems.
===============================

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (type: Path_fr, default: null)
  -t TRANSLATIONS [TRANSLATIONS ...], --translations TRANSLATIONS [TRANSLATIONS ...]
                        (type: Path_fr, default: null)
  -r REFERENCES, --references REFERENCES
                        (type: Path_fr, default: null)
  -d SACREBLEU_DATASET, --sacrebleu_dataset SACREBLEU_DATASET
                        (type: str, default: null)
  --batch_size BATCH_SIZE
                        (type: int, default: 16)
  --gpus GPUS           (type: int, default: 1)
  --quiet               Sets all loggers to ERROR level. (default: False)
  --only_system         Prints only the final system score. (default: False)
  --to_json TO_JSON     Exports results to a json file. (type: str, default: "")
  --model MODEL         COMET model to be used. (type: str, default: wmt20-comet-da)
  --model_storage_path MODEL_STORAGE_PATH
                        Path to the directory where models will be stored. By default
                        its saved in ~/.cache/torch/unbabel_comet/ (default: null)
  --num_workers NUM_WORKERS
                        Number of workers to use when loading data. (type: int, 
                        default: null)
  --disable_cache       Disables sentence embeddings caching. This makes inference
                        slower but saves memory. (default: False)
  --disable_length_batching
                        Disables length batching. This makes inference slower. 
                        (default: False)
  --print_cache_info    Print information about COMET cache. (default: False)
"""
import itertools
import json
import logging
import os

import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
from sacrebleu.utils import get_reference_files, get_source_file

from comet import download_model, load_from_checkpoint


def score_command() -> None:
    parser = ArgumentParser(description="Command for scoring MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr)
    parser.add_argument("-t", "--translations", type=Path_fr, nargs="+")
    parser.add_argument("-r", "--references", type=Path_fr)
    parser.add_argument("-d", "--sacrebleu_dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--quiet", action="store_true", help="Sets all loggers to ERROR level."
    )
    parser.add_argument(
        "--only_system", action="store_true", help="Prints only the final system score."
    )
    parser.add_argument(
        "--to_json",
        type=str,
        default="",
        help="Exports results to a json file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="Unbabel/wmt22-comet-da",
        help="COMET model to be used.",
    )
    parser.add_argument(
        "--model_storage_path",
        help=(
            "Path to the directory where models will be stored. "
            + "By default its saved in ~/.cache/torch/unbabel_comet/"
        ),
        default=None,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers to use when loading data.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disables sentence embeddings caching. This makes inference slower but saves memory.",
    )
    parser.add_argument(
        "--disable_length_batching",
        action="store_true",
        help="Disables length batching. This makes inference slower.",
    )
    parser.add_argument(
        "--print_cache_info",
        action="store_true",
        help="Print information about COMET cache.",
    )
    cfg = parser.parse_args()

    if cfg.quiet:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.ERROR)

    seed_everything(1)
    if cfg.sources is None and cfg.sacrebleu_dataset is None:
        parser.error(f"You must specify a source (-s) or a sacrebleu dataset (-d)")

    if cfg.sacrebleu_dataset is not None:
        if cfg.references is not None or cfg.sources is not None:
            parser.error(
                f"Cannot use sacrebleu datasets (-d) with manually-specified datasets (-s and -r)"
            )

        try:
            testset, langpair = cfg.sacrebleu_dataset.rsplit(":", maxsplit=1)
            cfg.sources = Path_fr(get_source_file(testset, langpair))
            cfg.references = Path_fr(get_reference_files(testset, langpair)[0])
        except ValueError:
            parser.error(
                "SacreBLEU testset format must be TESTSET:LANGPAIR, e.g., wmt20:de-en"
            )
        except Exception as e:
            import sys

            print("SacreBLEU error:", e, file=sys.stderr)
            sys.exit(1)

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    else:
        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)

    model = load_from_checkpoint(model_path)
    model.eval()

    if model.requires_references() and (cfg.references is None):
        parser.error(
            "{} requires -r/--references or -d/--sacrebleu_dataset.".format(cfg.model)
        )

    if not cfg.disable_cache:
        model.set_embedding_cache()

    with open(cfg.sources(), encoding="utf-8") as fp:
        sources = [line.strip() for line in fp.readlines()]

    translations = []
    for path_fr in cfg.translations:
        with open(path_fr(), encoding="utf-8") as fp:
            translations.append([line.strip() for line in fp.readlines()])

    if cfg.references is not None:
        with open(cfg.references(), encoding="utf-8") as fp:
            references = [line.strip() for line in fp.readlines()]
        data = {
            "src": [sources for _ in translations],
            "mt": translations,
            "ref": [references for _ in translations],
        }
    else:
        data = {"src": [sources for _ in translations], "mt": translations}

    if cfg.gpus > 1:
        # Flatten all data to score across multiple GPUs
        for k, v in data.items():
            data[k] = list(itertools.chain(*v))

        data = [dict(zip(data, t)) for t in zip(*data.values())]
        outputs = model.predict(
            samples=data,
            batch_size=cfg.batch_size,
            gpus=cfg.gpus,
            progress_bar=(not cfg.quiet),
            accelerator="auto",
            num_workers=cfg.num_workers,
            length_batching=(not cfg.disable_length_batching),
        )
        seg_scores = outputs.scores
        if "metadata" in outputs and "error_spans" in outputs.metadata:
            errors = outputs.metadata.error_spans
        else:
            errors = []
        
        if len(cfg.translations) > 1:
            seg_scores = np.array_split(seg_scores, len(cfg.translations))
            sys_scores = [sum(split) / len(split) for split in seg_scores]
            data = np.array_split(data, len(cfg.translations))
            errors = np.array_split(outputs.metadata.errors, len(cfg.translations))
        else:
            sys_scores = [
                outputs.system_score,
            ]
            seg_scores = [
                seg_scores,
            ]
            errors = [errors, ]
            data = [
                np.array(data),
            ]
    else:
        # If not using Multiple GPUs we will score each system independently
        # to maximize cache hits!
        seg_scores, sys_scores, errors = [], [], []
        new_data = []
        for i in range(len(cfg.translations)):
            sys_data = {k: v[i] for k, v in data.items()}
            sys_data = [dict(zip(sys_data, t)) for t in zip(*sys_data.values())]
            new_data.append(np.array(sys_data))
            outputs = model.predict(
                samples=sys_data,
                batch_size=cfg.batch_size,
                gpus=cfg.gpus,
                progress_bar=(not cfg.quiet),
                accelerator="cpu" if cfg.gpus == 0 else "auto",
                num_workers=cfg.num_workers,
                length_batching=(not cfg.disable_length_batching),
            )
            seg_scores.append(outputs.scores)
            sys_scores.append(outputs.system_score)
            if "metadata" in outputs and "error_spans" in outputs.metadata:
                errors.append(outputs.metadata.error_spans)
        data = new_data

    files = [path_fr.rel_path for path_fr in cfg.translations]
    data = {file: system_data.tolist() for file, system_data in zip(files, data)}
    for i in range(len(data[files[0]])):  # loop over (src, ref)
        for j in range(len(files)):  # loop of system
            data[files[j]][i]["COMET"] = seg_scores[j][i]
            if errors:
                data[files[j]][i]["errors"] = errors[j][i]
                
            if not cfg.only_system:
                print(
                    "{}\tSegment {}\tscore: {:.4f}".format(
                        files[j], i, seg_scores[j][i]
                    )
                )

    for j in range(len(files)):
        print("{}\tscore: {:.4f}".format(files[j], sys_scores[j]))

    if cfg.to_json != "":
        with open(cfg.to_json, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))

    if cfg.print_cache_info:
        print(model.retrieve_sentence_embedding.cache_info())


if __name__ == "__main__":
    score_command()
