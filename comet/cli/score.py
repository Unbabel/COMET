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
                        (required unless using -d, type: Path_fr)
  -t TRANSLATIONS, --translations TRANSLATIONS
                        (required, type: Path_fr)
  -r REFERENCES, --references REFERENCES
                        (type: Path_fr, default: None)
  -d SACREBLEU_TESTSET, --sacrebleu_dataset SACREBLEU_TESTSET
                        (optional, use in place of -s and -r, type: str
                         format TESTSET:LANGPAIR, e.g., wmt20:en-de)
  --to_json TO_JSON     (type: Union[bool, str], default: False)
  --model MODEL         (type: Union[str, Path_fr], default: wmt21-large-estimator)
  --batch_size BATCH_SIZE
                        (type: int, default: 8)
  --gpus GPUS           (type: int, default: 1)
  --accelerator {dp,ddp}
                        Pytorch Lightnining accelerator for multi-GPU. (type: str, default: ddp)
  --disable_bar         Disables progress bar. (default: True)
  --to_json TO_JSON     Exports results to a json file. (type: Union[bool, str], default: False)
  --model {emnlp20-comet-rank,wmt20-comet-da,wmt20-comet-qe-da,wmt21-comet-mqm,wmt21-cometinho-da,wmt21-comet-qe-mqm}
                        COMET model to be used. (type: Union[str, Path_fr], default: wmt20-comet-da)
  --mc_dropout MC_DROPOUT
                        Number of inference runs for each sample in MC Dropout. 
                        (type: Union[bool, int], default: False)
  --seed_everything SEED_EVERYTHING
                        Prediction seed. (type: int, default: 12)
"""
import itertools
import json
import multiprocessing
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
from sacrebleu.utils import get_reference_files, get_source_file

_REFLESS_MODELS = ["comet-qe"]


def score_command() -> None:
    parser = ArgumentParser(description="Command for scoring MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr)
    parser.add_argument("-t", "--translations", type=Path_fr, nargs="+")
    parser.add_argument("-r", "--references", type=Path_fr)
    parser.add_argument("-d", "--sacrebleu_dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="ddp",
        choices=["dp", "ddp"],
        help="Pytorch Lightnining accelerator for multi-GPU.",
    )
    parser.add_argument(
        "--to_json",
        type=Union[bool, str],
        default=False,
        help="Exports results to a json file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="wmt20-comet-da",
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
        "--mc_dropout",
        type=Union[bool, int],
        default=False,
        help="Number of inference runs for each sample in MC Dropout.",
    )
    parser.add_argument(
        "--seed_everything",
        help="Prediction seed.",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers to use when loading data.",
        type=int,
        default=multiprocessing.cpu_count(),
    )
    parser.add_argument(
        "--disable_bar", action="store_true", help="Disables progress bar."
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
    seed_everything(cfg.seed_everything)
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

    if (cfg.references is None) and (
        not any([i in cfg.model for i in _REFLESS_MODELS])
    ):
        parser.error(
            "{} requires -r/--references or -d/--sacrebleu_dataset.".format(cfg.model)
        )

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    elif cfg.model in available_metrics:
        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)
    else:
        parser.error(
            "{} is not a valid checkpoint path or model choice. Choose from {}".format(
                cfg.model, available_metrics.keys()
            )
        )
    model = load_from_checkpoint(model_path)
    model.eval()

    if not cfg.disable_cache:
        model.set_embedding_cache()

    with open(cfg.sources()) as fp:
        sources = [line.strip() for line in fp.readlines()]

    translations = []
    for path_fr in cfg.translations:
        with open(path_fr()) as fp:
            translations.append([line.strip() for line in fp.readlines()])

    if "comet-qe" in cfg.model:
        data = {"src": [sources for _ in translations], "mt": translations}
    else:
        with open(cfg.references()) as fp:
            references = [line.strip() for line in fp.readlines()]
        data = {
            "src": [sources for _ in translations],
            "mt": translations,
            "ref": [references for _ in translations],
        }

    if cfg.gpus > 1 and cfg.accelerator == "ddp":
        # Flatten all data to score across multiple GPUs
        for k, v in data.items():
            data[k] = list(itertools.chain(*v))
        data = [dict(zip(data, t)) for t in zip(*data.values())]

        gather_mean = [None for _ in range(cfg.gpus)]  # Only necessary for multigpu DDP
        gather_std = [None for _ in range(cfg.gpus)]  # Only necessary for multigpu DDP
        outputs = model.predict(
            samples=data,
            batch_size=cfg.batch_size,
            gpus=cfg.gpus,
            mc_dropout=cfg.mc_dropout,
            progress_bar=(not cfg.disable_bar),
            accelerator=cfg.accelerator,
            num_workers=cfg.num_workers,
            length_batching=(not cfg.disable_length_batching),
        )
        seg_scores = outputs[0]
        std_scores = None if len(outputs) == 2 else outputs[1]
        torch.distributed.all_gather_object(gather_mean, outputs[0])
        if len(outputs) == 3:
            torch.distributed.all_gather_object(gather_std, outputs[1])

        torch.distributed.barrier()  # Waits for all processes
        if torch.distributed.get_rank() == 0:
            seg_scores = list(itertools.chain(*gather_mean))
            if len(outputs) == 3:
                std_scores = list(itertools.chain(*gather_std))
            else:
                std_scores = None
        else:
            return

        if len(cfg.translations) > 1:
            seg_scores = np.array_split(seg_scores, len(cfg.translations))
            sys_scores = [sum(split) / len(split) for split in seg_scores]
            std_scores = (
                np.array_split(std_scores, len(cfg.translations))
                if std_scores
                else [None] * len(seg_scores)
            )
            data = np.array_split(data, len(cfg.translations))
        else:
            sys_scores = [
                sum(seg_scores) / len(seg_scores),
            ]
            seg_scores = [
                seg_scores,
            ]
            data = [
                np.array(data),
            ]
            std_scores = [
                std_scores,
            ]
    else:
        # If not using Multiple GPUs we will score each system independently
        # to maximize cache hits!
        seg_scores, std_scores, sys_scores = [], [], []
        new_data = []
        for i in range(len(cfg.translations)):
            sys_data = {k: v[i] for k, v in data.items()}
            sys_data = [dict(zip(sys_data, t)) for t in zip(*sys_data.values())]
            new_data.append(np.array(sys_data))
            outputs = model.predict(
                samples=sys_data,
                batch_size=cfg.batch_size,
                gpus=cfg.gpus,
                mc_dropout=cfg.mc_dropout,
                progress_bar=(not cfg.disable_bar),
                accelerator=cfg.accelerator,
                num_workers=cfg.num_workers,
                length_batching=(not cfg.disable_length_batching),
            )
            if len(outputs) == 3:
                seg_scores.append(outputs[0])
                std_scores.append(outputs[1])
            else:
                seg_scores.append(outputs[0])
                std_scores.append(None)

            sys_scores.append(sum(outputs[0]) / len(outputs[0]))
        data = new_data

    files = [path_fr.rel_path for path_fr in cfg.translations]
    data = {file: system_data.tolist() for file, system_data in zip(files, data)}
    for i in range(len(data[files[0]])):  # loop over (src, ref)
        for j in range(len(files)):  # loop of system
            data[files[j]][i]["COMET"] = seg_scores[j][i]
            if cfg.mc_dropout:
                data[files[j]][i]["variance"] = std_scores[j][i]
                print(
                    "{}\tSegment {}\tscore: {:.4f}\tvariance: {:.4f}".format(
                        files[j], i, seg_scores[j][i], std_scores[j][i]
                    )
                )
            else:
                print(
                    "{}\tSegment {}\tscore: {:.4f}".format(
                        files[j], i, seg_scores[j][i]
                    )
                )

    for j in range(len(files)):
        print("{}\tscore: {:.4f}".format(files[j], sys_scores[j]))

    if isinstance(cfg.to_json, str):
        with open(cfg.to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))

    if cfg.print_cache_info:
        print(model.retrieve_sentence_embedding.cache_info())
