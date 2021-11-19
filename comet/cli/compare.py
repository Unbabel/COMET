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
Command for comparing two MT systems.
======================================

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (required unless using -d, type: Path_fr)
  -x SYSTEM_X, --system_x SYSTEM_X
                        (required, type: Path_fr)
  -y SYSTEM_Y, --system_y SYSTEM_Y
                        (required, type: Path_fr)
  -r REFERENCES, --references REFERENCES
                        (type: Path_fr, default: None)
  -d SACREBLEU_TESTSET, --sacrebleu_dataset SACREBLEU_TESTSET
                        (optional, use in place of -s and -r, type: str
                         format TESTSET:LANGPAIR, e.g., wmt20:en-de)
  --batch_size BATCH_SIZE
                        (type: int, default: 8)
  --gpus GPUS           (type: int, default: 1)
  --num_splits NUM_SPLITS
                        Number of random partitions used in Bootstrap resampling. (type: int, default: 300)
  --sample_ratio SAMPLE_RATIO
                        Percentage of the testset to use in each bootstrap resampling partition. (type: float, default: 0.4)
  --to_json TO_JSON     Exports results to a json file. (type: Union[bool, str], default: False)
  --model {emnlp20-comet-rank,wmt20-comet-da,wmt20-comet-qe-da,wmt21-cometinho-da}
                        COMET model to be used. (type: Union[str, Path_fr], default: wmt20-comet-da)
  --seed_everything SEED_EVERYTHING
                        Prediction seed. (type: int, default: 12)

"""
import json
import multiprocessing
import os
from typing import Union

import numpy as np
import torch
from comet.cli.score import _REFLESS_MODELS
from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
from scipy import stats

_REFLESS_MODELS = ["comet-qe"]  # All reference-free metrics are named with 'comet-qe'
# Due to small numerical differences in scores we consider that any system comparison
# with a difference bellow EPS to be considered a tie.
EPS = 0.001


def compare_command() -> Union[None, int]:
    parser = ArgumentParser(description="Command for comparing two MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr)
    parser.add_argument("-x", "--system_x", type=Path_fr, required=True)
    parser.add_argument("-y", "--system_y", type=Path_fr, required=True)
    parser.add_argument("-r", "--references", type=Path_fr)
    parser.add_argument("-d", "--sacrebleu_dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--num_splits",
        type=int,
        default=300,
        help="Number of random partitions used in Bootstrap resampling.",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.4,
        help="Percentage of the testset to use in each split.",
    )
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
                cfg.model, list(available_metrics.keys())
            )
        )
    model = load_from_checkpoint(model_path)
    model.eval()

    if not cfg.disable_cache:
        model.set_embedding_cache()

    with open(cfg.sources()) as fp:
        sources = [line.strip() for line in fp.readlines()]

    with open(cfg.system_x()) as fp:
        system_x = [line.strip() for line in fp.readlines()]

    with open(cfg.system_y()) as fp:
        system_y = [line.strip() for line in fp.readlines()]

    if "comet-qe" in cfg.model:
        system_x = {"src": sources, "mt": system_x}
        system_y = {"src": sources, "mt": system_y}
    else:
        with open(cfg.references()) as fp:
            references = [line.strip() for line in fp.readlines()]
        system_x = {"src": sources, "mt": system_x, "ref": references}
        system_y = {"src": sources, "mt": system_y, "ref": references}

    system_x = [dict(zip(system_x, t)) for t in zip(*system_x.values())]
    system_y = [dict(zip(system_y, t)) for t in zip(*system_y.values())]

    if cfg.gpus > 1 and cfg.accelerator == "ddp":
        gather_outputs = [
            None for _ in range(cfg.gpus)
        ]  # Only necessary for multigpu DDP
        seperator_index = len(system_x)
        data = system_x + system_y
        outputs = model.predict(
            samples=data,
            batch_size=cfg.batch_size,
            gpus=cfg.gpus,
            progress_bar=(not cfg.disable_bar),
            accelerator=cfg.accelerator,
            num_workers=cfg.num_workers,
            length_batching=(not cfg.disable_length_batching),
        )
        seg_scores = outputs[0]
        torch.distributed.all_gather_object(gather_outputs, seg_scores)
        torch.distributed.barrier()  # Waits for all processes
        if torch.distributed.get_rank() == 0:
            seg_scores = [
                o[i] for i in range(len(gather_outputs[0])) for o in gather_outputs
            ]
        else:
            return 0

        x_seg_scores = seg_scores[:seperator_index]
        y_seg_scores = seg_scores[seperator_index:]

    else:  # This maximizes cache hits because batches will be equal!
        x_seg_scores, _ = model.predict(
            samples=system_x,
            batch_size=cfg.batch_size,
            gpus=cfg.gpus,
            progress_bar=(not cfg.disable_bar),
            accelerator=cfg.accelerator,
            num_workers=cfg.num_workers,
            length_batching=(not cfg.disable_length_batching),
        )
        y_seg_scores, _ = model.predict(
            samples=system_y,
            batch_size=cfg.batch_size,
            gpus=cfg.gpus,
            progress_bar=(not cfg.disable_bar),
            accelerator=cfg.accelerator,
            num_workers=cfg.num_workers,
            length_batching=(not cfg.disable_length_batching),
        )

    data = []
    for i, (x_score, y_score) in enumerate(zip(x_seg_scores, y_seg_scores)):
        print(
            "Segment {}\tsystem_x score: {:.4f}\tsystem_y score: {:.4f}".format(
                i, x_score, y_score
            )
        )
        data.append(
            {
                "src": system_x[i]["src"],
                "system_x": {"mt": system_x[i]["mt"], "score": x_score},
                "system_y": {"mt": system_y[i]["mt"], "score": y_score},
            }
        )
        if "comet-qe" not in cfg.model:
            data[-1]["ref"] = system_y[i]["ref"]

    n = len(sources)
    ids = list(range(n))
    sample_size = max(int(n * cfg.sample_ratio), 1)

    x_sys_scores, y_sys_scores = [], []
    win_count = [0, 0, 0]
    for _ in range(cfg.num_splits):
        # Subsample the gold and system outputs (with replacement)
        subsample_ids = np.random.choice(ids, size=sample_size, replace=True)
        subsample_x_scr = sum([x_seg_scores[i] for i in subsample_ids]) / sample_size
        subsample_y_scr = sum([y_seg_scores[i] for i in subsample_ids]) / sample_size

        if abs(subsample_x_scr - subsample_y_scr) < EPS:  # TIE
            win_count[2] += 1
        elif subsample_x_scr > subsample_y_scr:  # X WIN
            win_count[0] += 1
        else:  # subsample_y_scr > subsample_x_scr: # Y WIN
            win_count[1] += 1

        x_sys_scores.append(subsample_x_scr)
        y_sys_scores.append(subsample_y_scr)

    t_test_result = stats.ttest_rel(np.array(x_seg_scores), np.array(y_seg_scores))
    data.insert(
        0,
        {
            "bootstrap_resampling": {
                "x-mean": np.mean(np.array(x_sys_scores)),
                "y-mean": np.mean(np.array(y_sys_scores)),
                "ties (%)": win_count[2] / sum(win_count),
                "x_wins (%)": win_count[0] / sum(win_count),
                "y_wins (%)": win_count[1] / sum(win_count),
            },
            "paired_t-test": {
                "statistic": t_test_result.statistic,
                "p_value": t_test_result.pvalue,
            },
        },
    )
    print("\nBootstrap Resampling Results:")
    for k, v in data[0]["bootstrap_resampling"].items():
        print("{}:\t{:.4f}".format(k, v))

    print("\nPaired T-Test Results:")
    for k, v in data[0]["paired_t-test"].items():
        print("{}:\t{:.4f}".format(k, v))

    best_system = (
        cfg.system_x.rel_path
        if sum(x_seg_scores) > sum(y_seg_scores)
        else cfg.system_y.rel_path
    )
    worse_system = (
        cfg.system_x.rel_path
        if sum(x_seg_scores) < sum(y_seg_scores)
        else cfg.system_y.rel_path
    )
    if data[0]["paired_t-test"]["p_value"] <= 0.05:
        print("Null hypothesis rejected according to t-test.")
        print(f"Scores differ significantly across samples.")
        print(f"{best_system} outperforms {worse_system}.")
    else:
        print("Null hypothesis can't be rejected.\nBoth systems have equal averages.")

    if isinstance(cfg.to_json, str):
        with open(cfg.to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))

    if cfg.print_cache_info:
        print(model.retrieve_sentence_embedding.cache_info())


if __name__ == "__main__":
    compare_command()
