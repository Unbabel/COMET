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
                        (required, type: Path_fr)
  -t TRANSLATIONS, --translations TRANSLATIONS
                        (required, type: Path_fr)
  -r REFERENCES, --references REFERENCES
                        (type: Path_fr, default: null)
  --batch_size BATCH_SIZE
                        (type: int, default: 8)
  --gpus GPUS           (type: int, default: 1)
  --accelerator {dp,ddp,ddp_cpu}
                        Pytorch Lightnining accelerator for multi-GPU. (type: str, default: ddp)
  --disable_bar         Disables progress bar. (default: True)
  --to_json TO_JSON     Exports results to a json file. (type: Union[bool, str], default: False)
  --model {emnlp20-comet-rank,wmt20-comet-da,wmt20-comet-qe-da,wmt21-comet-mqm,wmt21-cometinho-da,wmt21-comet-qe-mqm}
                        COMET model to be used. (type: Union[str, Path_fr], default: wmt20-comet-da)
  --mc_dropout MC_DROPOUT
                        Number of inference runs for each sample in MC Dropout. (type: Union[bool, int], default: False)
  --seed_everything SEED_EVERYTHING
                        Prediction seed. (type: int, default: 12)
"""
import json
from typing import Dict, List, Optional, Union

import torch
from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything

_REFLESS_MODELS = ["comet-qe"]


def print_scores(
    seg_scores: List[float],
    sys_score: float,
    data: List[Dict[str, str]],
    std_scores: Optional[List[float]] = None,
    save_predictions: Union[str, bool] = False,
) -> None:
    for i, (score, sample) in enumerate(zip(seg_scores, data)):
        sample["COMET"] = score
        if std_scores:
            sample["variance"] = std_scores[i]
            print(
                "Segment {}\tscore: {:.4f}\tvariance: {:.4f}".format(
                    i, score, std_scores[i]
                )
            )
        else:
            print("Segment {}\tscore: {:.4f}".format(i, score))

    print("System score: {:.4f}".format(sys_score))
    if isinstance(save_predictions, str):
        with open(save_predictions, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(save_predictions))


def score_command() -> None:
    parser = ArgumentParser(description="Command for scoring MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-t", "--translations", type=Path_fr, required=True)
    parser.add_argument("-r", "--references", type=Path_fr)
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
        "--disable_bar", action="store_false", help="Disables progress bar."
    )
    parser.add_argument(
        "--to_json",
        type=Union[bool, str],
        default=False,
        help="Exports results to a json file.",
    )
    parser.add_argument(
        "--model",
        type=Union[str, Path_fr],
        required=False,
        default="wmt20-comet-da",
        choices=available_metrics.keys(),
        help="COMET model to be used.",
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
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    if (cfg.references is None) and (
        not any([i in cfg.model for i in _REFLESS_MODELS])
    ):
        parser.error("{} requires -r/--references.".format(cfg.model))

    model_path = (
        download_model(cfg.model) if cfg.model in available_metrics else cfg.model
    )
    model = load_from_checkpoint(model_path)
    model.eval()

    with open(cfg.sources()) as fp:
        sources = [line.strip() for line in fp.readlines()]

    with open(cfg.translations()) as fp:
        translations = [line.strip() for line in fp.readlines()]

    if "comet-qe" in cfg.model:
        data = {"src": sources, "mt": translations}
    else:
        with open(cfg.references()) as fp:
            references = [line.strip() for line in fp.readlines()]
        data = {"src": sources, "mt": translations, "ref": references}

    data = [dict(zip(data, t)) for t in zip(*data.values())]

    if cfg.mc_dropout:
        gather_mean = [None for _ in range(cfg.gpus)]  # Only necessary for multigpu DDP
        gather_std = [None for _ in range(cfg.gpus)]  # Only necessary for multigpu DDP

        mean_scores, std_scores, sys_score = model.predict(
            data,
            cfg.batch_size,
            cfg.gpus,
            cfg.mc_dropout,
            cfg.disable_bar,
            cfg.accelerator,
        )
        if cfg.gpus > 1 and cfg.accelerator == "ddp":
            torch.distributed.all_gather_object(gather_mean, mean_scores)
            torch.distributed.all_gather_object(gather_std, std_scores)
            torch.distributed.barrier()  # Waits for all processes

            if torch.distributed.get_rank() == 0:
                mean_scores = [
                    o[i] for i in range(len(gather_mean[0])) for o in gather_mean
                ]
                std_scores = [
                    o[i] for i in range(len(gather_std[0])) for o in gather_std
                ]
                sys_score = sum(mean_scores) / len(mean_scores)
                print_scores(
                    mean_scores,
                    sys_score,
                    data,
                    std_scores=std_scores,
                    save_predictions=cfg.to_json,
                )
        else:
            print_scores(
                mean_scores,
                sys_score,
                data,
                std_scores=std_scores,
                save_predictions=cfg.to_json,
            )

    else:
        gather_outputs = [
            None for _ in range(cfg.gpus)
        ]  # Only necessary for multigpu DDP
        seg_scores, sys_score = model.predict(
            data, cfg.batch_size, cfg.gpus, False, cfg.disable_bar, cfg.accelerator
        )
        if cfg.gpus > 1 and cfg.accelerator == "ddp":
            torch.distributed.all_gather_object(gather_outputs, seg_scores)
            torch.distributed.barrier()  # Waits for all processes
            if torch.distributed.get_rank() == 0:
                seg_scores = [
                    o[i] for i in range(len(gather_outputs[0])) for o in gather_outputs
                ]
                sys_score = sum(seg_scores) / len(seg_scores)
                print_scores(seg_scores, sys_score, data, save_predictions=cfg.to_json)
        else:
            print_scores(seg_scores, sys_score, data, save_predictions=cfg.to_json)
