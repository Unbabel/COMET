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
                        (required, type: Path_fr)
  -x SYSTEM_X, --system_x SYSTEM_X
                        (required, type: Path_fr)
  -y SYSTEM_Y, --system_y SYSTEM_Y
                        (required, type: Path_fr)
  -r REFERENCES, --references REFERENCES
                        (type: Path_fr, default: null)
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
from typing import Union

import numpy as np
from comet.cli.score import _REFLESS_MODELS
from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything


def compare_command() -> None:
    parser = ArgumentParser(description="Command for comparing two MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-x", "--system_x", type=Path_fr, required=True)
    parser.add_argument("-y", "--system_y", type=Path_fr, required=True)
    parser.add_argument("-r", "--references", type=Path_fr)
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
        help="Percentage of the testset to use in each bootstrap resampling partition.",
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

    with open(cfg.system_x()) as fp:
        system_x = [line.strip() for line in fp.readlines()]

    with open(cfg.system_y()) as fp:
        system_y = [line.strip() for line in fp.readlines()]

    if "refless" in cfg.model:
        system_x = {"src": sources, "mt": system_x}
        system_y = {"src": sources, "mt": system_y}
    else:
        with open(cfg.references()) as fp:
            references = [line.strip() for line in fp.readlines()]
        system_x = {"src": sources, "mt": system_x, "ref": references}
        system_y = {"src": sources, "mt": system_y, "ref": references}

    system_x = [dict(zip(system_x, t)) for t in zip(*system_x.values())]
    system_y = [dict(zip(system_y, t)) for t in zip(*system_y.values())]

    x_seg_scores, _ = model.predict(system_x, cfg.batch_size, cfg.gpus)
    y_seg_scores, _ = model.predict(system_y, cfg.batch_size, cfg.gpus)

    data = []
    for i, (x_score, y_score) in enumerate(zip(x_seg_scores, y_seg_scores)):
        print(
            "Segment {}\tsystem_x score: {:.4f}\tsystem_y score: {:.4f}".format(
                i, x_score, y_score
            )
        )
        data.append(
            {
                "src": system_x[0]["src"],
                "system_x": {"mt": system_x[0]["mt"], "score": x_score},
                "system_y": {"mt": system_y[0]["mt"], "score": y_score},
                "ref": system_y[0]["ref"],
            }
        )

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

        if subsample_x_scr > subsample_y_scr:
            win_count[0] += 1
        elif subsample_y_scr > subsample_x_scr:
            win_count[1] += 1
        else:
            win_count[2] += 1

        x_sys_scores.append(subsample_x_scr)
        y_sys_scores.append(subsample_y_scr)

    data.insert(
        0,
        {
            "x-mean": np.mean(np.array(x_sys_scores)),
            "y-mean": np.mean(np.array(y_sys_scores)),
            "ties (%)": win_count[2] / sum(win_count),
            "x_wins (%)": win_count[0] / sum(win_count),
            "y_wins (%)": win_count[1] / sum(win_count),
        },
    )
    for k, v in data[0].items():
        print("{}:\t{:.4f}".format(k, v))

    if isinstance(cfg.to_json, str):
        with open(cfg.to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))
