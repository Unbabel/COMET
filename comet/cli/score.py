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
                        (required, type: Path_fr)
  --to_json TO_JSON     (type: Union[bool, str], default: False)
  --model MODEL         (type: Union[str, Path_fr], default: wmt21-large-estimator)
  --batch_size BATCH_SIZE
                        (type: int, default: 32)
  --gpus GPUS           (type: int, default: 1)

"""
import json
from typing import Union
import multiprocessing

from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything


_REFLESS_MODELS = ["comet-qe"]


def score_command() -> None:
    parser = ArgumentParser(description="Command for scoring MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-t", "--translations", type=Path_fr, required=True)
    parser.add_argument("-r", "--references", type=Path_fr)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
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
        "--model_storage_path",
        help="Path to the directory where models will be stored.",
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
        "--disable_cache",
        action="store_true",
        help="Disables sentence embeddings caching. This makes inference slower but saves memory.",
    )
    parser.add_argument(
        "--disable_length_batching",
        action="store_true",
        help="Disables length batching. This makes inference slower.",
    )
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    if (cfg.references is None) and (
        not any([i in cfg.model for i in _REFLESS_MODELS])
    ):
        parser.error("{} requires -r/--references.".format(cfg.model))

    model_path = (
        download_model(cfg.model, saving_directory=cfg.model_storage_path)
        if cfg.model in available_metrics
        else cfg.model
    )
    model = load_from_checkpoint(model_path)
    model.eval()

    if not cfg.disable_cache:
        model.set_embedding_cache()

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
        mean_scores, std_scores, sys_score = model.predict(
            data, cfg.batch_size, cfg.gpus, cfg.mc_dropout, 
            num_workers=cfg.num_workers, length_batching=cfg.disable_length_batching
        )
        for i, (mean, std, sample) in enumerate(zip(mean_scores, std_scores, data)):
            print("Segment {}\tscore: {:.4f}\tvariance: {:.4f}".format(i, mean, std))
            sample["COMET"] = mean
            sample["variance"] = std

    else:
        predictions, sys_score = model.predict(
            data, cfg.batch_size, cfg.gpus, 
            num_workers=cfg.num_workers, length_batching=cfg.disable_length_batching
        )
        for i, (score, sample) in enumerate(zip(predictions, data)):
            print("Segment {}\tscore: {:.4f}".format(i, score))
            sample["COMET"] = score

    print("System score: {:.4f}".format(sys_score))
    if isinstance(cfg.to_json, str):
        with open(cfg.to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))
