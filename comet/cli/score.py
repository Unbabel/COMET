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
import multiprocessing
from typing import Union

import torch
from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader


def score_command() -> None:
    parser = ArgumentParser(description="Command for scoring MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-t", "--translations", type=Path_fr, required=True)
    parser.add_argument("-r", "--references", type=Path_fr, required=True)
    parser.add_argument("--to_json", type=Union[bool, str], default=False)
    parser.add_argument(
        "--model",
        type=Union[str, Path_fr],
        required=False,
        default="wmt21-large-estimator",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    cfg = parser.parse_args()

    model_path = (
        download_model(cfg.model) if cfg.model in available_metrics else cfg.model
    )
    model = load_from_checkpoint(model_path)
    model.eval()

    with open(cfg.sources()) as fp:
        sources = fp.readlines()

    with open(cfg.translations()) as fp:
        translations = fp.readlines()

    with open(cfg.references()) as fp:
        references = fp.readlines()

    data = {"src": sources, "mt": translations, "ref": references}
    data = [dict(zip(data, t)) for t in zip(*data.values())]

    dataloader = DataLoader(
        dataset=data,
        batch_size=cfg.batch_size,
        collate_fn=lambda x: model.prepare_sample(x, inference=True),
        num_workers=multiprocessing.cpu_count(),
    )
    trainer = Trainer(gpus=cfg.gpus, deterministic=True)
    predictions = trainer.predict(
        model, dataloaders=dataloader, return_predictions=True
    )
    predictions = torch.cat(predictions, dim=0).tolist()

    if isinstance(cfg.to_json, str):
        with open(cfg.to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))

    for i in range(len(predictions)):
        print("Segment {} score: {:.3f}".format(i, predictions[i]))

    print("System score: {:.3f}".format(sum(predictions) / len(predictions)))
