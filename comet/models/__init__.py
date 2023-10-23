# flake8: noqa
# -*- coding: utf-8 -*-
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
import os
from pathlib import Path
from typing import Union

import torch
import yaml
from huggingface_hub import snapshot_download

from .base import CometModel
from .multitask.unified_metric import UnifiedMetric
from .multitask.xcomet_metric import XCOMETMetric
from .ranking.ranking_metric import RankingMetric
from .regression.referenceless import ReferencelessRegression
from .regression.regression_metric import RegressionMetric
from .download_utils import download_model_legacy


str2model = {
    "referenceless_regression_metric": ReferencelessRegression,
    "regression_metric": RegressionMetric,
    "ranking_metric": RankingMetric,
    "unified_metric": UnifiedMetric,
    "xcomet_metric": XCOMETMetric,
}


def download_model(
    model: str,
    saving_directory: Union[str, Path, None] = None,
    local_files_only: bool = False,
) -> str:
    try:
        model_path = snapshot_download(
            repo_id=model, cache_dir=saving_directory, local_files_only=local_files_only
        )
    except Exception:
        try:
            checkpoint_path = download_model_legacy(model, saving_directory)
        except Exception:
            raise KeyError(f"Model '{model}' not supported by COMET.")
    else:
        checkpoint_path = os.path.join(*[model_path, "checkpoints", "model.ckpt"])
    return checkpoint_path


def load_from_checkpoint(
    checkpoint_path: str, reload_hparams: bool = False, strict: bool = False
) -> CometModel:
    """Loads models from a checkpoint path.

    Args:
        checkpoint_path (str): Path to a model checkpoint.
        reload_hparams (bool): hparams.yaml file located in the parent folder is
            only use for deciding the `class_identifier`. By setting this flag
            to True all hparams will be reloaded.
        strict (bool): Strictly enforce that the keys in checkpoint_path match the
            keys returned by this module's state dict. Defaults to False
    Return:
        COMET model.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise Exception(f"Invalid checkpoint path: {checkpoint_path}")

    parent_folder = checkpoint_path.parents[1]  # .parent.parent
    hparams_file = parent_folder / "hparams.yaml"

    if hparams_file.is_file():
        with open(hparams_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model_class = str2model[hparams["class_identifier"]]
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            load_pretrained_weights=False,
            hparams_file=hparams_file if reload_hparams else None,
            map_location=torch.device("cpu"),
            strict=strict,
        )
        return model
    else:
        raise Exception(f"hparams.yaml file is missing from {parent_folder}!")
