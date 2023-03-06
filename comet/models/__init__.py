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

import yaml
from huggingface_hub import snapshot_download

from .base import CometModel
from .multitask.unified_metric import UnifiedMetric
from .ranking.ranking_metric import RankingMetric
from .regression.referenceless import ReferencelessRegression
from .regression.regression_metric import RegressionMetric

str2model = {
    "referenceless_regression_metric": ReferencelessRegression,
    "regression_metric": RegressionMetric,
    "ranking_metric": RankingMetric,
    "unified_metric": UnifiedMetric,
}

def download_model(
    model: str, 
    saving_directory: Union[str, Path, None] = None
) -> str:
    model_path = snapshot_download(repo_id=model, cache_dir=saving_directory)
    checkpoint_path = os.path.join(*[model_path, "checkpoints", "model.ckpt"])
    return checkpoint_path

def load_from_checkpoint(checkpoint_path: str) -> CometModel:
    """Loads models from a checkpoint path.

    Args:
        checkpoint_path (str): Path to a model checkpoint.

    Return:
        COMET model.
    """
    if not os.path.exists(checkpoint_path):
        raise Exception(f"Invalid checkpoint path: {checkpoint_path}")
    
    parent_folder = os.path.join(*os.path.normpath(checkpoint_path).split(os.path.sep)[:-2])
    hparams_file = parent_folder.split(os.path.sep) + [
        "hparams.yaml"
    ]
    # If the checkpoint starts with the root folder then we have an absolute path
    # and we need to add root before joining the folders
    if checkpoint_path.startswith(os.sep):
        hparams_file = [
            os.path.abspath(os.sep),
        ] + hparams_file
    hparams_file = os.path.join(*hparams_file)
    
    if os.path.exists(hparams_file):
        with open(hparams_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model_class = str2model[hparams["class_identifier"]]
        model = model_class.load_from_checkpoint(
            checkpoint_path, load_pretrained_weights=False
        )
        return model
    else:
        raise Exception(f"hparams.yaml file is missing from {parent_folder}!")
