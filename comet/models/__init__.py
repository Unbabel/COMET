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

import yaml

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

available_metrics = {
    # ------------------------------------  LEGACY MODELS ------------------------------#
    # WMT20 Models
    "emnlp20-comet-rank": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/emnlp20-comet-rank.tar.gz",
    "wmt20-comet-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/wmt20-comet-da.tar.gz",
    "wmt20-comet-qe-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/wmt20-comet-qe-da.tar.gz",
    "wmt20-comet-qe-da-v2": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/wmt20-comet-qe-da-v2.tar.gz",
    # WMT21 Models
    "wmt21-comet-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-da.tar.gz",
    "wmt21-comet-mqm": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-mqm.tar.gz",
    "wmt21-cometinho-mqm": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-cometinho-mqm.tar.gz",
    "wmt21-cometinho-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-cometinho-da.tar.gz",
    "wmt21-comet-qe-mqm": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-mqm.tar.gz",
    "wmt21-comet-qe-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-da.tar.gz",
    # EAMT22 Models
    "eamt22-cometinho-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-cometinho-da.tar.gz",
    "eamt22-prune-comet-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-prune-comet-da.tar.gz",
    # ------------------------------------  NEW MODELS ------------------------------#
    # WMT22 Models
    "wmt22-comet-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt22/wmt22-comet-da.tar.gz",
    "wmt22-cometkiwi-da": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt22/wmt22-cometkiwi-da.tar.gz",
    "wmt22-seqtag-mqm": "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt22/wmt22-seqtag-mqm.tar.gz",
}


def load_from_checkpoint(checkpoint_path: str) -> CometModel:
    """Loads models from a checkpoint path.

    Args:
        checkpoint_path (str): Path to a model checkpoint.

    Return:
        COMET model.
    """
    if not os.path.exists(checkpoint_path):
        raise Exception(f"Invalid checkpoint path: {checkpoint_path}")

    hparams_file = os.path.normpath(checkpoint_path).split(os.path.sep)[:-2] + [
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
        model = model_class.load_from_checkpoint(checkpoint_path, **hparams)
        return model
    else:
        raise Exception(f"Cannot find hparams.yaml file in {hparams_file}!")
