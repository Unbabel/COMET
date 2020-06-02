# -*- coding: utf-8 -*-
import os

import click
import pandas as pd

from torchnlp.download import download_file_maybe_extract

from .estimators import CometEstimator, MetricEstimator
from .model_base import ModelBase
from .ranking import CometRanker, MetricRanker

str2model = {
    "CometEstimator": CometEstimator,
    "CometRanker": CometRanker,
    # Model that use reference only:
    "MetricEstimator": MetricEstimator,
    "MetricRanker": MetricRanker,
}

model2download = {
    "da-ranker-v1.0": "https://unbabel-experimental-models.s3.amazonaws.com/comet/share/da-ranker-v1.0.zip",
    "hter-estimator-v1.0": "https://unbabel-experimental-models.s3.amazonaws.com/comet/share/hter-estimator-v1.0.zip",
}


def download_model(comet_model: str, saving_directory: str = None) -> ModelBase:
    """ Function that loads pretrained models from AWS.
    :param comet_model: Name of the model to be loaded.
    :param saving_directory: RELATIVE path to the saving folder.
    
    Return:
        - Pretrained model.
    """
    if saving_directory is None:
        if "HOME" in os.environ:
            saving_directory = os.environ["HOME"] + "/.cache/torch/unbabel_comet/"
        else:
            raise Exception("HOME environment variable is not defined.")

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    if os.path.isdir(saving_directory + comet_model):
        click.secho(f"{comet_model} is already in cache.", fg="yellow")

    elif comet_model in model2download:
        download_file_maybe_extract(
            model2download[comet_model], directory=saving_directory,
        )

    else:
        raise Exception(f"{comet_model} is not a valid COMET model!")

    click.secho("Download succeeded. Loading model...", fg="yellow")

    if os.path.exists(saving_directory + comet_model + ".zip"):
        os.remove(saving_directory + comet_model + ".zip")

    elif os.path.exists(saving_directory + comet_model + ".tar.gz"):
        os.remove(saving_directory + comet_model + ".tar.gz")

    experiment_folder = saving_directory + comet_model
    checkpoints = [
        file for file in os.listdir(experiment_folder) if file.endswith(".ckpt")
    ]
    checkpoint = checkpoints[-1]
    checkpoint_path = experiment_folder + "/" + checkpoint
    return load_checkpoint(checkpoint_path)


def load_checkpoint(checkpoint: str) -> ModelBase:
    """ Function that loads a model from a checkpoint file.
    :param checkpoint: Path to the checkpoint file.
    
    Returns:
        - COMET Model
    """
    tags_csv_file = "/".join(checkpoint.split("/")[:-1] + ["meta_tags.csv"])

    if not os.path.exists(tags_csv_file):
        raise Exception("meta_tags.csv file is missing from checkpoint folder.")

    tags = pd.read_csv(tags_csv_file, header=None, index_col=0, squeeze=True).to_dict()
    model = str2model[tags["model"]].load_from_checkpoint(
        checkpoint, tags_csv=tags_csv_file
    )
    model.eval()
    model.freeze()
    return model
