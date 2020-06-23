# -*- coding: utf-8 -*-
import os
import pickle

import click
import pandas as pd

import wget
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

MODELS_URL = "https://unbabel-experimental-models.s3.amazonaws.com/comet/share/model2download.pkl" 


def model2download(saving_directory: str) -> dict:
    """ Download a dictionary with the mapping between models and downloading urls.
    :param saving_directory: RELATIVE path to the saving folder (must end with /).
    Return:
        - dictionary with the mapping between models and downloading urls.
    """
    if os.path.exists(saving_directory + "model2download.pkl"):
        os.remove(saving_directory + "model2download.pkl")
    
    wget.download(MODELS_URL, saving_directory + "model2download.pkl")
    with open(saving_directory + "model2download.pkl", "rb") as handle:
        return pickle.load(handle)


def download_model(comet_model: str, saving_directory: str = None) -> ModelBase:
    """ Function that loads pretrained models from AWS.
    :param comet_model: Name of the model to be loaded.
    :param saving_directory: RELATIVE path to the saving folder (must end with /).
    
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

    models = model2download(saving_directory)

    if os.path.isdir(saving_directory + comet_model):
        click.secho(f"{comet_model} is already in cache.", fg="yellow")

    elif comet_model in models:
        download_file_maybe_extract(models[comet_model], directory=saving_directory)

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
