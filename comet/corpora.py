# -*- coding: utf-8 -*-
r"""
Corpora
==============
    Available corpora to train/test COMET models.
"""
import os

import click

from torchnlp.download import download_file_maybe_extract

corpus2download = {
    "apequest": "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/comet/hter/apequest.zip",
    "qt21": "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/comet/hter/qt21.zip",
    "wmt17": "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/comet/relative-ranks/wmt17.zip",
    "wmt18": "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/comet/relative-ranks/wmt18.zip",
    "wmt19": "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/comet/relative-ranks/wmt19.zip",
    "wmt": "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/comet/relative-ranks/wmt.zip",
}


def download_corpus(corpus: str, saving_directory: str = None) -> None:
    """ Function that downloads a corpus from AWS.
    :param corpus: Name of the corpus to be loaded.
    :param saving_directory: RELATIVE path to the saving folder.
    """
    corpus = corpus.lower()
    if not saving_directory:
        saving_directory = "data/"

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    if os.path.isdir(saving_directory + corpus):
        click.secho(f"{corpus} is already in cache.", fg="yellow")
        return

    elif corpus in corpus2download:
        download_file_maybe_extract(
            corpus2download[corpus], directory=saving_directory,
        )

    else:
        raise Exception(f"{corpus} is not a valid corpus!")

    click.secho("Download succeeded.", fg="yellow")
    if os.path.exists(saving_directory + corpus + ".zip"):
        os.remove(saving_directory + corpus + ".zip")

    elif os.path.exists(saving_directory + corpus + ".tar.gz"):
        os.remove(saving_directory + corpus + ".tar.gz")

    else:
        click.secho("Fail to delete compressed file.", fg="red")
