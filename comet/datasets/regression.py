# -*- coding: utf-8 -*-
r"""
Regression Dataset
==============
    Regression Datasets are composed by source, MT and reference sentences as long as a 
    quality score (between 0 and 1) for the MT.
"""
import pandas as pd

from argparse import Namespace


def load_data(path: str):
    df = pd.read_csv(path)
    df = df[["src", "mt", "ref", "score"]]
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)
    df["ref"] = df["ref"].astype(str)
    df["score"] = df["score"].astype(float)
    return df.to_dict("records")


def regression_dataset(hparams: Namespace, train=True, val=True, test=True):
    """
    :param hparams: Namespace containg the path to the data files.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """
    func_out = []
    if train:
        func_out.append(load_data(hparams.train_path))
    if val:
        func_out.append(load_data(hparams.val_path))
    if test:
        func_out.append(load_data(hparams.test_path))

    return tuple(func_out)
