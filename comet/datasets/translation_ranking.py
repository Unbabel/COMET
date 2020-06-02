# -*- coding: utf-8 -*-
r"""
Translation Ranking Dataset
==============
    Translation Ranking Datasets are composed by source, reference, a postive hypothesis 
    and a negative hypothesis.
"""
import pandas as pd

from argparse import Namespace


def load_data(path: str):
    df = pd.read_csv(path)
    df = df[["src", "ref", "pos", "neg"]]
    df["src"] = df["src"].astype(str)
    df["ref"] = df["ref"].astype(str)
    df["pos"] = df["pos"].astype(str)
    df["neg"] = df["neg"].astype(str)
    return df.to_dict("records")


def ranking_dataset(hparams: Namespace, train=True, val=True, test=True):
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
