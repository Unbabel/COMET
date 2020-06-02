# -*- coding: utf-8 -*-
r"""
Metric Ranking Model (MetricRanker)
==============
    The goal of this model is to rank good translations closer to the reference text
    and bad translations further by a small margin.

    https://pytorch.org/docs/stable/nn.html#tripletmarginloss
"""
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch

from comet.models.ranking.ranking_base import RankingBase
from torchnlp.utils import collate_tensors


class MetricRanker(RankingBase):
    """
    Metric Ranking Model class that uses a pretrained encoder 
    to extract features from the sequences and then passes those features through a 
    Triplet Margin Loss.

    :param hparams: Namespace containing the hyperparameters.
    """

    def __init__(self, hparams: Namespace,) -> None:
        super().__init__(hparams)

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[Tuple[Dict[str, torch.Tensor], None], List[Dict[str, torch.Tensor]]]:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        :param inference: If set to to False, then the model expects 
            a MT and reference instead of anchor, pos, and neg segments.

        Returns:
            - Tuple with a dictionary containing the model inputs and None.
        or
            - List with source, MT and reference tokenized and vectorized.
        """
        sample = collate_tensors(sample)
        if inference:
            mt_inputs = self.encoder.prepare_sample(sample["mt"])
            ref_inputs = self.encoder.prepare_sample(sample["ref"])
            return mt_inputs, ref_inputs

        ref_inputs = self.encoder.prepare_sample(sample["ref"])
        pos_inputs = self.encoder.prepare_sample(sample["pos"])
        neg_inputs = self.encoder.prepare_sample(sample["neg"])

        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        pos_inputs = {"pos_" + k: v for k, v in pos_inputs.items()}
        neg_inputs = {"neg_" + k: v for k, v in neg_inputs.items()}

        return {**ref_inputs, **pos_inputs, **neg_inputs}, None
