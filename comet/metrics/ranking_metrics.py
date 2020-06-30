# -*- coding: utf-8 -*-
r"""
Ranking Metrics
==============
    Metrics to evaluate ranking quality of ranker models.
"""
from typing import Tuple

import torch

from pytorch_lightning.metrics import Metric


class WMTKendall(Metric):
    def __init__(self):
        super().__init__(name="kendall")

    def forward(
        self, distance_pos: torch.Tensor, distance_neg: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """ Computes the level of concordance, discordance and the WMT kendall tau metric
        
        :param distance_pos: distance between the positive samples and the anchor (aka reference)
        :param distance_neg: distance between the negative samples and the anchor (aka reference)

        Return:
            - Level of agreement
            - Number of times the positive sample is closer to the anchor.
            - Number of times the negative sample is closer to the anchor.
        """
        concordance = torch.sum((distance_pos < distance_neg).float())
        discordance = torch.sum((distance_pos >= distance_neg).float())
        kendall = (concordance - discordance) / (concordance + discordance)
        return kendall
