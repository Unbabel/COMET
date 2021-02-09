# -*- coding: utf-8 -*-
r"""
Ranking Metrics
==============
    Metrics to evaluate ranking quality of ranker models.
"""
import torch


class WMTKendall:
    def __init__(self):
        self.name = "kendall"

    def compute(
        self, distance_pos: torch.Tensor, distance_neg: torch.Tensor
    ) -> torch.Tensor:
        """Computes the level of concordance, discordance and the WMT kendall tau metric

        :param distance_pos: distance between the positive samples and the anchor/s
        :param distance_neg: distance between the negative samples and the anchor/s

        :return: Level of agreement, nº of positive sample closer to the anchor
        """
        concordance = torch.sum((distance_pos < distance_neg).float())
        discordance = torch.sum((distance_pos >= distance_neg).float())
        kendall = (concordance - discordance) / (concordance + discordance)
        return kendall
