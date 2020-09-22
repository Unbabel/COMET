# -*- coding: utf-8 -*-
r"""
Regression Metrics
==============
    Metrics to evaluate regression quality of estimator models.
"""
import warnings

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

import torch
from pytorch_lightning.metrics import Metric


class RegressionReport(Metric):
    def __init__(self):
        super().__init__(name="regression_report")
        self.metrics = [Pearson(), Kendall(), Spearman()]

    def forward(self, x: np.array, y: np.array) -> float:
        """Computes Kendall correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        :return: Kendall Tau correlation value.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {metric.name: metric(x, y) for metric in self.metrics}


class Kendall(Metric):
    def __init__(self):
        super().__init__(name="kendall")

    def forward(self, x: np.array, y: np.array) -> float:
        """Computes Kendall correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        :return: Kendall Tau correlation value.
        """
        return torch.tensor(kendalltau(x, y)[0], dtype=torch.float32)


class Pearson(Metric):
    def __init__(self):
        super().__init__(name="pearson")

    def forward(self, x: np.array, y: np.array) -> torch.Tensor:
        """Computes Pearson correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        :return: Pearson correlation value.
        """
        return torch.tensor(pearsonr(x, y)[0], dtype=torch.float32)


class Spearman(Metric):
    def __init__(self):
        super().__init__(name="spearman")

    def forward(self, x: np.array, y: np.array) -> float:
        """Computes Spearman correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        Return:
            - Spearman correlation value.
        """
        return torch.tensor(spearmanr(x, y)[0], dtype=torch.float32)
