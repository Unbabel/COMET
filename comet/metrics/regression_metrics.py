# -*- coding: utf-8 -*-
r"""
Regression Metrics
==============
    Metrics to evaluate regression quality of estimator models.
"""
import warnings

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr


class RegressionReport:
    def __init__(self):
        super().__init__()
        self.metrics = [Pearson(), Kendall(), Spearman()]

    def compute(self, x: np.array, y: np.array) -> float:
        """Computes Kendall correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        :return: Kendall Tau correlation value.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {metric.name: metric.compute(x, y) for metric in self.metrics}


class Kendall:
    def __init__(self):
        self.name = "kendall"

    def compute(self, x: np.array, y: np.array) -> float:
        """Computes Kendall correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        :return: Kendall Tau correlation value.
        """
        return torch.tensor(kendalltau(x, y)[0], dtype=torch.float32)


class Pearson:
    def __init__(self):
        self.name = "pearson"

    def compute(self, x: np.array, y: np.array) -> torch.Tensor:
        """Computes Pearson correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        :return: Pearson correlation value.
        """
        return torch.tensor(pearsonr(x, y)[0], dtype=torch.float32)


class Spearman:
    def __init__(self):
        self.name = "spearman"

    def compute(self, x: np.array, y: np.array) -> float:
        """Computes Spearman correlation.

        :param x: predicted scores.
        :param x: ground truth scores.

        Return:
            - Spearman correlation value.
        """
        return torch.tensor(spearmanr(x, y)[0], dtype=torch.float32)
