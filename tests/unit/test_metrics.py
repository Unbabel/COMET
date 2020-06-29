# -*- coding: utf-8 -*-
import unittest

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

from comet.metrics import RegressionReport, WMTKendall


class TestMetrics(unittest.TestCase):

    def test_regression_report(self):
        report = RegressionReport()
        a = np.array([0, 0, 0, 1, 1, 1, 1])
        b = np.arange(7)
        expected = {
            "pearson": 0.8660254037844386,
            "kendall": 0.7559289460184545,
            "spearman": 0.8660254037844388
        }
        result = report(a, b)
        self.assertDictEqual(result, expected)

    def test_wmt_kendall(self):
        metric = WMTKendall()
        
        pos = torch.tensor([0, 0.5, 1])
        neg = torch.tensor([1, 0.5, 0])

        expected = (1 - 2) / (1 + 2)
        self.assertEqual(metric(pos, neg), expected)
