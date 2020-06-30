# -*- coding: utf-8 -*-
import unittest

import numpy as np

import torch
from comet.metrics import RegressionReport, WMTKendall


class TestMetrics(unittest.TestCase):
    def test_regression_report(self):
        report = RegressionReport()
        a = np.array([0, 0, 0, 1, 1, 1, 1])
        b = np.arange(7)
        expected = {"pearson": 0.8660254, "kendall": 0.7559289, "spearman": 0.866025}
        result = report(a, b)
        self.assertDictEqual(
            {k: round(v, 5) for k, v in result.items()},
            {k: round(v, 5) for k, v in expected.items()},
        )

    def test_wmt_kendall(self):
        metric = WMTKendall()

        pos = torch.tensor([0, 0.5, 1])
        neg = torch.tensor([1, 0.5, 0])

        expected = (1 - 2) / (1 + 2)
        self.assertEqual(metric(pos, neg), expected)
