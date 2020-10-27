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
        expected = {
            "pearson": torch.tensor(0.8660254, dtype=torch.float32),
            "kendall": torch.tensor(0.7559289, dtype=torch.float32),
            "spearman": torch.tensor(0.866025, dtype=torch.float32),
        }
        result = report.compute(a, b)
        self.assertDictEqual(
            {k: round(v.item(), 4) for k, v in result.items()},
            {k: round(v.item(), 4) for k, v in expected.items()},
        )

    def test_wmt_kendall(self):
        metric = WMTKendall()

        pos = torch.tensor([0, 0.5, 1])
        neg = torch.tensor([1, 0.5, 0])

        expected = (1 - 2) / (1 + 2)

        self.assertEqual(metric.compute(pos, neg), expected)
