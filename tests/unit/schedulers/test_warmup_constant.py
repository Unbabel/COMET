# -*- coding: utf-8 -*-
import unittest

import torch

from comet.schedulers import WarmupConstant
from test_tube import HyperOptArgumentParser


class TestWarmupConstantScheduler(unittest.TestCase):
    @property
    def optimizer(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        return torch.optim.Adam(params)

    @property
    def hparams(self):
        parser = HyperOptArgumentParser()
        parser = WarmupConstant.add_scheduler_specific_args(parser)
        return parser.parse_args()

    def test_scheduler_init(self):
        assert WarmupConstant.from_hparams(self.optimizer, self.hparams)
