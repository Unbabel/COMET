# -*- coding: utf-8 -*-
import unittest

import torch

from comet.schedulers import LinearWarmup
from test_tube import HyperOptArgumentParser


class TestLinearWarmupScheduler(unittest.TestCase):
    @property
    def optimizer(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        return torch.optim.Adam(params)

    @property
    def hparams(self):
        parser = HyperOptArgumentParser()
        parser = LinearWarmup.add_scheduler_specific_args(parser)
        return parser.parse_args()

    def test_scheduler_init(self):
        assert LinearWarmup.from_hparams(self.optimizer, self.hparams)
