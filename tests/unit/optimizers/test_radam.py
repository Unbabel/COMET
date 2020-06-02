# -*- coding: utf-8 -*-
import unittest

import torch

from comet.optimizers import RAdam
from test_tube import HyperOptArgumentParser


class TestRAdamOptimizer(unittest.TestCase):
    @property
    def parameters(self):
        return [torch.nn.Parameter(torch.randn(2, 3, 4))]

    @property
    def hparams(self):
        parser = HyperOptArgumentParser()
        parser = RAdam.add_optim_specific_args(parser)
        return parser.parse_args()

    def test_from_hparams(self):
        assert RAdam.from_hparams(self.parameters, self.hparams)
