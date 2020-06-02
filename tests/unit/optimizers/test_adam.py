# -*- coding: utf-8 -*-
import unittest

import torch

from comet.optimizers import Adam
from test_tube import HyperOptArgumentParser


class TestAdamOptimizer(unittest.TestCase):
    @property
    def parameters(self):
        return [torch.nn.Parameter(torch.randn(2, 3, 4))]

    @property
    def hparams(self):
        parser = HyperOptArgumentParser()
        parser = Adam.add_optim_specific_args(parser)
        return parser.parse_args()

    def test_from_hparams(self):
        assert Adam.from_hparams(self.parameters, self.hparams)
