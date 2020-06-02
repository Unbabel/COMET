# -*- coding: utf-8 -*-
import unittest

import torch

from comet.optimizers import Adamax
from test_tube import HyperOptArgumentParser


class TestAdamaxOptimizer(unittest.TestCase):
    @property
    def parameters(self):
        return [torch.nn.Parameter(torch.randn(2, 3, 4))]

    @property
    def hparams(self):
        parser = HyperOptArgumentParser()
        parser = Adamax.add_optim_specific_args(parser)
        return parser.parse_args()

    def test_from_hparams(self):
        assert Adamax.from_hparams(self.parameters, self.hparams)
