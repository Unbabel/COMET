# -*- coding: utf-8 -*-
import unittest

import torch
from argparse import Namespace
from comet.schedulers import ConstantLR, WarmupConstant, LinearWarmup


def unwrap_schedule(optimizer, scheduler, num_steps=10):
    lrs = []
    for _ in range(num_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs


class TestSchedulers(unittest.TestCase):
    m = torch.nn.Linear(50, 50)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    num_steps = 10

    def test_constant_scheduler(self):
        scheduler = ConstantLR.from_hparams(self.optimizer, None)
        lrs = unwrap_schedule(self.optimizer, scheduler, self.num_steps)
        expected = [0.001] * self.num_steps
        self.assertListEqual(lrs, expected)

    def test_warmup_constant(self):
        hparams = Namespace(**{"warmup_steps": 4})
        scheduler = WarmupConstant.from_hparams(self.optimizer, hparams)
        lrs = unwrap_schedule(self.optimizer, scheduler, self.num_steps)
        expected = [
            0.00025,
            0.0005,
            0.00075,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
        ]
        self.assertListEqual(lrs, expected)

    def test_linear_warmup(self):
        hparams = Namespace(**{"warmup_steps": 2, "num_training_steps": 10})
        scheduler = LinearWarmup.from_hparams(
            self.optimizer, hparams, num_training_steps=10
        )
        lrs = unwrap_schedule(self.optimizer, scheduler, self.num_steps)
        expected = [
            0.0005,
            0.001,
            0.000875,
            0.00075,
            0.000625,
            0.0005,
            0.000375,
            0.00025,
            0.000125,
            0.0,
        ]
        self.assertListEqual(lrs, expected)
