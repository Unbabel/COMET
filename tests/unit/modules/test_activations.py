import unittest

import numpy as np
import torch
from torch import nn

from comet.modules.activations import Swish, build_activation


class TestActivations(unittest.TestCase):
    def test_build_activation(self):
        activation = build_activation("ReLU")
        self.assertIsInstance(activation, nn.ReLU)
        activation = build_activation("Tanh")
        self.assertIsInstance(activation, nn.Tanh)
        activation = build_activation("Swish")
        self.assertIsInstance(activation, Swish)

    def test_build_activation_with_nonexistent_function(self):
        with self.assertRaises(Exception) as context:
            build_activation("BadActivation")
        self.assertEqual(
            str(context.exception), "BadActivation invalid activation function."
        )

    def test_swish_result(self):
        input = torch.tensor([-1.0, 2.0, 3.0])
        activation = build_activation("Swish")
        result = activation(input)
        result = torch.round(result * 10 ** 4) / (10 ** 4)  # round to 4 decimals
        expected = torch.tensor(
            [
                -1 * (1 / (1 + np.exp(1))),
                2.0 * (1 / (1 + np.exp(-2))),
                3.0 * (1 / (1 + np.exp(-3))),
            ]
        )
        expected = (
            torch.round(expected * 10 ** 4) / (10 ** 4)
        ).float()  # round to 4 decimals
        self.assertTrue(torch.equal(expected, result))
