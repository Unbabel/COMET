# -*- coding: utf-8 -*-
r"""
Feed Forward 
==============
    Simple Feed Forward Neural Network module that can be used for classification or regression
"""
import torch
from torch import nn

from comet.modules.activations import build_activation


class FeedForward(nn.Module):
    """
    Feed Forward Neural Network.

    :param in_dim: Number input features.
    :param out_dim: Number of output features. Default is just a score.
    :param hidden_sizes: list with the size of the hidden layers.
        This parameter can also be a string with the sizes splited by a comma.
    :param activations: Name of the activation function to be used in the hidden layers.
    :param final_activation: Name of the final activation function or False if we dont
        want a final activation.
    :param dropout: dropout to be used in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: str = "3072,1536,768",
        activations: str = "Sigmoid",
        final_activation: str = "Sigmoid",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if isinstance(hidden_sizes, str):
            hidden_sizes = [int(x) for x in hidden_sizes.split(",")]

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        activation_func = build_activation(activations)

        self.ff = torch.nn.Sequential()
        self.ff.add_module("linear_1", nn.Linear(in_dim, hidden_sizes[0]))
        self.ff.add_module("activation_1", activation_func)
        self.ff.add_module("dropout_1", nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            self.ff.add_module(
                "linear_{}".format(i + 1),
                nn.Linear(int(hidden_sizes[i - 1]), hidden_sizes[i]),
            )
            self.ff.add_module("activation_{}".format(i + 1), activation_func)
            self.ff.add_module("dropout_{}".format(i + 1), nn.Dropout(dropout))

        self.ff.add_module(
            "linear_{}".format(len(hidden_sizes) + 1),
            nn.Linear(hidden_sizes[-1], int(out_dim)),
        )
        if final_activation:
            final_activation = build_activation(final_activation)
            self.ff.add_module(
                "activation_{}".format(len(hidden_sizes) + 1), final_activation
            )

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)
