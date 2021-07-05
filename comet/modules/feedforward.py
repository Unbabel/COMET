# -*- coding: utf-8 -*-
r"""
Feed Forward 
==============
    Feed Forward Neural Network module that can be used for classification or regression
"""
import torch
from torch import nn
from typing import List, Optional


class FeedForward(nn.Module):
    """
    Feed Forward Neural Network.

    :param in_dim: Number input features.
    :param out_dim: Number of output features. Default is just a score.
    :param hidden_sizes: List with hidden layer sizes.
    :param activations: Name of the activation function to be used in the hidden layers.
    :param final_activation: Name of the final activation function if any.
    :param dropout: dropout to be used in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: List[int] = [3072, 768],
        activations: str = "Sigmoid",
        final_activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))
        modules.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self.build_activation(final_activation))

        self.ff = nn.Sequential(*modules)

    def build_activation(self, activation: str) -> nn.Module:
        if hasattr(nn, activation):
            return getattr(nn, activation)()

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)
