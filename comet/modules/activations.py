# -*- coding: utf-8 -*-
r"""
Activation Functions 
==============
    Provides an easy to use interface to initialize different activation functions.
"""
import torch
from torch import nn


def build_activation(activation: str) -> nn.Module:
    """Builder function that returns a nn.module activation function.

    :param activation: string defining the name of the activation function.

    Activations available:
        Swish + every native pytorch activation function.
    """
    if hasattr(nn, activation):
        return getattr(nn, activation)()
    elif activation == "Swish":
        return Swish()
    else:
        raise Exception("{} invalid activation function.".format(activation))


def swish(input: torch.Tensor) -> torch.Tensor:
    """
    Applies Swish element-wise: A self-gated activation function
        swish(x) = x * sigmoid(x)
    """
    return input * torch.sigmoid(input)


class Swish(nn.Module):
    """
    Applies the Swish function element-wise:

        Swish(x) = x * sigmoid(x)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        -  Related paper:
        https://arxiv.org/pdf/1710.05941v1.pdf
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.
        """
        return swish(input)
