# -*- coding: utf-8 -*-
r"""
Layer-Wise Attention Mechanism
================================
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.

    If `do_layer_norm=True` then apply layer normalization to each tensor before weighting.

    If `dropout > 0`, then for each scalar weight, adjust its softmax weight mass to 0 with
    the dropout probability (i.e., setting the unnormalized weight to -inf). This effectively
    should redistribute dropped probability mass to all other weights.

    Original implementation:
        - https://github.com/Hyperparticle/udify
"""
from typing import List

import torch
from torch.nn import Parameter, ParameterList


class ScalarMixWithDropout(torch.nn.Module):
    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: list = None,
        trainable: bool = True,
        dropout: float = None,
        dropout_value: float = -1e20,
    ) -> None:
        super(ScalarMixWithDropout, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.dropout = dropout

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise Exception(
                "Length of initial_scalar_parameters {} differs \
                from mixture_size {}".format(
                    initial_scalar_parameters, mixture_size
                )
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]),
                    requires_grad=trainable,
                )
                for i in range(mixture_size)
            ]
        )

        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(dropout_value)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(
        self,
        tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.
        When `do_layer_norm=False` the `mask` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise Exception(
                "{} tensors were passed, but the module was initialized to \
                mix {} tensors.".format(
                    len(tensors), self.mixture_size
                )
            )

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2)
                / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        weights = torch.cat([parameter for parameter in self.scalar_parameters])

        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout, weights, self.dropout_fill
            )

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight
                    * _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )
            return self.gamma * sum(pieces)
