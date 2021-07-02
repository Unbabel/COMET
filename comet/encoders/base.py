# -*- coding: utf-8 -*-
r"""
Encoder Model base
====================
    Module defining the common interface between all pretrained encoder models.
"""
import abc
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for an encoder model.

    :param output_units: Number of output features that will be passed to the Estimator.
    """

    @property
    @abc.abstractmethod
    def output_units(self):
        """Max number of tokens the encoder handles."""
        pass

    @property
    @abc.abstractmethod
    def max_positions(self):
        """Max number of tokens the encoder handles."""
        pass

    @property
    @abc.abstractmethod
    def num_layers(self):
        """Number of model layers available."""
        pass

    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls, pretrained_model):
        """Function that loads a pretrained encoder and the respective tokenizer.

        :return: Encoder model
        """
        raise NotImplementedError

    def check_lengths(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Checks if lengths are not exceeded and warns user if so."""
        if lengths.max() > self.max_positions:
            warnings.warn(
                "Encoder max length exceeded ({} > {}).".format(
                    lengths.max(), self.max_positions
                ),
                category=RuntimeWarning,
            )
            lengths[lengths > self.max_positions] = self.max_positions
            tokens = tokens[:, : self.max_positions]

        return tokens, lengths

    def prepare_sample(self, sample: List[str]) -> Dict[str, torch.Tensor]:
        """Receives a list of strings and applies model specific tokenization and vectorization."""
        tokenizer_output = self.tokenizer(sample, return_tensors="pt", padding=True)
        return tokenizer_output

    def freeze(self) -> None:
        """Frezees the entire encoder network."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfrezees the entire encoder network."""
        for param in self.parameters():
            param.requires_grad = True

    @abc.abstractmethod
    def freeze_embeddings(self) -> None:
        """Frezees the embedding layer of the network to save some memory while training."""
        pass

    @abc.abstractmethod
    def layerwise_lr(self, lr: float, decay: float):
        """
        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        pass

    @abc.abstractmethod
    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pass
