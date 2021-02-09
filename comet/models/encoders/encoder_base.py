# -*- coding: utf-8 -*-
r"""
Encoder Model base
====================
    Module defining the common interface between all pretrained encoder models.
"""
import warnings
from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn

from comet.tokenizers import TextEncoderBase


class Encoder(nn.Module):
    """Base class for an encoder model.

    :param output_units: Number of output features that will be passed to the Estimator.
    """

    def __init__(
        self,
        tokenizer: TextEncoderBase,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @property
    def output_units(self):
        """ Max number of tokens the encoder handles. """
        return self._output_units

    @property
    def max_positions(self):
        """ Max number of tokens the encoder handles. """
        return self._max_pos

    @property
    def num_layers(self):
        """ Number of model layers available. """
        return self._n_layers

    @property
    def lm_head(self):
        """ Language modeling head. """
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, hparams: Namespace):
        """Function that loads a pretrained encoder and the respective tokenizer.

        :return: Encoder model
        """
        raise NotImplementedError

    def check_lengths(self, tokens: torch.Tensor, lengths: torch.Tensor):
        """ Checks if lengths are not exceeded and warns user if so."""
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

    def prepare_sample(self, sample: List[str]) -> (torch.Tensor, torch.Tensor):
        """ Receives a list of strings and applies model specific tokenization and vectorization."""
        tokens, lengths = self.tokenizer.batch_encode(sample)
        tokens, lengths = self.check_lengths(tokens, lengths)
        return {"tokens": tokens, "lengths": lengths}

    def freeze(self) -> None:
        """ Frezees the entire encoder network. """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """ Unfrezees the entire encoder network. """
        for param in self.parameters():
            param.requires_grad = True

    def freeze_embeddings(self) -> None:
        """ Frezees the embedding layer of the network to save some memory while training. """
        raise NotImplementedError

    def layerwise_lr(self, lr: float, decay: float):
        """
        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        raise NotImplementedError

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a batch of sequences.

        :param tokens: Torch tensor with the input sequences [batch_size x seq_len].
        :param lengths: Torch tensor with the lenght of each sequence [seq_len].

        :return: Dictionary with `sentemb` (tensor with dims [batch_size x output_units]), `wordemb`
            (tensor with dims [batch_size x seq_len x output_units]), `mask` (input mask),
            `all_layers` (List with word_embeddings from all layers, `extra` (model specific outputs).
        """
        raise NotImplementedError
