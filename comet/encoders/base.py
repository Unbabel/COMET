# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Encoder Model base
====================
    Module defining the common interface between all pretrained encoder models.
"""
import abc
from typing import Dict, List

import torch
import torch.nn as nn


class Encoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for an encoder model."""

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

    def prepare_sample(self, sample: List[str]) -> Dict[str, torch.Tensor]:
        """Receives a list of strings and applies tokenization and vectorization.

        :param sample: List with text segments to be tokenized and padded.

        :return: Dictionary with HF model inputs.
        """
        tokenizer_output = self.tokenizer(
            sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_positions - 2,
        )
        return tokenizer_output

    def freeze(self) -> None:
        """Frezees the entire encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfrezees the entire encoder."""
        for param in self.parameters():
            param.requires_grad = True

    @abc.abstractmethod
    def freeze_embeddings(self) -> None:
        """Frezees the embedding layer."""
        pass

    @abc.abstractmethod
    def layerwise_lr(self, lr: float, decay: float):
        """
        :param lr: Learning rate for the highest encoder layer.
        :param decay: decay percentage for the lower layers.

        :return: List of model parameters with layer-wise decay learning rate
        """
        pass

    @abc.abstractmethod
    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pass
