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
    def output_units(self) -> int:
        """Max number of tokens the encoder handles."""
        pass

    @property
    @abc.abstractmethod
    def max_positions(self) -> int:
        """Max number of tokens the encoder handles."""
        pass

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        """Number of model layers available."""
        pass

    @property
    @abc.abstractmethod
    def size_separator(self) -> int:
        """Size of the seperator.
        E.g: For BERT is just 1 ([SEP]) but models such as XLM-R use 2 (</s></s>).

        Returns:
            int: Number of tokens used between two segments.
        """
        pass

    @property
    @abc.abstractmethod
    def uses_token_type_ids(self) -> bool:
        """Whether or not the model uses token type ids to differentiate sentences.

        Returns:
            bool: True for models that use token_type_ids such as BERT.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls, pretrained_model: str):
        """Function that loads a pretrained encoder and the respective tokenizer.

        Returns:
            Encoder: Pretrained model from Hugging Face.
        """
        raise NotImplementedError

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
        """Calculates the learning rate for each layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            list: List of model parameters for all layers and the corresponding lr.
        """
        pass

    @abc.abstractmethod
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Encoder model forward

        Args:
            input_ids (torch.Tensor): tokenized batch.
            attention_mask (torch.Tensor): batch attention mask.

        Returns:
            Dict[str, torch.Tensor]: dictionary with 'sentemb', 'wordemb', 'all_layers'
                and 'attention_mask'.
        """
        pass

    @abc.abstractmethod
    def subword_tokenize_to_ids(self, tokens: List[str]) -> Dict[str, torch.Tensor]:
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.

        Args:
            tokens (List[str]): A sequence of strings, representing input tokens.

        Returns:
            Dict[str, torch.Tensor]: dict with 'input_ids', 'attention_mask',
                'subword_mask'
        """
        pass

    def prepare_sample(
        self, sample: List[str], word_level_training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Receives a list of strings and applies tokenization and vectorization.

        Args:
            sample (List[str]): List with text segments to be tokenized and padded.
            word_level_training (bool, optional): True for subword tokenization.
                Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: dict with 'input_ids', 'attention_mask' and
                'subword_mask' if word_level_training=True
        """
        if word_level_training:
            tokenizer_output = self.subword_tokenize_to_ids(sample)
            return tokenizer_output
        else:
            tokenizer_output = self.tokenizer(
                sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_positions - 2,
            )
            return tokenizer_output

    def pad_tensor(
        self, tensor: torch.Tensor, length: torch.Tensor, padding_index: int
    ) -> torch.Tensor:
        """Pad a tensor to length with padding_index.

        Args:
            tensor (torch.Tensor): Tensor to pad.
            length (torch.Tensor): Sequence length after padding.
            padding_index (int): Index to pad tensor with.

        Returns:
            torch.Tensor: Input batch
        """
        n_padding = length - tensor.shape[0]
        assert n_padding >= 0
        if n_padding == 0:
            return tensor
        padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
        return torch.cat((tensor, padding), dim=0)
