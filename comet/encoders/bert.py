# -*- coding: utf-8 -*-
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
BERT Encoder
==============
    Pretrained BERT encoder from Hugging Face.
"""
from typing import Dict

import torch
from comet.encoders.base import Encoder
from transformers import AutoModel, AutoTokenizer


class BERTEncoder(Encoder):
    """BERT encoder.

    :param pretrained_model: Pretrained model from hugging face.
    """

    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.model.encoder.output_hidden_states = True

    @property
    def output_units(self):
        """Max number of tokens the encoder handles."""
        return self.model.config.hidden_size

    @property
    def max_positions(self):
        """Max number of tokens the encoder handles."""
        return self.model.config.max_position_embeddings

    @property
    def num_layers(self):
        """Number of model layers available."""
        return self.model.config.num_hidden_layers + 1

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param pretrained_model: Name of the pretrain model to be loaded.

        :return: Encoder model
        """
        return BERTEncoder(pretrained_model)

    def freeze_embeddings(self) -> None:
        """Frezees the embedding layer."""
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def layerwise_lr(self, lr: float, decay: float):
        """
        :param lr: Learning rate for the highest encoder layer.
        :param decay: decay percentage for the lower layers.

        :return: List of model parameters with layer-wise decay learning rate
        """
        # Embedding Layer
        opt_parameters = [
            {
                "params": self.model.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        # All layers
        opt_parameters += [
            {
                "params": self.model.encoder.layer[i].parameters(),
                "lr": lr * decay ** i,
            }
            for i in range(self.num_layers - 2, 0, -1)
        ]
        return opt_parameters

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        last_hidden_states, pooler_output, all_layers = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "sentemb": pooler_output,
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }
