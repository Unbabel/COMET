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
from typing import Dict, Optional

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from comet.encoders.base import Encoder


class BERTEncoder(Encoder):
    """BERT encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
    """

    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model, use_fast=True
        )
        if load_pretrained_weights:
            self.model = BertModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = BertModel(
                BertConfig.from_pretrained(pretrained_model), add_pooling_layer=False
            )
        self.model.encoder.output_hidden_states = True

    @property
    def output_units(self) -> int:
        """Max number of tokens the encoder handles."""
        return self.model.config.hidden_size

    @property
    def max_positions(self) -> int:
        """Max number of tokens the encoder handles."""
        return self.model.config.max_position_embeddings - 2

    @property
    def num_layers(self) -> int:
        """Number of model layers available."""
        return self.model.config.num_hidden_layers + 1

    @property
    def size_separator(self) -> int:
        """Size of the seperator.
        E.g: For BERT is just 1 ([SEP]) but models such as XLM-R use 2 (</s></s>).

        Returns:
            int: Number of tokens used between two segments.
        """
        return 1

    @property
    def uses_token_type_ids(self) -> bool:
        """Whether or not the model uses token type ids to differentiate sentences.

        Returns:
            bool: True for models that use token_type_ids such as BERT.
        """
        return True

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face
        Returns:
            Encoder: XLMREncoder object.
        """
        return BERTEncoder(pretrained_model, load_pretrained_weights)

    def freeze_embeddings(self) -> None:
        """Frezees the embedding layer."""
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def layerwise_lr(self, lr: float, decay: float):
        """Calculates the learning rate for each layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            list: List of model parameters for all layers and the corresponding lr.
        """
        # Last layer keeps LR
        opt_parameters = [
            {
                "params": self.model.encoder.layer[-1].parameters(),
                "lr": lr,
            }
        ]
        # Decay at each layer.
        for i in range(2, self.num_layers):
            opt_parameters.append(
                {
                    "params": self.model.encoder.layer[-i].parameters(),
                    "lr": lr * decay ** (i - 1),
                }
            )
        # Embedding Layer
        opt_parameters.append(
            {
                "params": self.model.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        )
        return opt_parameters

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """BERT model forward

        Args:
            input_ids (torch.Tensor): tokenized batch.
            attention_mask (torch.Tensor): batch attention mask.
            token_type_ids (Optional[torch.tensor]): batch attention mask. Defaults to
                None

        Returns:
            Dict[str, torch.Tensor]: dictionary with 'sentemb', 'wordemb', 'all_layers'
                and 'attention_mask'.
        """
        last_hidden_states, pooler_output, all_layers = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
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
