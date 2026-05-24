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
XLM-RoBERTa Encoder
==============
    Pretrained XLM-RoBERTa  encoder from Hugging Face.
"""

from typing import Dict

import importlib_metadata
import packaging.version as packaging_version
import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

transformers_version = importlib_metadata.distribution('transformers').version
if packaging_version.parse(transformers_version) >= packaging_version.parse(
    'v5.0.0rc0'
):
    from transformers import XLMRobertaTokenizer as XLMRobertaTokenizer
else:
    from transformers import XLMRobertaTokenizerFast as XLMRobertaTokenizer

from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder


class XLMREncoder(BERTEncoder):
    """XLM-RoBERTA Encoder encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
        local_files_only (bool): Whether or not to only look at local files.
    """

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            pretrained_model, local_files_only=local_files_only
        )
        if load_pretrained_weights:
            self.model = XLMRobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = XLMRobertaModel(
                XLMRobertaConfig.from_pretrained(
                    pretrained_model, local_files_only=local_files_only
                ),
                add_pooling_layer=False,
            )
        self.model.encoder.output_hidden_states = True

    @property
    def size_separator(self):
        """Number of tokens used between two segments. For BERT is just 1 ([SEP])
        but models such as XLM-R use 2 (</s></s>)"""
        return 2

    @property
    def uses_token_type_ids(self):
        return False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face
            local_files_only (bool): Whether or not to only look at local files.

        Returns:
            Encoder: XLMREncoder object.
        """
        return XLMREncoder(
            pretrained_model, load_pretrained_weights, local_files_only
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )

        if len(output) < 3:
            last_hidden_states, all_layers = output
        else:
            last_hidden_states, _, all_layers = output

        return {
            'sentemb': last_hidden_states[:, 0, :],
            'wordemb': last_hidden_states,
            'all_layers': all_layers,
            'attention_mask': attention_mask,
        }

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.tokenizer.cls_token_id]
        sep = [self.tokenizer.sep_token_id]

        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep
