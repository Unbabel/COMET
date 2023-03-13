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
RemBERT Encoder
===============
    Pretrained RemBERT encoder from Google. This encoder is similar to BERT but uses 
    sentencepiece like XLMR.
"""
from transformers import RemBertConfig, RemBertModel, RemBertTokenizerFast

from comet.encoders.xlmr import Encoder, XLMREncoder


class RemBERTEncoder(XLMREncoder):
    """RemBERT encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
    """

    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = RemBertTokenizerFast.from_pretrained(
            pretrained_model, use_fast=True
        )
        if load_pretrained_weights:
            self.model = RemBertModel.from_pretrained(pretrained_model)
        else:
            self.model = RemBertModel(RemBertConfig.from_pretrained(pretrained_model))

        self.model.encoder.output_hidden_states = True

    @property
    def size_separator(self):
        """Number of tokens used between two segments. For BERT is just 1 ([SEP])"""
        return 1

    @property
    def uses_token_type_ids(self):
        return True

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str): Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face

        Returns:
            Encoder: XLMRXLEncoder object.
        """
        return RemBERTEncoder(pretrained_model, load_pretrained_weights)
