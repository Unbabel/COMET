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
MiniLM Encoder
==============
    Pretrained MiniLM encoder from Microsoft. This encoder uses a BERT 
    architecture with an XLMR tokenizer.
"""
from transformers import BertConfig, BertModel, XLMRobertaTokenizerFast

from comet.encoders.xlmr import Encoder, XLMREncoder


class MiniLMEncoder(XLMREncoder):
    """MiniLMEncoder encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
    """

    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            "xlm-roberta-base", use_fast=True
        )
        if load_pretrained_weights:
            self.model = BertModel.from_pretrained(pretrained_model)
        else:
            self.model = BertModel(BertConfig.from_pretrained(pretrained_model))

        self.model.encoder.output_hidden_states = True

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
        return MiniLMEncoder(pretrained_model, load_pretrained_weights)
