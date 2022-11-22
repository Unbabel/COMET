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
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder


class XLMREncoder(BERTEncoder):
    """XLM-RoBERTA Encoder encoder.

    :param pretrained_model: Pretrained model from hugging face.
    """

    def __init__(self, pretrained_model: str) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model)
        self.model = XLMRobertaModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
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
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param pretrained_model: Name of the pretrain model to be loaded.

        :return: Encoder model
        """
        return XLMREncoder(pretrained_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        last_hidden_states, _, all_layers = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }

    def is_word_piece(self, subword: str) -> bool:
        """Whether or not a string is a word piece

        Returns:
            bool: True for wordpieces (▁cos).
        """
        return not subword.startswith("▁")

    def convert_tokens_to_ids(self, tokens, seg_lengths, pad=True):
        token_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
        ids = torch.tensor(token_ids)
        assert ids.size(1) <= self.max_positions
        if pad:
            attention_mask = torch.zeros(ids.size()).to(ids)
            for i, l in enumerate(seg_lengths):
                attention_mask[i, :l] = 1
            return ids, attention_mask
        else:
            return ids

    def mask_start_tokens(self, subwords, seq_len, pad=True):
        def special_tokens():
            return [
                self.tokenizer.bos_token,
                self.tokenizer.eos_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ]

        mask = torch.zeros(len(subwords), np.max(seq_len))
        for i, sent in enumerate(subwords):
            for j, subword in enumerate(sent):
                if self.is_word_piece(subword) or subword in special_tokens():
                    continue
                else:
                    mask[i][j] = 1

        return mask

    def subword_tokenize(
        self, sentences: List[str], pad: bool = True
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Args:
            sentences (List[str]): A list of sentences.
            pad (bool): Adds padding.

        Returns:

        """
        subwords = list(map(self.tokenizer.tokenize, sentences))
        for i, subword in enumerate(subwords):
            # Truncate Long sentences!
            if len(subword) > self.max_positions:
                subword = subword[: self.max_positions - 2]
            subword = [self.tokenizer.bos_token] + subword + [self.tokenizer.eos_token]
            assert len(subword) <= self.max_positions
            subwords[i] = subword

        subword_lengths = list(map(len, subwords))
        max_len = np.max(subword_lengths)
        if pad:
            for i, subword in enumerate(subwords):
                for _ in range(max_len - len(subword)):
                    subword.append(self.tokenizer.pad_token)
                subwords[i] = subword
        subword_mask = self.mask_start_tokens(subwords, subword_lengths)
        return subwords, subword_mask, subword_lengths

    def subword_tokenize_to_ids(self, tokens) -> Dict[str, torch.Tensor]:
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.

        Args:
            tokens (List[str]): A sequence of strings, representing input tokens.

        Returns:
            Dict[str, torch.Tensor]: dict with 'input_ids', 'attention_mask',
                'subword_mask'
        """
        subwords, subword_masks, subword_lengths = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords, subword_lengths)
        input_dict = {
            "input_ids": subword_ids,
            "attention_mask": mask,
            "subword_mask": subword_masks,
        }
        return input_dict
