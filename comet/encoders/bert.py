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
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from comet.encoders.base import Encoder

logger = logging.getLogger(__name__)


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
        return self.model.config.max_position_embeddings - 2

    @property
    def num_layers(self):
        """Number of model layers available."""
        return self.model.config.num_hidden_layers + 1

    @property
    def size_separator(self):
        """Number of tokens used between two segments. For BERT is just 1 ([SEP])
        but models such as XLM-R use 2 (</s></s>)"""
        return 1

    @property
    def uses_token_type_ids(self):
        return True

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
                "lr": lr * decay**i,
            }
            for i in range(self.num_layers - 2, 0, -1)
        ]
        return opt_parameters

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
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

    def mask_start_tokens(self, subwords, sub_len, pad=True):
        mask = torch.zeros(len(subwords), np.max(sub_len))
        for i, sent in enumerate(subwords):
            for j, subword in enumerate(sent):
                if subword.startswith("▁") or j == 1:
                    mask[i][j] = 1
        return mask

    def subword_tokenize(self, tokens, pad=True):
        """
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
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

        :param tokens: A sequence of strings, representing input tokens.
        :return: dict with 'input_ids', 'attention_mask', 'subword_mask'
        """
        subwords, subword_masks, subword_lengths = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords, subword_lengths)
        input_dict = {
            "input_ids": subword_ids,
            "attention_mask": mask,
            "subword_mask": subword_masks,
        }
        return input_dict

    def concat_sequences(
        self, inputs: List[Dict[str, torch.Tensor]], word_outputs=False
    ) -> Dict[str, torch.Tensor]:
        """ Receives a list of model input sentences and concatenates them into
        one single sequence.

        Args:
            inputs (List[Dict[str, torch.Tensor]]): List of model inputs
            word_outputs (bool, optional): For word-level models we also need to 
                concatenate subword masks. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Returns a single model input with all sentences 
                concatenated into a single input.
        """
        concat_input_ids = []
        # Remove padding before concatenation
        for encoder_input in inputs:
            input_ids = encoder_input["input_ids"]
            input_ids = [
                x.masked_select(x.ne(self.tokenizer.pad_token_id)).tolist()
                for x in input_ids.unbind(dim=0)
            ]
            concat_input_ids.append(input_ids)

        # Concatenate inputs into a single batch
        batch_size = len(concat_input_ids[0])
        batch = []
        token_type_ids = []
        for i in range(batch_size):
            # Exclude [CLS] and [SEP] before build_inputs_with_special_tokens
            # because that method adds them again
            lengths = tuple(len(x[i][1:-1]) for x in concat_input_ids)

            # self.max_positions = 512 but we need to remove 4 aditional tokens
            # [CLS]...[SEP]...[SEP]...[SEP]
            special_tokens = 1 + len(inputs) * self.size_separator
            new_sequence = concat_input_ids[0][i]

            token_type_ids.append(
                torch.zeros(len(new_sequence[1:-1]) + 2, dtype=torch.int)
            )
            for j in range(1, len(inputs)):
                new_sequence = self.tokenizer.build_inputs_with_special_tokens(
                    new_sequence[1:-1], concat_input_ids[j][i][1:-1]
                )
            if sum(lengths) > self.max_positions - special_tokens:
                new_sequence = new_sequence[: self.max_positions]
                # logger.warning("Sequence with {} tokens truncated at {} tokens".format(sum(lengths), self.max_positions))

            batch.append(torch.tensor(new_sequence))

        lengths = [t.shape[0] for t in batch]
        max_len = max(lengths)
        padded = [
            self.pad_tensor(t, max_len, self.tokenizer.pad_token_id) for t in batch
        ]
        lengths = torch.tensor(lengths, dtype=torch.long)
        padded = torch.stack(padded, dim=0).contiguous()
        attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]

        if word_outputs:
            subword_mask = []
            for padseg, pmask in zip(padded, inputs[0]["subword_mask"]):
                sub_mask = torch.zeros(padseg.size())
                sub_mask[: pmask.size(-1)] = pmask
                subword_mask.append(sub_mask.unsqueeze(0))
            subword_mask = torch.cat(subword_mask, dim=0)

            encoder_input = {
                "input_ids": padded,
                "attention_mask": attention_mask,
                "subwords_mask": subword_mask,
            }

        else:
            encoder_input = {"input_ids": padded, "attention_mask": attention_mask}

        if self.uses_token_type_ids:
            token_type_ids = [self.pad_tensor(t, max_len, 1) for t in token_type_ids]
            token_type_ids = torch.stack(token_type_ids, dim=0).contiguous()
            encoder_input["token_type_ids"] = token_type_ids

        if word_outputs:
            return encoder_input, lengths, max_len

        return encoder_input
