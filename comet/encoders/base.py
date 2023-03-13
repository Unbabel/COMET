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
from typing import Dict, List, Tuple

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

    def add_span_tokens(self, begin_of_span: str, end_of_span: str) -> None:
        """For sequence tagging the training data will have special tokens
        delimiting the annotation spans such as <v> and </v>. We need to add those to the
        vocab.

        Args:
            begin_of_span (str): Token that marks a begin of span, e.g: <v>.
            end_of_span (src): Token that marks a end of span, e.g: </v>.
        """
        tokens = [begin_of_span, end_of_span]
        num_added_toks = self.tokenizer.add_tokens(tokens)
        assert len(tokens) == num_added_toks
        self.span_tokens = self.tokenizer.get_added_vocab()
        self.begin_of_span_id = self.span_tokens[begin_of_span]
        self.end_of_span_id = self.span_tokens[end_of_span]

    def extract_tag_spans(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """When doing sequence tagging we will encode sentences with annotation spans
        that are marked in text with special tokens (e.g <v> and </v>). Everything in
        between those delimiters will receive a label 1 while other tokens will receive
        a label 0.

        Args:
            begin_of_span (str): Token that marks a begin of span, e.g: <v>.
            end_of_span (src): Token that marks a end of span, e.g: </v>.

        Returns:
            Tuple[List[int], List[int]]: the input_ids without the begin_of_span and
                end_of_span token ids and a 'in_span_mask' where each token within a
                span is assigned a value 1
        """
        token_ids, in_span_mask = [], []
        for i in range(len(input_ids)):
            in_span = 0
            tag_seq, input_tokens = [], []
            for j in range(len(input_ids[i])):
                if input_ids[i][j] == self.begin_of_span_id:
                    in_span += 1

                elif input_ids[i][j] == self.end_of_span_id:
                    in_span -= 1

                elif in_span > 0:
                    input_tokens.append(input_ids[i][j])
                    tag_seq.append(1)

                else:
                    input_tokens.append(input_ids[i][j])
                    tag_seq.append(0)

            assert len(tag_seq) == len(input_tokens)
            if (self.begin_of_span_id in input_tokens) or (
                self.end_of_span_id in input_tokens
            ):
                raise Exception(
                    "Malformed sequence span found in {}".format(
                        self.tokenizer.decode(input_tokens)
                    )
                )

            token_ids.append(input_tokens)
            in_span_mask.append(tag_seq)
        return token_ids, in_span_mask

    def subword_tokenize(self, sample: List[str]) -> Dict[str, torch.Tensor]:
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.

        Args:
            tokens (List[str]): A sequence of strings, representing input tokens.

        Returns:
            Dict[str, torch.Tensor]: dict with 'input_ids', 'attention_mask',
                'subword_mask'
        """
        encoder_input = self.tokenizer(
            sample,
            truncation=True,
            max_length=self.max_positions - 2,
        )
        input_ids, in_span_mask = self.extract_tag_spans(encoder_input.input_ids)
        attention_mask = [[1 for _ in seq] for seq in input_ids]
        max_length = max([len(l) for l in input_ids])
        input_ids = self.pad_list(input_ids, max_length, self.tokenizer.pad_token_id)
        in_span_mask = self.pad_list(in_span_mask, max_length, -1)
        attention_mask = self.pad_list(attention_mask, max_length, 0)
        return {
            "input_ids": torch.tensor(input_ids),
            "in_span_mask": torch.tensor(in_span_mask),
            "attention_mask": torch.tensor(attention_mask),
        }

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
            return self.subword_tokenize(sample)
        else:
            tokenizer_output = self.tokenizer(
                sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_positions - 2,
            )
            return tokenizer_output

    def pad_list(
        self, token_ids: List[int], length: int, padding_index: int
    ) -> List[int]:
        """Pad a list to length with padding_index.

        Args:
            token_ids (List[int]): List to pad.
            length (int): Sequence length after padding.
            padding_index (int): Index to pad with.

        Returns:
            List[int]: extended list.
        """
        for seq in token_ids:
            seq.extend([padding_index] * (length - len(seq)))
        return token_ids

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

    def concat_sequences(
        self, inputs: List[Dict[str, torch.Tensor]], return_in_span_mask=False
    ) -> Dict[str, torch.Tensor]:
        """Receives a list of model input sentences and concatenates them into
        one single sequence.

        Args:
            inputs (List[Dict[str, torch.Tensor]]): List of model inputs
            return_in_span_mask (bool, optional): For word-level models we also need to
                pad in_span masks. Defaults to False.

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

            batch.append(torch.tensor(new_sequence))

        lengths = [t.shape[0] for t in batch]
        max_len = max(lengths)
        padded = [
            self.pad_tensor(t, max_len, self.tokenizer.pad_token_id) for t in batch
        ]
        lengths = torch.tensor(lengths, dtype=torch.long)
        padded = torch.stack(padded, dim=0).contiguous()
        attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
        if return_in_span_mask:
            in_span_mask = [
                self.pad_tensor(t, max_len, -1) for t in inputs[0]["in_span_mask"]
            ]
            in_span_mask = torch.stack(in_span_mask, dim=0).contiguous()
            encoder_input = {
                "input_ids": padded,
                "attention_mask": attention_mask,
                "in_span_mask": in_span_mask,
            }

        else:
            encoder_input = {"input_ids": padded, "attention_mask": attention_mask}

        if self.uses_token_type_ids:
            token_type_ids = [self.pad_tensor(t, max_len, 1) for t in token_type_ids]
            token_type_ids = torch.stack(token_type_ids, dim=0).contiguous()
            encoder_input["token_type_ids"] = token_type_ids

        return encoder_input, lengths, max_len
