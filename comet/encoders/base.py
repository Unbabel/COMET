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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tokenizers import Encoding

from comet.models.utils import LabelSet


class Encoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for an encoder model."""

    labelset = LabelSet()

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

    def align_tokens_and_annotations(
        self, tokenized: Encoding, annotations: List[dict]
    ) -> Tuple[List[int], List[int]]:
        """Inspired by: https://github.com/LightTag/sequence-labeling-with-transformers/"""
        tokens = tokenized.tokens
        aligned_labels = ["O"] * len(tokens)
        for anno in annotations:
            # A set that stores the token indices of the annotation
            annotation_token_ix_set = set()
            for char_ix in range(anno["start"], anno["end"]):
                token_ix = tokenized.char_to_token(char_ix)
                if token_ix is not None:
                    annotation_token_ix_set.add(token_ix)
            for _, token_ix in enumerate(sorted(annotation_token_ix_set)):
                prefix = "I"
                aligned_labels[token_ix] = f"{prefix}-{anno['severity']}"

        def get_label_id(item):
            """Aux function that replaces the `get` function from labels_to_id
            dictionary to raise an exception when a label is not found.
            """
            label = self.labelset.labels_to_id.get(item)
            if label is None:
                raise Exception(
                    f"{label} does not exist in self.labelset: {self.labelset.labels_to_id.keys()}"
                )
            return label

        return list(map(get_label_id, aligned_labels))

    def subword_tokenize(
        self, sample: List[str], annotations: List[dict]
    ) -> Dict[str, torch.Tensor]:
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
        input_ids, offsets, label_ids = [], [], []
        for i in range(len(sample)):
            tokenized_text, sent_annot = encoder_input[i], annotations[i]
            input_ids.append(tokenized_text.ids)
            label_ids.append(
                self.align_tokens_and_annotations(tokenized_text, sent_annot)
            )
            offsets.append(tokenized_text.offsets)

        attention_mask = [[1 for _ in seq] for seq in input_ids]
        max_length = max([len(l) for l in input_ids])
        input_ids = self.pad_list(input_ids, max_length, self.tokenizer.pad_token_id)
        label_ids = self.pad_list(label_ids, max_length, -1)
        attention_mask = self.pad_list(attention_mask, max_length, 0)
        return {
            "input_ids": torch.tensor(input_ids),
            "label_ids": torch.tensor(label_ids),
            "attention_mask": torch.tensor(attention_mask),
            "offsets": offsets,  # Used during inference
        }

    def prepare_sample(
        self,
        sample: List[str],
        word_level: bool = False,
        annotations: Optional[List[dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Receives a list of strings and applies tokenization and vectorization.

        Args:
            sample (List[str]): List with text segments to be tokenized and padded.
            word_level (bool, optional): True for subword tokenization.
                Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: dict with 'input_ids', 'attention_mask' and
                'subword_mask' if word_level_training=True
        """
        if word_level:
            if annotations is None:
                annotations = [[]] * len(sample)
            return self.subword_tokenize(sample, annotations)
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
        self, inputs: List[Dict[str, torch.Tensor]], return_label_ids=False
    ) -> Dict[str, torch.Tensor]:
        """Receives a list of model input sentences and concatenates them into
        one single sequence.

        Args:
            inputs (List[Dict[str, torch.Tensor]]): List of model inputs
            return_label_ids (bool, optional): For word-level models we also need to
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
        if return_label_ids:
            label_ids = [
                self.pad_tensor(t, max_len, -1) for t in inputs[0]["label_ids"]
            ]
            label_ids = torch.stack(label_ids, dim=0).contiguous()
            encoder_input = {
                "input_ids": padded,
                "attention_mask": attention_mask,
                "label_ids": label_ids,
                "mt_offsets": inputs[0]["offsets"],  # Used during inference
            }

        else:
            encoder_input = {"input_ids": padded, "attention_mask": attention_mask}

        if self.uses_token_type_ids:
            token_type_ids = [self.pad_tensor(t, max_len, 1) for t in token_type_ids]
            token_type_ids = torch.stack(token_type_ids, dim=0).contiguous()
            encoder_input["token_type_ids"] = token_type_ids

        return encoder_input, lengths, max_len
