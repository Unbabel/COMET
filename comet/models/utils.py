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
from typing import List

from torch.utils.data import Sampler
from transformers.utils import ModelOutput


class Prediction(ModelOutput):
    "Renamed ModelOutput"
    pass


class Target(ModelOutput):
    "Renamed ModelOutput into Targets to keep same behaviour"
    pass


class OrderedSampler(Sampler[int]):
    """
    Sampler that returns the indices in a deterministic order.
    """

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class LabelEncoder:
    def __init__(self, reserved_labels: List[str]):
        self.token_to_index = {
            token: index for index, token in enumerate(reserved_labels)
        }
        self.index_to_token = reserved_labels.copy()

    @property
    def vocab(self):
        return self.index_to_token

    @property
    def vocab_size(self):
        """
        Returns:
            int: Number of labels in the dictionary.
        """
        return len(self.vocab)

    def encode(self, sequence: List[str]):
        return [self.token_to_index[t] for t in sequence]

    def decode(self, sequence: List[int], join: bool):
        if join:
            return " ".join([self.index_to_token[idx] for idx in sequence])
        return [self.index_to_token[idx] for idx in sequence]

    def batch_encode(self, batch, split=True):
        if split:
            return [self.encode(seq.split()) for seq in batch]
        return [self.encode(seq) for seq in batch]

    def batch_decode(self, batch, join=True):
        return [self.decode(seq, join) for seq in batch]
