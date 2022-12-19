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

import torch
from torch.utils.data import Sampler
from transformers.utils import ModelOutput


class Prediction(ModelOutput):
    """Renamed ModelOutput"""

    pass


class Target(ModelOutput):
    """Renamed ModelOutput into Targets to keep same behaviour"""

    pass


class OrderedSampler(Sampler[int]):
    """Sampler that returns the indices in a deterministic order."""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def flatten_metadata(metadata):
    """Metadata from the model output can be in various forms and this function
    will gather all metadata and flatten everything.
    """
    metadata = Prediction(**{k: [dic[k] for dic in metadata] for k in metadata[0]})
    for k, v in metadata.items():
        if torch.is_tensor(v[0]):
            # If we have tensors we can use cat to flatten them.
            metadata[k] = torch.cat(v, dim=0).tolist()
        else:
            # for other predictions such as word tags we have to flatten the list.
            metadata[k] = [item for sublist in v for item in sublist]
    return metadata


def restore_list_order(sorted_list, sort_ids):
    """Restores the original ids of a given list."""
    unsorted_list = [None for _ in range(len(sorted_list))]
    for i, s in zip(sort_ids, sorted_list):
        unsorted_list[i] = s
    return unsorted_list
