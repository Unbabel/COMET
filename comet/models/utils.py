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
