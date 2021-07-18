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
WMT Kendall Tau
====================
    Kendall Tau like formulation used to measure agreement between relative ranks
    produced by humans and relative ranks produced by metrics.
"""
import torch
from torchmetrics import Metric


class WMTKendall(Metric):
    def __init__(self, dist_sync_on_step=False, prefix=""):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("concordance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("discordance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.prefix = prefix

    def update(self, distance_pos: torch.Tensor, distance_neg: torch.Tensor):
        assert distance_pos.shape == distance_neg.shape
        self.concordance = torch.sum((distance_pos < distance_neg).float())
        self.discordance = torch.sum((distance_pos >= distance_neg).float())

    def compute(self):
        return {
            self.prefix
            + "_kendall": (self.concordance - self.discordance)
            / (self.concordance + self.discordance)
        }
