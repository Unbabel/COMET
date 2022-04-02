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
Metrics
=======
    Regression and Ranking metrics to be used during training to measure 
    correlations with human judgements
"""
import torch
from torch import Tensor

from torchmetrics import Metric
from typing import Any, Callable, List, Optional
import scipy.stats as stats


class RegressionMetrics(Metric):
    is_differentiable = False
    higher_is_better = True
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        prefix: str = "",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.prefix = prefix

        
    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """ Computes spearmans correlation coefficient. """
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        kendall, _ = stats.kendalltau(preds.tolist(), target.tolist())
        spearman, _ = stats.spearmanr(preds.tolist(), target.tolist())
        pearson, _ = stats.pearsonr(preds.tolist(), target.tolist())
        return {
            self.prefix + "_kendall": kendall,
            self.prefix + "_spearman": spearman,
            self.prefix + "_pearson": pearson,         
        }

class WMTKendall(Metric):
    def __init__(
        self,
        prefix: str = "",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
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
