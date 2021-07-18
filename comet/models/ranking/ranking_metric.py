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
Ranking Metric
====================
    Translation Ranking metric was introduced by
        [Rei, et al. 2020](https://aclanthology.org/2020.emnlp-main.213/)
    and it is trained on top of Direct Assessment Relative Ranks (DARR) to encode
    `good` translations closer to the anchors (source & reference) than `worse`
    translations.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from comet.models.base import CometModel
from transformers import AdamW

from .wmt_kendall import WMTKendall


class RankingMetric(CometModel):
    """RankingMetric

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.05,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 8,
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "ranking_metric",
        )
        self.save_hyperparameters()

    def init_metrics(self):
        self.train_metrics = WMTKendall(prefix="train")
        self.val_metrics = WMTKendall(prefix="val")

    @property
    def loss(self):
        return torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets the optimizers to be used during training."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            params = layer_parameters + layerwise_attn_params
        else:
            params = layer_parameters

        optimizer = AdamW(
            params,
            lr=self.hparams.learning_rate,
            correct_bias=True,
        )
        # scheduler = self._build_scheduler(optimizer)
        return [optimizer], []

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Dict[str, torch.Tensor]:

        sample = {k: [dic[k] for dic in sample] for k in sample[0]}

        if inference:
            src_inputs = self.encoder.prepare_sample(sample["src"])
            mt_inputs = self.encoder.prepare_sample(sample["mt"])
            ref_inputs = self.encoder.prepare_sample(sample["ref"])

            ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
            src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
            mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}

            return {**ref_inputs, **src_inputs, **mt_inputs}

        ref_inputs = self.encoder.prepare_sample(sample["ref"])
        src_inputs = self.encoder.prepare_sample(sample["src"])
        pos_inputs = self.encoder.prepare_sample(sample["pos"])
        neg_inputs = self.encoder.prepare_sample(sample["neg"])

        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        pos_inputs = {"pos_" + k: v for k, v in pos_inputs.items()}
        neg_inputs = {"neg_" + k: v for k, v in neg_inputs.items()}

        return {**ref_inputs, **src_inputs, **pos_inputs, **neg_inputs}

    def forward(
        self,
        src_input_ids: torch.tensor,
        ref_input_ids: torch.tensor,
        pos_input_ids: torch.tensor,
        neg_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        ref_attention_mask: torch.tensor,
        pos_attention_mask: torch.tensor,
        neg_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)
        pos_sentemb = self.get_sentence_embedding(pos_input_ids, pos_attention_mask)
        neg_sentemb = self.get_sentence_embedding(neg_input_ids, neg_attention_mask)

        loss = self.loss(src_sentemb, pos_sentemb, neg_sentemb) + self.loss(
            ref_sentemb, pos_sentemb, neg_sentemb
        )

        distance_src_pos = F.pairwise_distance(pos_sentemb, src_sentemb)
        distance_ref_pos = F.pairwise_distance(pos_sentemb, ref_sentemb)
        # Harmonic mean between anchors and the positive example
        distance_pos = (2 * distance_src_pos * distance_ref_pos) / (
            distance_src_pos + distance_ref_pos
        )

        # Harmonic mean between anchors and the negative example
        distance_src_neg = F.pairwise_distance(neg_sentemb, src_sentemb)
        distance_ref_neg = F.pairwise_distance(neg_sentemb, ref_sentemb)
        distance_neg = (2 * distance_src_neg * distance_ref_neg) / (
            distance_src_neg + distance_ref_neg
        )

        return {
            "loss": loss,
            "distance_pos": distance_pos,
            "distance_neg": distance_neg,
        }

    def read_csv(self, path: str, regression: bool = False) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)

        if regression:
            df = df[["src", "mt", "ref", "score"]]
            df["src"] = df["src"].astype(str)
            df["mt"] = df["mt"].astype(str)
            df["ref"] = df["ref"].astype(str)
            df["score"] = df["score"].astype(float)
            return df.to_dict("records")

        df = df[["src", "pos", "neg", "ref"]]
        df["src"] = df["src"].astype(str)
        df["pos"] = df["pos"].astype(str)
        df["neg"] = df["neg"].astype(str)
        df["ref"] = df["ref"].astype(str)
        return df.to_dict("records")

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs one training step.
        This usually consists in the forward function followed by the loss function.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.

        :returns: dictionary containing the loss and the metrics to be added to the
            lightning logger.
        """
        batch_prediction = self.forward(**batch)
        loss_value = batch_prediction["loss"]

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Similar to the training step but with the model in eval mode.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.
        :param dataloader_idx: Integer displaying which dataloader this is.

        :returns: dictionary passed to the validation_end function.
        """
        batch_prediction = self.forward(**batch)
        loss_value = batch_prediction["loss"]
        self.log("val_loss", loss_value, on_step=True, on_epoch=True)

        # TODO: REMOVE if condition after torchmetrics bug fix
        if dataloader_idx == 0:
            self.train_metrics.update(
                batch_prediction["distance_pos"], batch_prediction["distance_neg"]
            )
        elif dataloader_idx == 1:
            self.val_metrics.update(
                batch_prediction["distance_pos"], batch_prediction["distance_neg"]
            )

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int],
    ) -> List[float]:
        src_sentemb = self.get_sentence_embedding(
            batch["src_input_ids"], batch["src_attention_mask"]
        )
        ref_sentemb = self.get_sentence_embedding(
            batch["ref_input_ids"], batch["ref_attention_mask"]
        )
        mt_sentemb = self.get_sentence_embedding(
            batch["mt_input_ids"], batch["mt_attention_mask"]
        )

        src_distance = F.pairwise_distance(mt_sentemb, src_sentemb)
        ref_distance = F.pairwise_distance(mt_sentemb, ref_sentemb)

        distances = (2 * ref_distance * src_distance) / (ref_distance + src_distance)
        return torch.ones_like(distances) / (1 + distances)
