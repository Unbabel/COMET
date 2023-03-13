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
from torch import nn
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup

from comet.models.base import CometModel
from comet.models.metrics import WMTKendall
from comet.models.utils import Prediction


class RankingMetric(CometModel):
    """RankingMetric

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.1.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to False.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        warmup_steps (int): Warmup steps for LR scheduler.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 1e-05.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 3e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'xlm-roberta-base'.
        pool (str): Type of sentence level pooling (options: 'max', 'cls', 'avg').
            Defaults to 'avg'
        layer (Union[str, int]): Encoder layer to be used for regression ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'softmax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'True'.
        loss (str): Loss function to be used. Defaults to 'triplet-margin'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.1,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        layer_transformation: str = "softmax",
        layer_norm: bool = True,
        loss: str = "triplet-margin",
        dropout: float = 0.1,
        batch_size: int = 8,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        load_pretrained_weights: bool = True
    ) -> None:
        super().__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            pool=pool,
            layer=layer,
            layer_transformation=layer_transformation,
            layer_norm=layer_norm,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="ranking_metric",
            load_pretrained_weights=load_pretrained_weights
        )
        self.save_hyperparameters()

    def init_metrics(self):
        """Initializes train/validation metrics."""
        self.train_metrics = WMTKendall(prefix="train")
        self.val_metrics = nn.ModuleList(
            [WMTKendall(prefix=d) for d in self.hparams.validation_data]
        )

    def requires_references(self) -> bool:
        return True

    @property
    def loss(self):
        return torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to configure optimizers and schedulers."""
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

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        # If warmup setps are not defined we don't need a scheduler.
        if self.hparams.warmup_steps < 2:
            return [optimizer], []

        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
        )
        return [optimizer], [scheduler]

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "fit"
    ) -> Dict[str, torch.Tensor]:
        """This method will be called by dataloaders to prepared data to input to the
        model.

        Args:
            sample (List[dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs. If stage == 'predict' we will return only the src, mt and ref
                input ids. Otherwise, during training/validation we will return the
                the input ids for src, pos, neg, and ref.
        """
        sample = {k: [str(dic[k]) for dic in sample] for k in sample[0]}

        if stage == "predict":
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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Ranking model forward method.

        Args:
            src_input_ids [torch.tensor]: input ids from source sentences.
            ref_input_ids [torch.tensor]: input ids from reference translations.
            pos_input_ids [torch.tensor]: input ids from positive samples.
            neg_input_ids [torch.tensor]: input ids from negative samples.
            src_attention_mask [torch.tensor]: Attention mask from source sentences.
            ref_attention_mask [torch.tensor]: Attention mask from reference
                translations.
            pos_attention_mask [torch.tensor]: Attention mask from positive samples.
            neg_attention_mask [torch.tensor]: Attention mask from negative samples.

        Return:
            Dictionary with triplet loss, distance between anchors and positive samples
            and  distance between anchors and negative samples.
        """
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

    def read_training_data(self, path: str) -> List[dict]:
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        df = df[["src", "pos", "neg", "ref"]]
        df["src"] = df["src"].astype(str)
        df["pos"] = df["pos"].astype(str)
        df["neg"] = df["neg"].astype(str)
        df["ref"] = df["ref"].astype(str)
        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        return self.read_training_data(path)

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Pytorch Lightning training step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.

        Returns:
            [torch.Tensor] Loss value
        """
        batch_prediction = self.forward(**batch)
        loss_value = batch_prediction["loss"]

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_idx > self.first_epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Pytorch Lightning validation step. Runs model and logs metircs.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
        """
        batch_prediction = self.forward(**batch)
        loss_value = batch_prediction["loss"]
        self.log("val_loss", loss_value, on_step=True, on_epoch=True)

        if dataloader_idx == 0:
            self.train_metrics.update(
                batch_prediction["distance_pos"], batch_prediction["distance_neg"]
            )
        elif dataloader_idx > 0:
            self.val_metrics[dataloader_idx - 1].update(
                batch_prediction["distance_pos"], batch_prediction["distance_neg"]
            )

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> List[float]:
        """Pytorch Lightning predict step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this sample is
                coming from.

        Return:
            Predicion object
        """

        def _predict_forward(batch):
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
            distances = (2 * ref_distance * src_distance) / (
                ref_distance + src_distance
            )
            return Prediction(
                scores=torch.ones_like(distances) / (1 + distances),
                metadata=Prediction(
                    src_scores=src_distance,
                    ref_scores=ref_distance,
                ),
            )

        if self.mc_dropout:
            raise NotImplementedError("MCD not implemented for this model!")

        else:
            return _predict_forward(batch)
