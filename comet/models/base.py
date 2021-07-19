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
CometModel
========================
    Abstract Model class that implements some of the Pytorch Lightning logic.
    Extend this class to create new model and metrics within COMET.
"""
import abc
import logging
import multiprocessing
from os import path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as ptl
import torch
from comet.encoders import str2encoder
from comet.modules import LayerwiseAttention
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset

from .pooling_utils import average_pooling, max_pooling

logger = logging.getLogger(__name__)


class CometModel(ptl.LightningModule, metaclass=abc.ABCMeta):
    """CometModel:

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
    :param class_identifier: subclass identifier.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-large",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
        class_identifier: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["train_data", "validation_data", "load_weights_from_checkpoint"]
        )
        self.encoder = str2encoder[self.hparams.encoder_model].from_pretrained(
            self.hparams.pretrained_model
        )
        self.epoch_nr = 0
        if self.hparams.layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=True,
            )
        else:
            self.layerwise_attention = None

        if self.hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        if self.hparams.keep_embeddings_frozen:
            self.encoder.freeze_embeddings()

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

        if load_weights_from_checkpoint is not None:
            if path.exists(load_weights_from_checkpoint):
                self.load_weights(load_weights_from_checkpoint)
            else:
                logger.warning(f"Path {load_weights_from_checkpoint} does not exist!")

        self.mc_dropout = False  # Flag used to control usage of MC Dropout

    def set_mc_dropout(self, value: bool):
        self.mc_dropout = value

    def load_weights(self, checkpoint: str) -> None:
        """Function that loads the weights from a given checkpoint file.
        Note:
            If the checkpoint model architecture is different then `self`, only
            the common parts will be loaded.

        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        logger.info(f"Loading weights from {checkpoint}.")
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    @abc.abstractmethod
    def read_csv(self):
        pass

    @abc.abstractmethod
    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], *args, **kwargs
    ):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    @abc.abstractmethod
    def init_metrics(self) -> None:
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def freeze_encoder(self) -> None:
        logger.info("Encoder model frozen.")
        self.encoder.freeze()

    @property
    def loss(self) -> None:
        return nn.MSELoss()

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.loss(predictions["score"].view(-1), targets["score"])

    def unfreeze_encoder(self) -> None:
        if self._frozen:
            if self.trainer.is_global_zero:
                logger.info("Encoder model fine-tuning")

            self.encoder.unfreeze()
            self._frozen = False
            if self.hparams.keep_embeddings_frozen:
                self.encoder.freeze_embeddings()

    def on_train_epoch_end(self) -> None:
        """Hook used to unfreeze encoder during training."""
        self.epoch_nr += 1
        if self.epoch_nr >= self.nr_frozen_epochs and self._frozen:
            self.unfreeze_encoder()
            self._frozen = False

    def get_sentence_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Function that extracts sentence embeddings for
            a single sentence.

        :param tokens: sequences [batch_size x seq_len]
        :param lengths: lengths [batch_size]

        :return: torch.Tensor [batch_size x hidden_size]
        """
        encoder_out = self.encoder(input_ids, attention_mask)
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )

        elif self.hparams.layer >= 0 and self.hparams.layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.hparams.layer]

        else:
            raise Exception("Invalid model layer {}.".format(self.hparams.layer))

        if self.hparams.pool == "default":
            sentemb = encoder_out["sentemb"]

        elif self.hparams.pool == "max":
            sentemb = max_pooling(
                input_ids, embeddings, self.encoder.tokenizer.pad_token_id
            )

        elif self.hparams.pool == "avg":
            sentemb = average_pooling(
                input_ids,
                embeddings,
                attention_mask,
                self.encoder.tokenizer.pad_token_id,
            )

        elif self.hparams.pool == "cls":
            sentemb = embeddings[:, 0, :]

        else:
            raise Exception("Invalid pooling technique.")

        return sentemb

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
    ) -> torch.Tensor:
        """
        Runs one training step and logs the training loss.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.

        :returns: Loss value
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

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
    ) -> None:
        """
        Runs one validation step and logs metrics.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.
        :param dataloader_idx: Integer displaying which dataloader this is.
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        self.log("val_loss", loss_value, on_step=True, on_epoch=True)

        # TODO: REMOVE if condition after torchmetrics bug fix
        if batch_prediction["score"].view(-1).size() != torch.Size([1]):
            if dataloader_idx == 0:
                self.train_metrics.update(
                    batch_prediction["score"].view(-1), batch_target["score"]
                )
            elif dataloader_idx == 1:
                self.val_metrics.update(
                    batch_prediction["score"].view(-1), batch_target["score"]
                )

    def on_predict_start(self) -> None:
        """Called when predict begins."""
        if self.mc_dropout:
            self.train()
        else:
            self.eval()

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Runs one prediction step and returns the predicted values.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.
        :param dataloader_idx: Integer displaying which dataloader this is.
        """
        if self.mc_dropout:
            mcd_outputs = torch.stack(
                [self(**batch)["score"].view(-1) for _ in range(self.mc_dropout)]
            )
            mcd_mean = mcd_outputs.mean(dim=0)
            mcd_std = mcd_outputs.std(dim=0)
            return mcd_mean, mcd_std

        return self(**batch)["score"].view(-1)

    def validation_epoch_end(self, *args, **kwargs) -> None:
        """ " Computes and logs metrics."""
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()
        self.val_metrics.reset()

    def setup(self, stage) -> None:
        """Data preparation function called before training by Lightning.

        :param stage: either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in (None, "fit"):
            self.train_dataset = self.read_csv(self.hparams.train_data)
            self.validation_dataset = self.read_csv(self.hparams.validation_data)

            self.epoch_total_steps = len(self.train_dataset) // (
                self.hparams.batch_size * max(1, self.trainer.num_gpus)
            )
            self.total_steps = self.epoch_total_steps * float(self.trainer.max_epochs)

            # Always validate the model with 2k examples to control overfit.
            train_subset = np.random.choice(a=len(self.train_dataset), size=2000)
            self.train_subset = Subset(self.train_dataset, train_subset)
            self.init_metrics()

    def train_dataloader(self) -> DataLoader:
        """Function that loads the train set."""
        return DataLoader(
            dataset=self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        return [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=min(8, multiprocessing.cpu_count()),
            ),
            DataLoader(
                dataset=self.validation_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=min(8, multiprocessing.cpu_count()),
            ),
        ]
