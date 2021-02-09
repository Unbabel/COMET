# -*- coding: utf-8 -*-
r"""
Model Base
==============
    Abstract base class used to build new modules inside COMET. 
    This class is just an extention of PyTorch Lightning main module:
    https://pytorch-lightning.readthedocs.io/en/0.8.4/lightning-module.html
"""
from argparse import Namespace
from os import path
from typing import Dict, Generator, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

import pytorch_lightning as ptl
from comet.models.encoders import Encoder, str2encoder
from comet.schedulers import str2scheduler


class ModelBase(ptl.LightningModule):
    """
    Extends PyTorch Lightning with a common structure and interface
    that will be shared across all architectures.

    :param hparams: Namespace with hyper-parameters
    """

    class ModelConfig:
        """
        The ModelConfig class is used to define model hyper-parameters that
        are used to initialize our Lightning Modules. These parameters are
        then overwritted with the values defined in the YAML file and coverted
        to a Namespace to initialize the model.

        :param model: Model class name (to be replaced with the model specified in the YAML)

        -------------------- Training Parameters -------------------------

        :param batch_size: Batch size used during training.

        :param nr_frozen_epochs: Number of epochs we keep the encoder model frozen.

        :param keep_embeddings_frozen: Keeping the embeddings frozen is a usefull way to save some GPU memory usage.
            This is critical to fine-tune large models in GPUs with less than 32GB memory.

        -------------------- Optimizer Parameters -------------------------

        :param optimizer: Optimizer class to be used.

        :param learning_rate: Overall learning rate.

        -------------------- Scheduler Parameters -------------------------

        :param scheduler: Scheduler class to be used.

        :param warmup_steps: Warmup steps (only used for schedulers with warmup period).

        -------------------- Architecture Parameters -------------------------

        :param encoder_model: Encoder class to be used.

        :param pretrained_model: Encoder checkpoint (e.g. xlmr.base vs xlmr.large)

        :param pool: Pooling technique to extract the sentence embeddings.
            Options: {max, avg, default, cls} where default uses the `default` sentence embedding
            returned by the encoder (e.g. BERT pooler_output) and `cls` is the first token of the
            sequence and depends on the selected layer.

        :param load_weights: Loads weights from a checkpoint file that match the architecture.

        -------------------- Data Parameters -------------------------

        :param train_path: Path to the training data.

        :param val_path: Path to the validation data.

        :param test_path: Path to the test data.

        :param loader_workers: Number of workers used to load and tokenize data during training.

        :param monitor: Metric to be displayed in tqdm bar. Same as trainer monitor flag!
        """

        model: str = None

        # Training details
        batch_size: int = 8
        nr_frozen_epochs: int = 0
        keep_embeddings_frozen: bool = False

        # Optimizer
        optimizer: str = "Adam"
        learning_rate: float = 1e-05

        # Scheduler
        scheduler: str = "constant"
        warmup_steps: int = None

        # Architecture Definition
        encoder_model: str = "XLMR"
        pretrained_model: str = "xlmr.base"
        pool: str = "avg"
        load_weights: str = False

        # Data
        train_path: str = None
        val_path: str = None
        test_path: str = None
        loader_workers: int = 8

        monitor: str = "kendall"

        def __init__(self, initial_data: dict) -> None:
            for key in initial_data:
                if hasattr(self, key):
                    setattr(self, key, initial_data[key])

        def namespace(self) -> Namespace:
            return Namespace(
                **{
                    name: getattr(self, name)
                    for name in dir(self)
                    if not callable(getattr(self, name)) and not name.startswith("__")
                }
            )

    def __init__(self, hparams: Namespace) -> None:
        super(ModelBase, self).__init__()
        if isinstance(hparams, dict):
            self.hparams = Namespace(**hparams)
        else:
            self.hparams = hparams
        self.encoder = self._build_encoder()

        # Model initialization
        self._build_model()

        # Loss criterion initialization.
        self._build_loss()

        # The encoder always starts in a frozen state.
        if self.hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        if (
            hasattr(self.hparams, "keep_embeddings_frozen")
            and self.hparams.keep_embeddings_frozen
        ):
            self.encoder.freeze_embeddings()

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def _build_loss(self):
        """ Initializes the loss function/s. """
        pass

    def _build_model(self) -> ptl.LightningModule:
        """
        Initializes the estimator architecture.
        """
        # Compatibility with previous COMET versions
        if (
            hasattr(self.hparams, "load_weights")
            and self.hparams.load_weights
            and path.exists(self.hparams.load_weights)
        ):
            click.secho(f"Loading weights from {self.hparams.load_weights}", fg="red")
            self.load_weights(self.hparams.load_weights)

    def _build_encoder(self) -> Encoder:
        """
        Initializes the encoder.
        """
        try:
            return str2encoder[self.hparams.encoder_model].from_pretrained(self.hparams)
        except KeyError:
            raise Exception(f"{self.hparams.encoder_model} invalid encoder model!")

    def _build_optimizer(self, parameters: Generator) -> torch.optim.Optimizer:
        """
        Initializes the Optimizer.

        :param parameters: Module.parameters.
        """
        if hasattr(torch.optim, self.hparams.optimizer):
            return getattr(torch.optim, self.hparams.optimizer)(
                params=parameters, lr=self.hparams.learning_rate
            )
        else:
            raise Exception(f"{self.hparams.optimizer} invalid optimizer!")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Initializes the Scheduler.

        :param optimizer: PyTorch optimizer
        """
        self.epoch_total_steps = len(self.train_dataset) // (
            self.hparams.batch_size * max(1, self.trainer.num_gpus)
        )
        self.total_steps = self.epoch_total_steps * float(self.trainer.max_epochs)
        try:
            return {
                "scheduler": str2scheduler[self.hparams.scheduler].from_hparams(
                    optimizer, self.hparams, num_training_steps=self.total_steps
                ),
                "interval": "step",  # called after each training step
            }
        except KeyError:
            raise Exception(f"{self.hparams.scheduler} invalid scheduler!")

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        return df.to_dict("records")

    def freeze_encoder(self) -> None:
        """ Freezes the encoder layer. """
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            if self.trainer.is_global_zero:
                click.secho("\nEncoder model fine-tuning", fg="red")
            self.encoder.unfreeze()
            self._frozen = False

            if (
                hasattr(self.hparams, "keep_embeddings_frozen")
                and self.hparams.keep_embeddings_frozen
            ):
                self.encoder.freeze_embeddings()

    def on_epoch_end(self):
        """ Hook used to unfreeze encoder during training. """
        if self.current_epoch + 1 >= self.nr_frozen_epochs and self._frozen:
            self.unfreeze_encoder()
            self._frozen = False

    def predict(
        self, samples: Dict[str, str]
    ) -> (Dict[str, Union[str, float]], List[float]):
        """Function that runs a model prediction,

        :param samples: dictionary with expected model sequences.
            You can also pass a list of dictionaries to predict an entire batch.

        :return: Dictionary with input samples + scores and list with just the scores.
        """
        pass

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        PyTorch Forward.

        :return: Dictionary with model outputs to be passed to the loss function.
        """
        pass

    def compute_loss(
        self, model_out: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes Loss value according to a loss function.

        :param model_out: model specific output.
        :param targets: Target score values [batch_size]
        """
        pass

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: List of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets). If `inference=True`
            returns only the model inputs.
        """
        pass

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """
        Function for setting up the optimizers and the schedulers to be used during training.

        :returns: List with as many optimizers as we need and a list with the respective schedulers.
        """
        optimizer = self._build_optimizer(self.parameters())
        scheduler = self._build_scheduler(optimizer)
        return [optimizer], [scheduler]

    def compute_metrics(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Function that computes metrics of interest based on the list of outputs
        you defined in validation_step.
        """
        pass

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs one training step.
        This usually consists in the forward function followed by the loss function.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.

        :returns: dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

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
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        return {
            "val_loss": loss_value,
            "val_prediction": batch_prediction,
            "val_target": batch_target,
        }

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Function that takes as input a list of dictionaries returned by the validation_step
            and measures the model performance accross the entire validation set.

        :param outputs:

        :returns: Dictionary with metrics to be added to the lightning logger.
        """
        train_outs, val_outs = outputs
        train_loss = torch.stack([x["val_loss"] for x in train_outs]).mean()
        val_loss = torch.stack([x["val_loss"] for x in val_outs]).mean()

        # Store Metrics for Reporting...
        val_metrics = self.compute_metrics(val_outs)
        val_metrics["avg_loss"] = val_loss
        self.log_dict(val_metrics, prog_bar=True)

        train_metrics = self.compute_metrics(train_outs)
        train_metrics["avg_loss"] = train_loss
        self.log_dict({"train_" + k: v for k, v in train_metrics.items()})

    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """ Redirects to the validation_step function """
        return self.validation_step(batch, batch_nb, 0)

    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """ Computes metrics. """
        return self.compute_metrics(outputs)

    def setup(self, stage) -> None:
        """Data preparation function called before training by Lightning.
        Equivalent to the prepare_data in previous Lightning Versions"""
        self.train_dataset = self.read_csv(self.hparams.train_path)
        self.val_dataset = self.read_csv(self.hparams.val_path)

        # Always validate the model with 2k examples from training to control overfit.
        train_subset = np.random.choice(a=len(self.train_dataset), size=2000)
        self.train_subset = Subset(self.train_dataset, train_subset)

        if self.hparams.test_path:
            self.test_dataset = self.read_csv(self.hparams.test_path)

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            dataset=self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=self.hparams.loader_workers,
            ),
            DataLoader(
                dataset=self.val_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=self.hparams.loader_workers,
            ),
        ]

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )
