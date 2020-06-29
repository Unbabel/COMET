# -*- coding: utf-8 -*-
r"""
Model Base
==============
    Abstract base class used to build new modules inside COMET.
"""
from abc import ABCMeta, abstractmethod
from argparse import Namespace
from typing import Dict, Generator, Iterable, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

import pytorch_lightning as ptl
from comet.models.encoders import Encoder, str2encoder
from comet.schedulers import str2scheduler


class ModelBase(ABCMeta, ptl.LightningModule):
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

        :param model: Model class namae (to be replaced with the model specified in the YAML)
         
        -------------------- Training Parameters -------------------------

        :param batch_size: Batch size used during training.
        
        :param nr_frozen_epochs: Number of epochs we keep the encoder model frozen.
        
        -------------------- Optimizer Parameters -------------------------

        :param optimizer: Optimizer class to be used.

        :param encoder_learning_rate: Learning rate used for the encoder model.
        
        :param learning_rate: Overall learning rate.
        
        -------------------- Scheduler Parameters -------------------------
        
        :param scheduler: Scheduler class to be used.

        :param warmup_steps: Warmup steps (only used for schedulers with warmup period).
        
        :param num_training_steps: Number of training steps after which the LR goes to 0.

        -------------------- Architecture Parameters -------------------------

        :param encoder_model: Encoder class to be used.

        :param pretrained_model: Encoder checkpoint (e.g. xlmr.base vs xlmr.large)
        
        :param layer: Encoder layer to extract the embeddings. If 'mix' all layers are combined
            with layer-wise attention.

        :param scalar_mix_dropout: Dropout probability for the layer-wise attention pooling.

        :param pool: Pooling technique to extract the sentence embeddings. 
            Options: {max, avg, default, cls} where default uses the 'default' sentence embedding
            returned by the encoder (e.g. BERT pooler_output) and 'cls' is the first token of the 
            sequence and depends on the selected layer.
        
        -------------------- Data Parameters -------------------------

        :param train_path: Path to the training data.

        :param val_path: Path to the validation data.

        :param test_path: Path to the test data.

        :param loader_workers: Number of workers used to load and tokenize data during training.

        """
        
        model: str = None

        # Training details
        batch_size: int = 8
        nr_frozen_epochs: int = 1

        # Optimizer
        optimizer: str = "Adam"
        encoder_learning_rate: float = 1e-05
        learning_rate: float = 1e-05

        # Scheduler
        scheduler: str = "constant"
        warmup_steps: int = None
        num_training_steps: int = None

        # Architecture Definition
        encoder_model: str = "XLM-R"
        pretrained_model: str = "xlm-roberta-base"
        layer: str = "mix"
        scalar_mix_dropout: float = 0.0
        pool: str = "avg"

        # Data
        train_path: str = None
        val_path: str = None
        test_path: str = None   
        loader_workers: int = 8

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
        self.hparams = hparams
        self.encoder = self._build_encoder()

        # Model initialization
        self._build_model()

        # Loss criterion initialization.
        self._build_loss()

        # The encoder always starts in a frozen state.
        if hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        self.nr_frozen_epochs = hparams.nr_frozen_epochs

    @abstractmethod
    def _build_loss(self):
        """ Initializes the loss function/s. """
        pass
    
    @abstractmethod
    def _build_model(self) -> ptl.LightningModule:
        """
        Initializes the estimator architecture.
        """
        pass

    @abstractmethod
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
        try:
            return str2scheduler[self.hparams.scheduler].from_hparams(
                optimizer, self.hparams
            )
        except KeyError:
            Exception(f"{self.hparams.scheduler} invalid scheduler!")

    def read_csv(self, path: str) -> List[dict]:
        """ Reads a comma separated value file.
        
        :param path: path to a csv file.
        
        Return:
            - List of records as dictionaries
        """
        df = pd.read_csv(path)
        return df.to_dict("records")

    def freeze_encoder(self) -> None:
        """ Freezes the encoder layer. """
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            click.secho("\nEncoder model fine-tuning", fg="red")
            self.encoder.unfreeze()
            self._frozen = False

    def on_epoch_end(self):
        """ Hook used to unfreeze encoder during training. """
        if self.current_epoch + 1 >= self.nr_frozen_epochs and self._frozen:
            self.unfreeze_encoder()
            self._frozen = False

    @abstractmethod
    def predict(
        self, samples: Dict[str, str]
    ) -> (Dict[str, Union[str, float]], List[float]):
        """ Function that runs a model prediction,
        :param samples: dictionary with expected model sequences. 
            You can also pass a list of dictionaries to predict an entire batch.
        
        Return: Dictionary with input samples + scores and list just the scores.
        """
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        PyTorch Forward.
        Return: Dictionary with model outputs to be passed to the loss function.
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self, model_out: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes Loss value according to a loss function.
        :param model_out: model specific output.
        :param targets: Target score values [batch_size]
        """
        pass
    
    @abstractmethod
    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.
        :param sample: List of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        Returns:
            - Tuple with 2 dictionaries: model inputs and targets
        or
            - Dictionary with model inputs
        """
        pass

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """ 
        Function for setting up the optimizers and the schedulers to be used during training.
        
        Returns:
            - List with as many optimizers as we need
            - List with the respective schedulers.
        """
        optimizer = self._build_optimizer(self.parameters())
        scheduler = self._build_scheduler(optimizer)
        return [optimizer], [scheduler]

    def compute_metrics(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """ 
        Private function that computes metrics of interest based on the list of outputs 
        you defined in validation_step.
        """
        raise NotImplementedError

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

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        return {"loss": loss_value}

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

        Returns:
            - dictionary passed to the validation_end function.
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
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        train_outs, val_outs = outputs
        train_loss = torch.stack([x["val_loss"] for x in train_outs]).mean()
        val_loss = torch.stack([x["val_loss"] for x in val_outs]).mean()

        # Store Metrics for Reporting...
        train_metrics = self.compute_metrics(train_outs)
        val_metrics = self.compute_metrics(val_outs)
        
        logging_metrics = {
            "train": train_metrics, 
            **val_metrics
        }
        
        return {"log": {"val_loss": val_loss, "train_loss": train_loss, **logging_metrics}}
        
    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_nb, 0)

    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.compute_metrics(outputs)

    def setup(self, stage) -> None:
        """ Data preparation function called before training by Lightning.
            Equivalent to the prepare_data in previous Lightning Versions """
        self.train_dataset = self.read_csv(self.hparams.train_path)
        self.val_dataset = self.read_csv(self.hparams.val_path)
        
        # Always validate the model with 2k examples from training to control overfit.
        train_dataset = self.read_csv(self.hparams.train_path)
        train_subset = np.random.choice(a=len(train_dataset), size=2000) 
        self.train_subset = Subset(train_dataset, train_subset)
        
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
            )
        ]

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )
