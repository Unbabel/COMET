# -*- coding: utf-8 -*-
r"""
Estimator Base Model
==============
    Abstract base class used to build new estimator models
    inside COMET.
"""
from argparse import Namespace
from typing import Dict, Iterable, List, Union

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from comet.metrics import RegressionReport
from comet.models.model_base import ModelBase
from comet.models.utils import move_to_cpu, move_to_cuda


class Estimator(ModelBase):
    """
    Estimator base class that uses an Encoder to encode sequences.

    :param hparams: Namespace containing the hyperparameters.
    """

    class ModelConfig(ModelBase.ModelConfig):
        # Estimator specific hyper-parameters
        loss: str = "mse"
        hidden_sizes: str = "1536,768"
        activations: str = "Tanh"
        dropout: float = 0.1

    def __init__(self, hparams: Namespace) -> None:
        super().__init__(hparams)


    def _build_model(self) -> ModelBase:
        """
        Initializes the estimator architecture.
        """
        super()._build_model()
        self.metrics = RegressionReport()

    def _build_loss(self):
        """ Initializes the loss function/s. """
        super()._build_loss()
        if self.hparams.loss == "mse":
            self.loss = nn.MSELoss(reduction="sum")
        elif self.hparams.loss == "binary_xent":
            self.loss = nn.BCELoss(reduction="sum")
        else:
            raise Exception("{} is not a valid loss option.".format(self.hparams.loss))
    
    def read_csv(self, path: str) -> List[dict]:
        """ Reads a comma separated value file.
        
        :param path: path to a csv file.
        
        Return:
            - List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "ref", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].astype(str)
        df["score"] = df["score"].astype(float)
        return df.to_dict("records")

    def predict(
        self, samples: Dict[str, str], cuda: bool = False, show_progress: bool = False
    ) -> (Dict[str, Union[str, float]], List[float]):
        """ Function that runs a model prediction,
        :param samples: List of dictionaries with 'mt' and 'ref' keys.
        :param cuda: Flag that runs inference using 1 single GPU.
        :param show_progress: Flag to show progress during inference of multiple examples.
        
        Return: 
            - Dictionary with original samples + predicted scores and list of predicted scores
        """
        if self.training:
            self.eval()

        if cuda and torch.cuda.is_available():
            self.to("cuda")

        with torch.no_grad():
            batches = [
                samples[i : i + self.hparams.batch_size]
                for i in range(0, len(samples), self.hparams.batch_size)
            ]
            model_inputs = []
            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Preparing batches...",
                    dynamic_ncols=True,
                    leave=None,
                )
            for batch in batches:
                batch = self.prepare_sample(batch, inference=True)
                model_inputs.append(batch)
                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Scoring hypothesis...",
                    dynamic_ncols=True,
                    leave=None,
                )
            scores = []
            for model_input in model_inputs:
                if cuda and torch.cuda.is_available():
                    model_input = move_to_cuda(model_input)
                    model_out = self.forward(**model_input)
                    model_out = move_to_cpu(model_out)
                else:
                    model_out = self.forward(**model_input)

                model_scores = model_out["score"].numpy().tolist()
                for i in range(len(model_scores)):
                    scores.append(model_scores[i][0])

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        assert len(scores) == len(samples)
        for i in range(len(scores)):
            samples[i]["predicted_score"] = scores[i]
        return samples, scores

    def compute_loss(
        self, model_out: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes Loss value according to a loss function.
        :param model_out: model specific output. Must contain a key 'score' with 
            a tensor [batch_size x 1] with model predictions
        :param targets: Target score values [batch_size]
        """
        return self.loss(model_out["score"].view(-1), targets["score"])

    def compute_metrics(self, outputs: List[Dict[str, torch.Tensor]]) -> dict:
        """ 
        Private function that computes metrics of interest based on model predictions and 
        respective targets.
        """
        predictions = (
            torch.cat([batch["val_prediction"]["score"].view(-1) for batch in outputs])
            .cpu()
            .numpy()
        )
        targets = (
            torch.cat([batch["val_target"]["score"] for batch in outputs]).cpu().numpy()
        )
        return self.metrics(predictions, targets)
