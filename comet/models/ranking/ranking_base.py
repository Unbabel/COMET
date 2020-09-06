# -*- coding: utf-8 -*-
r"""
Translation Ranking Base Model
==============================
    Abstract base class used to build new ranking systems inside COMET.
    This task consists of ranking "good" translations above "worse" ones.
"""
from argparse import Namespace
from typing import Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from comet.models.model_base import ModelBase
from comet.models.utils import average_pooling, max_pooling, move_to_cuda
from comet.modules.scalar_mix import ScalarMixWithDropout
from comet.metrics import WMTKendall


class RankingBase(ModelBase):
    """
    Ranking Model base class used to fine-tune pretrained models such as XLM-R
    to produce better sentence embeddings by optmizing Triplet Margin Loss.

    :param hparams: Namespace containing the hyperparameters.
    """

    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__(hparams)

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["src", "ref", "pos", "neg"]]
        df["src"] = df["src"].astype(str)
        df["ref"] = df["ref"].astype(str)
        df["pos"] = df["pos"].astype(str)
        df["neg"] = df["neg"].astype(str)
        return df.to_dict("records")

    def _build_loss(self):
        """ Initializes the loss function/s. """
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def _build_model(self) -> ModelBase:
        """
        Initializes the ranking model architecture.
        """
        super()._build_model()
        self.metrics = WMTKendall()
        if self.hparams.encoder_model != "LASER":
            self.layer = (
                int(self.hparams.layer)
                if self.hparams.layer != "mix"
                else self.hparams.layer
            )

            self.scalar_mix = (
                ScalarMixWithDropout(
                    mixture_size=self.encoder.num_layers,
                    dropout=self.hparams.scalar_mix_dropout,
                    do_layer_norm=True,
                )
                if self.layer == "mix" and self.hparams.pool != "default"
                else None
            )

    def compute_loss(self, model_out: Dict[str, torch.Tensor], *args) -> torch.Tensor:
        """
        Computes Triplet Margin Loss.
        
        :param model_out: model specific output with reference, pos and neg
            sentence embeddings.
        """
        ref = model_out["ref_sentemb"]
        positive = model_out["pos_sentemb"]
        negative = model_out["neg_sentemb"]
        return self.loss(ref, positive, negative)

    def compute_metrics(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """  Computes WMT19 shared task kendall tau like metric. """
        distance_pos, distance_neg = [], []
        for minibatch in outputs:
            minibatch = minibatch["val_prediction"]
            ref_embedding = minibatch["ref_sentemb"]
            pos_embedding = minibatch["pos_sentemb"]
            neg_embedding = minibatch["neg_sentemb"]
            distance_ref_pos = F.pairwise_distance(pos_embedding, ref_embedding)
            distance_ref_neg = F.pairwise_distance(neg_embedding, ref_embedding)
            distance_pos.append(distance_ref_pos)
            distance_neg.append(distance_ref_neg)

        return {
            "kendall": self.metrics(torch.cat(distance_pos), torch.cat(distance_neg))
        }

    def predict(
        self, samples: Dict[str, str], cuda: bool = False, show_progress: bool = False
    ) -> (Dict[str, Union[str, float]], List[float]):
        """Function that runs a model prediction,
        
        :param samples: List of dictionaries with 'mt' and 'ref' keys.
        :param cuda: Flag that runs inference using 1 single GPU.
        :param show_progress: Flag to show progress during inference of multiple examples.

        :return: Dictionary with original samples + predicted scores and list of predicted scores
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
                model_inputs.append(self.prepare_sample(batch, inference=True))
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
                mt_input, ref_input = model_input
                if cuda and torch.cuda.is_available():
                    mt_embeddings = self.get_sentence_embedding(
                        **move_to_cuda(mt_input)
                    )
                    ref_embeddings = self.get_sentence_embedding(
                        **move_to_cuda(ref_input)
                    )
                    distances = F.pairwise_distance(mt_embeddings, ref_embeddings).cpu()
                else:
                    mt_embeddings = self.get_sentence_embedding(**mt_input)
                    ref_embeddings = self.get_sentence_embedding(**ref_input)
                    distances = F.pairwise_distance(mt_embeddings, ref_embeddings)

                distances = distances.numpy().tolist()
                for i in range(len(distances)):
                    scores.append(1 / (1 + distances[i]))

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        assert len(scores) == len(samples)
        for i in range(len(scores)):
            samples[i]["predicted_score"] = scores[i]
        return samples, scores

    def get_sentence_embedding(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Auxiliar function that extracts sentence embeddings for
            a single sentence.
        
        :param tokens: sequences [batch_size x seq_len]
        :param lengths: lengths [batch_size]
        
        :return: torch.Tensor [batch_size x hidden_size]
        """
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        if self.trainer and self.trainer.use_dp and self.trainer.num_gpus > 1:
            tokens = tokens[:, : lengths.max()]

        encoder_out = self.encoder(tokens, lengths)
        # for LASER we dont care about the word embeddings
        if self.hparams.encoder_model == "LASER":
            pass
        elif self.scalar_mix:
            embeddings = self.scalar_mix(encoder_out["all_layers"], encoder_out["mask"])
        elif self.layer > 0 and self.layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.layer]
        else:
            raise Exception("Invalid model layer {}.".format(self.layer))

        if self.hparams.pool == "default" or self.hparams.encoder_model == "LASER":
            sentemb = encoder_out["sentemb"]

        if self.hparams.pool == "max":
            sentemb = max_pooling(
                tokens, embeddings, self.encoder.tokenizer.padding_index
            )

        elif self.hparams.pool == "avg":
            sentemb = average_pooling(
                tokens,
                embeddings,
                encoder_out["mask"],
                self.encoder.tokenizer.padding_index,
            )

        elif self.hparams.pool == "cls":
            sentemb = embeddings[:, 0, :]

        else:
            raise Exception("Invalid pooling technique.")

        return sentemb

    def forward(
        self,
        ref_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        neg_tokens: torch.Tensor,
        ref_lengths: torch.Tensor,
        pos_lengths: torch.Tensor,
        neg_lengths: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes the reference, positive samples and negative samples
        and returns embeddings for the triplet.

        :param ref_tokens: reference sequences [batch_size x ref_seq_len]
        :param pos_tokens: positive sequences [batch_size x pos_seq_len]
        :param neg_tokens: negative sequences [batch_size x neg_seq_len]
        :param ref_lengths: reference lengths [batch_size]
        :param pos_lengths: positive lengths [batch_size]
        :param neg_lengths: negative lengths [batch_size]

        :return: Dictionary with model outputs to be passed to the loss function.
        """
        return {
            "ref_sentemb": self.get_sentence_embedding(ref_tokens, ref_lengths),
            "pos_sentemb": self.get_sentence_embedding(pos_tokens, pos_lengths),
            "neg_sentemb": self.get_sentence_embedding(neg_tokens, neg_lengths),
        }
