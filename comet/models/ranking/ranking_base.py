# -*- coding: utf-8 -*-
r"""
Translation Ranking Base Model
==============================
    Abstract base class used to build new ranking systems inside COMET.
    This task consists of ranking "good" translations above "worse" ones.
"""
from argparse import Namespace
from typing import List

import pandas as pd
import torch
import torch.nn as nn

from comet.models.model_base import ModelBase
from comet.models.utils import average_pooling, max_pooling
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
        elif self.layer >= 0 and self.layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.layer]
        else:
            raise Exception("Invalid model layer {}.".format(self.layer))

        if self.hparams.pool == "default" or self.hparams.encoder_model == "LASER":
            sentemb = encoder_out["sentemb"]

        elif self.hparams.pool == "max":
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
