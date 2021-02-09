# -*- coding: utf-8 -*-
r"""
Quality Estimator Model
=======================
    Quality Estimator Model estimates a quality score for the hyphotesis (e.g: the MT text)
    by looking only at source and MT.
"""
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch

from comet.models.estimators import CometEstimator, Estimator
from comet.modules.feedforward import FeedForward
from comet.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors


class QualityEstimator(CometEstimator):
    """
    Estimator class that uses a pretrained encoder to extract features from
    the sequences and then passes those features to a feed forward estimator.

    :param hparams: Namespace containing the hyperparameters.
    """

    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__(hparams)

    def _build_model(self) -> Estimator:
        """
        Initializes the estimator architecture.
        """
        super()._build_model()
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

        self.ff = FeedForward(
            in_dim=self.encoder.output_units * 4,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
        )

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["mt", "src", "score"]]
        df["mt"] = df["mt"].astype(str)
        df["src"] = df["src"].astype(str)
        df["score"] = df["score"].astype(float)
        return df.to_dict("records")

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = collate_tensors(sample)
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        src_inputs = self.encoder.prepare_sample(sample["src"])

        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}

        inputs = {**mt_inputs, **src_inputs}

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def forward(
        self,
        mt_tokens: torch.tensor,
        src_tokens: torch.tensor,
        mt_lengths: torch.tensor,
        src_lengths: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes both Source, MT and returns a quality score.

        :param mt_tokens: MT sequences [batch_size x mt_seq_len]
        :param src_tokens: SRC sequences [batch_size x src_seq_len]
        :param mt_lengths: MT lengths [batch_size]
        :param src_lengths: SRC lengths [batch_size]

        :return: Dictionary with model outputs to be passed to the loss function.
        """
        mt_sentemb = self.get_sentence_embedding(mt_tokens, mt_lengths)
        src_sentemb = self.get_sentence_embedding(src_tokens, src_lengths)

        diff_src = torch.abs(mt_sentemb - src_sentemb)
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, src_sentemb, prod_src, diff_src), dim=1
        )
        return {"score": self.ff(embedded_sequences)}
