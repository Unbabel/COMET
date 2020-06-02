# -*- coding: utf-8 -*-
r"""
Metric Estimator Model
==============
    Metric Estimator Model estimates a quality score for the hyphotesis (e.g: the MT text)
    by looking only at reference and MT.
"""
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch

from comet.models.estimators import CometEstimator, Estimator
from comet.models.utils import average_pooling, max_pooling
from comet.modules.feedforward import FeedForward
from comet.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors


class MetricEstimator(CometEstimator):
    """
    Estimator class that uses a pretrained encoder to extract features from
    the sequences and then passes those features to a feed forward estimator.

    :param hparams: Namespace containing the hyperparameters.
    """

    def __init__(self, hparams: Namespace,) -> None:
        super().__init__(hparams)

    def _build_model(self) -> Estimator:
        """
        Initializes the estimator architecture.
        """
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

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        Returns:
            - Tuple with 2 dictionaries: model inputs and targets
        or
            - Dictionary with model inputs
        """
        sample = collate_tensors(sample)
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = self.encoder.prepare_sample(sample["ref"])

        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}

        inputs = {**mt_inputs, **ref_inputs}

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def forward(
        self,
        mt_tokens: torch.tensor,
        ref_tokens: torch.tensor,
        mt_lengths: torch.tensor,
        ref_lengths: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes both Source, MT and Reference and returns a quality score.

        :param mt_tokens: MT sequences [batch_size x mt_seq_len]
        :param ref_tokens: REF sequences [batch_size x ref_seq_len]
        :param mt_lengths: MT lengths [batch_size]
        :param ref_lengths: REF lengths [batch_size]

        Return: Dictionary with model outputs to be passed to the loss function.
        """
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        if self.trainer and self.trainer.use_dp and self.trainer.num_gpus > 1:
            mt_tokens = mt_tokens[:, : mt_lengths.max()]
            ref_tokens = ref_tokens[:, : ref_lengths.max()]

        encoder_out_mt = self.encoder(mt_tokens, mt_lengths)
        encoder_out_ref = self.encoder(ref_tokens, ref_lengths)

        # for LASER we dont care about the word embeddings
        if self.hparams.encoder_model == "LASER":
            pass
        elif self.scalar_mix:
            mt_embeddings = self.scalar_mix(
                encoder_out_mt["all_layers"], encoder_out_mt["mask"]
            )
            ref_embeddings = self.scalar_mix(
                encoder_out_ref["all_layers"], encoder_out_ref["mask"]
            )
        elif self.layer > 0 and self.layer < self.encoder.num_layers:
            mt_embeddings = encoder_out_mt["all_layers"][self.layer]
            ref_embeddings = encoder_out_ref["all_layers"][self.layer]
        else:
            raise Exception("Invalid model layer {}.".format(self.layer))

        if self.hparams.pool == "default" or self.hparams.encoder_model == "LASER":
            mt_sentemb = encoder_out_mt["sentemb"]
            ref_sentemb = encoder_out_ref["sentemb"]

        elif self.hparams.pool == "max":
            mt_sentemb = max_pooling(
                mt_tokens, mt_embeddings, self.encoder.tokenizer.padding_index
            )
            ref_sentemb = max_pooling(
                ref_tokens, ref_embeddings, self.encoder.tokenizer.padding_index
            )

        elif self.hparams.pool == "avg":
            mt_sentemb = average_pooling(
                mt_tokens,
                mt_embeddings,
                encoder_out_mt["mask"],
                self.encoder.tokenizer.padding_index,
            )
            ref_sentemb = average_pooling(
                ref_tokens,
                ref_embeddings,
                encoder_out_ref["mask"],
                self.encoder.tokenizer.padding_index,
            )

        elif self.hparams.pool == "cls":
            mt_sentemb = mt_embeddings[:, 0, :]
            ref_sentemb = ref_embeddings[:, 0, :]

        else:
            raise Exception("Invalid pooling technique.")

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        prod_ref = mt_sentemb * ref_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, ref_sentemb, prod_ref, diff_ref), dim=1
        )
        return {"score": self.ff(embedded_sequences)}
