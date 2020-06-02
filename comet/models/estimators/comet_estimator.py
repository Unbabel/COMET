# -*- coding: utf-8 -*-
r"""
Comet Estimator Model
==============
    Comet Estimator predicts a quality score for the 
    hyphotesis (e.g: the MT text) by looking at reference, source and MT.
"""
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch

from comet.models.estimators.estimator_base import Estimator
from comet.models.utils import average_pooling, max_pooling
from comet.modules.feedforward import FeedForward
from comet.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors


class CometEstimator(Estimator):
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
            in_dim=self.encoder.output_units * 6,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
        )

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.ff.parameters()},
            {
                "params": self.encoder.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        if self.hparams.encoder_model != "LASER" and self.scalar_mix:
            parameters.append(
                {
                    "params": self.scalar_mix.parameters(),
                    "lr": self.hparams.encoder_learning_rate,
                }
            )
        optimizer = self._build_optimizer(parameters)
        scheduler = self._build_scheduler(optimizer)
        return [optimizer], [scheduler]

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
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = self.encoder.prepare_sample(sample["ref"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        inputs = {**src_inputs, **mt_inputs, **ref_inputs}

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def forward(
        self,
        src_tokens: torch.tensor,
        mt_tokens: torch.tensor,
        ref_tokens: torch.tensor,
        src_lengths: torch.tensor,
        mt_lengths: torch.tensor,
        ref_lengths: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes both Source, MT and Reference and returns a quality score.

        :param src_tokens: SRC sequences [batch_size x src_seq_len]
        :param mt_tokens: MT sequences [batch_size x mt_seq_len]
        :param ref_tokens: REF sequences [batch_size x ref_seq_len]
        :param src_lengths: SRC lengths [batch_size]
        :param mt_lengths: MT lengths [batch_size]
        :param ref_lengths: REF lengths [batch_size]

        Return: Dictionary with model outputs to be passed to the loss function.
        """
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        if self.trainer and self.trainer.use_dp and self.trainer.num_gpus > 1:
            src_tokens = src_tokens[:, : src_lengths.max()]
            mt_tokens = mt_tokens[:, : mt_lengths.max()]
            ref_tokens = ref_tokens[:, : ref_lengths.max()]

        encoder_out_src = self.encoder(src_tokens, src_lengths)
        encoder_out_mt = self.encoder(mt_tokens, mt_lengths)
        encoder_out_ref = self.encoder(ref_tokens, ref_lengths)

        # for LASER we dont care about the word embeddings
        if self.hparams.encoder_model == "LASER":
            pass
        elif self.scalar_mix:
            src_embeddings = self.scalar_mix(
                encoder_out_src["all_layers"], encoder_out_src["mask"]
            )
            mt_embeddings = self.scalar_mix(
                encoder_out_mt["all_layers"], encoder_out_mt["mask"]
            )
            ref_embeddings = self.scalar_mix(
                encoder_out_ref["all_layers"], encoder_out_ref["mask"]
            )
        elif self.layer > 0 and self.layer < self.encoder.num_layers:
            src_embeddings = encoder_out_src["all_layers"][self.layer]
            mt_embeddings = encoder_out_mt["all_layers"][self.layer]
            ref_embeddings = encoder_out_ref["all_layers"][self.layer]
        else:
            raise Exception("Invalid model layer {}.".format(self.layer))

        if self.hparams.pool == "default" or self.hparams.encoder_model == "LASER":
            src_sentemb = encoder_out_src["sentemb"]
            mt_sentemb = encoder_out_mt["sentemb"]
            ref_sentemb = encoder_out_ref["sentemb"]

        elif self.hparams.pool == "max":
            src_sentemb = max_pooling(
                src_tokens, src_embeddings, self.encoder.tokenizer.padding_index
            )
            mt_sentemb = max_pooling(
                mt_tokens, mt_embeddings, self.encoder.tokenizer.padding_index
            )
            ref_sentemb = max_pooling(
                ref_tokens, ref_embeddings, self.encoder.tokenizer.padding_index
            )

        elif self.hparams.pool == "avg":
            src_sentemb = average_pooling(
                src_tokens,
                src_embeddings,
                encoder_out_src["mask"],
                self.encoder.tokenizer.padding_index,
            )
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
            src_sentemb = src_embeddings[:, 0, :]
            mt_sentemb = mt_embeddings[:, 0, :]
            ref_sentemb = ref_embeddings[:, 0, :]

        else:
            raise Exception("Invalid pooling technique.")

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
        )
        return {"score": self.ff(embedded_sequences)}
