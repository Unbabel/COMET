# -*- coding: utf-8 -*-
r"""
Comet Estimator Model
================================
    Comet Estimator predicts a quality score for the 
    hyphotesis (e.g: the MT text) by looking at reference, source and MT.
"""
import random
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch

from comet.models.estimators.estimator_base import Estimator
from comet.modules.feedforward import FeedForward
from comet.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors


class CometEstimator(Estimator):
    """
    Estimator class that uses a pretrained encoder to extract features from
    the sequences and then passes those features to a feed forward estimator.

    :param hparams: Namespace containing the hyperparameters.
    """

    class ModelConfig(Estimator.ModelConfig):
        switch_prob: float = 0.0

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

        input_emb_sz = (
            self.encoder.output_units * 6
            if self.hparams.pool != "cls+avg"
            else self.encoder.output_units * 2 * 6
        )

        self.ff = FeedForward(
            in_dim=input_emb_sz,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=(
                self.hparams.final_activation
                if hasattr(
                    self.hparams, "final_activation"
                )  # compatability with older checkpoints!
                else "Sigmoid"
            ),
        )

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """ Sets different Learning rates for different parameter groups. """
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        ff_parameters = [
            {"params": self.ff.parameters(), "lr": self.hparams.learning_rate}
        ]

        if self.hparams.encoder_model != "LASER" and self.scalar_mix:
            scalar_mix_parameters = [
                {
                    "params": self.scalar_mix.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]

            optimizer = self._build_optimizer(
                layer_parameters + ff_parameters + scalar_mix_parameters
            )
        else:
            optimizer = self._build_optimizer(layer_parameters + ff_parameters)
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

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = collate_tensors(sample)
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = self.encoder.prepare_sample(sample["ref"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}

        if "alt" in sample:
            alt_inputs = self.encoder.prepare_sample(sample["alt"])
            alt_inputs = {"alt_" + k: v for k, v in alt_inputs.items()}
            inputs = {**src_inputs, **mt_inputs, **ref_inputs, **alt_inputs}

        else:
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
        alt_tokens: torch.tensor = None,
        alt_lengths: torch.tensor = None,
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

        :param alt_tokens: Alternative REF sequences [batch_size x alt_seq_len]
        :param alt_lengths: Alternative REF lengths [batch_size]

        :return: Dictionary with model outputs to be passed to the loss function.
        """
        src_sentemb = self.get_sentence_embedding(src_tokens, src_lengths)
        mt_sentemb = self.get_sentence_embedding(mt_tokens, mt_lengths)
        ref_sentemb = self.get_sentence_embedding(ref_tokens, ref_lengths)

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        if (
            not hasattr(
                self.hparams, "switch_prob"
            )  # compatability with older checkpoints!
            or self.hparams.switch_prob <= 0.0
        ):
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
            )
            score = self.ff(embedded_sequences)

            if (alt_tokens is not None) and (alt_lengths is not None):

                alt_sentemb = self.get_sentence_embedding(alt_tokens, alt_lengths)

                diff_alt = torch.abs(mt_sentemb - alt_sentemb)
                prod_alt = mt_sentemb * alt_sentemb

                embedded_sequences = torch.cat(
                    (mt_sentemb, alt_sentemb, prod_alt, diff_alt, prod_src, diff_src),
                    dim=1,
                )
                score = (score + self.ff(embedded_sequences)) / 2

            return {"score": score}

        if self.training:
            switch = random.random() < self.hparams.switch_prob

            if switch:
                embedded_sequences = torch.cat(
                    (mt_sentemb, ref_sentemb, prod_src, diff_src, prod_ref, diff_ref),
                    dim=1,
                )
            else:
                embedded_sequences = torch.cat(
                    (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
                    dim=1,
                )
            return {"score": self.ff(embedded_sequences)}

        elif (alt_tokens is not None) and (alt_lengths is not None):
            # Switcheroo Inference!
            alt_sentemb = self.get_sentence_embedding(alt_tokens, alt_lengths)
            diff_alt = torch.abs(mt_sentemb - alt_sentemb)
            prod_alt = mt_sentemb * alt_sentemb

            # Source + MT + Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
            )
            src_mt_ref = self.ff(embedded_sequences)

            # Reference + MT + Source
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_src, diff_src, prod_ref, diff_ref), dim=1
            )
            ref_mt_src = self.ff(embedded_sequences)

            # Source + MT + Alternative Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, alt_sentemb, prod_alt, diff_alt, prod_src, diff_src), dim=1
            )
            src_mt_alt = self.ff(embedded_sequences)

            # Alternative Reference + MT + Source
            embedded_sequences = torch.cat(
                (mt_sentemb, alt_sentemb, prod_src, diff_src, prod_alt, diff_alt), dim=1
            )
            alt_mt_src = self.ff(embedded_sequences)

            # Alternative Reference + MT + Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, alt_sentemb, prod_alt, diff_alt, prod_ref, diff_ref), dim=1
            )
            alt_mt_ref = self.ff(embedded_sequences)

            # Reference + MT + Alternative Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_alt, diff_alt), dim=1
            )
            ref_mt_alt = self.ff(embedded_sequences)

            score = torch.stack(
                [src_mt_ref, ref_mt_src, src_mt_alt, alt_mt_src, alt_mt_ref, ref_mt_alt]
            )
            confidence = 1 - score.std(dim=0)

            return {"score": score.mean(dim=0) * confidence, "confidence": confidence}

        else:
            # Usual scoring
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
            )
            score = self.ff(embedded_sequences) * (1 - self.hparams.switch_prob)

            # Switch src and reference embeddings
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_src, diff_src, prod_ref, diff_ref), dim=1
            )
            return {
                "score": score + self.ff(embedded_sequences) * self.hparams.switch_prob
            }
