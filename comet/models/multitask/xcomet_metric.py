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
XCOMET Metric
==============
    eXplainable Metric is a multitask metric that performs error span detection along with
    sentence-level regression. It can also be used for QE (reference-free evaluation).
"""

from typing import Dict, List, Optional, Union

import torch
from torch import nn

from comet.models.multitask.unified_metric import UnifiedMetric
from comet.models.utils import Prediction
from comet.modules import FeedForward


class XCOMETMetric(UnifiedMetric):
    """eXplainable COMET is same has Unified Metric but overwrites predict function.
    This way we can control better for the models inference.

    To cast back XCOMET models into UnifiedMetric (and vice-versa) we can simply run
    model.__class__ = UnifiedMetric

    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 1.0e-06,
        learning_rate: float = 3.66e-06,
        layerwise_decay: float = 0.983,
        encoder_model: str = "XLM-RoBERTa-XL",
        pretrained_model: str = "facebook/xlm-roberta-xl",
        sent_layer: Union[str, int] = "mix",
        layer_transformation: str = "sparsemax",
        layer_norm: bool = False,
        word_layer: int = 36,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [2560, 1280],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        word_level_training: bool = True,
        error_labels: List[str] = ["minor", "major", "critical"],
        loss_lambda: float = 0.055,
        cross_entropy_weights: Optional[List[float]] = [0.08, 0.486, 0.505, 0.533],
        load_pretrained_weights: bool = True,
    ) -> None:
        super(UnifiedMetric, self).__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            layer=sent_layer,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="xcomet_metric",
            load_pretrained_weights=load_pretrained_weights,
        )
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        assert error_labels == ["minor", "major", "critical"]
        self.hparams.input_segments = ["mt", "src", "ref"]
        self.word_level = True
        self.encoder.labelset = self.label_encoder
        self.hidden2tag = nn.Linear(self.encoder.output_units, self.num_classes)
        self.input_tags = False  # unused

        # By default 3rd input [mt:src:ref] has 50% weight,
        # 2nd input [mt:ref] 33% and 1st input [mt:src] has 16%
        self.input_weights_spans = torch.tensor([0.1667, 0.3333, 0.5])

        # The final score is a weighted average between different scores.
        # First weight is for [mt:src], second for [mt:ref], third for [mt:src:ref] and
        # last weight is for MQM computed score.
        self.score_weights = [0.12, 0.33, 0.33, 0.22]

        # This is None by default and we will use argmax during decoding yet, to control over
        # precision and recall we can set it to another value.
        self.decoding_threshold = None

        self.init_losses()
        self.save_hyperparameters()

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """PyTorch Lightning predict_step

        Args:
            batch (Dict[str, torch.Tensor]): The output of your prepare_sample function
            batch_idx (Optional[int], optional): Integer displaying which batch this is
                Defaults to None.
            dataloader_idx (Optional[int], optional): Integer displaying which
                dataloader this is. Defaults to None.

        Returns:
            Prediction: Model Prediction
        """

        def _compute_mqm_from_spans(error_spans):
            scores = []
            for sentence_spans in error_spans:
                sentence_score = 0
                for annotation in sentence_spans:
                    if annotation["severity"] == "minor":
                        sentence_score += 1
                    elif annotation["severity"] == "major":
                        sentence_score += 5
                    elif annotation["severity"] == "critical":
                        sentence_score += 10

                if sentence_score > 25:
                    sentence_score = 25

                scores.append(sentence_score)

            # Rescale between 0 and 1
            scores = (torch.tensor(scores) * -1 + 25) / 25
            return scores

        # XCOMET is suposed to be used with a reference thus 3 different inputs.
        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Regression scores are weighted with self.score_weights
            regression_scores = torch.stack(
                [
                    torch.where(pred.score > 1.0, 1.0, pred.score) * w
                    for pred, w in zip(predictions, self.score_weights[:3])
                ],
                dim=0,
            ).sum(dim=0)
            mt_mask = batch[0]["label_ids"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()

            # Weighted average of the softmax probs along the different inputs.
            subword_probs = [
                nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :] * w
                for w, o in zip(self.input_weights_spans, predictions)
            ]
            subword_probs = torch.sum(torch.stack(subword_probs), dim=0)
            error_spans = self.decode(
                subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
            )
            mqm_scores = _compute_mqm_from_spans(error_spans)
            final_scores = (
                regression_scores
                + mqm_scores.to(regression_scores.device) * self.score_weights[3]
            )
            batch_prediction = Prediction(
                scores=final_scores,
                metadata=Prediction(
                    src_scores=predictions[0].score,
                    ref_scores=predictions[1].score,
                    unified_scores=predictions[2].score,
                    mqm_scores=mqm_scores,
                    error_spans=error_spans,
                ),
            )

        # XCOMET if reference is not available we fall back to QE model.
        else:
            model_output = self.forward(**batch[0])
            regression_score = torch.where(
                model_output.score > 1.0, 1.0, model_output.score
            )
            mt_mask = batch[0]["label_ids"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            subword_probs = nn.functional.softmax(model_output.logits, dim=2)[
                :, :seq_len, :
            ]
            error_spans = self.decode(
                subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
            )
            mqm_scores = _compute_mqm_from_spans(error_spans)
            final_scores = (
                regression_score * sum(self.score_weights[:3])
                + mqm_scores.to(regression_score.device) * self.score_weights[3]
            )
            batch_prediction = Prediction(
                scores=final_scores,
                metadata=Prediction(
                    src_scores=regression_score,
                    mqm_scores=mqm_scores,
                    error_spans=error_spans,
                ),
            )
        return batch_prediction
