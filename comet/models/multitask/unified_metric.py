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
Unified Metric
==============
    Unified Metric is a multitask metric that performs word-level and segment-level 
    evaluation in a multitask manner. It can also be used with and without reference 
    translations.
    
    Inspired on [UniTE](https://arxiv.org/pdf/2204.13346.pdf)
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
from transformers.optimization import (Adafactor,
                                       get_constant_schedule_with_warmup)

from comet.models.base import CometModel
from comet.models.metrics import MCCMetric, RegressionMetrics
from comet.models.utils import LabelSet, Prediction, Target
from comet.modules import FeedForward


class UnifiedMetric(CometModel):
    """UnifiedMetric is a multitask metric that performs word-level classification along
    with sentence-level regression. This metric has the ability to work with and without
    reference translations.

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.9.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to True.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        warmup_steps (int): Warmup steps for LR scheduler.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 3.0e-06.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 3.0e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'microsoft/infoxlm-large'.
        sent_layer (Union[str, int]): Encoder layer to be used for regression task ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'sparsemax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'False'.
        word_layer (int): Encoder layer to be used for word-level classification. Defaults
            to 24.
        loss (str): Loss function to be used. Defaults to 'mse'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
        hidden_sizes (List[int]): Size of hidden layers used in the regression head.
            Defaults to [3072, 1024].
        activations (Optional[str]): Activation function used in the regression head.
            Defaults to 'Tanh'.
        final_activation (Optional[str]): Activation function used in the last layer of
            the regression head. Defaults to None.
        input_segments (Optional[List[str]]): List with input segment names to be used.
            Defaults to ["mt", "src", "ref"].
        word_level_training (bool): If True, the model is trained with multitask
            objective. Defaults to False.
        loss_lambda (float): Weight assigned to the word-level loss. Defaults to 0.65.
        error_labels (List[str]): List of severity labels for word-level training.
            Defaults to ['minor', 'major'].
        cross_entropy_weights (Optional[List[float]]):  Weights for each label in the
            error_labels + weight for the default 'O' label. Defaults to None.
        load_pretrained_weights (Bool): If set to False it avoids loading the weights
            of the pretrained model (e.g. XLM-R) before it loads the COMET checkpoint
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.9,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 3.0e-06,
        learning_rate: float = 3.0e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "microsoft/infoxlm-large",
        sent_layer: Union[str, int] = "mix",
        layer_transformation: str = "sparsemax",
        layer_norm: bool = True,
        word_layer: int = 24,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        input_segments: List[str] = ["mt", "src", "ref"],
        word_level_training: bool = False,
        loss_lambda: float = 0.65,
        error_labels: List[str] = ["minor", "major"],
        cross_entropy_weights: Optional[List[float]] = None,
        load_pretrained_weights: bool = True,
    ) -> None:
        super().__init__(
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
            class_identifier="unified_metric",
            load_pretrained_weights=load_pretrained_weights,
        )
        self.save_hyperparameters()
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        self.word_level = word_level_training
        if word_level_training:
            self.encoder.labelset = self.label_encoder
            self.hidden2tag = nn.Linear(self.encoder.output_units, self.num_classes)

        if len(self.hparams.input_segments) == 3:
            # By default 3rd input [mt:src:ref] has 50% weight,
            # 2nd input [mt:ref] 33% and 1st input [mt:src] has 16%
            self.input_weights_spans = torch.tensor([0.1667, 0.3333, 0.5])

        # This is None by default and we will use argmax during decoding yet, to control over
        # precision and recall we can set it to another value.
        self.decoding_threshold = None
        self.init_losses()

    def set_input_weights_spans(self, weights: torch.Tensor):
        """Used to set input weights in another.

        Args:
            weights (torch.Tensor): Tensor (size 3) with input weights."""
        assert weights.shape == (3,)
        self.input_weights_spans = weights

    def set_decoding_threshold(self, threshold: float = 0.5):
        """Used during decoding to control over precision and recall. It always assumes
        that the first label corresponds to "no-error" and the remaining labels
        correspond to different severities.

        When set to a value, the following rule is used to decide if a subword belong to
        an error: torch.sum(probs[1:]) > threshold.

        Args:
            threshold (float): Threshold to decide when"""
        self.decoding_threshold = threshold

    def init_metrics(self):
        """Initializes training and validation metrics"""
        # Train and Dev correlation metrics
        self.train_corr = RegressionMetrics(prefix="train")
        self.val_corr = nn.ModuleList(
            [RegressionMetrics(prefix=d) for d in self.hparams.validation_data]
        )
        if self.hparams.word_level_training:
            self.label_encoder = LabelSet(self.hparams.error_labels)
            self.num_classes = len(self.label_encoder.labels_to_id)
            # Train and Dev MCC
            self.train_mcc = MCCMetric(num_classes=self.num_classes, prefix="train")
            self.val_mcc = nn.ModuleList(
                [
                    MCCMetric(num_classes=self.num_classes, prefix=d)
                    for d in self.hparams.validation_data
                ]
            )

    def init_losses(self) -> None:
        """Initializes Loss functions to be used."""
        self.sentloss = nn.MSELoss()
        if self.word_level:
            if self.hparams.cross_entropy_weights:
                assert len(self.hparams.cross_entropy_weights) == self.num_classes
                loss_weights = torch.tensor(self.hparams.cross_entropy_weights)
            else:
                loss_weights = None

            self.wordloss = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=-1, weight=loss_weights
            )

    def requires_references(self) -> bool:
        """Unified models can be developed to exclusively use [mt, ref] or to use both
        [mt, src, ref]. Models developed to use the source will work in a quality
        estimation scenario but models trained with [mt, ref] won't!

        Return:
            [bool]: True if the model was trained to work exclusively with references.
        """
        if self.hparams.input_segments == ["mt", "ref"]:
            return True
        return False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to initialize a training Optimizer and learning
        rate scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
                List with Optimizers and a List with lr_schedulers.
        """
        params = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        params += [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.word_level:
            params += [
                {
                    "params": self.hidden2tag.parameters(),
                    "lr": self.hparams.learning_rate,
                },
            ]

        if self.layerwise_attention:
            params += [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        # If warmup setps are not defined we don't need a scheduler.
        if self.hparams.warmup_steps < 1:
            return [optimizer], []

        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
        )
        return [optimizer], [scheduler]

    def read_training_data(self, path: str) -> List[dict]:
        """Reads a csv file with training data.

        Args:
            path (str): Path to the csv file to be loaded.

        Returns:
            List[dict]: Returns a list of training examples.
        """
        df = pd.read_csv(path)
        # Deep copy input segments
        columns = self.hparams.input_segments[:]
        # Make sure everything except score is str type
        for col in columns:
            df[col] = df[col].astype(str)
        columns.append("score")
        df["score"] = df["score"].astype("float16")
        df = df[columns]
        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Reads a csv file with validation data.

        Args:
            path (str): Path to the csv file to be loaded.

        Returns:
            List[dict]: Returns a list of validation examples.
        """
        df = pd.read_csv(path)
        # Deep copy input segments
        columns = self.hparams.input_segments[:]
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
        # Make sure everything except score is str type
        for col in columns:
            df[col] = df[col].astype(str)
        columns.append("score")
        df["score"] = df["score"].astype("float16")
        df = df[columns]
        return df.to_dict("records")

    def concat_inputs(
        self,
        input_sequences: Tuple[Dict[str, torch.Tensor]],
        unified_input: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Prepares tokenized src, ref and mt for joint encoding by putting
        everything into a single contiguous sequence.

        Args:
            input_sequences (Tuple[Dict[str, torch.Tensor]]): Tokenized Source, MT and
                Reference.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Contiguous sequence.
        """
        model_inputs = OrderedDict()
        # If we are using source and reference we will have to create 3 different input
        if unified_input:
            mt_src, mt_ref = input_sequences[:2], [
                input_sequences[0],
                input_sequences[2],
            ]
            src_input, _, _ = self.encoder.concat_sequences(
                mt_src, return_label_ids=self.word_level
            )
            ref_input, _, _ = self.encoder.concat_sequences(
                mt_ref, return_label_ids=self.word_level
            )
            full_input, _, _ = self.encoder.concat_sequences(
                input_sequences, return_label_ids=self.word_level
            )
            model_inputs["inputs"] = (src_input, ref_input, full_input)
            model_inputs["mt_length"] = input_sequences[0]["attention_mask"].sum(dim=1)
            return model_inputs

        # Otherwise we will have one single input sequence that concatenates the MT
        # with SRC/REF.
        else:
            model_inputs["inputs"] = (
                self.encoder.concat_sequences(
                    input_sequences, return_label_ids=self.word_level
                )[0],
            )
            model_inputs["mt_length"] = input_sequences[0]["attention_mask"].sum(dim=1)
        return model_inputs

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "fit"
    ) -> Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Tokenizes input data and prepares targets for training.

        Args:
            sample (List[Dict[str, Union[str, float]]]): Mini-batch
            stage (str, optional): Model stage ('train' or 'predict'). Defaults to "fit".

        Returns:
            Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: Model input
                and targets.
        """
        inputs = {k: [d[k] for d in sample] for k in sample[0]}
        input_sequences = [
            self.encoder.prepare_sample(inputs["mt"], self.word_level, None),
        ]

        src_input, ref_input = False, False
        if ("src" in inputs) and ("src" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(inputs["src"]))
            src_input = True

        if ("ref" in inputs) and ("ref" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(inputs["ref"]))
            ref_input = True

        unified_input = src_input and ref_input
        model_inputs = self.concat_inputs(input_sequences, unified_input)
        if stage == "predict":
            return model_inputs["inputs"]

        scores = [float(s) for s in inputs["score"]]
        targets = Target(score=torch.tensor(scores, dtype=torch.float))

        if "system" in inputs:
            targets["system"] = inputs["system"]

        if self.word_level:
            # Labels will be the same accross all inputs because we are only
            # doing sequence tagging on the MT. We will only use the mask corresponding
            # to the MT segment.
            seq_len = model_inputs["mt_length"].max()
            targets["mt_length"] = model_inputs["mt_length"]
            targets["labels"] = model_inputs["inputs"][0]["label_ids"][:, :seq_len]

        return model_inputs["inputs"], targets

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            input_ids (torch.Tensor): Input sequence.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids for
                BERT-like models. Defaults to None.

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Dict[str, torch.Tensor]: Sentence scores and word-level logits (if
                word_level_training = True)
        """
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )

        # Word embeddings used for the word-level classification task
        if self.word_level:
            if (
                isinstance(self.hparams.word_layer, int)
                and 0 <= self.hparams.word_layer < self.encoder.num_layers
            ):
                wordemb = encoder_out["all_layers"][self.hparams.word_layer]
            else:
                raise Exception(
                    "Invalid model word layer {}.".format(self.hparams.word_layer)
                )

        # embeddings used for the sentence-level regression task
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )
        elif (
            isinstance(self.hparams.sent_layer, int)
            and 0 <= self.hparams.sent_layer < self.encoder.num_layers
        ):
            embeddings = encoder_out["all_layers"][self.hparams.sent_layer]
        else:
            raise Exception(
                "Invalid model sent layer {}.".format(self.hparams.word_layer)
            )
        sentemb = embeddings[:, 0, :] # We take the CLS token as sentence-embedding
        
        if self.word_level:
            sentence_output = self.estimator(sentemb)
            word_output = self.hidden2tag(wordemb)
            return Prediction(score=sentence_output.view(-1), logits=word_output)

        return Prediction(score=self.estimator(sentemb).view(-1))

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """Receives model batch prediction and respective targets and computes
        a loss value

        Args:
            prediction (Prediction): Batch prediction
            target (Target): Batch targets

        Returns:
            torch.Tensor: Loss value
        """
        sentence_loss = self.sentloss(prediction.score, target.score)
        if self.word_level:
            predictions = prediction.logits.reshape(-1, self.num_classes)
            targets = target.labels.reshape(-1).type(torch.LongTensor).cuda()
            word_loss = self.wordloss(predictions, targets)
            return sentence_loss * (1 - self.hparams.loss_lambda) + word_loss * (
                self.hparams.loss_lambda
            )
        else:
            return sentence_loss

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_nb: int
    ) -> torch.Tensor:
        """Pytorch Lightning training_step.

        Args:
            batch (Tuple[Dict[str, torch.Tensor]]): The output of your prepare_sample
                function.
            batch_nb (int): Integer displaying which batch this is.

        Returns:
            torch.Tensor: Loss value
        """
        batch_input, batch_target = batch
        # When using references our loss will be computed with 3 different forward
        # passes. Loss = L src + L ref + L src_and_ref
        predictions = [self.forward(**input_seq) for input_seq in batch_input]
        loss_value = 0
        for pred in predictions:
            if self.word_level:
                seq_len = batch_target.mt_length.max()
                pred.logits = pred.logits[:, :seq_len, :]
            loss_value += self.compute_loss(pred, batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.first_epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log(
            "train_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            batch_size=batch_target.score.shape[0],
            sync_dist=True,
        )
        return loss_value

    def validation_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_nb: int, dataloader_idx: int
    ) -> None:
        """Pytorch Lightning validation_step.

        Args:
            batch (Tuple[Dict[str, torch.Tensor]]): The output of your prepare_sample
                function.
            batch_nb (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this is.
        """
        batch_input, batch_target = batch
        predictions = [self.forward(**input_seq) for input_seq in batch_input]
        # Final score is the average of the 3 scores when using references.
        scores = torch.stack([pred.score for pred in predictions], dim=0).mean(dim=0)
        if self.word_level:
            seq_len = batch_target.mt_length.max()
            # Final probs for each word is the average of the 3 forward passes.
            subword_probs = [
                nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :]
                for o in predictions
            ]
            subword_probs = torch.mean(torch.stack(subword_probs), dim=0)
            # Removing masked targets and the corresponding logits.
            # This includes subwords and padded tokens.
            probs = subword_probs.reshape(-1, self.num_classes)
            targets = batch_target.labels.reshape(-1)
            mask = targets != -1
            probs, targets = probs[mask, :], targets[mask].int()

        if dataloader_idx == 0:
            self.train_corr.update(scores, batch_target.score)
            if self.word_level:
                self.train_mcc.update(probs, targets)

        elif dataloader_idx > 0:
            self.val_corr[dataloader_idx - 1].update(
                scores,
                batch_target.score,
                batch_target["system"] if "system" in batch_target else None,
            )
            if self.word_level:
                self.val_mcc[dataloader_idx - 1].update(probs, targets)

    # Overwriting this method to log correlation and classification metrics
    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics."""
        self.log_dict(self.train_corr.compute(), prog_bar=False, sync_dist=True)
        self.train_corr.reset()

        if self.word_level:
            self.log_dict(self.train_mcc.compute(), prog_bar=False, sync_dist=True)
            self.train_mcc.reset()

        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            corr_metrics = self.val_corr[i].compute()
            self.val_corr[i].reset()
            if self.word_level:
                cls_metric = self.val_mcc[i].compute()
                self.val_mcc[i].reset()
                results = {**corr_metrics, **cls_metric}
            else:
                results = corr_metrics

            # Log to tensorboard the results for this validation set.
            self.log_dict(results, prog_bar=False, sync_dist=True)
            val_metrics.append(results)

        average_results = {"val_" + k.split("_")[-1]: [] for k in val_metrics[0].keys()}
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + k.split("_")[-1]].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()},
            prog_bar=True,
            sync_dist=True,
        )

    def set_mc_dropout(self, value: int):
        """Sets Monte Carlo Dropout runs per sample.

        Args:
            value (int): number of runs per sample.
        """
        raise NotImplementedError("MCD not implemented for this model!")

    def decode(
        self,
        subword_probs: torch.Tensor,
        input_ids: torch.Tensor,
        mt_offsets: torch.Tensor,
    ) -> List[Dict]:
        """Decode error spans from subwords.

        Args:
            subword_probs (torch.Tensor): probabilities of each label for each subword.
            input_ids (torch.Tensor): input ids from the model.
            mt_offsets (torch.Tensor): subword offsets.

        Return:
            List with of dictionaries with text, start, end, severity and a
            confidence score which is the average of the probs for that label.
        """
        decoded_output = []
        for i in range(len(mt_offsets)):
            seq_len = len(mt_offsets[i])
            error_spans, in_span, span = [], False, {}
            for token_id, probs, token_offset in zip(
                input_ids[i, :seq_len], subword_probs[i][:seq_len], mt_offsets[i]
            ):
                if self.decoding_threshold:
                    if torch.sum(probs[1:]) > self.decoding_threshold:
                        probability, label_value = torch.topk(probs[1:], 1)
                        label_value += 1  # offset from removing label 0
                    else:
                        # This is just to ensure same format but at this point
                        # we will only look at label 0 and its prob
                        probability, label_value = torch.topk(probs[0], 1)
                else:
                    probability, label_value = torch.topk(probs, 1)

                # Some torch versions topk returns a shape 1 tensor with only
                # a item inside
                label_value = (
                    label_value.item()
                    if label_value.dim() < 1
                    else label_value[0].item()
                )
                label = self.label_encoder.ids_to_label.get(label_value)
                # Label set:
                # O I-minor I-major
                # Begin of annotation span
                if label.startswith("I") and not in_span:
                    in_span = True
                    span["tokens"] = [
                        token_id,
                    ]
                    span["severity"] = label.split("-")[1]
                    span["offset"] = list(token_offset)
                    span["confidence"] = [
                        probability,
                    ]

                # Inside an annotation span
                elif label.startswith("I") and in_span:
                    span["tokens"].append(token_id)
                    span["confidence"].append(probability)
                    # Update offset end
                    span["offset"][1] = token_offset[1]

                # annotation span finished.
                elif label == "O" and in_span:
                    error_spans.append(span)
                    in_span, span = False, {}

            sentence_output = []
            for span in error_spans:
                sentence_output.append(
                    {
                        "text": self.encoder.tokenizer.decode(span["tokens"]),
                        "confidence": torch.concat(span["confidence"]).mean().item(),
                        "severity": span["severity"],
                        "start": span["offset"][0],
                        "end": span["offset"][1],
                    }
                )
            decoded_output.append(sentence_output)
        return decoded_output

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
        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Final score is the average of the 3 scores!
            avg_scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            batch_prediction = Prediction(
                scores=avg_scores,
                metadata=Prediction(
                    src_scores=predictions[0].score,
                    ref_scores=predictions[1].score,
                    unified_scores=predictions[2].score,
                ),
            )
            if self.word_level:
                mt_mask = batch[0]["label_ids"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = [
                    nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :] * w
                    for w, o in zip(self.input_weights_spans, predictions)
                ]
                subword_probs = torch.sum(torch.stack(subword_probs), dim=0)
                error_spans = self.decode(
                    subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
                )
                batch_prediction.metadata["error_spans"] = error_spans

        else:
            model_output = self.forward(**batch[0])
            batch_prediction = Prediction(scores=model_output.score)
            if self.word_level:
                mt_mask = batch[0]["label_ids"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = nn.functional.softmax(model_output.logits, dim=2)[
                    :, :seq_len, :
                ]
                error_spans = self.decode(
                    subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
                )
                batch_prediction = Prediction(
                    scores=model_output.score,
                    metadata=Prediction(error_spans=error_spans),
                )
        return batch_prediction
