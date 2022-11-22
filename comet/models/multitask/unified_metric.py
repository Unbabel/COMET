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

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers.optimization import Adafactor

from comet.models.base import CometModel
from comet.models.metrics import MCCMetric, RegressionMetrics
from comet.models.utils import LabelEncoder, Prediction, Target
from comet.modules import FeedForward, LayerwiseAttention


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
        word_weights (List[float]): Loss weight for OK/BAD tags. Defaults to [0.15,
            0.85].
        loss_lambda (foat): Weight assigned to the word-level loss. Defaults to 0.65.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.9,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
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
        word_weights: List[float] = [0.15, 0.85],
        loss_lambda: float = 0.65,
    ) -> None:
        super().__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
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
        )
        self.save_hyperparameters()
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        # CZ-TODO: remove hardcoding of num_classes for wordlevel
        self.hidden2tag = nn.Linear(self.encoder.output_units, 2)

        if self.hparams.sent_layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                layer_transformation=layer_transformation,
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=self.hparams.layer_norm,
            )
        else:
            self.layerwise_attention = None

        self.input_segments = input_segments
        self.word_level = word_level_training
        if word_level_training:
            self.label_encoder = LabelEncoder(reserved_labels=["OK", "BAD"])
        self.init_losses()

    def init_metrics(self):
        """Initializes training and validation metrics"""
        # Train and Dev correlation metrics
        self.train_corr = RegressionMetrics(prefix="train")
        self.val_corr = nn.ModuleList(
            [RegressionMetrics(prefix=d) for d in self.hparams.validation_data]
        )
        if self.hparams.word_level_training:
            # Train and Dev MCC
            self.train_mcc = MCCMetric(num_classes=2, prefix="train")
            self.val_mcc = nn.ModuleList(
                [MCCMetric(num_classes=2, prefix=d) for d in self.hparams.validation_data]
            )

    def init_losses(self) -> None:
        """Initializes Loss functions to be used."""
        self.sentloss = nn.MSELoss()
        if self.word_level:
            self.wordloss = nn.CrossEntropyLoss(
                reduction="mean",
                weight=torch.tensor(self.hparams.word_weights),
                ignore_index=-1,
            )

    def is_referenceless(self) -> bool:
        return True

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to initialize a training Optimizer and learning
        rate scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
                List with Optimizers and a List with lr_schedulers.
        """
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]

        word_level_parameters = [
            {"params": self.hidden2tag.parameters(), "lr": self.hparams.learning_rate},
        ]

        layerwise_attn_params = []
        if self.layerwise_attention:
            layerwise_attn_params.append(
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            )

        if self.layerwise_attention:
            params = (
                layer_parameters
                + top_layers_parameters
                + word_level_parameters
                + layerwise_attn_params
            )
        else:
            params = layer_parameters + top_layers_parameters + word_level_parameters

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        return [optimizer], []

    def read_training_data(self, path: str) -> List[dict]:
        """Reads a csv file with training data.

        Args:
            path (str): Path to the csv file to be loaded.

        Returns:
            List[dict]: Returns a list of training examples.
        """
        df = pd.read_csv(path)
        # Deep copy input segments
        columns = self.input_segments[:]
        # Add word level labels if word_level
        if self.word_level:
            columns.append("labels")
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
        columns = self.input_segments[:]
        # Add word level labels if word_level
        if self.word_level:
            columns.append("labels")
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

    def prepare_target(
        self, labels: List[List[int]], subword_masks: List[List[float]], max_len: int
    ) -> torch.Tensor:
        """Prepares the target labels for word-level training.

        Args:
            labels (List[List[int]]): Encoded OK/BAD tags
            subword_masks (List[List[float]]): Mask indicating the which tokens are
                subword prefixes.
            max_len (int): Longest sequence

        Returns:
            torch.Tensor: Word-level targets.
        """
        expanded_labels = torch.sub(
            torch.zeros(subword_masks.size(0), max_len),
            torch.ones(subword_masks.size(0), max_len),
            alpha=1,
        )
        for i in range(len(subword_masks)):
            cnt = 0
            for k in range(max_len):
                if subword_masks[i][k] == 1:
                    expanded_labels[i][k] = labels[i][cnt]
                    cnt += 1
                    
        return expanded_labels

    def concat_inputs(
        self, input_sequences: Tuple[Dict[str, torch.Tensor]]
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
        if len(input_sequences) == 3:
            mt_src, mt_ref = input_sequences[:1], [
                input_sequences[0],
                input_sequences[2],
            ]
            src_input, _, src_max_len = self.encoder.concat_sequences(
                mt_src, word_outputs=self.word_level
            )
            ref_input, _, ref_max_len = self.encoder.concat_sequences(
                mt_ref, word_outputs=self.word_level
            )
            full_input, _, full_max_len = self.encoder.concat_sequences(
                input_sequences, word_outputs=self.word_level
            )
            mt_length = input_sequences[0]["attention_mask"].sum(dim=1)
            
            src_input["mt_length"] = mt_length
            ref_input["mt_length"] = mt_length
            full_input["mt_length"] = mt_length
            model_inputs["inputs"] = (src_input, ref_input, full_input)
            
            if self.word_level:
                shorter_input = np.argmin([src_max_len, ref_max_len, full_max_len])
                min_len = min(src_max_len, ref_max_len, full_max_len)
                model_inputs["subwords_mask"] = model_inputs["inputs"][shorter_input][
                    "subwords_mask"
                ]
                model_inputs["min_len"] = min_len
            
            return model_inputs

        # Otherwise we will have one single input sequence that concatenates the MT
        # with SRC/REF.
        else:
            model_inputs["inputs"] = (
                self.encoder.concat_sequences(input_sequences, word_outputs=self.word_level)[0],
            )
        return model_inputs

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Tokenizes input data and prepares targets for training.

        Args:
            sample (List[Dict[str, Union[str, float]]]): Mini-batch
            stage (str, optional): Model stage ('train' or 'predict'). Defaults to "train".

        Returns:
            Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: Model input
                and targets.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        input_sequences = [
            self.encoder.prepare_sample(sample["mt"], self.word_level),
        ]

        if ("src" in sample) and ("src" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(sample["src"], self.word_level))

        if ("ref" in sample) and ("ref" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(sample["ref"], self.word_level))
        
        model_inputs = self.concat_inputs(input_sequences)
        if stage == "predict":
            return model_inputs["inputs"]

        targets = Target(score=torch.tensor(sample["score"], dtype=torch.float))
        if self.word_level:
            #  Word Labels will be exactly the same accross all inputs!
            #  We will choose the smallest segment
            targets["labels"] = self.prepare_target(
                self.label_encoder.batch_encode(sample["labels"], split=True),
                model_inputs["subwords_mask"],
                model_inputs["min_len"],
            )
        return model_inputs["inputs"], targets

    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        token_type_ids: Optional[torch.tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            input_ids (torch.tensor): Input sequence.
            attention_mask (torch.tensor): Attention mask.
            token_type_ids (Optional[torch.tensor], optional): Token type ids for
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

        # Word embeddings used for the sentence-level regression task
        if self.layerwise_attention:
            sentemb = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )[:, 0, :]

        elif (
            isinstance(self.hparams.sent_layer, int)
            and 0 <= self.hparams.sent_layer < self.encoder.num_layers
        ):
            sentemb = encoder_out["all_layers"][self.hparams.sent_layer][:, 0, :]
        else:
            raise Exception(
                "Invalid model sent layer {}.".format(self.hparams.word_layer)
            )

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
            sentence_loss = self.sentloss(prediction.score, target.score)
            predictions = prediction.logits.reshape(-1, 2)
            targets = target.labels.view(-1).type(torch.LongTensor).cuda()
            word_loss = self.wordloss(predictions, targets)
            return sentence_loss * (1 - self.hparams.loss_lambda) + word_loss * (
                self.hparams.loss_lambda
            )
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
        if len(batch_input) == 3:
            # In UniTE training is made of 3 losses:
            #    Lsrc + Lref + Lsrc+ref
            # For that reason we have to perform 3 forward passes and sum
            # the respective losses.
            predictions = [self.forward(**input_seq) for input_seq in batch_input]
            loss_value = 0
            for pred in predictions:
                # We created the target according to the shortest segment.
                # We have to remove padding for all predictions
                if self.word_level:
                    seq_len = batch_target.labels.shape[1]
                    pred.logits = pred.logits[:, :seq_len, :]
                
                loss_value += self.compute_loss(pred, batch_target)

        else:
            batch_prediction = self.forward(**batch_input[0])
            loss_value = self.compute_loss(batch_prediction, batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.first_epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
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
        if len(batch_input) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch_input]
            # Final score is the average of the 3 scores!
            scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            batch_prediction = Prediction(score=scores)

            if self.word_level:
                seq_len = batch_target.labels.shape[1]
                # Final Logits for each word is the average of the 3 scores!
                batch_prediction["logits"] = (
                    predictions[0].logits[:, :seq_len, :]
                    + predictions[1].logits[:, :seq_len, :]
                    + predictions[2].logits[:, :seq_len, :]
                ) / 3
        else:
            batch_prediction = self.forward(**batch_input[0])

        if self.word_level:
            # Removing masked targets and the corresponding logits.
            # This includes subwords and padded tokens.
            logits = batch_prediction.logits.reshape(-1, 2)
            targets = batch_target.labels.view(-1)
            mask = targets != -1
            logits, targets = logits[mask, :], targets[mask].int()

        if dataloader_idx == 0:
            self.train_corr.update(batch_prediction.score, batch_target.score)
            if self.word_level:
                self.train_mcc.update(logits, targets)

        elif dataloader_idx > 0:
            self.val_corr[dataloader_idx - 1].update(
                batch_prediction.score, 
                batch_target.score,
                batch_target["system"] if "system" in batch_target else None,
            )
            if self.word_level:
                self.val_mcc[dataloader_idx - 1].update(logits, targets)

    # Overwriting this method to log correlation and classification metrics
    def validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics."""
        self.log_dict(self.train_corr.compute(), prog_bar=False)
        self.train_corr.reset()

        if self.word_level:
            self.log_dict(self.train_mcc.compute(), prog_bar=False)
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
            self.log_dict(results, prog_bar=False)
            val_metrics.append(results)

        average_results = {"val_" + k.split("_")[-1]: [] for k in val_metrics[0].keys()}
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + k.split("_")[-1]].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()}, prog_bar=True
        )

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """Pytorch Lightning predict_step

        Args:
            batch (Dict[str, torch.Tensor]): The output of your prepare_sample function
            batch_idx (Optional[int], optional): Integer displaying which batch this is
                Defaults to None.
            dataloader_idx (Optional[int], optional): Integer displaying which
                dataloader this is. Defaults to None.

        Returns:
            Prediction: Model Prediction
        """

        def decode_labels(logits, subword_mask):
            predicted_tags = logits.argmax(dim=2)
            probabilities = nn.functional.softmax(logits, dim=2)
            word_labels, word_probs = [], []
            for i in range(predicted_tags.shape[0]):
                mask, tags = subword_mask[i, :], predicted_tags[i, :]
                tag_sequence = torch.masked_select(tags, mask).tolist()
                word_probs.append(probabilities[i, mask, :].tolist())
                word_labels.append(self.label_encoder.decode(tag_sequence, join=False))

            return word_labels, word_probs

        if self.mc_dropout:
            raise NotImplementedError("MCD not implemented for this model!")

        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Final score is the average of the 3 scores!
            avg_scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            batch_prediction = Prediction(
                score=avg_scores,
                metadata=Prediction(
                    src_score=predictions[0].score,
                    ref_score=predictions[1].score,
                    unified_score=predictions[2].score,
                ),
            )
            if self.word_level:
                # For world-level tagging we will have to convert logits into tag sequences.
                min_len = min([o.logits.shape[1] for o in predictions])
                min_input = np.argmin([o.logits.shape[1] for o in predictions])
                subword_mask = batch[min_input]["subwords_mask"] == 1
                logits = [o.logits[:, :min_len, :] for o in predictions]
                logits = torch.mean(torch.stack(logits), dim=0)
                word_labels, word_logits = decode_labels(logits, subword_mask)
                batch_prediction.metadata["word_labels"] = word_labels
                batch_prediction.metadata["word_probs"] = word_logits
        else:
            batch_prediction = self.forward(**batch[0])
            if self.word_level:
                subword_mask = batch[0]["subwords_mask"] == 1
                batch_prediction = Prediction(
                    score=batch_prediction.score,
                    metadata=Prediction(
                        word_labels=decode_labels(batch_prediction.logits, subword_mask)
                    ),
                )
        return batch_prediction
