# -*- coding: utf-8 -*-
r"""
Estimator Base Model
=======================
    Abstract base class used to build new estimator models
    inside COMET.
"""
from argparse import Namespace
from typing import Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from comet.metrics import RegressionReport
from comet.models.model_base import ModelBase
from comet.models.utils import average_pooling, max_pooling, move_to_cpu, move_to_cuda


class Estimator(ModelBase):
    """
    Estimator base class that uses an Encoder to encode sequences.

    :param hparams: Namespace containing the hyperparameters.
    """

    class ModelConfig(ModelBase.ModelConfig):
        """
        Estimator ModelConfig:

        --------------------------- Encoder -----------------------------------------

        :param encoder_learning_rate: Learning rate used for the encoder model.

        :param layerwise_decay: Decay for the layer wise learning rates. If 1.0 no decay is applied.

        :param layer: Layer that will be used to extract embeddings. If 'mix' embeddings
            from all layers are combined with a layer-wise attention mechanism

        :param scalar_mix_dropout: If layer='mix' we can regularize layer's importance by
            with a given probability setting that weight to - inf before softmax.

        ------------------------- Feed Forward ---------------------------------------

        :param loss: Loss function to be used (options: binary_xent, mse).

        :param hidden_sizes: String with size of the hidden layers in the feedforward.

        :param activations: Activation functions to be used in the feedforward

        :param dropout: Dropout probability to be applied to the feedforward

        :param final_activation: Activation function to be applied after getting the
            final regression score. Set to False if you wish to perform an 'unbounded' regression.
        """

        encoder_learning_rate: float = 1e-06
        layerwise_decay: float = 1.0
        layer: str = "mix"
        scalar_mix_dropout: float = 0.0

        loss: str = "mse"
        hidden_sizes: str = "1024"
        activations: str = "Tanh"
        dropout: float = 0.1
        final_activation: str = "Sigmoid"

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
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "ref", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].astype(str)
        df["score"] = df["score"].astype(float)
        return df.to_dict("records")

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
        return self.metrics.compute(predictions, targets)

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

        elif self.hparams.pool == "cls+avg":
            cls_sentemb = embeddings[:, 0, :]
            avg_sentemb = average_pooling(
                tokens,
                embeddings,
                encoder_out["mask"],
                self.encoder.tokenizer.padding_index,
            )
            sentemb = torch.cat((cls_sentemb, avg_sentemb), dim=1)
        else:
            raise Exception("Invalid pooling technique.")

        return sentemb

    def predict(
        self,
        samples: List[Dict[str, str]],
        cuda: bool = False,
        show_progress: bool = False,
        batch_size: int = -1,
    ) -> (Dict[str, Union[str, float]], List[float]):
        """Function that runs a model prediction,

        :param samples: List of dictionaries with 'mt' and 'ref' keys.
        :param cuda: Flag that runs inference using 1 single GPU.
        :param show_progress: Flag to show progress during inference of multiple examples.
        :para batch_size: Batch size used during inference. By default uses the same batch size used during training.

        :return: Dictionary with original samples, predicted scores and langid results for SRC and MT
            + list of predicted scores
        """
        if self.training:
            self.eval()

        if cuda and torch.cuda.is_available():
            self.to("cuda")

        batch_size = self.hparams.batch_size if batch_size < 1 else batch_size
        with torch.no_grad():
            batches = [
                samples[i : i + batch_size] for i in range(0, len(samples), batch_size)
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

    def document_predict(
        self,
        documents: List[Dict[str, List[str]]],
        cuda: bool = False,
        show_progress: bool = False,
    ) -> (Dict[str, Union[str, float]], List[float]):
        """Function that scores entire documents by processing all segments in parallel.

        :param documents: List of dictionaries with 'mt', 'src' and 'ref' keys where each key is
            a list of segments.
        :param cuda: Flag that runs inference using 1 single GPU.
        :param show_progress: Flag to show progress during inference of multiple examples.

        :return: tuple with Dictionary with original samples and predicted document score, micro
            average scores, macro average scores.
        """
        if self.training:
            self.eval()

        if cuda and torch.cuda.is_available():
            self.to("cuda")

        inputs, lengths = [], []
        for d in documents:
            d = [dict(zip(d, t)) for t in zip(*d.values())]
            # For very long documents we need to create chunks.
            # (64 sentences per chunk)
            if len(d) > 64:
                document_chunks, document_lengths = [], []
                chunks = [d[i : i + 64] for i in range(0, len(d), 64)]
                for chunk in chunks:
                    chunk = self.prepare_sample(chunk, inference=True)
                    document_lengths.append(chunk["mt_lengths"])
                    if cuda and torch.cuda.is_available():
                        document_chunks.append(chunk)
                lengths.append(torch.cat(document_lengths, dim=0))
                inputs.append(document_chunks)
            else:
                d_input = self.prepare_sample(d, inference=True)
                lengths.append(d_input["mt_lengths"])
                if cuda and torch.cuda.is_available():
                    inputs.append(d_input)

        micro_average, average = [], []
        for doc, seg_lengths in tqdm(
            zip(inputs, lengths),
            total=len(inputs),
            desc="Scoring Documents ...",
            dynamic_ncols=True,
            leave=None,
        ):
            if isinstance(doc, list):
                seg_scores = []
                for chunk in doc:
                    model_output = self.forward(**move_to_cuda(chunk))
                    seg_scores.append(move_to_cpu(model_output)["score"].view(1, -1)[0])
                seg_scores = torch.cat(seg_scores, dim=0)
            else:
                model_output = self.forward(**move_to_cuda(doc))
                seg_scores = move_to_cpu(model_output)["score"].view(1, -1)[0]

            # Invert segment-level scores for HTER
            # seg_scores = torch.ones_like(seg_scores) -  seg_scores
            micro = (seg_scores * seg_lengths).sum() / seg_lengths.sum()
            macro = seg_scores.sum() / seg_scores.size()[0]
            micro_average.append(micro.item())
            average.append(macro.item())

        assert len(micro_average) == len(documents)
        for i in range(len(documents)):
            documents[i]["predicted_score"] = micro_average[i]

        return documents, micro_average, average
