# -*- coding: utf-8 -*-
r"""
Comet Ranker Model
======================
    The goal of this model is to rank good translations closer to the reference and source text
    and bad translations further by a small margin.

    https://pytorch.org/docs/stable/nn.html#tripletmarginloss
"""
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from comet.models.ranking.ranking_base import RankingBase
from comet.models.utils import move_to_cuda
from torchnlp.utils import collate_tensors


class CometRanker(RankingBase):
    """
    Comet Ranker class that uses a pretrained encoder to extract features
    from the sequences and then passes those features through a Triplet Margin Loss.

    :param hparams: Namespace containing the hyperparameters.
    """

    def __init__(self, hparams: Namespace) -> None:
        super().__init__(hparams)

    def compute_metrics(self, outputs: List[Dict[str, torch.Tensor]]) -> dict:
        """  Computes WMT19 shared task kendall tau like metric. """
        distance_pos, distance_neg = [], []
        for minibatch in outputs:
            minibatch = minibatch["val_prediction"]
            src_embedding = minibatch["src_sentemb"]
            ref_embedding = minibatch["ref_sentemb"]
            pos_embedding = minibatch["pos_sentemb"]
            neg_embedding = minibatch["neg_sentemb"]

            distance_src_pos = F.pairwise_distance(pos_embedding, src_embedding)
            distance_ref_pos = F.pairwise_distance(pos_embedding, ref_embedding)
            harmonic_distance_pos = (2 * distance_src_pos * distance_ref_pos) / (
                distance_src_pos + distance_ref_pos
            )
            distance_pos.append(harmonic_distance_pos)

            distance_src_neg = F.pairwise_distance(neg_embedding, src_embedding)
            distance_ref_neg = F.pairwise_distance(neg_embedding, ref_embedding)
            harmonic_distance_neg = (2 * distance_src_neg * distance_ref_neg) / (
                distance_src_neg + distance_ref_neg
            )
            distance_neg.append(harmonic_distance_neg)

        return {
            "kendall": self.metrics.compute(
                torch.cat(distance_pos), torch.cat(distance_neg)
            )
        }

    def compute_loss(self, model_out: Dict[str, torch.Tensor], *args) -> torch.Tensor:
        """
        Computes Triplet Margin Loss for both the reference and the source.

        :param model_out: model specific output with src_anchor, ref_anchor, pos and neg
            sentence embeddings.
        """
        ref_anchor = model_out["ref_sentemb"]
        src_anchor = model_out["src_sentemb"]
        positive = model_out["pos_sentemb"]
        negative = model_out["neg_sentemb"]
        return self.loss(src_anchor, positive, negative) + self.loss(
            ref_anchor, positive, negative
        )

    def predict(
        self,
        samples: Dict[str, str],
        cuda: bool = False,
        show_progress: bool = False,
        batch_size: int = -1,
    ) -> (Dict[str, Union[str, float]], List[float]):
        """Function that runs a model prediction,

        :param samples: List of dictionaries with 'mt' and 'ref' keys.
        :param cuda: Flag that runs inference using 1 single GPU.
        :param show_progress: Flag to show progress during inference of multiple examples.
        :para batch_size: Batch size used during inference. By default uses the same batch size used during training.

        :return: Dictionary with model outputs
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
                    total=len(batches), desc="Preparing batches....", dynamic_ncols=True
                )
            for batch in batches:
                model_inputs.append(self.prepare_sample(batch, inference=True))
                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

            if show_progress:
                pbar = tqdm(
                    total=len(batches), desc="Scoring hypothesis...", dynamic_ncols=True
                )

            distance_weighted, distance_src, distance_ref = [], [], []
            for k, model_input in enumerate(model_inputs):
                src_input, mt_input, ref_input, alt_input = model_input
                if cuda and torch.cuda.is_available():
                    src_embeddings = self.get_sentence_embedding(
                        **move_to_cuda(src_input)
                    )
                    mt_embeddings = self.get_sentence_embedding(
                        **move_to_cuda(mt_input)
                    )
                    ref_embeddings = self.get_sentence_embedding(
                        **move_to_cuda(ref_input)
                    )
                    ref_distances = F.pairwise_distance(
                        mt_embeddings, ref_embeddings
                    ).cpu()
                    src_distances = F.pairwise_distance(
                        mt_embeddings, src_embeddings
                    ).cpu()

                    # When 2 references are given the distance to the reference is the Min between
                    # both references.
                    if alt_input is not None:
                        alt_embeddings = self.get_sentence_embedding(
                            **move_to_cuda(alt_input)
                        )
                        alt_distances = F.pairwise_distance(
                            mt_embeddings, alt_embeddings
                        ).cpu()
                        ref_distances = torch.stack([ref_distances, alt_distances])
                        ref_distances = ref_distances.min(dim=0).values

                else:
                    src_embeddings = self.get_sentence_embedding(**src_input)
                    mt_embeddings = self.get_sentence_embedding(**mt_input)
                    ref_embeddings = self.get_sentence_embedding(**ref_input)
                    ref_distances = F.pairwise_distance(mt_embeddings, ref_embeddings)
                    src_distances = F.pairwise_distance(mt_embeddings, src_embeddings)

                # Harmonic mean between the distances:
                distances = (2 * ref_distances * src_distances) / (
                    ref_distances + src_distances
                )
                src_distances = ref_distances.numpy().tolist()
                ref_distances = ref_distances.numpy().tolist()
                distances = distances.numpy().tolist()

                for i in range(len(distances)):
                    distance_weighted.append(1 / (1 + distances[i]))
                    distance_src.append(1 / (1 + src_distances[i]))
                    distance_ref.append(1 / (1 + ref_distances[i]))

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        assert len(distance_weighted) == len(samples)
        scores = []
        for i in range(len(samples)):
            scores.append(distance_weighted[i])
            samples[i]["predicted_score"] = scores[-1]
            samples[i]["reference_distance"] = distance_ref[i]
            samples[i]["source_distance"] = distance_src[i]

        return samples, scores

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[Tuple[Dict[str, torch.Tensor], None], List[Dict[str, torch.Tensor]]]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to to False, then the model expects
            a MT and reference instead of anchor, pos, and neg segments.

        :return: Tuple with a dictionary containing the model inputs and None OR List
            with source, MT and reference tokenized and vectorized.
        """
        sample = collate_tensors(sample)
        if inference:
            src_inputs = self.encoder.prepare_sample(sample["src"])
            mt_inputs = self.encoder.prepare_sample(sample["mt"])
            ref_inputs = self.encoder.prepare_sample(sample["ref"])
            alt_inputs = (
                self.encoder.prepare_sample(sample["alt"]) if "alt" in sample else None
            )
            return src_inputs, mt_inputs, ref_inputs, alt_inputs

        ref_inputs = self.encoder.prepare_sample(sample["ref"])
        src_inputs = self.encoder.prepare_sample(sample["src"])
        pos_inputs = self.encoder.prepare_sample(sample["pos"])
        neg_inputs = self.encoder.prepare_sample(sample["neg"])

        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        pos_inputs = {"pos_" + k: v for k, v in pos_inputs.items()}
        neg_inputs = {"neg_" + k: v for k, v in neg_inputs.items()}

        return {**ref_inputs, **src_inputs, **pos_inputs, **neg_inputs}, torch.empty(0)

    def forward(
        self,
        src_tokens: torch.tensor,
        ref_tokens: torch.tensor,
        pos_tokens: torch.tensor,
        neg_tokens: torch.tensor,
        src_lengths: torch.tensor,
        ref_lengths: torch.tensor,
        pos_lengths: torch.tensor,
        neg_lengths: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes the anchor, positive samples and negative samples
        and returns embeddings for the triplet.

        :param src_tokens: anchor sequences [batch_size x anchor_seq_len]
        :param ref_tokens: anchor sequences [batch_size x anchor_seq_len]
        :param pos_tokens: positive sequences [batch_size x pos_seq_len]
        :param neg_tokens: negative sequences [batch_size x neg_seq_len]
        :param src_lengths: anchor lengths [batch_size]
        :param ref_lengths: anchor lengths [batch_size]
        :param pos_lengths: positive lengths [batch_size]
        :param neg_lengths: negative lengths [batch_size]

        :return: Dictionary with model outputs to be passed to the loss function.
        """
        return {
            "src_sentemb": self.get_sentence_embedding(src_tokens, src_lengths),
            "ref_sentemb": self.get_sentence_embedding(ref_tokens, ref_lengths),
            "pos_sentemb": self.get_sentence_embedding(pos_tokens, pos_lengths),
            "neg_sentemb": self.get_sentence_embedding(neg_tokens, neg_lengths),
        }
