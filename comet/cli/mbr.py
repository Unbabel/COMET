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
"""
Command for Minimum Bayes Risk Decoding.
========================================

This script is inspired in Chantal Amrhein script used in:
    Title: Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes Risk Decoding: A Case Study for COMET
    URL: https://arxiv.org/abs/2202.05148

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (type: Path_fr, default: null)
  -t TRANSLATIONS, --translations TRANSLATIONS
                        (type: Path_fr, default: null)
  --batch_size BATCH_SIZE
                        (type: int, default: 8)
  --num_samples NUM_SAMPLES
                        (required, type: int)
  --model MODEL         COMET model to be used. (type: str, default: wmt20-comet-da)
  --model_storage_path MODEL_STORAGE_PATH
                        Path to the directory where models will be stored. By default its saved in ~/.cache/torch/unbabel_comet/ (default: null)
  -o OUTPUT, --output OUTPUT
                        Best candidates after running MBR decoding. (type: str, default: mbr_result.txt)
"""
import os
from typing import List, Tuple

import torch
from comet.download_utils import download_model
from comet.models import RegressionMetric, available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from tqdm import tqdm


def build_embeddings(
    sources: List[str],
    translations: List[str],
    model: RegressionMetric,
    batch_size: int,
) -> Tuple[torch.Tensor]:
    """Tokenization and respective encoding of source and translation sentences using
    a RegressionMetric model.

    :param sources: List of source sentences.
    :param translations: List of translation sentences.
    :param model: RegressionMetric model that will be used to embed sentences.
    :param batch_size: batch size used during encoding.

    :return: source and MT embeddings.
    """
    # TODO: Optimize this function to have faster MBR decoding!
    src_batches = [
        sources[i : i + batch_size] for i in range(0, len(sources), batch_size)
    ]
    src_inputs = [model.encoder.prepare_sample(batch) for batch in src_batches]
    mt_batches = [
        translations[i : i + batch_size]
        for i in range(0, len(translations), batch_size)
    ]
    mt_inputs = [model.encoder.prepare_sample(batch) for batch in mt_batches]

    src_embeddings = []
    with torch.no_grad():
        for batch in src_inputs:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            src_embeddings.append(
                model.get_sentence_embedding(input_ids, attention_mask)
            )
    src_embeddings = torch.vstack(src_embeddings)

    mt_embeddings = []
    with torch.no_grad():
        for batch in tqdm(mt_inputs, desc="Encoding sentences...", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            mt_embeddings.append(
                model.get_sentence_embedding(input_ids, attention_mask)
            )
    mt_embeddings = torch.vstack(mt_embeddings)

    return src_embeddings, mt_embeddings


def mbr_decoding(
    src_embeddings: torch.Tensor, mt_embeddings: torch.Tensor, model: RegressionMetric
) -> torch.Tensor:
    """Performs MBR Decoding for each translation for a given source.

    :param src_embeddings: Embeddings of source sentences.
    :param mt_embeddings: Embeddings of MT sentences.
    :param model: RegressionMetric Model.

    :return:
        Returns a [n_sent x num_samples] matrix M where each line represents a source sentence
        and each column a given sample.
        M[i][j] is the MBR score of sample j for source i.
    """
    n_sent, num_samples, _ = mt_embeddings.shape
    mbr_matrix = torch.zeros(n_sent, num_samples)
    with torch.no_grad():
        # Loop over all source sentences
        for i in tqdm(
            range(mbr_matrix.shape[0]), desc="MBR Scores...", dynamic_ncols=True
        ):
            source = src_embeddings[i, :].repeat(num_samples, 1)
            # Loop over all hypothesis
            for j in range(mbr_matrix.shape[1]):
                translation = mt_embeddings[i, j, :].repeat(num_samples, 1)
                # Score current hypothesis against all others
                pseudo_refs = mt_embeddings[i, :]
                scores = model.estimate(source, translation, pseudo_refs)[
                    "score"
                ].squeeze(1)
                scores = torch.cat([scores[0:j], scores[j + 1 :]])
                mbr_matrix[i, j] = scores.mean()

    return mbr_matrix


def mbr_command() -> None:
    parser = ArgumentParser(description="Command for Minimum Bayes Risk Decoding.")
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-t", "--translations", type=Path_fr, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="wmt20-comet-da",
        help="COMET model to be used.",
    )
    parser.add_argument(
        "--model_storage_path",
        help=(
            "Path to the directory where models will be stored. "
            + "By default its saved in ~/.cache/torch/unbabel_comet/"
        ),
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Best candidates after running MBR decoding.",
    )
    cfg = parser.parse_args()

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    elif cfg.model in available_metrics:
        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)
    else:
        parser.error(
            "{} is not a valid checkpoint path or model choice. Choose from {}".format(
                cfg.model, available_metrics.keys()
            )
        )
    model = load_from_checkpoint(model_path)

    if not isinstance(model, RegressionMetric) or model.is_referenceless():
        raise Exception(
            "Incorrect model ({}). MBR command only works with Reference-based Regression models!".format(
                model.__class__.__name__
            )
        )

    model.eval()
    model.cuda()

    with open(cfg.sources()) as fp:
        sources = [line.strip() for line in fp.readlines()]

    with open(cfg.translations()) as fp:
        translations = [line.strip() for line in fp.readlines()]

    src_embeddings, mt_embeddings = build_embeddings(
        sources, translations, model, cfg.batch_size
    )
    mt_embeddings = mt_embeddings.reshape(len(sources), cfg.num_samples, -1)
    mbr_matrix = mbr_decoding(src_embeddings, mt_embeddings, model)
    translations = [
        translations[i : i + cfg.num_samples]
        for i in range(0, len(translations), cfg.num_samples)
    ]
    assert len(sources) == len(translations)

    best_candidates = []
    for i, samples in enumerate(translations):
        best_cand_idx = torch.argmax(mbr_matrix[i, :])
        best_candidates.append(samples[best_cand_idx])

    with open(cfg.output, "w") as fp:
        for sample in best_candidates:
            fp.write(sample + "\n")


if __name__ == "__main__":
    mbr_command()
