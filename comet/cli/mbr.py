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
    Title: Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes
    Risk Decoding: A Case Study for COMET
    URL: https://aclanthology.org/2022.aacl-main.83/

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (required, type: Path_fr)
  -t TRANSLATIONS, --translations TRANSLATIONS
                        (required, type: Path_fr)
  --num_samples NUM_SAMPLES
                        (required, type: int)
  --batch_size BATCH_SIZE
                        (type: int, default: 16)
  --rerank_top_k RERANK_TOP_K
                       Chooses the topK candidates according to --qe_model before
                       applying MBR. Disabled by default. (type: int, default: 0)
  --qe_model QE_MODEL   Reference Free model used for reranking before MBR. (type: str,
                        default: Unbabel/wmt23-cometkiwi-da-xl)
  --model MODEL         COMET model to be used. 
                        (type: str, default: Unbabel/wmt23-comet-da-xl)
  --model_storage_path MODEL_STORAGE_PATH
                        Path to the directory where models will be stored. By default
                        its saved in ~/.cache/torch/unbabel_comet/ (default: null)
  -o OUTPUT, --output OUTPUT
                        Best candidates after running MBR decoding. (required, type: str)
"""
import os
from typing import List, Tuple

import numpy as np
import torch
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from tqdm import tqdm

from comet.models import RegressionMetric, download_model, load_from_checkpoint


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

    Args:
        src_embeddings (torch.Tensor): Embeddings of source sentences. [n_sent x hidden_size]
        mt_embeddings (torch.Tensor): Embeddings of MT sentences. [n_sent x num_samples x hidden_size]
        model (RegressionMetric): RegressionMetric Model.

    Return:
        torch.Tensor: matrice [n_sent x num_samples] where each line represents a
        source sentence and each column a given sample.
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
                scores = model.estimate(source, translation, pseudo_refs)["score"]
                scores = torch.cat([scores[0:j], scores[j + 1 :]])
                mbr_matrix[i, j] = scores.mean()

    return mbr_matrix

def rerank_top_k(
    sources: List[str],
    translations: List[str],
    qe_model: str,
    batch_size: int,
    gpus: int,
    num_samples: int,
    topk: int,
) -> Tuple[List[str], List[str]]:
    """Performs QE reranking.

    Args:
        sources (List[str]): Embeddings of source sentences.
        translations (List[str]: Embeddings of MT sentences.
        qe_model (str): Reference-free model used for reranking.
        batch_size (int): Batch size used during inference.
        num_samples (int): Number of candidates per source.
        topk (int): Number of top k samples.

    Return:
        List[str]: Top k candidate translations for each source
    """
    translations = [
        translations[i : i + num_samples]
        for i in range(0, len(translations), num_samples)
    ]
    assert len(translations) == len(sources)
    data = []
    for i in range(len(sources)):
        for j in range(num_samples):
            data.append({"src": sources[i], "mt": translations[i][j]})

    model_output = qe_model.predict(data, batch_size=batch_size, gpus=gpus)
    seg_scores = np.array(model_output.scores).reshape(len(sources), num_samples)
    topk_indices = np.argsort(seg_scores, axis=1)
    topk_translations = []
    for i in range(len(sources)):
        topk_translations += [translations[i][idx] for idx in topk_indices[i][::-1][:topk]]

    return topk_translations

def mbr_command() -> None:
    parser = ArgumentParser(description="Command for Minimum Bayes Risk Decoding.")
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-t", "--translations", type=Path_fr, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=0,
        help=(
            "Chooses the topK candidates according to --qe_model before applying MBR."
            + " Disabled by default."
        ),
    )
    parser.add_argument(
        "--qe_model",
        type=str,
        required=False,
        default="Unbabel/wmt22-cometkiwi-da",
        help="Reference Free model used for reranking before MBR.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="Unbabel/wmt22-comet-da",
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

    with open(cfg.sources(), encoding="utf-8") as fp:
        sources = [line.strip() for line in fp.readlines()]

    with open(cfg.translations(), encoding="utf-8") as fp:
        translations = [line.strip() for line in fp.readlines()]

    num_samples = cfg.num_samples
    # Running QE reranking before MBR!
    if cfg.rerank_top_k > 0:
        if (
            cfg.qe_model.endswith(".ckpt")
            and os.path.exists(cfg.qe_model)
        ):
            qe_model_path = cfg.qe_model
        else:
            qe_model_path = download_model(
                cfg.qe_model, saving_directory=cfg.model_storage_path
            )
        assert (
            cfg.rerank_top_k < cfg.num_samples
        ), "--rerank_top_k needs to be smaller than number of candidates provided!"
        model = load_from_checkpoint(qe_model_path)
        assert (
            not model.requires_references()
        ), "--qe_model expects a Reference Free model!"

        translations = rerank_top_k(
            sources, translations, model, cfg.batch_size, cfg.gpus, cfg.num_samples, cfg.rerank_top_k
        )
        num_samples = cfg.rerank_top_k

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    else:
        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)
    
    model = load_from_checkpoint(model_path)
    model.eval()
    model.cuda()
    if not isinstance(model, RegressionMetric):
        raise Exception(
            "Invalid model ({}). MBR command only works with Reference-based Regression models!".format(
                model.__class__.__name__
            )
        )
        
    src_embeddings, mt_embeddings = build_embeddings(
        sources, translations, model, cfg.batch_size
    )
    mt_embeddings = mt_embeddings.reshape(len(sources), num_samples, -1)
    mbr_matrix = mbr_decoding(src_embeddings, mt_embeddings, model)
    translations = [
        translations[i : i + num_samples]
        for i in range(0, len(translations), num_samples)
    ]
    assert len(sources) == len(translations)

    best_candidates = []
    for i, samples in enumerate(translations):
        best_cand_idx = torch.argmax(mbr_matrix[i, :])
        best_candidates.append(samples[best_cand_idx])

    with open(cfg.output, "w", encoding="utf-8") as fp:
        for sample in best_candidates:
            fp.write(sample + "\n")


if __name__ == "__main__":
    mbr_command()
