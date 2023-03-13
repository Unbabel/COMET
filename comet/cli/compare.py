#!/usr/bin/env python3

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
Command for comparing multiple MT systems.
===========================================

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (type: Path_fr, default: null)
  -r REFERENCES, --references REFERENCES
                        (type: Path_fr, default: null)
  -t [TRANSLATIONS [TRANSLATIONS ...]], --translations [TRANSLATIONS [TRANSLATIONS ..]]
                        (type: Path_fr, default: null)
  -d SACREBLEU_DATASET, --sacrebleu_dataset SACREBLEU_DATASET
                        (type: str, default: null)
  --batch_size BATCH_SIZE
                        (type: int, default: 8)
  --gpus GPUS           (type: int, default: 1)
  --quiet               Sets all loggers to ERROR level. (default: False)
  --only_system         Prints only the final system score. (default: False)
  --num_splits NUM_SPLITS
                        Number of random partitions used in Bootstrap resampling.
                        (type: int, default: 300)
  --sample_ratio SAMPLE_RATIO
                        Percentage of the testset to use in each split. (type: float,
                        default: 0.4)
  --t_test_alternative T_TEST_ALTERNATIVE
                        Alternative hypothesis from scipy.stats.ttest_rel. The
                        following options are available: 'two-sided', 'less', 
                        'greater'. Defaults to 'less' (type: str, default: two-sided)
  --to_json TO_JSON     Exports results to a json file. (type: str, default: "")
  --model MODEL         COMET model to be used. (type: str, default: wmt20-comet-da)
  --model_storage_path MODEL_STORAGE_PATH
                        Path to the directory where models will be stored. By default
                        its saved in ~/.cache/torch/unbabel_comet/ (default: null)
  --num_workers NUM_WORKERS
                        Number of workers to use when loading data. (type: int,
                        default: null)
  --disable_cache       Disables sentence embeddings caching. This makes inference
                        slower but saves memory. (default: False)
  --disable_length_batching
                        Disables length batching. This makes inference slower.
                        (default: False)
  --print_cache_info    Print information about COMET cache. (default: False)
"""
import json
import os
from itertools import combinations
from typing import Dict, Generator, List, Tuple, Union

import numpy as np
from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
from sacrebleu.utils import get_reference_files, get_source_file
from scipy import stats
from tabulate import tabulate

from comet import download_model, load_from_checkpoint

Statistical_test_info = Dict[str, Union[Path_fr, Dict[str, float]]]

# Due to small numerical differences in scores we consider that any system comparison
# with a difference bellow EPS to be considered a tie.
EPS = 0.001


def display_statistical_results(data: Statistical_test_info) -> None:
    """Print out the T-test results for a system pair.

    Args:
        data (Statistical_test_info): Stats to be printed out.
    """
    print("==========================")
    print("x_name:", data["x_name"].rel_path)
    print("y_name:", data["y_name"].rel_path)

    print("\nBootstrap Resampling Results:")
    for k, v in data["bootstrap_resampling"].items():
        print("{}:\t{:.4f}".format(k, v))

    print("\nPaired T-Test Results:")
    for k, v in data["paired_t-test"].items():
        print("{}:\t{:.4f}".format(k, v))

    x_seg_scores = data["bootstrap_resampling"]["x-mean"]
    y_seg_scores = data["bootstrap_resampling"]["y-mean"]
    best_system = (
        data["x_name"].rel_path
        if x_seg_scores > y_seg_scores
        else data["y_name"].rel_path
    )
    worse_system = (
        data["x_name"].rel_path
        if x_seg_scores < y_seg_scores
        else data["y_name"].rel_path
    )
    if data["paired_t-test"]["p_value"] <= 0.05:
        print("Null hypothesis rejected according to t-test.")
        print("Scores differ significantly across samples.")
        print(f"{best_system} outperforms {worse_system}.")
    else:
        print("Null hypothesis can't be rejected.\nBoth systems have equal averages.")


def t_tests_summary(
    t_test_results: List[Statistical_test_info],
    translations: Tuple[Path_fr],
    threshold_p_value: float = 0.05,
) -> None:
    """Prints T-tests Summary

    Args:
        t_test_results (List[Statistical_test_info]): List of stats between systems.
        translations (Tuple[Path_fr]): Path to each system.
        threshold_p_value (float): Threshold for p_value. Defaults to 0.05.
    """
    n = len(translations)
    name2id = {name: i for i, name in enumerate(translations)}
    grid = [[None] * n for name in translations]
    for t_test in t_test_results:
        p_value = t_test["paired_t-test"]["p_value"]
        x_id = name2id[t_test["x_name"]]
        y_id = name2id[t_test["y_name"]]
        grid[x_id][y_id] = False
        grid[y_id][x_id] = False
        if p_value < threshold_p_value:
            x_seg_scores = t_test["bootstrap_resampling"]["x-mean"]
            y_seg_scores = t_test["bootstrap_resampling"]["y-mean"]
            if x_seg_scores > y_seg_scores:
                grid[x_id][y_id] = True
            else:
                grid[y_id][x_id] = True

    # Add the row's name aka the system's name.
    grid = [(name,) + tuple(row) for name, row in zip(translations, grid)]

    print("Summary")
    print("If system_x is better than system_y then:")
    print(
        f"Null hypothesis rejected according to t-test with p_value={threshold_p_value}."
    )
    print("Scores differ significantly across samples.")
    print(tabulate(grid, headers=("system_x \ system_y",) + translations))


def calculate_bootstrap(
    x_sys_scores: np.ndarray, y_sys_scores: np.ndarray, x_name: Path_fr, y_name: Path_fr
) -> Statistical_test_info:
    """Calculate bootstrap score, wins and ties for a system pair.

    Args:
        x_sys_scores (np.ndarray): array of num_splits comet scores for system x
        y_sys_scores (np.ndarray): array of num_splits comet scores for system y
        x_name (str): system x's name
        y_name (str): system y's name

    Return:
        Statistical_test_info: Stats from bootstrap resampling.
    """
    num_splits = x_sys_scores.shape[0]
    delta = x_sys_scores - y_sys_scores
    ties = np.absolute(delta)
    ties = float(len(ties[ties < EPS]))
    x_wins = float(len(delta[delta >= EPS]))
    y_wins = float(len(delta[delta <= -EPS]))
    return {
        "x_name": x_name,
        "y_name": y_name,
        "bootstrap_resampling": {
            "x-mean": float(np.mean(x_sys_scores)),
            "y-mean": float(np.mean(y_sys_scores)),
            "ties (%)": ties / num_splits,
            "x_wins (%)": x_wins / num_splits,
            "y_wins (%)": y_wins / num_splits,
        },
    }


def pairwise_bootstrap(
    sys_scores: np.ndarray, systems: List[Path_fr]
) -> Generator[Statistical_test_info, None, None]:
    """Calculates the bootstrap resampling between all systems' permutations.

    Args:
        sys_scores (np.ndarray): comet scores [num_systems x num_splits]
        systems (List[Path_fr]): List with paths to each system.

    Return:
        Generator(Statistical_test_info): bootstrap resampling stats between systems.
    """
    assert sys_scores.shape[0] == len(systems), "Each system should have its sys_score."

    pairs = combinations(zip(systems, sys_scores), 2)
    for (x_name, x_sys_scores), (y_name, y_sys_scores) in pairs:
        yield calculate_bootstrap(x_sys_scores, y_sys_scores, x_name, y_name)


def bootstrap_resampling(
    seg_scores: np.ndarray, sample_size: int, num_splits: int
) -> np.array:
    """Computes Bootstrap Resampling.

    seg_scores (np.ndarray): scores for each systems' translation, aka a score matrix
        [num_systems X num_sentences]
    sample_size (int): Number of examples for each partition.
    num_splits (int): Number of partitions.

    Return:
        np.array: scores matrice [num_systems X num_splits]
    """
    population_size = seg_scores.shape[1]
    # Subsample the gold and system outputs (with replacement)
    subsample_ids = np.random.choice(
        population_size, size=(sample_size, num_splits), replace=True
    )
    subsamples = np.take(
        seg_scores, subsample_ids, axis=1
    )  # num_systems x sample_size x num_splits
    sys_scores = np.mean(subsamples, axis=1)  # num_systems x num_splits
    return sys_scores


def score(cfg: Namespace, systems: List[Dict[str, List[str]]]) -> np.ndarray:
    """Scores each systems with a given model.

    Args:
        cfg (Namespace): comet-compare configs.
        systems (List[Dict[str, List[str]]]): List with translations for each system.

    Return:
        np.ndarray: segment-level scores flatten.
    """
    model = load_from_checkpoint(cfg.model_path)
    model.eval()

    if not cfg.disable_cache:
        model.set_embedding_cache()

    if cfg.print_cache_info:
        print(model.retrieve_sentence_embedding.cache_info())

    if cfg.gpus > 1:
        # Create a single list that contains all systems' src, ref & translation.
        samples = [
            dict(zip(system.keys(), values))
            for system in systems
            for values in zip(*system.values())
        ]

        outputs = model.predict(
            samples=samples,
            batch_size=cfg.batch_size,
            gpus=cfg.gpus,
            progress_bar=(not cfg.quiet),
            accelerator="auto",
            num_workers=cfg.num_workers,
            length_batching=(not cfg.disable_length_batching),
        )
        seg_scores = outputs.scores
    else:
        # This maximizes cache hits because batches will be equal!
        seg_scores = []
        for system in systems:
            samples = [dict(zip(system, t)) for t in zip(*system.values())]
            outputs = model.predict(
                samples=samples,
                batch_size=cfg.batch_size,
                gpus=cfg.gpus,
                progress_bar=(not cfg.quiet),
                accelerator="cpu" if cfg.gpus == 0 else "auto",
                num_workers=cfg.num_workers,
                length_batching=(not cfg.disable_length_batching),
            )
            seg_scores += outputs.scores

    n = len(systems[0]["src"])
    # [grouper](https://docs.python.org/3/library/itertools.html#itertools-recipes)
    seg_scores = list(zip(*[iter(seg_scores)] * n))
    seg_scores = np.array(seg_scores, dtype="float32")  # num_systems x num_translations
    return seg_scores


def get_cfg() -> Namespace:
    """Parse the CLI options and arguments.

    Return:
        Namespace: comet-compare configs.
    """
    parser = ArgumentParser(
        description="Command for comparing multiple MT systems' translations."
    )
    parser.add_argument("-s", "--sources", type=Path_fr)
    parser.add_argument("-r", "--references", type=Path_fr)
    parser.add_argument("-t", "--translations", nargs="*", type=Path_fr)
    parser.add_argument("-d", "--sacrebleu_dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--quiet", action="store_true", help="Sets all loggers to ERROR level."
    )
    parser.add_argument(
        "--only_system", action="store_true", help="Prints only the final system score."
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=300,
        help="Number of random partitions used in Bootstrap resampling.",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.4,
        help="Percentage of the testset to use in each split.",
    )
    parser.add_argument(
        "--t_test_alternative",
        type=str,
        default="two-sided",
        help=(
            "Alternative hypothesis from scipy.stats.ttest_rel. The following options"
            + " are available: 'two-sided', 'less', 'greater'. Defaults to 'two-sided'"
        ),
    )
    parser.add_argument(
        "--to_json",
        type=str,
        default="",
        help="Exports results to a json file.",
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
        "--num_workers",
        help="Number of workers to use when loading data.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help=(
            "Disables sentence embeddings caching."
            + " This makes inference slower but saves memory."
        ),
    )
    parser.add_argument(
        "--disable_length_batching",
        action="store_true",
        help="Disables length batching. This makes inference slower.",
    )
    parser.add_argument(
        "--print_cache_info",
        action="store_true",
        help="Print information about COMET cache.",
    )
    cfg = parser.parse_args()

    if cfg.sources is None and cfg.sacrebleu_dataset is None:
        parser.error(f"You must specify a source (-s) or a sacrebleu dataset (-d)")

    if cfg.sacrebleu_dataset is not None:
        if cfg.references is not None or cfg.sources is not None:
            parser.error(
                f"Cannot use sacrebleu datasets (-d) with manually-specified datasets (-s and -r)"
            )

        try:
            testset, langpair = cfg.sacrebleu_dataset.rsplit(":", maxsplit=1)
            cfg.sources = Path_fr(get_source_file(testset, langpair))
            cfg.references = Path_fr(get_reference_files(testset, langpair)[0])

        except ValueError:
            parser.error(
                "SacreBLEU testset format must be TESTSET:LANGPAIR, e.g., wmt20:de-en"
            )
        except Exception as e:
            import sys

            print("SacreBLEU error:", e, file=sys.stderr)
            sys.exit(1)

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        cfg.model_path = cfg.model

    else:
        cfg.model_path = download_model(
            cfg.model, saving_directory=cfg.model_storage_path
        )

    return cfg, parser


def compare_command() -> None:
    """CLI that uses comet to compare multiple systems in a pairwise manner."""
    cfg, parser = get_cfg()
    seed_everything(1)

    model = load_from_checkpoint(cfg.model_path)
    model.eval()

    if model.requires_references() and (cfg.references is None):
        parser.error(
            "{} requires -r/--references or -d/--sacrebleu_dataset.".format(cfg.model)
        )

    if not cfg.disable_cache:
        model.set_embedding_cache()

    if cfg.print_cache_info:
        print(model.retrieve_sentence_embedding.cache_info())

    assert len(cfg.translations) > 1, "You must provide at least 2 translation files"

    with open(cfg.sources(), encoding="utf-8") as fp:
        sources = [line.strip() for line in fp.readlines()]

    translations = []
    for system in cfg.translations:
        with open(system, mode="r", encoding="utf-8") as fp:
            translations.append([line.strip() for line in fp.readlines()])

    if cfg.references is not None:
        with open(cfg.references(), encoding="utf-8") as fp:
            references = [line.strip() for line in fp.readlines()]
        systems = [
            {"src": sources, "mt": system, "ref": references} for system in translations
        ]
    else:
        systems = [{"src": sources, "mt": system} for system in translations]

    seg_scores = score(cfg, systems)
    population_size = seg_scores.shape[1]
    sys_scores = bootstrap_resampling(
        seg_scores,
        sample_size=max(int(population_size * cfg.sample_ratio), 1),
        num_splits=cfg.num_splits,
    )
    results = list(pairwise_bootstrap(sys_scores, cfg.translations))

    # Paired T_Test Results:
    pairs = combinations(zip(cfg.translations, seg_scores), 2)
    for (x_name, x_seg_scores), (y_name, y_seg_scores) in pairs:
        ttest_result = stats.ttest_rel(
            x_seg_scores, y_seg_scores, alternative=cfg.t_test_alternative
        )
        for res in results:
            if res["x_name"] == x_name and res["y_name"] == y_name:
                res["paired_t-test"] = {
                    "statistic": ttest_result.statistic,
                    "p_value": ttest_result.pvalue,
                }

    info = {
        "model": cfg.model,
        "statistical_results": results,
        "source": sources,
        "translations": [
            {
                "name": name,
                "mt": trans,
                "scores": scores.tolist(),
            }
            for name, trans, scores in zip(cfg.translations, translations, seg_scores)
        ],
    }

    if references is not None:
        info["reference"] = references

    for data in results:
        display_statistical_results(data)

    print()
    t_tests_summary(results, tuple(cfg.translations))

    if cfg.to_json != "":
        with open(cfg.to_json, "w", encoding="utf-8") as outfile:
            json.dump(info, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))


if __name__ == "__main__":
    compare_command()
