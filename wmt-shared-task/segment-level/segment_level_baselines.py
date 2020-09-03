f"""
Shell script tho reproduce results for BLEURTscores in data from WMT18/19 Metrics Shared task.
"""
import argparse

import pandas as pd

from sacrebleu import corpus_bleu, sentence_chrf
from sacremoses import MosesTokenizer
from tqdm import tqdm


def compute_kendall(
    hyp1_scores: list, hyp2_scores: list, dataframe: pd.DataFrame
) -> (int, list):
    """ Computes the official WMT19 shared task Kendall correlation score. """
    assert len(hyp1_scores) == len(hyp2_scores) == len(data)
    conc, disc = 0, 0
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Kendall eval..."):
        if hyp1_scores[i] > hyp2_scores[i]:
            conc += 1
        else:
            disc += 1

    return (conc - disc) / (conc + disc)


def run_sentence_bleu(candidates: list, references: list, language: str) -> list:
    """ Runs sentence BLEU from Sacrebleu. """
    tokenizer = MosesTokenizer(lang=language)
    candidates = [tokenizer.tokenize(mt, return_str=True) for mt in candidates]
    references = [tokenizer.tokenize(ref, return_str=True) for ref in references]
    assert len(candidates) == len(references)
    bleu_scores = []
    for i in tqdm(range(len(candidates)), desc="Running BLEU..."):
        bleu_scores.append(corpus_bleu([candidates[i],], [[references[i],]]).score)
    return bleu_scores


def run_sentence_chrf(candidates: list, references: list) -> list:
    """ Runs sentence chrF from Sacrebleu. """
    assert len(candidates) == len(references)
    chrf_scores = []
    for i in tqdm(range(len(candidates)), desc="Running chrF..."):
        chrf_scores.append(
            sentence_chrf(hypothesis=candidates[i], reference=references[i]).score
        )
    return chrf_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates BLEU/chrF against relative preferences."
    )
    parser.add_argument(
        "--test_path",
        default="wmt-metrics/wmt19/de-en/relative-ranks.csv",
        help="Path to the test dataframe with relative preferences.",
        type=str,
    )
    parser.add_argument(
        "--language", default="en", help="Target language of the testset.", type=str,
    )
    parser.add_argument(
        "--run_wmt18",
        default=False,
        help="Runs entire WMT18 evaluation.",
        action="store_true",
    )
    parser.add_argument(
        "--run_wmt19",
        default=False,
        help="Runs entire WMT19 evaluation.",
        action="store_true",
    )

    args = parser.parse_args()
    if args.run_wmt18:
        lps = [
            "en-cs",
            "en-de",
            "en-et",
            "en-fi",
            "en-ru",
            "en-tr",
            "en-zh",
            "cs-en",
            "de-en",
            "et-en",
            "fi-en",
            "ru-en",
            "tr-en",
            "zh-en",
        ]
        kendall_scores = {}
        for lp in lps:
            data = pd.read_csv(f"wmt-metrics/wmt18/{lp}/relative-ranks.csv")
            scores_hyp1 = run_sentence_bleu(
                candidates=[str(s) for s in data.hyp1],
                references=list(data.ref),
                language=lp.split("-")[1],
            )

            scores_hyp2 = run_sentence_bleu(
                candidates=[str(s) for s in data.hyp2],
                references=list(data.ref),
                language=lp.split("-")[1],
            )
            bleu_kendall = compute_kendall(scores_hyp1, scores_hyp2, data)
            print(f"SentenceBLEU Kendall for {lp} : {bleu_kendall}")
            scores_hyp1 = run_sentence_chrf(
                candidates=[str(s) for s in data.hyp1], references=list(data.ref)
            )

            scores_hyp2 = run_sentence_chrf(
                candidates=[str(s) for s in data.hyp2], references=list(data.ref)
            )
            kendall_chrf = compute_kendall(scores_hyp1, scores_hyp2, data)
            print(f"chrF Kendall for {lp} : {kendall_chrf}\n")
            kendall_scores[lp] = [bleu_kendall, kendall_chrf]
        print(kendall_scores)

    elif args.run_wmt19:
        lps = [
            "en-cs",
            "en-de",
            "en-fi",
            "en-gu",
            "en-kk",
            "en-lt",
            "en-ru",
            "en-zh",
            "de-en",
            "fi-en",
            "gu-en",
            "kk-en",
            "lt-en",
            "ru-en",
            "zh-en",
            "de-cs",
            "de-fr",
            "fr-de",
        ]
        kendall_scores = {}
        for lp in lps:
            data = pd.read_csv(f"wmt-metrics/wmt19/{lp}/relative-ranks.csv")
            scores_hyp1 = run_sentence_bleu(
                candidates=[str(s) for s in data.hyp1],
                references=list(data.ref),
                language=lp.split("-")[1],
            )

            scores_hyp2 = run_sentence_bleu(
                candidates=[str(s) for s in data.hyp2],
                references=list(data.ref),
                language=lp.split("-")[1],
            )
            bleu_kendall = compute_kendall(scores_hyp1, scores_hyp2, data)
            print(f"SentenceBLEU Kendall for {lp} : {bleu_kendall}")
            scores_hyp1 = run_sentence_chrf(
                candidates=[str(s) for s in data.hyp1], references=list(data.ref)
            )

            scores_hyp2 = run_sentence_chrf(
                candidates=[str(s) for s in data.hyp2], references=list(data.ref)
            )
            kendall_chrf = compute_kendall(scores_hyp1, scores_hyp2, data)
            print(f"chrF Kendall for {lp} : {kendall_chrf}\n")
            kendall_scores[lp] = [bleu_kendall, kendall_chrf]
        print(kendall_scores)

    else:
        data = pd.read_csv(args.test_path)
        scores_hyp1 = run_sentence_bleu(
            candidates=[str(s) for s in data.hyp1],
            references=list(data.ref),
            language=args.language,
        )

        scores_hyp2 = run_sentence_bleu(
            candidates=[str(s) for s in data.hyp2],
            references=list(data.ref),
            language=args.language,
        )
        kendall = compute_kendall(scores_hyp1, scores_hyp2, data)
        print("SentenceBLEU Kendall for {} : {}".format(args.test_path, kendall))
        scores_hyp1 = run_sentence_chrf(
            candidates=[str(s) for s in data.hyp1], references=list(data.ref)
        )

        scores_hyp2 = run_sentence_chrf(
            candidates=[str(s) for s in data.hyp2], references=list(data.ref)
        )
        kendall = compute_kendall(scores_hyp1, scores_hyp2, data)
        print("chrF Kendall for {} : {}\n".format(args.test_path, kendall))
