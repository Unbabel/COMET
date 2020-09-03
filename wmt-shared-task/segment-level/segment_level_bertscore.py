f"""
Shell script tho reproduce results for BERTScores in data from WMT18/19 Metrics Shared task.
"""
import argparse

import bert_score
import pandas as pd
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


def run_bertscore(
    mt: list, ref: list, model_type="xlm-roberta-base", language=False, idf=False
) -> (list, list, list):
    """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
    if language:
        precison, recall, f1 = bert_score.score(
            cands=mt,
            refs=ref,
            idf=idf,
            batch_size=32,
            lang=language,
            rescale_with_baseline=False,
            verbose=True,
            nthreads=4,
        )
    else:
        precison, recall, f1 = bert_score.score(
            cands=mt,
            refs=ref,
            idf=idf,
            batch_size=32,
            model_type=model_type,
            rescale_with_baseline=False,
            verbose=True,
            nthreads=4,
        )
    return precison, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates BERTScores against relative preferences."
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
        "--model_type",
        default=None,
        help="Specifies particular encoder model to be used when computing BERTScores",
        type=str,
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
    parser.add_argument(
        "--idf",
        default=False,
        help="Runs BERTScores with inverse-document-frequency.",
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
            if args.model_type:
                hyp1_precision, hyp1_recall, hyp1_f1 = run_bertscore(
                    [str(s) for s in data.hyp1],
                    list(data.ref),
                    model_type=args.model_type,
                    idf=args.idf,
                )
                hyp2_precision, hyp2_recall, hyp2_f1 = run_bertscore(
                    [str(s) for s in data.hyp2],
                    list(data.ref),
                    model_type=args.model_type,
                    idf=args.idf,
                )
            else:
                hyp1_precision, hyp1_recall, hyp1_f1 = run_bertscore(
                    [str(s) for s in data.hyp1],
                    list(data.ref),
                    language=lp.split("-")[-1],
                    idf=args.idf,
                )
                hyp2_precision, hyp2_recall, hyp2_f1 = run_bertscore(
                    [str(s) for s in data.hyp2],
                    list(data.ref),
                    language=lp.split("-")[-1],
                    idf=args.idf,
                )

            print("Results for {}".format(lp))
            precision_kendall = compute_kendall(hyp1_precision, hyp2_precision, data)
            print(f"BERTScore Precision Kendall: {precision_kendall}")

            recall_kendall = compute_kendall(hyp1_recall, hyp2_recall, data)
            print(f"BERTScore Recall Kendall: {recall_kendall}")

            f1_kendall = compute_kendall(hyp1_f1, hyp2_f1, data)
            print(f"BERTScore F1 Kendall: {f1_kendall}\n")
            kendall_scores[lp] = [precision_kendall, recall_kendall, f1_kendall]
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
            if args.model_type:
                hyp1_precision, hyp1_recall, hyp1_f1 = run_bertscore(
                    [str(s) for s in data.hyp1],
                    list(data.ref),
                    model_type=args.model_type,
                    idf=args.idf,
                )
                hyp2_precision, hyp2_recall, hyp2_f1 = run_bertscore(
                    [str(s) for s in data.hyp2],
                    list(data.ref),
                    model_type=args.model_type,
                    idf=args.idf,
                )
            else:
                hyp1_precision, hyp1_recall, hyp1_f1 = run_bertscore(
                    [str(s) for s in data.hyp1],
                    list(data.ref),
                    language=lp.split("-")[-1],
                    idf=args.idf,
                )
                hyp2_precision, hyp2_recall, hyp2_f1 = run_bertscore(
                    [str(s) for s in data.hyp2],
                    list(data.ref),
                    language=lp.split("-")[-1],
                    idf=args.idf,
                )

            print("Results for {}".format(lp))
            precision_kendall = compute_kendall(hyp1_precision, hyp2_precision, data)
            print(f"BERTScore Precision Kendall: {precision_kendall}")

            recall_kendall = compute_kendall(hyp1_recall, hyp2_recall, data)
            print(f"BERTScore Recall Kendall: {recall_kendall}")

            f1_kendall = compute_kendall(hyp1_f1, hyp2_f1, data)
            print(f"BERTScore F1 Kendall: {f1_kendall}\n")
            kendall_scores[lp] = [precision_kendall, recall_kendall, f1_kendall]
        print(kendall_scores)

    else:
        data = pd.read_csv(args.test_path)

        if args.model_type:
            hyp1_precision, hyp1_recall, hyp1_f1 = run_bertscore(
                [str(s) for s in data.hyp1],
                list(data.ref),
                model_type=args.model_type,
                idf=args.idf,
            )
            hyp2_precision, hyp2_recall, hyp2_f1 = run_bertscore(
                [str(s) for s in data.hyp2],
                list(data.ref),
                model_type=args.model_type,
                idf=args.idf,
            )
        else:
            hyp1_precision, hyp1_recall, hyp1_f1 = run_bertscore(
                [str(s) for s in data.hyp1],
                list(data.ref),
                language=args.language,
                idf=args.idf,
            )
            hyp2_precision, hyp2_recall, hyp2_f1 = run_bertscore(
                [str(s) for s in data.hyp2],
                list(data.ref),
                language=args.language,
                idf=args.idf,
            )

        print("Results for {}".format(args.test_path))
        precision_kendall = compute_kendall(hyp1_precision, hyp2_precision, data)
        print(f"BERTScore Precision Kendall: {precision_kendall}")

        recall_kendall = compute_kendall(hyp1_recall, hyp2_recall, data)
        print(f"BERTScore Recall Kendall: {recall_kendall}")

        f1_kendall = compute_kendall(hyp1_f1, hyp2_f1, data)
        print(f"BERTScore F1 Kendall: {f1_kendall}\n")
