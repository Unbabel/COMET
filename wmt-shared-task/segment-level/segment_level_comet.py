f"""
Shell script tho reproduce results for COMET models in data from WMT18/19 Metrics Shared task.
"""
import argparse
import json

import pandas as pd
from tqdm import tqdm

from comet.models import load_checkpoint


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


def create_samples(dataframe: pd.DataFrame):
    """ Dataframe to dictionary. """
    hyp1_samples, hyp2_samples = [], []
    for i, row in dataframe.iterrows():
        hyp1_samples.append(
            {"src": str(row["src"]), "ref": str(row["ref"]), "mt": str(row["hyp1"])}
        )
        hyp2_samples.append(
            {"src": str(row["src"]), "ref": str(row["ref"]), "mt": str(row["hyp2"])}
        )
    return hyp1_samples, hyp2_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates a COMET model against relative preferences."
    )
    parser.add_argument(
        "--checkpoint",
        default="_ckpt_epoch_1.ckpt",
        help="Path to the Model checkpoint we want to test.",
        type=str,
    )
    parser.add_argument(
        "--test_path",
        default="wmt-metrics/wmt19/de-en/relative-ranks.csv",
        help="Path to the test dataframe with relative preferences.",
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
        "--invert_score",
        default=False,
        help="Inverts the score (e.g 1 - HTER).",
        action="store_true",
    )
    parser.add_argument(
        "--cuda", default=False, help="Uses cuda.", action="store_true",
    )
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)

    if args.run_wmt18:
        lps = ["en-cs", "en-de", "en-fi", "en-tr", "cs-en", "de-en", "fi-en", "tr-en"]
        kendall_scores = {}
        for lp in lps:
            data = pd.read_csv(f"wmt-metrics/wmt18/{lp}/relative-ranks.csv")
            hyp1_samples, hyp2_samples = create_samples(data)
            print("Running model for hypothesis 1:")
            _, hyp1_scores = model.predict(
                hyp1_samples, cuda=args.cuda, show_progress=True
            )
            if args.invert_score:
                hyp1_scores = [1 - score for score in hyp1_scores]

            print("Running model for hypothesis 2:")
            _, hyp2_scores = model.predict(
                hyp2_samples, cuda=args.cuda, show_progress=True
            )
            if args.invert_score:
                hyp2_scores = [1 - score for score in hyp2_scores]

            kendall = compute_kendall(hyp1_scores, hyp2_scores, data)
            print("Results for {}".format(lp))
            print(f"Kendall correlation: {kendall}")
            kendall_scores[lp] = kendall
        print(json.dumps(kendall_scores, indent=4))
        print ("Average Kendall: {}".format(sum(kendall_scores.values())/len(kendall_scores)))

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
            hyp1_samples, hyp2_samples = create_samples(data)
            print("Running model for hypothesis 1:")
            _, hyp1_scores = model.predict(
                hyp1_samples, cuda=args.cuda, show_progress=True
            )
            if args.invert_score:
                hyp1_scores = [1 - score for score in hyp1_scores]

            print("Running model for hypothesis 2:")
            _, hyp2_scores = model.predict(
                hyp2_samples, cuda=args.cuda, show_progress=True
            )
            if args.invert_score:
                hyp2_scores = [1 - score for score in hyp2_scores]

            kendall = compute_kendall(hyp1_scores, hyp2_scores, data)
            print("Results for {}".format(lp))
            print(f"Kendall correlation: {kendall}\n")
            kendall_scores[lp] = kendall
        print(json.dumps(kendall_scores, indent=4))
        print ("Average Kendall: {}".format(sum(kendall_scores.values())/len(kendall_scores)))

    else:
        data = pd.read_csv(args.test_path)
        hyp1_samples, hyp2_samples = create_samples(data)
        print("Running model for hypothesis 1:")
        _, hyp1_scores = model.predict(hyp1_samples, cuda=args.cuda, show_progress=True)
        if args.invert_score:
            hyp1_scores = [1 - score for score in hyp1_scores]

        print("Running model for hypothesis 2:")
        _, hyp2_scores = model.predict(hyp2_samples, cuda=args.cuda, show_progress=True)
        if args.invert_score:
            hyp2_scores = [1 - score for score in hyp2_scores]

        kendall = compute_kendall(hyp1_scores, hyp2_scores, data)
        print("Results for {}".format(args.test_path))
        print(f"Kendall correlation: {kendall}")
