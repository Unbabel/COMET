f"""
Shell script tho reproduce results for BLEURT in data from WMT19 Metrics Shared task.
"""
import sys
from datetime import datetime

import pandas as pd

from bleurt import score

startTime = datetime.now()


def compute_kendall(
    hyp1_scores: list, hyp2_scores: list, dataframe: pd.DataFrame
) -> (int, list):
    """ Computes the official WMT19 shared task Kendall correlation score. """
    assert len(hyp1_scores) == len(hyp2_scores) == len(data)
    conc, disc = 0, 0
    for i, row in data.iterrows():
        if hyp1_scores[i] > hyp2_scores[i]:
            conc += 1
        else:
            disc += 1

    return (conc - disc) / (conc + disc)


def run_bleurt(
    candidates: list, references: list, checkpoint: str = "bleurt/bleurt-large-512"
):
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references, candidates)
    return scores


def save_to_file(filename: str, scores: list):
    with open(filename, "w") as fp:
        for value in scores:
            fp.write(f"{value}\n")


if __name__ == "__main__":
    args = sys.argv
    print(args)
    checkpoint = args[1]
    modelname = checkpoint.split("/")[1]

    lps = ["fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en", "de-en"]

    correlations = []
    for lp in lps:
        lp_startTime = datetime.now()
        data = pd.read_csv(f"wmt-metrics/wmt19/{lp}/relative-ranks.csv")
        print("Scoring hypothesis 1...")
        scores_hyp1 = run_bleurt(
            [str(s) for s in data.hyp1], list(data.ref), checkpoint
        )

        save_to_file(f"wmt-metrics/wmt19/{lp}/{modelname}_hypothesis1.txt", scores_hyp1)

        print("Scoring hypothesis 2...")
        scores_hyp2 = run_bleurt(
            [str(s) for s in data.hyp2], list(data.ref), checkpoint
        )
        save_to_file(f"wmt-metrics/wmt19/{lp}/{modelname}_hypothesis2.txt", scores_hyp2)

        kendall = compute_kendall(scores_hyp1, scores_hyp2, data)
        print(f"BLEURT Kendall for {lp}: {kendall}")
        print("Runtime: {}\n".format(datetime.now() - lp_startTime))

    print("Total Runtime: {}".format(datetime.now() - startTime))
