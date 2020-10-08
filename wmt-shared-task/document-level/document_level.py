f"""
Shell script to Evaluate COMET at Document-level.
"""
import argparse
import json

from comet.models import load_checkpoint
from scipy.stats import pearsonr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates a COMET model against relative preferences."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Model checkpoint we want to test.",
        type=str,
    )
    parser.add_argument(
        "--test_path",
        required=True,
        help="Path to the test json with translated documents.",
        type=str,
    )

    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)
    seg_micro_scores, seg_macro_scores, seg_y = [], [], []
    sys_micro, sys_macro, sys_y = [], [], []
    with open(args.test_path) as json_file:
        data = json.load(json_file)
        for system in data:
            print ("Scoring {} system:".format(system))
            docs = [data[system][d] for d in data[system]]
            human_scores = [d['z_score'] for d in docs]
            _, micro_scores, macro_scores = model.document_predict(docs, cuda=True, show_progress=True)
            
            sys_micro.append(sum(micro_scores)/len(micro_scores))
            sys_macro.append(sum(macro_scores)/len(macro_scores))
            sys_y.append(sum(human_scores)/len(human_scores))
            print ("MICRO {} system-level score: {}".format(system, sys_micro[-1]))
            print ("MACRO {} system-level score: {}".format(system, sys_macro[-1]))
            print ("HUMAN {} system-level score: {}".format(system, sys_y[-1]))
    
    print ("MICRO Segment-level Pearson: {}".format(pearsonr(seg_micro_scores, seg_y)[0]))
    print ("MACRO Segment-level Pearson: {}".format(pearsonr(seg_macro_scores, seg_y)[0]))
    print ("MICRO System-level Pearson: {}".format(pearsonr(sys_micro, sys_y)[0]))
    print ("MACRO System-level Pearson: {}".format(pearsonr(sys_macro, sys_y)[0]))
