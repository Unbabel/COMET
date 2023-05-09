import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc_score(subword_scores):
    all_preds = []
    gold_truth = []
    for batch in subword_scores:
        y_hat = batch["subword_scores"].reshape(-1)
        y = batch["in_span_mask"].reshape(-1)
        mask = y != -1
        y_hat = torch.masked_select(y_hat, mask)
        y = torch.masked_select(y, mask)
        all_preds += y_hat.tolist()
        gold_truth += y.tolist()
        
    return roc_auc_score(gold_truth, all_preds)


def compute_average_precision_score(subword_scores):
    res = []
    for batch in subword_scores:
        for i in range(batch["in_span_mask"].shape[0]):
            mt_length = (batch["in_span_mask"][i] != -1).sum()
            y_hat = batch["subword_scores"][i][:mt_length]
            y = batch["in_span_mask"][i][:mt_length]
            
            # Check if annotation is missing.
            # Sometimes a sentence has no errors and thuswe skip.
            if (y == 1).sum() == 0:
                continue
            
            res.append(average_precision_score(y, y_hat))
                
    return sum(res) / len(res)


def compute_compute_recall_topk(subword_scores):

    def recall_at_topk(model_scores, gold_score):
        idxs = np.argsort(model_scores)[::-1][:sum(gold_score)]
        return len([idx for idx in idxs if gold_score[idx] == 1])/sum(gold_score)
    
    res = []
    for batch in subword_scores:
        for i in range(batch["in_span_mask"].shape[0]):
            mt_length = (batch["in_span_mask"][i] != -1).sum()
            y_hat = batch["subword_scores"][i][:mt_length]
            y = batch["in_span_mask"][i][:mt_length]
            
            # check if annotation is missing
            if (y == 1).sum() == 0:
                continue
            
            res.append(recall_at_topk(y_hat.numpy(), y.numpy()))
    return sum(res)/len(res)

def get_score_matrix(all_subword_scores, num_layers, num_heads, eval_func):
    scores = np.zeros((num_layers, num_heads))
    for layer_id in range(num_layers):
        for head_id in range(num_heads):
            subword_scores = [{
                "subword_scores": s["subword_scores"][layer_id][head_id],
                "in_span_mask": s["in_span_mask"],
                "input_ids": s["input_ids"]
            } for s in all_subword_scores]
            scores[layer_id][head_id] = eval_func(subword_scores)
    return scores


def top_k_max_indices(matrix, k):
    # Create an array of zeros with the same shape as the input matrix
    max_indices = np.zeros_like(matrix)

    # Sort the input matrix in non-decreasing order
    sorted_matrix = np.sort(matrix, axis=None)

    # Get the top k max values from the sorted matrix
    top_k_max_values = sorted_matrix[-k:]

    # Loop over the top k max values
    positions = []
    for i, max_value in enumerate(top_k_max_values):
        # Find the positions of the max value in the input matrix
        pos = np.where(matrix == max_value)
        positions.append((pos[0][0], pos[1][0]))

    return positions