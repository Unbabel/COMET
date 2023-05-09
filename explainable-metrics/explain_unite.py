""" Script to evaluate UniTE model explanations on a given dataset. 

python explain_unite.py -m PATH/TO/UNITE.ckpt -t PATH/TO/DATA --batch_size 8

"""
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from utils import (compute_auc_score, compute_compute_recall_topk,
                   get_score_matrix, top_k_max_indices)

from comet import download_model, load_from_checkpoint
from comet.models import UnifiedMetric

CUDA = "cuda:0"

def prepare_data(
    model: UnifiedMetric, 
    samples: List[Dict[str, str]], 
    batch_size: int = 8
):
    """ Builds batches to input the model.

    Args:
        model (UnifiedMetric): UniTE model.
        samples (List[Dict[str, str]]): List with samples (dict with src, mt, ref)
        batch_size (int, optional): Batch size. Defaults to 2.

    Returns:
        List[dict]: List with model inputs
    """
    model.word_level = True    
    batch_inputs = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    model_inputs = []
    for sample in tqdm(batch_inputs):
        model_inputs.append(model.prepare_sample(sample, stage="predict"))
        
    return model_inputs

def alignment_explanations(
    model: UnifiedMetric, 
    batch_input: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """ Generates explanation through embedding alignments between MT and reference.

    Args:
        model (UnifiedMetric): UniTE model.
        batch_input (Dict[str, torch.Tensor]): Model input

    Returns:
        torch.Tensor: Alignment scores.
    """
    with torch.no_grad():
        _, hidden_states, attn_mask = attention_and_hiddens(
            model, batch_input["input_ids"], batch_input["attention_mask"]
        )
        embeddings = model.layerwise_attention(hidden_states, attn_mask)
        embeddings = embeddings.cpu()
    
    mt_mask = batch_input["in_span_mask"] != -1
    mt_length = mt_mask.sum(dim=1)
    seq_len = attn_mask.sum(dim=1)
    
    subword_scores = []
    for i in range(len(mt_length)):
        mt_embeddings = embeddings[i, :mt_length[i]]
        ref_embeddings = embeddings[i, mt_length[i]: seq_len[i]]
        
        mt_embeddings.div_(torch.norm(mt_embeddings, dim=-1).unsqueeze(-1))
        ref_embeddings.div_(torch.norm(ref_embeddings, dim=-1).unsqueeze(-1))
        # remove CLS and EOS
        sim = torch.mm(mt_embeddings, ref_embeddings.transpose(0, 1))[1:-1]
        alignment = sim.max(dim=1)
        sim_scores = 1- alignment.values
        subword_scores.append([0, ] +  sim_scores.tolist() + [0, ])
        
        # Pad to max len
        subword_scores[-1] = subword_scores[-1] + [0]*(mt_length.max() -  len(subword_scores[-1]))
        
    return torch.tensor(subword_scores)

def mock_forward(
    model: UnifiedMetric, 
    encoder_embeddings: torch.Tensor, 
    attention_mask: torch.Tensor
) -> None:
    """ This functione runs forward and computes a loss on a fake target.
    Required to extract gradients.

    Args:
        model (UnifiedMetric): UniTE Metric
        encoder_embeddings (torch.Tensor): embeddings.
        attention_mask (torch.Tensor): attention mask.
    """
    model.zero_grad()
    sentemb = model.layerwise_attention(encoder_embeddings, attention_mask)[:, 0, :]
    pred_score = model.estimator(sentemb).view(-1)
    dummy_loss = torch.sum((pred_score) ** 2)
    dummy_loss.backward()
    
def attention_and_hiddens(
    model: UnifiedMetric, 
    token_ids: torch.Tensor, 
    attention_mask: torch.Tensor
) -> Tuple[torch.Tensor]:
    """ Extracts attention heads and representations from all layers.

    Args:
        model (UnifiedMetric): UniTE model
        token_ids (torch.Tensor): model input.
        attention_mask (torch.Tensor): Attention mask

    Returns:
        Tuple[torch.Tensor]: Tuple with attention, representations from all layers and
        the respective attention mask.
    """
    out = model.encoder(token_ids.to(CUDA), attention_mask.to(CUDA))
    token_ids.cpu(), attention_mask.cpu()
    return (
        torch.stack(out["attention"]), 
        torch.stack(out['all_layers']), 
        out["attention_mask"]
    )

def attention_x_grad_explanations(
    model: UnifiedMetric, 
    batch_input: torch.Tensor, 
    multiply_by_grad: bool = True
) -> torch.Tensor:
    """ Returns explanations that result from multiplying attention with gradients.

    Args:
        model (UnifiedMetric): UniTE model.
        batch_input (torch.Tensor): Model input
        multiply_by_grad (bool, optional): If set to False the returned score 
            is just the attention score. Defaults to True.

    Returns:
        torch.Tensor: Explanation scores
    """
    num_layers = model.encoder.model.config.num_hidden_layers
    num_heads = model.encoder.model.config.num_attention_heads
    
    if multiply_by_grad:
        for param in model.parameters():
            param.requires_grad = True
    
    attention, hidden_states, attn_mask = attention_and_hiddens(
        model, batch_input["input_ids"].to(CUDA), batch_input["attention_mask"].to(CUDA)
    )
    
    if multiply_by_grad:
        mock_forward(model, hidden_states, batch_input["attention_mask"].to(CUDA))
        
    hidden_states = hidden_states.transpose(0, 1)
    attention = attention.transpose(0, 1)
    
    mt_mask = batch_input["in_span_mask"] != -1
    mt_length = mt_mask.sum(dim=1)
    seq_len = mt_length.max()
    
    layer_scores = []
    for layer_id in range(num_layers):
        if multiply_by_grad:
            self_attn_module = model.encoder.model.encoder.layer[layer_id].attention.self
            v_grad = self_attn_module.value_layer.grad
            v_grad_norm = torch.norm(v_grad.detach(), p=2, dim=-1)
            attention[:, layer_id] = attention[:, layer_id] * v_grad_norm.unsqueeze(-2)
            
        head_scores = [] 
        for head_id in range(num_heads):
            attn_sum = (
                (attention[:, layer_id, head_id] * attn_mask.unsqueeze(-1).float()
            ).sum(1))[:, :seq_len]
            head_scores.append(attn_sum.detach().cpu())
        layer_scores.append(head_scores)

    return layer_scores

def ensemble_topk_features(
    model: UnifiedMetric, 
    batch_input: Dict[str, torch.Tensor], 
    top_k: List[int], 
    multiply_by_grad: bool = True
) -> torch.Tensor:
    """Ensembles features from topk heads.

    Args:
        model (UnifiedMetric): UniTE metric.
        batch_input (Dict[str, torch.Tensor]): Model input.
        top_k (List[int]): List with positions of the best performing attention heads
        multiply_by_grad (bool, optional): If set to False the returned score 
            is just the attention score. Defaults to True.

    Returns:
        torch.Tensor: Explanation scores.
    """
    if multiply_by_grad:
        for param in model.parameters():
            param.requires_grad = True
    
    attention, hidden_states, attn_mask = attention_and_hiddens(
        model, batch_input["input_ids"].to(CUDA), batch_input["attention_mask"].to(CUDA)
    )
    
    if multiply_by_grad:
        mock_forward(model, hidden_states, batch_input["attention_mask"].to(CUDA))
        
    hidden_states = hidden_states.transpose(0, 1)
    attention = attention.transpose(0, 1)
    
    mt_mask = batch_input["in_span_mask"] != -1
    mt_length = mt_mask.sum(dim=1)
    seq_len = mt_length.max()
    
    topk_features = []
    for i in range(len(top_k)):
        layer_id, head_id = top_k[i]
        
        self_attn_module = model.encoder.model.encoder.layer[layer_id].attention.self
        v_grad = self_attn_module.value_layer.grad
        v_grad_norm = torch.norm(v_grad.detach(), p=2, dim=-1)

        attention[:, layer_id] = attention[:, layer_id] * v_grad_norm.unsqueeze(-2)
        attn_sum = (
            (attention[:, layer_id, head_id] * attn_mask.unsqueeze(-1).float()
        ).sum(1))[:, :seq_len].detach().cpu()
        topk_features.append(attn_sum)
        
    subword_scores = torch.stack(topk_features)
    return subword_scores.mean(dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates Model."
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model or Path to a checkpoint file.",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    )
    parser.add_argument(
        "-t", "--testset", 
        required=True,
        help="Testset Path.", 
        action='append', nargs='+'
    )
    parser.add_argument(
        "--metric",
        default="AUC",
        type=str,
    )
    args = parser.parse_args()
    
    if args.metric == "recall":
        metric = compute_compute_recall_topk
    else:
        metric = compute_auc_score
    
    if "ckpt" in args.model:
        model = load_from_checkpoint(args.model)
    else:
        model_path = download_model(args.model)
        model = load_from_checkpoint(model_path)
    
    # Setup Model:
    model.encoder.add_span_tokens("<v>", "</v>")
    model.to(CUDA)
    model.eval()
    
    num_layers = model.encoder.model.config.num_hidden_layers
    num_heads = model.encoder.model.config.num_attention_heads
    
    for testset in args.testset[0]:    
        data = pd.read_csv(testset)
        data.mt = data.annotation # change MT for Annotation
        model_inputs = prepare_data(model, data.to_dict("records"), args.batch_size)
        
        #  ------------------------- SRC Alignment Explanations --------------------
        align_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[0].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            align_scores = alignment_explanations(model, batch)
            align_subword_scores.append({
                "subword_scores": align_scores,
                "input_ids": batch["input_ids"][:, :seq_len],
                "in_span_mask": batch["in_span_mask"][:, :seq_len]
            })
            
        score = metric(align_subword_scores)
        print ("SRC Align. {}: {}".format(args.metric, score))
        
        #  ------------------------- REF Alignment Explanations --------------------
        align_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[1].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            align_scores = alignment_explanations(model, batch)
            align_subword_scores.append({
                "subword_scores": align_scores,
                "input_ids": batch["input_ids"][:, :seq_len],
                "in_span_mask": batch["in_span_mask"][:, :seq_len]
            })
            
        score = metric(align_subword_scores)
        print ("REF Align. {}: {}".format(args.metric, score))
        
        #  ------------------------- Uni Alignment Explanations --------------------
        align_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[2].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            align_scores = alignment_explanations(model, batch)
            align_subword_scores.append({
                "subword_scores": align_scores,
                "input_ids": batch["input_ids"][:, :seq_len],
                "in_span_mask": batch["in_span_mask"][:, :seq_len]
            })
            
        score = metric(align_subword_scores)
        print ("Uni Align. {}: {}".format(args.metric, score))
        
        # ------------------------- SOURCE explanations --------------------
        src_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[0].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            src_subword_scores.append({
                "subword_scores": attention_x_grad_explanations(model, batch, multiply_by_grad=True),
                "input_ids": batch["input_ids"][:, :seq_len],
                "in_span_mask": batch["in_span_mask"][:, :seq_len]
            })
            
        score_matrix = get_score_matrix(src_subword_scores, num_layers, num_heads, metric)
        print ("SRC Attention x Grad {}: {}".format(args.metric, score_matrix.max()))
        print ("Top 5 Heads:")
        topk_heads = top_k_max_indices(score_matrix, 5)
        for pos in topk_heads:
            print (f"{pos}")
            
        # ------------------------- REFERENCE explanations --------------------
        ref_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[1].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            ref_subword_scores.append({
                "subword_scores": attention_x_grad_explanations(model, batch, multiply_by_grad=True),
                "input_ids": batch["input_ids"].cpu()[:, :seq_len],
                "in_span_mask": batch["in_span_mask"].cpu()[:, :seq_len]
            })
            
        score_matrix = get_score_matrix(ref_subword_scores, num_layers, num_heads, metric)
        print ("REF Attention x Grad {}: {}".format(args.metric, score_matrix.max()))
        print ("Top 5 Heads:")
        topk_heads = top_k_max_indices(score_matrix, 5)
        for pos in topk_heads:
            print (f"{pos}")

        # ------------------------- ENSEMBLE top 5 --------------------
        ensemble_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[2].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            ensemble_subword_scores.append({
                "subword_scores": ensemble_topk_features(model, batch, topk_heads, multiply_by_grad=True),
                "input_ids": batch["input_ids"].cpu()[:, :seq_len],
                "in_span_mask": batch["in_span_mask"].cpu()[:, :seq_len]
            })
        
        score = metric(ensemble_subword_scores)
        print ("Top-5 Ref Attention x Grad {}: {}".format(args.metric, score))
        
        # ------------------------- UNIFIED explanations --------------------
        uni_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[2].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            uni_subword_scores.append({
                "subword_scores": attention_x_grad_explanations(model, batch, multiply_by_grad=True),
                "input_ids": batch["input_ids"].cpu()[:, :seq_len],
                "in_span_mask": batch["in_span_mask"].cpu()[:, :seq_len]
            })
            
        score_matrix = get_score_matrix(uni_subword_scores, num_layers, num_heads, metric)
        print ("Uni Attention x Grad {}: {}".format(args.metric, score_matrix.max()))
        print ("Top 5 Heads:")
        topk_heads = top_k_max_indices(score_matrix, 5)
        for pos in topk_heads:
            print (f"{pos}")

        # ------------------------- ENSEMBLE top 5 --------------------
        ensemble_subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v for k, v in batch[2].items()}
            mt_mask = batch["in_span_mask"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            ensemble_subword_scores.append({
                "subword_scores": ensemble_topk_features(model, batch, topk_heads, multiply_by_grad=True),
                "input_ids": batch["input_ids"].cpu()[:, :seq_len],
                "in_span_mask": batch["in_span_mask"].cpu()[:, :seq_len]
            })
        
        score = metric(ensemble_subword_scores)
        print ("Top-5 UNI Attention x Grad {}: {}".format(args.metric, score))
        
