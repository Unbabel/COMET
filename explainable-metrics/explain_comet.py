""" Script to evaluate COMET model explanations on a given dataset. 

python explain_comet.py -m PATH/TO/UNITE.ckpt -t PATH/TO/DATA --batch_size 8

"""
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from utils import (compute_auc_score, compute_compute_recall_topk,
                   get_score_matrix, top_k_max_indices)

from comet import download_model, load_from_checkpoint
from comet.models import RegressionMetric
from comet.models.pooling_utils import average_pooling

CUDA = "cuda:0"

def prepare_data(
    model: RegressionMetric, 
    samples: List[Dict[str, str]], 
    batch_size: int = 8
):
    """ Builds batches to input the model.

    Args:
        model (RegressionMetric): COMET model.
        samples (List[Dict[str, str]]): List with samples (dict with src, mt, ref)
        batch_size (int, optional): Batch size. Defaults to 2.

    Returns:
        List[dict]: List with model inputs
    """
    batch_inputs = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    model_inputs = []
    for batch in tqdm(batch_inputs):
        batch = {k: [str(dic[k]) for dic in batch] for k in batch[0]}
        src_inputs = model.encoder.prepare_sample(batch["src"])
        
        # If word_level_training = True 
        # we will keep a mask for words inside annotation spans
        mt_inputs  = model.encoder.prepare_sample(batch["mt"], word_level_training=True)
        ref_inputs = model.encoder.prepare_sample(batch["ref"])
        
        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        
        model_inputs.append({**src_inputs, **mt_inputs, **ref_inputs})
        
    return model_inputs


def get_word_embeddings(
    model: RegressionMetric, 
    token_ids: torch.Tensor, 
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """ This method returns the word embeddings for a single sentence (e.g the MT)

    Args:
        model (RegressionMetric): COMET model.
        token_ids (torch.Tensor): input ids.
        attention_mask (torch.Tensor): attention mask.

    Returns:
        torch.Tensor: Word embeddings for the input sentence
    """
    with torch.no_grad():
        encoder_out = model.encoder(token_ids.to(CUDA), attention_mask.to(CUDA))
        embeddings = model.layerwise_attention(
            encoder_out["all_layers"], attention_mask
        )
        return embeddings

def alignment_explanations(
    model: RegressionMetric, 
    batch_input: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor]:
    """ Creates explanations from embedding alignments for a given input.

    Args:
        model (RegressionMetric): COMET model
        batch_input (Dict[torch.Tensor]): Input batch
    
    Returns:
        Tuple[torch.Tensor]: source and reference explanations.
    """
    def get_similarity_scores(mt_embs, ref_embs):
        """Return max sim score for each word in the MT compared to reference
        embeddings (reference can be also the source for QE).
        """
        ref_embs.div_(torch.norm(ref_embs, dim=-1).unsqueeze(-1))
        mt_embs.div_(torch.norm(mt_embs, dim=-1).unsqueeze(-1))
        sim = torch.bmm(mt_embs, ref_embs.transpose(1, 2))
        alignment = sim.max(dim=2)
        sim_scores = 1- alignment.values
        return sim_scores.cpu()
        
    mt_embs = get_word_embeddings(
        model=model,
        token_ids=batch_input["mt_input_ids"], 
        attention_mask=batch_input["mt_attention_mask"]
    )
    src_embs = get_word_embeddings(
        model=model,
        token_ids=batch_input["src_input_ids"], 
        attention_mask=batch_input["src_attention_mask"]
    )
    ref_embs = get_word_embeddings(
        model=model,
        token_ids=batch_input["ref_input_ids"], 
        attention_mask=batch_input["ref_attention_mask"]
    )
    ref_explanations = get_similarity_scores(mt_embs, ref_embs)
    src_explanations = get_similarity_scores(mt_embs, src_embs)
    src_ref_explanations = get_similarity_scores(
        mt_embs, torch.cat((src_embs, ref_embs), dim=1)
    )
    return src_explanations, ref_explanations, src_ref_explanations
        
def mock_forward(
    model: RegressionMetric, 
    batch_input: torch.Tensor
) -> Tuple[torch.Tensor]:
    """ This function runs forward and computes a loss on a fake target.
    Required to extract gradients.

    Args:
        model (RegressionMetric): COMET model
        batch_input (torch.Tensor): input batch

    Returns:
        Tuple[torch.tensor]: MT attention, hidden states and attention mask.
    """
    src_sentemb = model.get_sentence_embedding(
        batch_input["src_input_ids"], 
        batch_input["src_attention_mask"]
    )
    ref_sentemb = model.get_sentence_embedding(
        batch_input["ref_input_ids"], 
        batch_input["ref_attention_mask"]
    )
        
    # MT Embeddings
    encoder_out = model.encoder(
        batch_input["mt_input_ids"], 
        batch_input["mt_attention_mask"]
    )
    # Stored for later
    mt_attention = torch.stack(encoder_out["attention"])
    mt_hidden_states = torch.stack(encoder_out['all_layers'])
        
    mt_embs = model.layerwise_attention(
        encoder_out["all_layers"], 
        batch_input["mt_attention_mask"]
    )
    mt_sentemb = average_pooling(
        batch_input["mt_input_ids"],
        mt_embs,
        batch_input["mt_attention_mask"],
        model.encoder.tokenizer.pad_token_id,
    )
        
    pred_score = model.estimate(src_sentemb, mt_sentemb, ref_sentemb).score
    dummy_loss = torch.sum((pred_score) ** 2)
    dummy_loss.backward()
    return mt_attention, mt_hidden_states, encoder_out["attention_mask"]
      
def attention_x_grad_explanations(
    model: RegressionMetric, 
    batch_input: Dict[str, torch.Tensor], 
    multiply_by_grad: bool = True
) -> torch.Tensor:
    """ Extracts attention heads and representations from all layers.

    Args:
        model (UnifiedMetric): UniTE model
        token_ids (torch.Tensor): model input.
        attention_mask (torch.Tensor): Attention mask

    Returns:
        Tuple[torch.Tensor]: Tuple with attention, representations from all layers and
        the respective attention mask.
    """
    num_layers = model.encoder.model.config.num_hidden_layers
    num_heads = model.encoder.model.config.num_attention_heads

    if multiply_by_grad:
        for param in model.parameters():
            param.requires_grad = True
        attention, hidden_states, attn_mask = mock_forward(model, batch_input)
    
    else:
        out = model.encoder(
            batch_input["mt_input_ids"], 
            batch_input["mt_attention_mask"]
        )
        attention, hidden_states, attn_mask = (
            torch.stack(out["attention"]), 
            torch.stack(out['all_layers']), 
            out["attention_mask"]
        )
        
    hidden_states = hidden_states.transpose(0, 1)
    attention = attention.transpose(0, 1)
    
    mt_mask = batch_input["mt_in_span_mask"] != -1
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
    model: RegressionMetric, 
    batch_input: Dict[str, torch.Tensor], 
    top_k: List[int], 
    multiply_by_grad: bool = True
) -> torch.Tensor:
    """Ensembles features from topk heads.

    Args:
        model (RegressionMetric): COMET metric.
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
        attention, hidden_states, attn_mask = mock_forward(model, batch_input)
    else:
        out = model.encoder(
            batch_input["mt_input_ids"], 
            batch_input["mt_attention_mask"]
        )
        attention, hidden_states, attn_mask = (
            torch.stack(out["attention"]), 
            torch.stack(out['all_layers']), 
            out["attention_mask"]
        )
        
    hidden_states = hidden_states.transpose(0, 1)
    attention = attention.transpose(0, 1)
    
    mt_mask = batch_input["mt_in_span_mask"] != -1
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
        ).sum(1))[:, :seq_len]
        topk_features.append(attn_sum)
        
    subword_scores = torch.stack(topk_features)
    return subword_scores.mean(dim=0).detach().cpu()


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
    
    if "ckpt" in args.model:
        model = load_from_checkpoint(args.model)
    else:
        model_path = download_model(args.model)
        model = load_from_checkpoint(model_path)
    
    if args.metric == "recall":
        metric = compute_compute_recall_topk
    else:
        metric = compute_auc_score
        
    # Setup Model:
    model.encoder.add_span_tokens("<v>", "</v>")
    model.to(CUDA)
    model.eval()
    
    for testset in args.testset[0]:
        data = pd.read_csv(testset)
        data.mt = data.annotation # change MT for Annotation
        model_inputs = prepare_data(model, data.to_dict("records"), args.batch_size)

        #  ------------------------- Alignment Explanations --------------------
        ref_subword_scores, src_subword_scores, src_ref_subword_scores = [], [], []
        for batch in tqdm(model_inputs):
            batch = {k: v.to(CUDA) for k, v in batch.items()}
            src_align, ref_align, src_ref_align = alignment_explanations(model, batch)
            ref_subword_scores.append({
                "subword_scores": ref_align,
                "input_ids": batch["mt_input_ids"].cpu(),
                "in_span_mask": batch["mt_in_span_mask"].cpu()
                
            })
            src_subword_scores.append({
                "subword_scores": src_align,
                "input_ids": batch["mt_input_ids"].cpu(),
                "in_span_mask": batch["mt_in_span_mask"].cpu()
            })
            src_ref_subword_scores.append({
                "subword_scores": src_ref_align,
                "input_ids": batch["mt_input_ids"].cpu(),
                "in_span_mask": batch["mt_in_span_mask"].cpu()
            })
            
        src_align_score = metric(src_subword_scores)
        ref_align_score = metric(ref_subword_scores)
        src_ref_align_score = metric(src_ref_subword_scores)
        print ("Src Align {}: {}".format(args.metric, src_align_score))
        print ("Ref Align {}: {}".format(args.metric, ref_align_score))
        print ("Src+Ref Align {}: {}".format(args.metric, src_ref_align_score))
    
        # -------------------- Attention x Grad explanations -------------
        subword_scores = []
        for batch in tqdm(model_inputs):
            batch = {k: v.to(CUDA) for k, v in batch.items()}
            subword_scores.append({
                "subword_scores": attention_x_grad_explanations(model, batch, multiply_by_grad=True),
                "input_ids": batch["mt_input_ids"].cpu(),
                "in_span_mask": batch["mt_in_span_mask"].cpu()
            })
        
        num_layers = model.encoder.model.config.num_hidden_layers
        num_heads = model.encoder.model.config.num_attention_heads
        score_matrix = get_score_matrix(subword_scores, num_layers, num_heads, metric)
        print ("Attention x Grad {}: {}".format(args.metric, score_matrix.max()))
        print ("Top 5 Heads:")
        topk_heads = top_k_max_indices(score_matrix, 5)
        for pos in topk_heads:
            print (f"{pos}")
        
        