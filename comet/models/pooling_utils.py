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
import torch
from typing import List, Union

# From https://github.com/amazon-science/doc-mt-metrics/blob/5385cc28930aae9924edcb3201645dd3810b12c0/COMET/comet/models/pooling_utils.py#L18
def find_start_inds_and_mask_tokens(
        mask: torch.Tensor,
        tokens: torch.Tensor,
        separator_index: int,
) -> Union[List[int], torch.Tensor]:
    """Finds the starting indices of each sentence for multi-sentence sequences and 
    creates a new mask to omit all context sentences from the pooling function.

    Args:
        mask: Padding mask [batch_size x seq_length]
        tokens: Word ids [batch_size x seq_length]
        separator_index: Separator token index.
    """
    start_inds = []
    ctx_mask = mask
    for i, sent in enumerate(tokens):
        # find all separator tokens in the sequence
        separators = (sent == separator_index).nonzero()
        if len(separators) > 1:
            # if there are more than one find where the last sentence starts
            ind = separators[-2].cpu().numpy().item()
            start_inds.append(ind)
            ctx_mask[i, 1:ind+1] = 0
        else:
            start_inds.append(0)
    return start_inds, ctx_mask

def average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
    separator_index: int,
    enable_context: bool = False
) -> torch.Tensor:
    """Average pooling method.

    Args:
        tokens (torch.Tensor): Word ids [batch_size x seq_length]
        embeddings (torch.Tensor): Word embeddings [batch_size x seq_length x
            hidden_size]
        mask (torch.Tensor): Padding mask [batch_size x seq_length]
        padding_index (torch.Tensor): Padding value.

    Return:
        torch.Tensor: Sentence embedding
    """
    if enable_context:
        start_inds, ctx_mask = find_start_inds_and_mask_tokens(mask, tokens, separator_index)
        wordemb = mask_fill_index(0.0, tokens, embeddings, start_inds, padding_index)
        sentemb = torch.sum(wordemb, 1)
        sum_mask = ctx_mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    else: 
        wordemb = mask_fill(0.0, tokens, embeddings, padding_index)
        sentemb = torch.sum(wordemb, 1)
        sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


def max_pooling(
    tokens: torch.Tensor, embeddings: torch.Tensor, padding_index: int
) -> torch.Tensor:
    """Max pooling method.

    Args:
        tokens (torch.Tensor): Word ids [batch_size x seq_length]
        embeddings (torch.Tensor): Word embeddings [batch_size x seq_length x
            hidden_size]
        padding_index (int):Padding value.

    Return:
        torch.Tensor: Sentence embedding
    """
    return mask_fill(float("-inf"), tokens, embeddings, padding_index).max(dim=1)[0]

# From https://github.com/amazon-science/doc-mt-metrics/blob/5385cc28930aae9924edcb3201645dd3810b12c0/COMET/comet/models/pooling_utils.py#L18
def mask_fill_index(
        fill_value: float,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        start_inds: list,
        padding_index: int,
) -> torch.Tensor:
    """
    Masks embeddings representing padded elements and context sentences for multi-sentence sequences.

    Args:
        fill_value: the value to fill the embeddings belonging to padded tokens.
        tokens: The input sequences [bsz x seq_len].
        embeddings: word embeddings [bsz x seq_len x hiddens].
        start_inds: Start of sentence indices.
        padding_index: Index of the padding token.
    
    Return:
        torch.Tensor: Sentence embedding
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    padding_maks2 = torch.zeros(tokens.shape, dtype=torch.bool, device=padding_mask.device)
    for i, start in enumerate(start_inds):
        padding_maks2[i, 1: start+1] = True
    padding_mask = torch.logical_or(padding_mask, padding_maks2.unsqueeze(-1))
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)

def mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Method that masks embeddings representing padded elements.

    Args:
        fill_value (float): the value to fill the embeddings belonging to padded tokens
        tokens (torch.Tensor): Word ids [batch_size x seq_length]
        embeddings (torch.Tensor): Word embeddings [batch_size x seq_length x
            hidden_size]
        padding_index (int):Padding value.

    Return:
        torch.Tensor: Word embeddings [batch_size x seq_length x hidden_size]
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)
