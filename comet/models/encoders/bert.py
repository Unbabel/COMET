# -*- coding: utf-8 -*-
r"""
BERT Encoder
==============
    Pretrained BERT encoder from Hugging Face.
"""
from argparse import Namespace
from typing import Dict

import torch
from transformers import AutoModel

from comet.models.encoders.encoder_base import Encoder
from comet.tokenizers import HFTextEncoder
from torchnlp.utils import lengths_to_mask


class BERTEncoder(Encoder):
    """ BERT encoder.
    :param tokenizer: BERT text encoder.
    :param hparams: ArgumentParser.

    Check the available models here: 
        https://huggingface.co/transformers/pretrained_models.html
    """

    def __init__(self, tokenizer: HFTextEncoder, hparams: Namespace,) -> None:
        super().__init__(tokenizer)
        self.model = AutoModel.from_pretrained(hparams.pretrained_model)
        self.model.encoder.output_hidden_states = True
        self._output_units = self.model.config.hidden_size
        self._n_layers = self.model.config.num_hidden_layers + 1
        self._max_pos = self.model.config.max_position_embeddings

    @classmethod
    def from_pretrained(cls, hparams: Namespace) -> Encoder:
        """ Function that loads a pretrained encoder from Hugging Face.
        :param hparams: Namespace.

        Returns:
            - Encoder model
        """
        tokenizer = HFTextEncoder(model=hparams.pretrained_model)
        model = BERTEncoder(tokenizer=tokenizer, hparams=hparams)
        return model

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a batch of sequences.
        :param tokens: Torch tensor with the input sequences [batch_size x seq_len].
        :param lengths: Torch tensor with the length of each sequence [seq_len].

        Returns: 
            - 'sentemb': tensor [batch_size x 1024] with the sentence encoding.
            - 'wordemb': tensor [batch_size x seq_len x 1024] with the word level embeddings.
            - 'mask': torch.Tensor [seq_len x batch_size] 
            - 'all_layers': List with the word_embeddings returned by each layer.
            - 'extra': tuple with the last_hidden_state [batch_size x seq_len x hidden_size],
                the pooler_output representing the entire sentence and the word embeddings for 
                all BERT layers (list of tensors [batch_size x seq_len x hidden_size])
            
        """
        mask = lengths_to_mask(lengths, device=tokens.device)
        last_hidden_states, pooler_output, all_layers = self.model(tokens, mask)
        return {
            "sentemb": pooler_output,
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "mask": mask,
            "extra": (last_hidden_states, pooler_output, all_layers),
        }
