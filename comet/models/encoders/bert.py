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
    """BERT encoder.
    :param tokenizer: BERT text encoder.
    :param hparams: ArgumentParser.
    """

    def __init__(
        self,
        tokenizer: HFTextEncoder,
        hparams: Namespace,
    ) -> None:
        super().__init__(tokenizer)
        self.model = AutoModel.from_pretrained(hparams.pretrained_model)
        self.model.encoder.output_hidden_states = True
        self._output_units = self.model.config.hidden_size
        self._n_layers = self.model.config.num_hidden_layers + 1
        self._max_pos = self.model.config.max_position_embeddings

    @classmethod
    def from_pretrained(cls, hparams: Namespace) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param hparams: Namespace.

        :return: Encoder model
        """
        tokenizer = HFTextEncoder(model=hparams.pretrained_model)
        model = BERTEncoder(tokenizer=tokenizer, hparams=hparams)
        return model

    def freeze_embeddings(self) -> None:
        """ Frezees the embedding layer of the network to save some memory while training. """
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def layerwise_lr(self, lr: float, decay: float):
        """
        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        # Embedding Layer
        opt_parameters = [
            {
                "params": self.model.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        # All layers
        opt_parameters += [
            {
                "params": self.model.encoder.layer[l].parameters(),
                "lr": lr * decay ** l,
            }
            for l in range(self.num_layers - 2, 0, -1)
        ]
        return opt_parameters

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a batch of sequences.

        :param tokens: Torch tensor with the input sequences [batch_size x seq_len].
        :param lengths: Torch tensor with the lenght of each sequence [seq_len].

        :return: Dictionary with `sentemb` (tensor with dims [batch_size x output_units]), `wordemb`
            (tensor with dims [batch_size x seq_len x output_units]), `mask` (input mask),
            `all_layers` (List with word_embeddings from all layers), `extra` (tuple with the
            last_hidden_state, the pooler_output representing  the entire sentence and the word
            embeddings for all BERT layers).
        """
        mask = lengths_to_mask(lengths, device=tokens.device)
        last_hidden_states, pooler_output, all_layers = self.model(
            tokens, mask, output_hidden_states=True, return_dict=False
        )
        return {
            "sentemb": pooler_output,
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "mask": mask,
            "extra": (last_hidden_states, pooler_output, all_layers),
        }
