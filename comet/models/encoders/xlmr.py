# -*- coding: utf-8 -*-
r"""
Fairseq XLM-R Encoder Model
==============
    Pretrained XLM-RoBERTa from Fairseq framework.
    https://github.com/pytorch/fairseq/tree/master/examples/xlmr
"""
import os
from argparse import Namespace
from typing import Dict

import torch

from comet.models.encoders.encoder_base import Encoder
from comet.tokenizers import XLMRTextEncoder
from fairseq.models.roberta import XLMRModel
from torchnlp.download import download_file_maybe_extract
from torchnlp.utils import lengths_to_mask

XLMR_LARGE_URL = "https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz"
XLMR_LARGE_MODEL_NAME = "xlmr.large/model.pt"

XLMR_BASE_URL = "https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz"
XLMR_BASE_MODEL_NAME = "xlmr.base/model.pt"

XLMR_LARGE_V0_URL = "https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.v0.tar.gz"
XLMR_LARGE_V0_MODEL_NAME = "xlmr.large.v0/model.pt"

XLMR_BASE_V0_URL = "https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.v0.tar.gz"
XLMR_BASE_V0_MODEL_NAME = "xlmr.base.v0/model.pt"

if "HOME" in os.environ:
    saving_directory = os.environ["HOME"] + "/.cache/torch/unbabel_comet/"
else:
    raise Exception("HOME environment variable is not defined.")


class XLMREncoder(Encoder):
    """
    XLM-RoBERTa encoder from Fairseq.
    
    :param xlmr: XLM-R model to be used.
    :param tokenizer: XLM-R model tokenizer to be used.
    :param hparams: Namespace.
    """

    def __init__(
        self, xlmr: XLMRModel, tokenizer: XLMRTextEncoder, hparams: Namespace
    ) -> None:
        super().__init__(tokenizer)
        self._output_units = 768 if "base" in hparams.pretrained_model else 1024
        self._n_layers = 13 if "base" in hparams.pretrained_model else 25
        self._max_pos = 512
        self.model = xlmr

    @classmethod
    def from_pretrained(cls, hparams: Namespace):
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)

        pretrained_model = hparams.pretrained_model
        if pretrained_model == "xlmr.base":
            download_file_maybe_extract(
                XLMR_BASE_URL,
                directory=saving_directory,
                check_files=[XLMR_BASE_MODEL_NAME],
            )

        elif pretrained_model == "xlmr.large":
            download_file_maybe_extract(
                XLMR_LARGE_URL,
                directory=saving_directory,
                check_files=[XLMR_LARGE_MODEL_NAME],
            )
        elif pretrained_model == "xlmr.base.v0":
            download_file_maybe_extract(
                XLMR_BASE_V0_URL,
                directory=saving_directory,
                check_files=[XLMR_BASE_V0_MODEL_NAME],
            )

        elif pretrained_model == "xlmr.large.v0":
            download_file_maybe_extract(
                XLMR_LARGE_V0_URL,
                directory=saving_directory,
                check_files=[XLMR_LARGE_V0_MODEL_NAME],
            )
        else:
            raise Exception(f"{pretrained_model} is an invalid XLM-R model.")

        xlmr = XLMRModel.from_pretrained(
            saving_directory + pretrained_model, checkpoint_file="model.pt"
        )
        xlmr.eval()
        tokenizer = XLMRTextEncoder(
            xlmr.encode, xlmr.task.source_dictionary.__dict__["indices"]
        )
        return XLMREncoder(xlmr=xlmr, tokenizer=tokenizer, hparams=hparams)

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
            - 'all_layers': List with the word_embeddings returned by each layer.
            - 'mask': torch.Tensor [seq_len x batch_size] 
            - 'extra': tuple with all XLM-R layers (list of tensors [batch_size x seq_len x hidden_size]) 
        """
        mask = lengths_to_mask(lengths, device=tokens.device)
        all_layers = self.model.extract_features(tokens, return_all_hiddens=True)
        return {
            "sentemb": all_layers[-1][:, 0, :],
            "wordemb": all_layers[-1],
            "all_layers": all_layers,
            "mask": mask,
            "extra": (all_layers),
        }
