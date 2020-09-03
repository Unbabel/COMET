# -*- coding: utf-8 -*-
from typing import Callable, Dict

import torch

from torchnlp.encoders.text.text_encoder import TextEncoder

from .tokenizer_base import TextEncoderBase


class XLMRTextEncoder(TextEncoderBase):
    """
    XLM-RoBERTa encoder from Fairseq.

    :param tokenizer_func: XLM tokenization function.
        This can be easily obtain from the fairseq model (e.g: xlmr.encode callable)
    :param vocabulary: the dictionary containing the XLM-R vocabulary.
        This can be easily obtain from the fairseq model
        (e.g: xlmr.task.source_dictionary.__dict__['indices'])
    """

    def __init__(self, encode_func: Callable, vocabulary: Dict[str, int]) -> None:
        super().__init__()

        self.encode_func = encode_func
        # Properties from the base class
        self.stoi = vocabulary
        self.itos = {v: k for k, v in vocabulary.items()}
        self._pad_index = self.stoi["<pad>"]
        self._eos_index = self.stoi["</s>"]
        self._unk_index = self.stoi["<unk>"]
        self._bos_index = self.stoi["<s>"]
        self._mask_index = self.stoi["<mask>"]

    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = TextEncoder.encode(self, sequence)
        return self.encode_func(sequence)
