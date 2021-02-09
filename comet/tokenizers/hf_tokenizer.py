# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer

from torchnlp.encoders.text.text_encoder import TextEncoder

from .tokenizer_base import TextEncoderBase


class HFTextEncoder(TextEncoderBase):
    """
    Wrapper arround transformers AutoTokenizer:
        - https://huggingface.co/transformers/model_doc/auto.html#autotokenizer

    :param model: model to be used.
    """

    def __init__(self, model: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Properties from the base class
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}
        self._bos_index = self.tokenizer.cls_token_id
        self._pad_index = self.tokenizer.pad_token_id
        self._eos_index = self.tokenizer.sep_token_id
        self._unk_index = self.tokenizer.unk_token_id
        self._mask_index = self.tokenizer.mask_token_id

    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = TextEncoder.encode(self, sequence)
        return torch.tensor(self.tokenizer(sequence, truncation=False)["input_ids"])
