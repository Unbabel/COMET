# -*- coding: utf-8 -*-
import os
from typing import Dict, List

import fastBPE
from torchnlp.download import download_file_maybe_extract

from .tokenizer_base import TextEncoderBase

L93_CODES_URL = "https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes"
L93_VOCAB_URL = "https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab"

L93_CODES_FILE = "93langs.fcodes"
L93_VOCAB_FILE = "93langs.fvocab"

if "HOME" in os.environ:
    saving_directory = os.environ["HOME"] + "/.cache/torch/unbabel_comet/"
else:
    raise Exception("HOME environment variable is not defined.")


class FastBPEEncoder(TextEncoderBase):
    """
    FastBPE LASER tokenizer.

    :param dictionary: LASER vocabulary
    """

    def __init__(self, dictionary: Dict[str, int]) -> None:
        super().__init__()

        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)

        download_file_maybe_extract(
            L93_CODES_URL, directory=saving_directory, check_files=[L93_CODES_FILE]
        )

        download_file_maybe_extract(
            L93_VOCAB_URL, directory=saving_directory, check_files=[L93_VOCAB_FILE]
        )

        self.bpe = fastBPE.fastBPE(
            saving_directory + L93_CODES_FILE, saving_directory + L93_VOCAB_FILE
        )
        self.bpe_symbol = "@@ "

        # Properties from the base class
        self.stoi = dictionary
        self.itos = [key for key in dictionary.keys()]
        self._pad_index = dictionary["<pad>"]
        self._eos_index = dictionary["</s>"]
        self._unk_index = dictionary["<unk>"]
        self._mask_index = None

    def tokenize(self, sequence: str) -> List[str]:
        return self.bpe.apply([sequence.lower()])[0].split()
