# -*- coding: utf-8 -*-
import os
import unittest

import torch

from comet.tokenizers import XLMRTextEncoder
from fairseq.models.roberta import XLMRModel
from torchnlp.download import download_file_maybe_extract


class TestXLMRTextEncoder(unittest.TestCase):
    download_file_maybe_extract(
        "https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz",
        directory=os.environ["HOME"] + "/.cache/torch/unbabel_comet/",
        check_files=["xlmr.base/model.pt"],
    )
    xlmr = XLMRModel.from_pretrained(
        os.environ["HOME"] + "/.cache/torch/unbabel_comet/xlmr.base",
        checkpoint_file="model.pt",
    )
    original_vocab = xlmr.task.source_dictionary.__dict__["indices"]
    tokenizer = XLMRTextEncoder(xlmr.encode, original_vocab)

    def test_unk_property(self):
        self.assertEqual(self.tokenizer.unk_index, self.original_vocab["<unk>"])

    def test_pad_property(self):
        self.assertEqual(self.tokenizer.padding_index, self.original_vocab["<pad>"])

    def test_bos_property(self):
        self.assertEqual(self.tokenizer.bos_index, self.original_vocab["<s>"])

    def test_eos_property(self):
        self.assertEqual(self.tokenizer.eos_index, self.original_vocab["</s>"])

    def test_mask_property(self):
        self.assertEqual(self.tokenizer.mask_index, self.original_vocab["<mask>"])

    def test_vocab_property(self):
        self.assertEqual(self.tokenizer.vocab, self.original_vocab)

    def test_vocab_size_property(self):
        self.assertEqual(self.tokenizer.vocab_size, len(self.original_vocab))

    def test_encode(self):
        sentence = "Hello, my dog is cute"
        expected = self.xlmr.encode(sentence)
        result = self.tokenizer.encode(sentence)
        self.assertTrue(torch.equal(expected, result))
        # Make sure the bos and eos tokens were added.
        self.assertEqual(result[0], self.tokenizer.bos_index)
        self.assertEqual(result[-1], self.tokenizer.eos_index)

    def test_batch_encode(self):
        # Test batch_encode.
        batch = ["Hello, my dog is cute", "hello world!"]
        encoded_batch, lengths = self.tokenizer.batch_encode(batch)

        self.assertTrue(torch.equal(encoded_batch[0], self.tokenizer.encode(batch[0])))
        self.assertTrue(
            torch.equal(encoded_batch[1][: lengths[1]], self.tokenizer.encode(batch[1]))
        )
        self.assertEqual(lengths[0], len(self.xlmr.encode("Hello, my dog is cute")))
        self.assertEqual(lengths[1], len(self.xlmr.encode("hello world!")))

        # Check if last sentence is padded.
        self.assertEqual(encoded_batch[1][-1], self.tokenizer.padding_index)
        self.assertEqual(encoded_batch[1][-2], self.tokenizer.padding_index)
