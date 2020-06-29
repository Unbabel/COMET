# -*- coding: utf-8 -*-
import unittest

import torch
from transformers import AutoTokenizer

from comet.tokenizers import HFTextEncoder


class TestHFTextEncoder(unittest.TestCase):
    tokenizer = HFTextEncoder("bert-base-multilingual-cased")
    original_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def test_unk_property(self):
        self.assertEqual(self.tokenizer.unk_index, 100)

    def test_pad_property(self):
        self.assertEqual(self.tokenizer.padding_index, 0)

    def test_bos_property(self):
        self.assertEqual(self.tokenizer.bos_index, 101)

    def test_eos_property(self):
        self.assertEqual(self.tokenizer.eos_index, 102)

    def test_mask_property(self):
        self.assertEqual(self.tokenizer.mask_index, 103)

    def test_vocab_property(self):
        self.assertDictEqual(self.tokenizer.vocab, self.original_tokenizer.vocab)

    def test_vocab_size_property(self):
        self.assertEqual(self.tokenizer.vocab_size, 119_547)

    def test_encode(self):
        sentence = "Hello, my dog is cute"
        expected = self.original_tokenizer.encode(sentence)
        result = self.tokenizer.encode(sentence)
        self.assertTrue(torch.equal(torch.tensor(expected), result))
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
        self.assertEqual(
            lengths[0], len(self.original_tokenizer.encode("Hello, my dog is cute"))
        )
        self.assertEqual(
            lengths[1], len(self.original_tokenizer.encode("hello world!"))
        )

        # Check if last sentence is padded.
        self.assertEqual(encoded_batch[1][-1], self.tokenizer.padding_index)
        self.assertEqual(encoded_batch[1][-2], self.tokenizer.padding_index)
