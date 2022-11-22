# -*- coding: utf-8 -*-
import unittest

from comet.encoders.minilm import MiniLMEncoder


class TestMiniLMEncoder(unittest.TestCase):
    """MiniLMV2 uses XLM-R tokenizer thus, most tests are copy of XLMREncoder
    """
    minilm = MiniLMEncoder.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

    def test_num_layers(self):
        self.assertEqual(self.minilm.num_layers, 13)

    def test_output_units(self):
        self.assertEqual(self.minilm.output_units, 384)

    def test_max_positions(self):
        self.assertEqual(self.minilm.max_positions, 510)

    def test_prepare_sample(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.minilm.prepare_sample(sample)
        self.assertIn("input_ids", model_input)
        self.assertIn("attention_mask", model_input)

    def test_forward(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.minilm.prepare_sample(sample)
        model_output = self.minilm(**model_input)
        self.assertIn("wordemb", model_output)
        self.assertIn("sentemb", model_output)
        self.assertIn("all_layers", model_output)
        self.assertIn("attention_mask", model_output)

    def test_subword_tokenize(self):
        samples = ['the olympiacos revived <EOS>', "This is COMET ! <EOS>"]
        subwords, subword_mask, subword_lengths = self.minilm.subword_tokenize(samples, pad=True)
        expected_tokenization = [
            ['<s>', '▁the', '▁olymp', 'ia', 'cos', '▁revi', 'ved', '▁<', 'E', 'OS', '>', '</s>'],
            ['<s>', '▁This', '▁is', '▁COME', 'T', '▁!', '▁<', 'E', 'OS', '>', '</s>', '<pad>']
        ]
        expected_mask = [
            [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
        ]
        expected_lengths = [12, 11]
        self.assertListEqual(subwords, expected_tokenization)
        self.assertListEqual(subword_mask.int().tolist(), expected_mask)
        self.assertListEqual(subword_lengths, expected_lengths)

    def test_mask_start_tokens(self):
        tokenized_sentence = [
            ['<s>', '▁After', '▁the', '▁Greek', '▁war', '▁of', '▁in', 'dependence', '▁from', '▁the', '▁o', 'ttoman', '</s>'],
            ['<s>', '▁she', '▁was', '▁given', '▁the', '▁uk', '▁s', 'qui', 'r', 'rel', '</s>'],
        ]
        expected_mask = [
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        ]
        seq_len = [13, 11]
        mask = self.minilm.mask_start_tokens(tokenized_sentence, seq_len)
        self.assertListEqual(mask.int().tolist(), expected_mask)
    
    def test_concat_sequences(self):
        """Basic testcase to check if we can joint two sequences into a contiguous input"""
        source = ["Bem vindos ao COMET", "Isto é um exemplo!"]
        translations = ["Welcome to COMET!", "This is an example!"]
        source_input = self.minilm.prepare_sample(source)
        translations_input = self.minilm.prepare_sample(translations)
        expected_tokens = [
            [     0,  97043,   8530,    232,    940, 179951,    618,      2,      2,  91334,     47, 179951,    618,     38,      2],
            [     0,  40088,    393,    286,  15946,     38,      2,      2,   3293,     83,    142,  27781,     38,      2,      1],
        ]
        seq_size = [15, 14]
        continuous_input = self.minilm.concat_sequences([source_input, translations_input])
        self.assertListEqual(continuous_input[0]["input_ids"].tolist(), expected_tokens)
        self.assertListEqual(continuous_input[1].tolist(), seq_size)
