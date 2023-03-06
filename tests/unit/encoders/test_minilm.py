# -*- coding: utf-8 -*-
import unittest

from comet.encoders.minilm import MiniLMEncoder


class TestMiniLMEncoder(unittest.TestCase):
    """MiniLMV2 uses XLM-R tokenizer thus, most tests are copy of minilmEncoder"""

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

    def test_concat_sequences(self):
        """Basic testcase to check if we can joint two sequences into a contiguous input"""
        source = ["Bem vindos ao <v>COMET</v>", "Isto é um exemplo!"]
        translations = ["Welcome to COMET!", "This is an example!"]
        self.minilm.add_span_tokens("<v>", "</v>")
        source_input = self.minilm.prepare_sample(source, word_level_training=True)
        translations_input = self.minilm.prepare_sample(translations)
        expected_tokens = [
            [
                0,
                97043,
                8530,
                232,
                940,
                6,
                179951,
                618,
                2,
                2,
                91334,
                47,
                179951,
                618,
                38,
                2,
            ],
            [0, 40088, 393, 286, 15946, 38, 2, 2, 3293, 83, 142, 27781, 38, 2, 1, 1],
        ]
        expected_in_span_mask = [
            [0, 0, 0, 0, 0, 0, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ]
        seq_size = [16, 14]
        continuous_input = self.minilm.concat_sequences(
            [source_input, translations_input], return_in_span_mask=True
        )
        self.assertListEqual(continuous_input[0]["input_ids"].tolist(), expected_tokens)
        self.assertListEqual(continuous_input[1].tolist(), seq_size)
        self.assertListEqual(
            continuous_input[0]["in_span_mask"].tolist(), expected_in_span_mask
        )