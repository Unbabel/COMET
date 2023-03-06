# -*- coding: utf-8 -*-
import unittest

from comet.encoders.rembert import RemBERTEncoder


class TestRemBERTEncoder(unittest.TestCase):

    bert = RemBERTEncoder.from_pretrained("google/rembert")

    def test_num_layers(self):
        self.assertEqual(self.bert.num_layers, 33)

    def test_output_units(self):
        self.assertEqual(self.bert.output_units, 1152)

    def test_max_positions(self):
        self.assertEqual(self.bert.max_positions, 510)

    def test_prepare_sample(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.bert.prepare_sample(sample)
        self.assertIn("input_ids", model_input)
        self.assertIn("attention_mask", model_input)

    def test_forward(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.bert.prepare_sample(sample)
        model_output = self.bert(**model_input)
        self.assertIn("wordemb", model_output)
        self.assertIn("sentemb", model_output)
        self.assertIn("all_layers", model_output)
        self.assertIn("attention_mask", model_output)

    def test_concat_sequences(self):
        """Basic testcase to check if we can joint two sequences into a contiguous input"""
        translations = ["Bem vindos ao <v>COMET</v>", "Isto é um exemplo!"]
        source = ["Welcome to COMET!", "This is an example!"]
        self.bert.add_span_tokens("<v>", "</v>")
        translations_input = self.bert.prepare_sample(
            translations, word_level_training=True
        )
        source_input = self.bert.prepare_sample(source)
        expected_tokens = [
            [
                312,
                58230,
                16833,
                759,
                1425,
                573,
                148577,
                862,
                313,
                26548,
                596,
                148577,
                862,
                646,
                313,
            ],
            [312, 58378, 921, 835, 17293, 646, 313, 1357, 619, 666, 7469, 646, 313, 0, 0],
        ]
        expected_in_span_mask = [
            [0, 0, 0, 0, 0, 0, 1,  1,  0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        ]
        expected_token_type_ids = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        seq_size = [15, 13]
        continuous_input = self.bert.concat_sequences(
            [translations_input, source_input], return_in_span_mask=True
        )
        self.assertListEqual(continuous_input[0]["input_ids"].tolist(), expected_tokens)
        self.assertListEqual(
            continuous_input[0]["token_type_ids"].tolist(), expected_token_type_ids
        )
        self.assertListEqual(continuous_input[1].tolist(), seq_size)
        self.assertListEqual(
            continuous_input[0]["in_span_mask"].tolist(), expected_in_span_mask
        )
