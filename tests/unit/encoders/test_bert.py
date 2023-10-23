# -*- coding: utf-8 -*-
import unittest

from comet.encoders.bert import BERTEncoder


class TestBERTEncoder(unittest.TestCase):
    bert = BERTEncoder.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

    def test_num_layers(self):
        self.assertEqual(self.bert.num_layers, 3)

    def test_output_units(self):
        self.assertEqual(self.bert.output_units, 128)

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
        translations = ["Bem vindos ao COMET", "Isto é um exemplo!"]
        source = ["Welcome to COMET!", "This is an example!"]
        annotations = [
            [{"start": 14, "end": 19, "text": "COMET", "severity": "major"}],
            [],
        ]
        translations_input = self.bert.prepare_sample(
            translations, word_level=True, annotations=annotations
        )
        source_input = self.bert.prepare_sample(source)
        expected_tokens = [
            [
                101,
                2022,
                2213,
                19354,
                12269,
                20118,
                15699,
                102,
                6160,
                2000,
                15699,
                999,
                102,
                0,
                0,
                0,
                0,
            ],
            [
                101,
                21541,
                2080,
                1041,
                8529,
                4654,
                6633,
                24759,
                2080,
                999,
                102,
                2023,
                2003,
                2019,
                2742,
                999,
                102,
            ],
        ]
        expected_labels = [
            [0, 0, 0, 0, 0, 0, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
        ]
        expected_token_type_ids = [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
        seq_size = [13, 17]
        continuous_input = self.bert.concat_sequences(
            [translations_input, source_input], return_label_ids=True
        )
        self.assertListEqual(continuous_input[0]["input_ids"].tolist(), expected_tokens)
        self.assertListEqual(
            continuous_input[0]["token_type_ids"].tolist(), expected_token_type_ids
        )
        self.assertListEqual(continuous_input[1].tolist(), seq_size)
        self.assertListEqual(continuous_input[0]["label_ids"].tolist(), expected_labels)
