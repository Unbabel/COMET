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
        self.assertEqual(self.bert.max_positions, 512)

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
