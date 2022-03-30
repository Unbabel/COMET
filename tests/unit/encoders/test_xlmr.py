# -*- coding: utf-8 -*-
import unittest

from comet.encoders.xlmr import XLMREncoder


class TestXLMREncoder(unittest.TestCase):

    xlmr = XLMREncoder.from_pretrained("Unbabel/xlm-roberta-comet-small")

    def test_num_layers(self):
        self.assertEqual(self.xlmr.num_layers, 7)

    def test_output_units(self):
        self.assertEqual(self.xlmr.output_units, 384)

    def test_max_positions(self):
        self.assertEqual(self.xlmr.max_positions, 514)

    def test_prepare_sample(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.xlmr.prepare_sample(sample)
        self.assertIn("input_ids", model_input)
        self.assertIn("attention_mask", model_input)

    def test_forward(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.xlmr.prepare_sample(sample)
        model_output = self.xlmr(**model_input)
        self.assertIn("wordemb", model_output)
        self.assertIn("sentemb", model_output)
        self.assertIn("all_layers", model_output)
        self.assertIn("attention_mask", model_output)
