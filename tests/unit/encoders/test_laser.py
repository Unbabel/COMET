# -*- coding: utf-8 -*-
import unittest

import torch

from comet.models.encoders import LASEREncoder


class TestLASEREncoder(unittest.TestCase):
    """
    There is not much we can test in our encoders...

    The only thing we are interested is in maintaining a well defined interface
    between all encoders.
    """

    model = LASEREncoder.from_pretrained(None)
    bpe_tokenizer = model.tokenizer

    def test_num_layers(self):
        self.assertEqual(self.model.num_layers, 1)

    def test_output_units(self):
        self.assertEqual(self.model.output_units, 1024)

    def test_prepare_sample(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.model.prepare_sample(sample)
        self.assertIn("tokens", model_input)
        self.assertIn("lengths", model_input)

        expected = self.bpe_tokenizer.encode(sample[0])
        self.assertTrue(torch.equal(expected, model_input["tokens"][0]))
        self.assertEqual(expected.size()[0], model_input["lengths"][0])

    def test_forward(self):
        sample = ["hello world!", "This is a batch"]
        model_input = self.model.prepare_sample(sample)
        model_out = self.model(**model_input)

        self.assertIn("wordemb", model_out)
        self.assertIn("sentemb", model_out)
        self.assertIn("all_layers", model_out)
        self.assertIn("mask", model_out)
        self.assertIn("extra", model_out)

        self.assertEqual(len(model_out["all_layers"]), self.model.num_layers)
        self.assertEqual(self.model.output_units, model_out["sentemb"].size()[1])
        self.assertEqual(self.model.output_units, model_out["wordemb"].size()[2])
