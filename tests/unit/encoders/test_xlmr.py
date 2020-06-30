# -*- coding: utf-8 -*-
import unittest
from argparse import Namespace

import torch

from comet.models.encoders import XLMREncoder


class TestXLMREncoder(unittest.TestCase):
    """ 
    There is not much we can test in our encoders... 
    
    The only thing we are interested is in maintaining a well defined interface 
    between all encoders.
    """

    hparams = Namespace(pretrained_model="xlmr.base")
    model_base = XLMREncoder.from_pretrained(hparams)

    def test_num_layers(self):
        self.assertEqual(self.model_base.num_layers, 13)

    def test_output_units(self):
        self.assertEqual(self.model_base.output_units, 768)

    def test_max_positions(self):
        self.assertEqual(self.model_base.max_positions, 512)

    def test_prepare_sample(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]

        model_input = self.model_base.prepare_sample(sample)
        self.assertIn("tokens", model_input)
        self.assertIn("lengths", model_input)

        # Sanity Check: This is already checked when testing the tokenizer.
        expected = self.model_base.model.encode(sample[0])
        self.assertTrue(torch.equal(expected, model_input["tokens"][0]))
        self.assertEqual(len(expected), model_input["lengths"][0])

    def test_forward(self):
        sample = ["hello world!", "This is a batch"]
        model_input = self.model_base.prepare_sample(sample)
        model_out = self.model_base(**model_input)
        self.assertIn("wordemb", model_out)
        self.assertIn("sentemb", model_out)
        self.assertIn("all_layers", model_out)
        self.assertIn("mask", model_out)
        self.assertIn("extra", model_out)
        self.assertEqual(len(model_out["all_layers"]), self.model_base.num_layers)
        self.assertEqual(self.model_base.output_units, model_out["sentemb"].size()[1])
        self.assertEqual(self.model_base.output_units, model_out["wordemb"].size()[2])
