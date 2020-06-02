# -*- coding: utf-8 -*-
import unittest

import torch

from comet.models.encoders import LASEREncoder


class TestLASEREncoder(unittest.TestCase):
    model = LASEREncoder.from_pretrained(None)
    bpe_tokenizer = model.tokenizer

    def test_num_layers(self):
        assert self.model.num_layers == 1

    def test_output_units(self):
        assert self.model.output_units == 1024

    def test_prepare_sample(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]
        model_input = self.model.prepare_sample(sample)
        assert "tokens" in model_input
        assert "lengths" in model_input

        expected = self.bpe_tokenizer.encode(sample[0])
        assert torch.equal(expected, model_input["tokens"][0])
        assert expected.size()[0] == model_input["lengths"][0]

    def test_forward(self):
        sample = ["hello world!", "This is a batch"]
        model_input = self.model.prepare_sample(sample)
        model_out = self.model(**model_input)

        assert "wordemb" in model_out
        assert "sentemb" in model_out
        assert "all_layers" in model_out
        assert "mask" in model_out
        assert "extra" in model_out

        assert len(model_out["all_layers"]) == self.model.num_layers
        assert self.model.output_units == model_out["sentemb"].size()[1]
        assert self.model.output_units == model_out["wordemb"].size()[2]
