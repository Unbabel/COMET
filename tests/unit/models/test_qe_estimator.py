# -*- coding: utf-8 -*-
import unittest
from argparse import Namespace
from io import StringIO

from comet.models import QualityEstimator


class TestQualityEstimator(unittest.TestCase):

    hparams = Namespace(
        **{
            "encoder_model": "LASER",
            "pretrained_model": None,
            "nr_frozen_epochs": 0,
            "loss": "mse",
            # FeedForward Definition
            "pool": "default",
            "hidden_sizes": "1024",
            "activations": "Tanh",
            "dropout": 0.1,
            "final_activation": False,
        }
    )
    estimator = QualityEstimator(hparams)

    def test_read_csv(self):
        csv_file = StringIO("src,mt,score\n" "ola mundo,hey world!,0.8\n")
        expected = [{"src": "ola mundo", "mt": "hey world!", "score": 0.8}]

        data = self.estimator.read_csv(csv_file)
        self.assertListEqual(data, expected)

        # Test ignore extra columns
        csv_file = StringIO("src,mt,score,id\n" "ola mundo,hey world!,0.8,10299\n")
        data = self.estimator.read_csv(csv_file)
        self.assertListEqual(data, expected)

    def test_prepare_sample_to_forward(self):
        """ Test compatability between prepare_sample and forward functions. """
        sample = [
            {"src": "ola mundo", "mt": "hey world!", "score": 0.8},
            {"src": "ola mundo", "mt": "hey world!", "score": 0.8},
        ]

        model_input, target = self.estimator.prepare_sample(sample)
        model_output = self.estimator(**model_input)
        self.assertTrue(model_output["score"].shape[0] == 2)
        self.assertTrue(model_output["score"].shape[1] == 1)
