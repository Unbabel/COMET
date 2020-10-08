# -*- coding: utf-8 -*-
import unittest
from argparse import Namespace
from io import StringIO

import numpy as np
import torch
from comet.models import CometEstimator
from comet.models.utils import average_pooling, max_pooling


class TestCometEstimator(unittest.TestCase):

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
    estimator = CometEstimator(hparams)

    def test_read_csv(self):
        csv_file = StringIO("src,ref,mt,score\n" "ola mundo,hi world,hey world!,0.8\n")
        expected = [
            {"src": "ola mundo", "ref": "hi world", "mt": "hey world!", "score": 0.8}
        ]

        data = self.estimator.read_csv(csv_file)
        self.assertListEqual(data, expected)

        # Test ignore extra columns
        csv_file = StringIO(
            "src,ref,mt,score,id\n" "ola mundo,hi world,hey world!,0.8,10299\n"
        )
        data = self.estimator.read_csv(csv_file)
        self.assertListEqual(data, expected)

    def test_compute_metrics(self):
        dummy_outputs = [
            {
                "val_prediction": {
                    "score": torch.tensor(np.array([0, 0, 0, 1, 1, 1, 1])),
                },
                "val_target": {
                    "score": torch.tensor(np.arange(7)),
                },
            }
        ]
        expected = {
            "pearson": torch.tensor(0.8660254, dtype=torch.float32),
            "kendall": torch.tensor(0.7559289, dtype=torch.float32),
            "spearman": torch.tensor(0.866025, dtype=torch.float32),
        }
        metrics = self.estimator.compute_metrics(dummy_outputs)
        self.assertDictEqual(
            {k: round(v.item(), 4) for k, v in metrics.items()},
            {k: round(v.item(), 4) for k, v in expected.items()},
        )

    def test_prepare_sample_to_forward(self):
        """ Test compatability between prepare_sample and forward functions. """
        sample = [
            {"src": "ola mundo", "ref": "hi world", "mt": "hey world!", "score": 0.8},
            {"src": "ola mundo", "ref": "hi world", "mt": "hey world!", "score": 0.8},
        ]

        model_input, target = self.estimator.prepare_sample(sample)
        model_output = self.estimator(**model_input)
        self.assertTrue(model_output["score"].shape[0] == 2)
        self.assertTrue(model_output["score"].shape[1] == 1)

    def test_get_sentence_embedding(self):
        self.estimator.scalar_mix = None
        self.estimator.layer = 0

        # tokens from ["hello world", "how are your?"]
        tokens = torch.tensor([[29733, 4139, 1, 1], [2231, 137, 57374, 8]])
        lengths = torch.tensor([2, 4])
        encoder_out = self.estimator.encoder(tokens, lengths)

        # Expected sentence output with pool = 'max'
        hparams = Namespace(
            # LASER always used default... we need to pretend our encoder is another one
            **{"encoder_model": "other", "pool": "max"}
        )
        self.estimator.hparams = hparams
        # Max pooling is tested in test_utils.py
        expected = max_pooling(tokens, encoder_out["wordemb"], 1)
        sentemb = self.estimator.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))

        # Expected sentence output with pool = 'avg'
        hparams = Namespace(
            # LASER always used default... we need to pretend our encoder is another one
            **{"encoder_model": "other", "pool": "avg"}
        )
        self.estimator.hparams = hparams
        # AVG pooling is tested in test_utils.py
        expected = average_pooling(
            tokens, encoder_out["wordemb"], encoder_out["mask"], 1
        )
        sentemb = self.estimator.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))

        # Expected sentence output with pool = 'cls'
        hparams = Namespace(
            # LASER always used default... we need to pretend our encoder is another one
            **{"encoder_model": "other", "pool": "cls"}
        )
        self.estimator.hparams = hparams
        expected = encoder_out["wordemb"][:, 0, :]
        sentemb = self.estimator.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))

        # Expected sentence output with pool = 'cls+avg'
        hparams = Namespace(
            # LASER always used default... we need to pretend our encoder is another one
            **{"encoder_model": "other", "pool": "cls+avg"}
        )
        self.estimator.hparams = hparams
        cls_embedding = encoder_out["wordemb"][:, 0, :]
        avg_embedding = average_pooling(
            tokens, encoder_out["wordemb"], encoder_out["mask"], 1
        )
        expected = torch.cat((cls_embedding, avg_embedding), dim=1)
        sentemb = self.estimator.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))

        # Expected sentence output with pool = 'default'
        hparams = Namespace(**{"encoder_model": "LASER", "pool": "default"})
        self.estimator.hparams = hparams
        expected = encoder_out["sentemb"]
        sentemb = self.estimator.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))
