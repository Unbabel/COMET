# -*- coding: utf-8 -*-
import unittest
from argparse import Namespace
from io import StringIO

from comet.models import CometRanker
from comet.models.utils import average_pooling, max_pooling

import torch


class TestCometRanker(unittest.TestCase):

    hparams = Namespace(
        **{"encoder_model": "LASER", "pretrained_model": None, "nr_frozen_epochs": 0}
    )
    ranker = CometRanker(hparams)

    def test_read_csv(self):
        csv_file = StringIO(
            "src,ref,pos,neg\n" "ola mundo,hi world,hey world,hey world!\n"
        )
        expected = [
            {
                "src": "ola mundo",
                "ref": "hi world",
                "pos": "hey world",
                "neg": "hey world!",
            }
        ]

        data = self.ranker.read_csv(csv_file)
        self.assertListEqual(data, expected)

        # Test ignore extra columns
        csv_file = StringIO(
            "src,ref,pos,neg,id\n" "ola mundo,hi world,hey world,hey world!,10293\n"
        )
        data = self.ranker.read_csv(csv_file)
        self.assertListEqual(data, expected)

    def test_compute_metrics(self):
        dummy_outputs = [
            {
                "val_prediction": {
                    "src_sentemb": torch.tensor(
                        [[0.4963, 0.7682, 0.0885], [0.1320, 0.3074, 0.6341]]
                    ),
                    "ref_sentemb": torch.tensor(
                        [[0.4901, 0.8964, 0.4556], [0.6323, 0.3489, 0.4017]]
                    ),
                    "pos_sentemb": torch.tensor(
                        [[0.0223, 0.1689, 0.2939], [0.5185, 0.6977, 0.8000]]
                    ),
                    "neg_sentemb": torch.tensor(
                        [[0.1610, 0.2823, 0.6816], [0.9152, 0.3971, 0.8742]]
                    ),
                }
            }
        ]
        # Distance from positive embedding to source: [0.7912, 0.5738]
        # Distance from positive embedding to reference: [0.8800, 0.5415]
        # Harmonic mean: [0.8332, 0.5572]

        # Distance from negative embedding to source: [0.8369, 0.8240]
        # Distance from positive embedding to reference: [0.7325, 0.5528]
        # Harmonic mean: [0.7812, 0.6617]
        expected = {"kendall": torch.tensor(0.0)}
        metrics = self.ranker.compute_metrics(dummy_outputs)
        self.assertDictEqual(metrics, expected)

    def test_prepare_sample_to_forward(self):
        """ Test compatability between prepare_sample and forward functions. """
        sample = [
            {
                "src": "hello world",
                "ref": "ola mundo",
                "pos": "ola mundo",
                "neg": "oi mundo",
            }
        ]
        model_input, target = self.ranker.prepare_sample(sample)
        model_output = self.ranker(**model_input)
        self.assertTrue(model_output["src_sentemb"].shape[0] == 1)
        self.assertTrue(model_output["ref_sentemb"].shape[0] == 1)
        self.assertTrue(model_output["pos_sentemb"].shape[0] == 1)
        self.assertTrue(model_output["neg_sentemb"].shape[0] == 1)
        self.assertTrue(model_output["src_sentemb"].shape[1] == 1024)
        self.assertTrue(model_output["ref_sentemb"].shape[1] == 1024)
        self.assertTrue(model_output["pos_sentemb"].shape[1] == 1024)
        self.assertTrue(model_output["neg_sentemb"].shape[1] == 1024)

    def test_get_sentence_embedding(self):
        self.ranker.scalar_mix = None
        self.ranker.layer = 0

        # tokens from ["hello world", "how are your?"]
        tokens = torch.tensor([[29733, 4139, 1, 1], [2231, 137, 57374, 8]])
        lengths = torch.tensor([2, 4])

        encoder_out = self.ranker.encoder(tokens, lengths)

        # Expected sentence output with pool = 'default'
        hparams = Namespace(**{"encoder_model": "LASER", "pool": "default"})
        self.ranker.hparams = hparams
        expected = encoder_out["sentemb"]
        sentemb = self.ranker.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))

        # Expected sentence output with pool = 'max'
        hparams = Namespace(
            # LASER always used default... we need to pretend our encoder is another one
            **{"encoder_model": "other", "pool": "max"}
        )
        self.ranker.hparams = hparams
        # Max pooling is tested in test_utils.py
        expected = max_pooling(tokens, encoder_out["wordemb"], 1)
        sentemb = self.ranker.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))

        # Expected sentence output with pool = 'avg'
        hparams = Namespace(
            # LASER always used default... we need to pretend our encoder is another one
            **{"encoder_model": "other", "pool": "avg"}
        )
        self.ranker.hparams = hparams
        # AVG pooling is tested in test_utils.py
        expected = average_pooling(
            tokens, encoder_out["wordemb"], encoder_out["mask"], 1
        )
        sentemb = self.ranker.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))

        # Expected sentence output with pool = 'cls'
        hparams = Namespace(
            # LASER always used default... we need to pretend our encoder is another one
            **{"encoder_model": "other", "pool": "cls"}
        )
        self.ranker.hparams = hparams
        expected = encoder_out["wordemb"][:, 0, :]
        sentemb = self.ranker.get_sentence_embedding(tokens, lengths)
        self.assertTrue(torch.equal(sentemb, expected))
