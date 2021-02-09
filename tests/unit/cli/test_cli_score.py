# -*- coding: utf-8 -*-
import json
import os
import shutil
import unittest

from click.testing import CliRunner
from comet.cli import score
from comet.models import download_model
from tests.data import DATA_PATH


class TestScoreCli(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(DATA_PATH + "emnlp-base-da-ranker")
        shutil.rmtree(DATA_PATH + "wmt-large-da-estimator-1719")
        os.remove(DATA_PATH + "results.json")

    def setUp(self):
        self.runner = CliRunner()

    def test_score_ranker_cpu(self):
        download_model("emnlp-base-da-ranker", DATA_PATH)
        args = [
            "--model",
            DATA_PATH + "emnlp-base-da-ranker/_ckpt_epoch_0.ckpt",
            "-s",
            DATA_PATH + "src.en",
            "-h",
            DATA_PATH + "mt.de",
            "-r",
            DATA_PATH + "ref.de",
            "--cpu",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("COMET system score: ", result.stdout)

    def test_score_estimator_cpu(self):
        download_model("wmt-large-da-estimator-1719", DATA_PATH)
        args = [
            "--model",
            DATA_PATH + "wmt-large-da-estimator-1719/_ckpt_epoch_1.ckpt",
            "-s",
            DATA_PATH + "src.en",
            "-h",
            DATA_PATH + "mt.de",
            "-r",
            DATA_PATH + "ref.de",
            "--to_json",
            DATA_PATH + "results.json",
            "--cpu",
            "--batch_size",
            "32",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Predictions saved in: {}".format(DATA_PATH + "results.json"), result.stdout
        )
        self.assertTrue(os.path.exists(DATA_PATH + "results.json"))
        with open(DATA_PATH + "results.json") as json_file:
            samples = json.load(json_file)
            for s in samples:
                self.assertIn("predicted_score", s)
        self.assertIn("COMET system score: ", result.stdout)
