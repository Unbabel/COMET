# -*- coding: utf-8 -*-
import unittest

from click.testing import CliRunner
from tests.data import DATA_PATH

from comet.cli import download


class TestDownloadCli(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_download_inexistence_model(self):
        result = self.runner.invoke(
            download,
            ["-m", "inexistence-model", "--saving_path", DATA_PATH],
            catch_exceptions=False,
        )
        self.assertIn(
            "Error: Invalid value for '--model' / '-m': invalid choice: inexistence-model. (choose from ",
            result.stdout,
        )
        self.assertEqual(result.exit_code, 2)

    def test_download_inexistence_corpus(self):
        result = self.runner.invoke(
            download,
            ["-d", "inexistence-corpus", "--saving_path", DATA_PATH],
            catch_exceptions=False,
        )
        self.assertIn(
            "Error: Invalid value for '--data' / '-d': invalid choice: inexistence-corpus. (choose from ",
            result.stdout,
        )
        self.assertEqual(result.exit_code, 2)

    def test_missing_saving_dir(self):
        result = self.runner.invoke(
            download, ["-d", "apequest"], catch_exceptions=False
        )
        self.assertIn("Error: Missing option '--saving_path'.", result.stdout)
        self.assertEqual(result.exit_code, 2)
