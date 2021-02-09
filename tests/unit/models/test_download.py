# -*- coding: utf-8 -*-
import io
import os
import shutil
import unittest
import unittest.mock

from comet.models import CometEstimator, download_model, load_checkpoint, model2download
from tests.data import DATA_PATH


class TestDownload(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(DATA_PATH + "wmt-base-da-estimator-1719")
        # os.remove(DATA_PATH + "public-models.yaml")

    def test_load_unvalid_checkpoint(self):
        with self.assertRaises(Exception) as context:
            load_checkpoint("folder/that/does/not/exist/")
        self.assertEqual(
            str(context.exception), "folder/that/does/not/exist/ file not found!"
        )

    def test_model2download(self):
        model2link = model2download(DATA_PATH)
        self.assertTrue(os.path.exists(DATA_PATH + "public-models.yaml"))
        # Double call should overwrite file.
        model2link = model2download(DATA_PATH)
        self.assertIsInstance(model2link, dict)
        os.remove(DATA_PATH + "public-models.yaml")

    def test_model2download_bad_saving_dir(self):
        self.assertRaises(
            FileNotFoundError, lambda: model2download("folder/that/does/not/exist/")
        )

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_download_model(self, mock_stdout):
        model = download_model("wmt-base-da-estimator-1719", DATA_PATH)
        self.assertIsInstance(model, CometEstimator)
        self.assertIn("Download succeeded. Loading model...", mock_stdout.getvalue())
        download_model("wmt-base-da-estimator-1719", DATA_PATH)
        self.assertIn("is already in cache.", mock_stdout.getvalue())

    def test_download_wrong_model(self):
        with self.assertRaises(Exception) as context:
            download_model("WrongModel", DATA_PATH)
        self.assertEqual(
            str(context.exception), "WrongModel is not a valid COMET model!"
        )
