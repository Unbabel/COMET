# -*- coding: utf-8 -*-
import unittest
import os
import shutil
from tests.data import DATA_PATH
from comet import download_model
from comet.models import load_from_checkpoint


class TestDownloadModel(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--wmt22-comet-da"))
        shutil.rmtree(os.path.join(DATA_PATH, "eamt22-cometinho-da"))

    def test_download_from_aws(self):
        data_path = download_model("eamt22-cometinho-da", saving_directory=DATA_PATH)
        load_from_checkpoint(data_path)

    def test_download_fail(self):
        self.assertRaises(KeyError, download_model, "this_model_does_not_exist")

    def test_download_from_hf(self):
        data_path = download_model("Unbabel/wmt22-comet-da", saving_directory=DATA_PATH)
        load_from_checkpoint(data_path)
