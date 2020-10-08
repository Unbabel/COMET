# -*- coding: utf-8 -*-
import io
import os
import shutil
import unittest

from tests.data import DATA_PATH

from comet.corpora import download_corpus


class TestDownloadCorpus(unittest.TestCase):
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_download_apequest(self, mock_stdout):
        download_corpus("apequest", DATA_PATH)
        self.assertTrue(os.path.isdir(DATA_PATH + "apequest"))
        self.assertIn("Download succeeded.", mock_stdout.getvalue())
        download_corpus("apequest", DATA_PATH)
        self.assertIn("apequest is already in cache.", mock_stdout.getvalue())
        shutil.rmtree(DATA_PATH + "apequest")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_download_qt21(self, mock_stdout):
        download_corpus("qt21", DATA_PATH)
        self.assertTrue(os.path.isdir(DATA_PATH + "qt21"))
        self.assertIn("Download succeeded.", mock_stdout.getvalue())
        shutil.rmtree(DATA_PATH + "qt21")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_download_wmt(self, mock_stdout):
        download_corpus("wmt-metrics", DATA_PATH)
        self.assertTrue(os.path.isdir(DATA_PATH + "wmt-metrics"))
        self.assertIn("Download succeeded.", mock_stdout.getvalue())
        shutil.rmtree(DATA_PATH + "wmt-metrics")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_download_docwmt19(self, mock_stdout):
        download_corpus("doc-wmt19", DATA_PATH)
        self.assertTrue(os.path.isdir(DATA_PATH + "doc-wmt19"))
        self.assertIn("Download succeeded.", mock_stdout.getvalue())
        shutil.rmtree(DATA_PATH + "doc-wmt19")
