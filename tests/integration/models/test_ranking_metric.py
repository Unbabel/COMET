# -*- coding: utf-8 -*-
import multiprocessing
import os
import shutil
import unittest

import torch
from comet.models import RankingMetric
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.trainer import Trainer
from scipy.stats import pearsonr
from tests.data import DATA_PATH
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


class TestRankingMetric(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "checkpoints"))

    def test_training(self):
        seed_everything(12)
        trainer = Trainer(
            gpus=0,
            max_epochs=4,
            deterministic=True,
            checkpoint_callback=True,
            default_root_dir=DATA_PATH,
            logger=False,
            weights_summary=None,
            progress_bar_refresh_rate=0,
        )
        model = RankingMetric(
            encoder_model="BERT",
            pretrained_model="google/bert_uncased_L-2_H-128_A-2",
            train_data=os.path.join(DATA_PATH, "test_ranking_data.csv"),
            validation_data=os.path.join(DATA_PATH, "test_ranking_data.csv"),
            layerwise_decay=0.95,
            batch_size=32,
            learning_rate=1e-04,
            encoder_learning_rate=1e-04,
        )
        trainer.fit(model)
        self.assertTrue(
            os.path.exists(
                os.path.join(DATA_PATH, "checkpoints", "epoch=3-step=15.ckpt")
            )
        )
        saved_model = RankingMetric.load_from_checkpoint(
            os.path.join(DATA_PATH, "checkpoints", "epoch=3-step=15.ckpt")
        )
        dataset = saved_model.read_csv(
            os.path.join(DATA_PATH, "test_regression_data.csv"), regression=True
        )
        y = [s["score"] for s in dataset]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=256,
            collate_fn=lambda x: saved_model.prepare_sample(x, inference=True),
            num_workers=multiprocessing.cpu_count(),
        )
        y_hat = (
            torch.cat(
                trainer.predict(dataloaders=dataloader, return_predictions=True), dim=0
            )
            .cpu()
            .tolist()
        )
        # This shouldn't break!
        pearsonr(y_hat, y)[0]
