# -*- coding: utf-8 -*-
import os
import shutil
import unittest
import warnings

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
            devices=1 if torch.cuda.device_count() > 0 else 0,
            accelerator="auto",
            max_epochs=22,
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = RankingMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-04,
            learning_rate=1e-04,
            layerwise_decay=0.95,
            encoder_model="BERT",
            pretrained_model="google/bert_uncased_L-2_H-128_A-2",
            pool="cls",
            layer="mix",
            layer_transformation="softmax",
            layer_norm=True,
            dropout=0.1,
            batch_size=32,
            train_data=[os.path.join(DATA_PATH, "ranking_data.csv")],
            validation_data=[os.path.join(DATA_PATH, "ranking_data.csv")],
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)
        self.assertTrue(
            os.path.exists(
                os.path.join(DATA_PATH, "checkpoints", "epoch=21-step=154.ckpt")
            )
        )
        saved_model = RankingMetric.load_from_checkpoint(
            os.path.join(DATA_PATH, "checkpoints", "epoch=21-step=154.ckpt")
        )
        dataset = saved_model.read_validation_data(
            os.path.join(DATA_PATH, "ranking_data.csv")
        )

        # Scores for "superior" translations
        pos_translations = [
            {"src": s["src"], "mt": s["pos"], "ref": s["ref"]} for s in dataset
        ]
        dataloader = DataLoader(
            dataset=pos_translations,
            batch_size=256,
            collate_fn=lambda x: saved_model.prepare_sample(x, stage="predict"),
            num_workers=2,
        )
        predictions = trainer.predict(
            ckpt_path="best", dataloaders=dataloader, return_predictions=True
        )
        y_pos = torch.cat([p["scores"] for p in predictions], dim=0)

        # Scores for "worse" translations
        neg_translations = [
            {"src": s["src"], "mt": s["neg"], "ref": s["ref"]} for s in dataset
        ]
        dataloader = DataLoader(
            dataset=neg_translations,
            batch_size=256,
            collate_fn=lambda x: saved_model.prepare_sample(x, stage="predict"),
            num_workers=2,
        )
        predictions = trainer.predict(
            ckpt_path="best", dataloaders=dataloader, return_predictions=True
        )
        y_neg = torch.cat([p["scores"] for p in predictions], dim=0)
        ## This shouldn't break!
        pearsonr(y_pos, y_neg)[0]

        concordance = torch.sum((y_pos > y_neg))
        discordance = torch.sum((y_pos <= y_neg))
        kendall = (concordance - discordance) / (concordance + discordance)
        assert kendall > 0.7
