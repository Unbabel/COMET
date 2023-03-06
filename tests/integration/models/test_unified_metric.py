# -*- coding: utf-8 -*-
import math
import os
import shutil
import unittest
import warnings

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.trainer import Trainer
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from comet.models import UnifiedMetric
from tests.data import DATA_PATH

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


class TestUnifiedMetric(unittest.TestCase):
    """Testing if we can overfit a small dataset using UnifiedMetric class."""

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "checkpoints"))
        
    def test_regression_with_references(self):
        seed_everything(12)
        trainer = Trainer(
            devices=1 if torch.cuda.device_count() > 0 else 0,
            accelerator="auto",
            max_epochs=8,
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = UnifiedMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-03,
            learning_rate=1e-03,
            layerwise_decay=0.95,
            encoder_model="BERT",
            pretrained_model="google/bert_uncased_L-2_H-128_A-2",
            sent_layer="mix",
            layer_transformation="softmax",
            layer_norm=False,
            loss="mse",
            dropout=0.1,
            batch_size=32,
            train_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            validation_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            hidden_sizes=[384],
            activations="Tanh",
            final_activation=None,
            input_segments=["mt", "src", "ref"],
            word_level_training=False,
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)
        self.assertTrue(
            os.path.exists(
                os.path.join(DATA_PATH, "checkpoints", "epoch=7-step=104.ckpt")
            )
        )

        saved_model = UnifiedMetric.load_from_checkpoint(
            os.path.join(DATA_PATH, "checkpoints", "epoch=7-step=104.ckpt")
        )
        dataset = saved_model.read_validation_data(
            os.path.join(DATA_PATH, "regression_data.csv")
        )
        y = [s["score"] for s in dataset]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=256,
            collate_fn=lambda x: saved_model.prepare_sample(x, stage="predict"),
            num_workers=2,
        )
        predictions = trainer.predict(
            ckpt_path="best", dataloaders=dataloader, return_predictions=True
        )
        y_hat = torch.cat([p.scores for p in predictions], dim=0).tolist()
        assert pearsonr(y_hat, y)[0] > 0.9

    def test_regression_without_references(self):
        seed_everything(12)
        trainer = Trainer(
            devices=1 if torch.cuda.device_count() > 0 else 0,
            accelerator="auto",
            max_epochs=10,
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = UnifiedMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-03,
            learning_rate=1e-03,
            layerwise_decay=0.95,
            encoder_model="BERT",
            pretrained_model="google/bert_uncased_L-2_H-128_A-2",
            sent_layer="mix",
            layer_transformation="softmax",
            layer_norm=False,
            loss="mse",
            dropout=0.1,
            batch_size=32,
            train_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            validation_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            hidden_sizes=[384],
            activations="Tanh",
            final_activation=None,
            input_segments=["mt", "src"],
            word_level_training=False,
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)
        self.assertTrue(
            os.path.exists(
                os.path.join(DATA_PATH, "checkpoints", "epoch=9-step=130.ckpt")
            )
        )

        saved_model = UnifiedMetric.load_from_checkpoint(
            os.path.join(DATA_PATH, "checkpoints", "epoch=9-step=130.ckpt"),
            input_segments=["mt", "src"],
        )
        dataset = saved_model.read_validation_data(
            os.path.join(DATA_PATH, "regression_data.csv")
        )
        y = [s["score"] for s in dataset]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=256,
            collate_fn=lambda x: saved_model.prepare_sample(x, stage="predict"),
            num_workers=2,
        )
        predictions = trainer.predict(
            ckpt_path="best", dataloaders=dataloader, return_predictions=True
        )
        y_hat = torch.cat([p.scores for p in predictions], dim=0).tolist()
        assert pearsonr(y_hat, y)[0] > 0.9

    def test_multitask_with_references(self):
        seed_everything(12)
        trainer = Trainer(
            devices=1 if torch.cuda.device_count() > 0 else 0,
            accelerator="auto",
            max_epochs=15,  # This model takes a while to overfit.
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = UnifiedMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-03,
            learning_rate=1e-03,
            layerwise_decay=0.95,
            encoder_model="BERT",
            pretrained_model="google/bert_uncased_L-2_H-128_A-2",
            sent_layer="mix",
            word_layer=2,
            layer_transformation="softmax",
            layer_norm=False,
            loss="mse",
            dropout=0.1,
            batch_size=64,
            train_data=[os.path.join(DATA_PATH, "multitask_data.csv")],
            validation_data=[os.path.join(DATA_PATH, "multitask_data.csv")],
            hidden_sizes=[384],
            activations="Tanh",
            final_activation=None,
            input_segments=["mt", "src", "ref"],
            word_level_training=True,
            word_weights=[0.15, 0.85],
            loss_lambda=0.9,  # Giving more weight to word level.
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)
        self.assertTrue(
            os.path.exists(
                os.path.join(DATA_PATH, "checkpoints", "epoch=14-step=60.ckpt")
            )
        )
        saved_model = UnifiedMetric.load_from_checkpoint(
            os.path.join(DATA_PATH, "checkpoints", "epoch=14-step=60.ckpt"),
        )
        dataset = saved_model.read_validation_data(
            os.path.join(DATA_PATH, "multitask_data.csv")
        )
        y_reg = [s["score"] for s in dataset]
        predictions = saved_model.predict(
            dataset, gpus=0, batch_size=8, length_batching=True
        )
        assert pearsonr(predictions.scores, y_reg)[0] > 0.7
        assert pearsonr(predictions.metadata.src_scores, y_reg)[0] > 0.7
        assert pearsonr(predictions.metadata.ref_scores, y_reg)[0] > 0.7
        assert pearsonr(predictions.metadata.unified_scores, y_reg)[0] > 0.7
        word_level_example = [
            ("michael", 0),
            ("jackson", 0),
            ("tr", 0),
            ("##ug", 0),
            ("k", 0),
            ("##le", 0),
            ("##be", 0),
            ("##band", 0),
            ("auf", 0),
            ("der", 0),
            ("nas", 0),
            ("##e", 0),
            (",", 0),
            ("um", 0),
            ("ti", 0),
            ("##tel", 0),
            ("##sei", 0),
            ("##ten", 0),
            ("zu", 0), # Annotation 
            ("be", 1), # Annotation
            ("##ko", 1),  # Annotation
            ("##mme", 1),  # Annotation
            ("##n", 1),  # Annotation
            (",", 1),  # Annotation
            ("be", 1), # This one the model just does not learn.
            ("##ha", 0), 
            ("##upt", 0),
            ("##et", 0),
            ("eh", 0),
            ("##ema", 0),
            ("##li", 0),
            ("##ger", 0),
            ("bodyguard", 0),
        ]
        word_level_example = [subword[1] for subword in word_level_example]
        subword_scores_example = [
            0 if subword[1] < 0.5 else 1
            for subword in predictions.metadata.subword_scores[0]
        ]
        self.assertListEqual(word_level_example, subword_scores_example)
