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
            devices="auto",
            accelerator="auto",
            max_epochs=10,
            deterministic=True,
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = UnifiedMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-04,
            learning_rate=1e-04,
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
                os.path.join(DATA_PATH, "checkpoints", "epoch=9-step=130.ckpt")
            )
        )

        saved_model = UnifiedMetric.load_from_checkpoint(
            os.path.join(DATA_PATH, "checkpoints", "epoch=9-step=130.ckpt")
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
        y_hat = torch.cat([p.score for p in predictions], dim=0).tolist()
        assert pearsonr(y_hat, y)[0] > 0.85

    def test_regression_without_references(self):
        seed_everything(12)
        trainer = Trainer(
            devices="auto",
            accelerator="auto",
            max_epochs=10,
            deterministic=True,
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = UnifiedMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-04,
            learning_rate=1e-04,
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
        y_hat = torch.cat([p.score for p in predictions], dim=0).tolist()
        assert pearsonr(y_hat, y)[0] > 0.82
    
    def test_multitask_with_references(self):

        def matthews_correlation_coef(y_hat, y):
            tp, tn, fp, fn = 0, 0, 0, 0
            for pred, label in zip(y_hat, y):
                assert len(pred) == len(label)
                for i in range(len(pred)):
                    if pred[i] == label[i] and pred[i] == "BAD":
                        tp += 1
                    elif pred[i] == label[i] and pred[i] == "OK":
                        tn += 1
                    elif pred[i] != label[i] and pred[i] == "BAD":
                        fp += 1
                    elif pred[i] != label[i] and pred[i] == "OK":
                        fn += 1
            return ((tp*tn) - (fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        seed_everything(12)
        trainer = Trainer(
            devices="auto",
            accelerator="auto",
            max_epochs=15,
            deterministic=True,
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = UnifiedMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-04,
            learning_rate=1e-04,
            layerwise_decay=0.95,
            encoder_model="XLM-RoBERTa",
            pretrained_model="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",
            sent_layer="mix",
            word_layer=6,
            layer_transformation="softmax",
            layer_norm=False,
            loss="mse",
            dropout=0.1,
            batch_size=32,
            train_data=[os.path.join(DATA_PATH, "multitask_data.csv")],
            validation_data=[os.path.join(DATA_PATH, "multitask_data.csv")],
            hidden_sizes=[384],
            activations="Tanh",
            final_activation=None,
            input_segments=["mt", "src", "ref"],
            word_level_training=True,
            word_weights=[0.15, 0.85],
            loss_lambda=0.9, # Giving more weight to word level.
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)
        self.assertTrue(
            os.path.exists(
                os.path.join(DATA_PATH, "checkpoints", "epoch=14-step=105.ckpt")
            )
        )
        saved_model = UnifiedMetric.load_from_checkpoint(
            os.path.join(DATA_PATH, "checkpoints", "epoch=14-step=105.ckpt"),
        )
        dataset = saved_model.read_validation_data(
            os.path.join(DATA_PATH, "multitask_data.csv")
        )
        y_cls = [s["labels"].split() for s in dataset]
        y_reg = [s["score"] for s in dataset]
        predictions = saved_model.predict(dataset, gpus=0, batch_size=8, length_batching=True, accelerator="ddp")
        y_hat_cls = predictions.metadata.word_labels
        y_hat_reg = predictions.scores
        assert pearsonr(y_hat_reg, y_reg)[0] > 0.75
        assert matthews_correlation_coef(y_hat_cls, y_cls) > 0.75
        
