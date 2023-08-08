# -*- coding: utf-8 -*-
import os
import shutil
import sys
import unittest
import warnings
from functools import partial
from pathlib import Path

import torch
from comet.models import RegressionMetric
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.trainer import Trainer
from scipy.stats import pearsonr
from tests.data import DATA_PATH
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


@unittest.skipIf(True, "ignore")
class TestRegressionMetric(unittest.TestCase):
    """Testing if we can overfit a small dataset using RegressionMetric class."""

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "checkpoints"))

    def test_training(self):
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
        model = RegressionMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-04,
            learning_rate=1e-04,
            layerwise_decay=0.95,
            encoder_model="BERT",
            pretrained_model="google/bert_uncased_L-2_H-128_A-2",
            pool="avg",
            layer="mix",
            layer_transformation="sparsemax",
            layer_norm=True,
            loss="mse",
            dropout=0.1,
            batch_size=32,
            train_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            validation_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            hidden_sizes=[384],
            activations="Tanh",
            final_activation=None,
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

        saved_model = RegressionMetric.load_from_checkpoint(
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
        y_hat = torch.cat([p.scores for p in predictions], dim=0).tolist()
        assert pearsonr(y_hat, y)[0] > 0.85
        logging.debug("ASSERTED", pearsonr(y_hat, y)[0])


@unittest.skipIf(
    torch.cuda.device_count() < 2,
    f"Skipping distributed regression test because we have insufficient hardware ({torch.cuda.device_count()} devices, "
    f"need at least 2)"
)
class TestRegressionMetricDistributed(unittest.TestCase):
    """Testing if we can overfit a small dataset using RegressionMetric class."""

    # Commenting this out as it seems to be executed prematurely or something? Not sure
    # @classmethod
    # def tearDownClass(cls):
    #     shutil.rmtree(os.path.join(DATA_PATH, "checkpoints"))

    @classmethod
    def setUpClass(cls):
        Path(DATA_PATH).mkdir(exist_ok=True)

    def test_training(self):
        seed_everything(12)
        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            strategy="auto",
            max_epochs=2,
            enable_checkpointing=True,
            default_root_dir=DATA_PATH,
            logger=False,
            enable_progress_bar=True,
        )
        model = RegressionMetric(
            nr_frozen_epochs=1,
            keep_embeddings_frozen=False,
            optimizer="AdamW",
            encoder_learning_rate=1e-04,
            learning_rate=1e-04,
            layerwise_decay=0.95,
            encoder_model="BERT",
            pretrained_model="google/bert_uncased_L-2_H-128_A-2",
            pool="avg",
            layer="mix",
            layer_transformation="sparsemax",
            layer_norm=True,
            loss="mse",
            dropout=0.1,
            batch_size=32,
            train_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            validation_data=[os.path.join(DATA_PATH, "regression_data.csv")],
            hidden_sizes=[384],
            activations="Tanh",
            final_activation=None,
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)
        # Training seems to work fine?
        # But evaluation is not. Seems that something goes wrong when gathering from multiple processes

        ckpt_path = next(Path(DATA_PATH).glob("checkpoints/epoch=*"))
        saved_model = RegressionMetric.load_from_checkpoint(ckpt_path)
        dataset = saved_model.read_validation_data(
            os.path.join(DATA_PATH, "regression_data.csv")
        )
        y = [s["score"] for s in dataset]

        import logging
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=256,
            collate_fn=partial(saved_model.prepare_sample, stage="predict"),
            num_workers=2,
        )
        predictions = trainer.predict(
            ckpt_path="best",
            dataloaders=dataloader, return_predictions=True
        )
        logging.debug(f"dataset: {len(dataset)}")
        y_hat = torch.cat([p.scores for p in predictions], dim=0).tolist()
        logging.debug(f"y size: {len(y)}")
        logging.debug(f"y_hat size: {len(y_hat)}")
        assert pearsonr(y_hat, y)[0] > 0.85
        logging.debug("ASSERTED", pearsonr(y_hat, y)[0])

if __name__ == '__main__':
    import logging

    logging.basicConfig( stream=sys.stderr )
    logging.getLogger().setLevel( logging.DEBUG )
    unittest.main()