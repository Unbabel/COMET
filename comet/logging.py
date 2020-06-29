# -*- coding: utf-8 -*-
r"""
Loggers
==============
    COMET Loggers used for training.
"""
from datetime import datetime

import click
import pandas as pd

import pytorch_lightning as pl
from comet.models.utils import apply_to_sample
from pytorch_lightning.loggers import LightningLoggerBase, TestTubeLogger
from pytorch_lightning.utilities import rank_zero_only


def setup_testube_logger():
    """ Function that sets the TestTubeLogger to be used. """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    return TestTubeLogger(
        save_dir="experiments/", version=dt_string, name="lightning_logs",
    )


class CliLoggingCallback(pl.Callback):
    """ Logger Callback that echos results during training. """

    _stack: list = [] # stack to keep metrics from all epochs

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        metrics = LightningLoggerBase._flatten_dict(metrics, '_')
        metrics = apply_to_sample(lambda x: x.item(), metrics)
        self._stack.append(metrics)
        
        click.secho(
            "\n{}".format(
                pd.DataFrame(
                    data=[metrics],
                    index=["Epoch " + str(pl_module.current_epoch + 1)],
                )
            ),
            fg="yellow",
        )

    @rank_zero_only
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        click.secho(f"Training Report Experiment: {pl_module.logger.version}", fg="yellow")
        index_column = ["Epoch " + str(i+1) for i in range(len(self._stack))]
        df = pd.DataFrame(self._stack, index=index_column)
        click.secho("{}".format(df), fg="yellow")
