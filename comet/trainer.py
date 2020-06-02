# -*- coding: utf-8 -*-
import os
from argparse import Namespace
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
)
from pytorch_lightning.logging import TestTubeLogger


class TrainerConfig:
    """ 
    The TrainerConfig class is used to define default hyper-parameters that
    are used to initialize our Lightning Trainer. These parameters are then overwritted 
    with the values defined in the YAML file.

    -------------------- General Parameters -------------------------

    :param seed: Training seed.

    :param deterministic: If true enables cudnn.deterministic. Might make your system 
        slower, but ensures reproducibility.

    :param model: Model class we want to train.

    :param verbode: verbosity mode.
    
    -------------------- Model Checkpoint & Early Stopping -------------------------

    :param early_stopping: If true enables EarlyStopping.
    
    :param save_top_k: If save_top_k == k, the best k models according to the metric 
        monitored will be saved.
    
    :param monitor: Metric to be monitored.
    
    :param save_weights_only: Saves only the weights of the model.
    
    :param period: Interval (number of epochs) between checkpoints.
    
    :param metric_mode: One of {min, max}. In min mode, training will stop when the 
        metric monitored has stopped decreasing; in max mode it will stop when the 
        metric monitored has stopped increasing.
    
    :param min_delta: Minimum change in the monitored metric to qualify as an improvement.

    :param patience: Number of epochs with no improvement after which training will be stopped.
    """

    seed: int = 3
    deterministic: bool = True
    model: str = None
    verbose: bool = False

    # Model Checkpoint & Early Stopping
    early_stopping: bool = True
    save_top_k: int = 1
    monitor: str = "kendall"
    save_weights_only: bool = False
    period: int = 1
    metric_mode: str = "max"
    min_delta: float = 0.0
    patience: int = 1

    def __init__(self, initial_data: dict) -> None:
        trainer_attr = Trainer.default_attributes()
        for key in trainer_attr:
            setattr(self, key, trainer_attr[key])

        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])

    def namespace(self) -> Namespace:
        return Namespace(
            **{
                name: getattr(self, name)
                for name in dir(self)
                if not callable(getattr(self, name)) and not name.startswith("__")
            }
        )


def build_trainer(hparams: Namespace) -> Trainer:
    """
    :param hparams: Namespace

    Returns:
        - pytorch_lightning Trainer
    """

    def setup_testube_logger():
        """ Function that sets the TestTubeLogger to be used. """
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        return TestTubeLogger(
            save_dir="experiments/", version=dt_string, name="lightning_logs",
        )

    if hparams.early_stopping:
        # Enable Early stopping
        early_stop_callback = EarlyStopping(
            monitor=hparams.monitor,
            min_delta=hparams.min_delta,
            patience=hparams.patience,
            verbose=hparams.verbose,
            mode=hparams.metric_mode,
        )
    else:
        early_stop_callback = None

    trainer = Trainer(
        logger=setup_testube_logger(),
        deterministic=hparams.deterministic,
        checkpoint_callback=True,
        early_stop_callback=early_stop_callback,
        default_save_path="experiments/",
        gradient_clip_val=hparams.gradient_clip_val,
        gpus=hparams.gpus,
        overfit_pct=hparams.overfit_pct,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        train_percent_check=hparams.train_percent_check,
        val_percent_check=hparams.val_percent_check,
        val_check_interval=hparams.val_check_interval,
        log_save_interval=hparams.log_save_interval,
        row_log_interval=hparams.row_log_interval,
        distributed_backend=hparams.distributed_backend,
        precision=hparams.precision,
        weights_summary="top",
        profiler=hparams.profiler,
        log_gpu_memory="all",
        callbacks=[LearningRateLogger()],
    )

    ckpt_path = os.path.join(
        trainer.default_save_path,
        trainer.logger.name,
        f"version_{trainer.logger.version}",
        "checkpoints",
    )

    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=hparams.verbose,
        monitor=hparams.monitor,
        save_weights_only=hparams.save_weights_only,
        period=hparams.period,
        mode=hparams.metric_mode,
    )
    trainer.checkpoint_callback = checkpoint_callback
    return trainer
