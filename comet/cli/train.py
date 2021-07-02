from typing import Optional

from comet.models import CometModel, RegressionMetric, ReferencelessRegression, RankingMetric
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer


def train_command() -> None:
    parser = ArgumentParser(
        description="Command for training COMET models."
    )
    parser.add_argument(
        "--seed_everything",
        type=Optional[int],
        default=12,
        help="Set to an int to run seed_everything with this value before classes instantiation",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_class_arguments(CometModel, "model")
    parser.add_subclass_arguments(RegressionMetric, "regression_metric")
    parser.add_subclass_arguments(ReferencelessRegression, "referenceless_regression_metric")
    parser.add_subclass_arguments(RankingMetric, "ranking_metric")
    parser.add_subclass_arguments(EarlyStopping, "early_stopping")
    parser.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    parser.add_subclass_arguments(Trainer, "trainer")
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    checkpoint_callback = ModelCheckpoint(
        **namespace_to_dict(cfg.model_checkpoint.init_args)
    )
    early_stop_callback = EarlyStopping(
        **namespace_to_dict(cfg.early_stopping.init_args)
    )
    trainer_args = namespace_to_dict(cfg.trainer.init_args)
    trainer_args["callbacks"] = [early_stop_callback, checkpoint_callback]
    trainer = Trainer(**trainer_args)

    if cfg.regression_metric is not None:
        model = RegressionMetric(**namespace_to_dict(cfg.regression_metric.init_args))
    elif cfg.referenceless_regression_metric is not None:
        model = ReferencelessRegression(**namespace_to_dict(cfg.referenceless_regression_metric.init_args))
    elif cfg.ranking_metric is not None:
        model = RankingMetric(**namespace_to_dict(cfg.ranking_metric.init_args))
    else:
        raise Exception("Model configurations missing!")

    trainer.fit(model)