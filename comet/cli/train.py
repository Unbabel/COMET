# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

Command for training new Metrics.
=================================

e.g:
```
    comet-train --cfg configs/models/regression_metric.yaml
```

For more details run the following command:
```
    comet-train --help
```
"""
import json

from comet.models import (
    CometModel,
    RankingMetric,
    ReferencelessRegression,
    RegressionMetric,
)
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer


def train_command() -> None:
    parser = ArgumentParser(description="Command for training COMET models.")
    parser.add_argument(
        "--seed_everything",
        type=int,
        default=12,
        help="Training Seed.",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_class_arguments(CometModel, "model")
    parser.add_subclass_arguments(RegressionMetric, "regression_metric")
    parser.add_subclass_arguments(
        ReferencelessRegression, "referenceless_regression_metric"
    )
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
    print("TRAINER ARGUMENTS: ")
    print(json.dumps(trainer_args, indent=4, default=lambda x: x.__dict__))
    trainer = Trainer(**trainer_args)

    print("MODEL ARGUMENTS: ")
    if cfg.regression_metric is not None:
        print(
            json.dumps(
                cfg.regression_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        model = RegressionMetric(**namespace_to_dict(cfg.regression_metric.init_args))
    elif cfg.referenceless_regression_metric is not None:
        print(
            json.dumps(
                cfg.referenceless_regression_metric.init_args,
                indent=4,
                default=lambda x: x.__dict__,
            )
        )
        model = ReferencelessRegression(
            **namespace_to_dict(cfg.referenceless_regression_metric.init_args)
        )
    elif cfg.ranking_metric is not None:
        print(
            json.dumps(
                cfg.ranking_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        model = RankingMetric(**namespace_to_dict(cfg.ranking_metric.init_args))
    else:
        raise Exception("Model configurations missing!")

    trainer.fit(model)
