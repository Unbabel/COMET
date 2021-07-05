from .regression.regression_metric import RegressionMetric
from .ranking.ranking_metric import RankingMetric
from .regression.referenceless import ReferencelessRegression
from .base import CometModel

import os
import yaml

str2model = {
    "referenceless_regression_metric": ReferencelessRegression,
    "regression_metric": RegressionMetric,
    "ranking_metric": RankingMetric
}




def load_from_checkpoint(checkpoint_path: str):
    hparams_file = "/".join(checkpoint_path.split("/")[:-2] + ["hparams.yaml"])
    if os.path.exists(hparams_file):
        with open(hparams_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model_class = str2model[hparams["class_identifier"]]
        model = model_class.load_from_checkpoint(checkpoint_path)
        return model
    else:
        raise Exception("hparams.yaml file is missing!")