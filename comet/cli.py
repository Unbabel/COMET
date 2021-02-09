# -*- coding: utf-8 -*-
r"""
COMET command line interface (CLI)
==============
Composed by 4 main commands:
    train       Used to train a machine translation metric.
    score       Uses COMET to score a list of MT outputs.
    download    Used to download corpora or pretrained metric.
"""
import json
import os

import click
import yaml

from comet.corpora import corpus2download, download_corpus
from comet.models import download_model, load_checkpoint, model2download, str2model
from comet.trainer import TrainerConfig, build_trainer
from pytorch_lightning import seed_everything


@click.group()
def comet():
    pass


@comet.command()
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
def train(config):
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)

    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Print Trainer parameters into terminal
    result = "Hyperparameters:\n"
    for k, v in train_configs.namespace().__dict__.items():
        result += "{0:30}| {1}\n".format(k, v)
    click.secho(f"{result}", fg="blue", nl=False)

    # Build Model
    try:
        model_config = str2model[train_configs.model].ModelConfig(yaml_file)
        model = str2model[train_configs.model](model_config.namespace())
    except KeyError:
        raise Exception(f"Invalid model {train_configs.model}!")

    result = ""
    for k, v in model_config.namespace().__dict__.items():
        result += "{0:30}| {1}\n".format(k, v)
    click.secho(f"{result}", fg="cyan")

    # Train model
    click.secho(f"{model.__class__.__name__} train starting:", fg="yellow")
    trainer.fit(model)


@comet.command()
@click.option(
    "--model",
    default="wmt-large-da-estimator-1719",
    help="Name of the pretrained model OR path to a model checkpoint.",
    show_default=True,
    type=str,
)
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source segments.",
    type=click.File(),
)
@click.option(
    "--hypothesis",
    "-h",
    required=True,
    help="MT outputs.",
    type=click.File(),
)
@click.option(
    "--reference",
    "-r",
    required=True,
    help="Reference segments.",
    type=click.File(),
)
@click.option(
    "--cuda/--cpu",
    default=True,
    help="Flag that either runs inference on cuda or in cpu.",
    show_default=True,
)
@click.option(
    "--batch_size",
    default=-1,
    help="Batch size used during inference. By default uses the same batch size used during training.",
    type=int,
)
@click.option(
    "--to_json",
    default=False,
    help="Creates and exports model predictions to a JSON file.",
    type=str,
    show_default=True,
)
def score(model, source, hypothesis, reference, cuda, batch_size, to_json):
    source = [s.strip() for s in source.readlines()]
    hypothesis = [s.strip() for s in hypothesis.readlines()]
    reference = [s.strip() for s in reference.readlines()]
    data = {"src": source, "mt": hypothesis, "ref": reference}
    data = [dict(zip(data, t)) for t in zip(*data.values())]

    model = load_checkpoint(model) if os.path.exists(model) else download_model(model)
    data, scores = model.predict(data, cuda, show_progress=True, batch_size=batch_size)

    if isinstance(to_json, str):
        with open(to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        click.secho(f"Predictions saved in: {to_json}.", fg="yellow")

    for i in range(len(scores)):
        click.secho("Segment {} score: {:.3f}".format(i, scores[i]), fg="yellow")
    click.secho(
        "COMET system score: {:.3f}".format(sum(scores) / len(scores)), fg="yellow"
    )


@comet.command()
@click.option(
    "--data",
    "-d",
    type=click.Choice(corpus2download.keys(), case_sensitive=False),
    multiple=True,
    help="Public corpora to download.",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(model2download().keys(), case_sensitive=False),
    multiple=True,
    help="Pretrained models to download.",
)
@click.option(
    "--saving_path",
    type=str,
    help="Relative path to save the downloaded files.",
    required=True,
)
def download(data, model, saving_path):
    for d in data:
        download_corpus(d, saving_path)

    for m in model:
        download_model(m, saving_path)
