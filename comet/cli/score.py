import multiprocessing

import torch
from jsonargparse import ArgumentParser
from torch.utils.data import DataLoader
from jsonargparse.typing import Path_fr
from typing import Union
from comet.models import available_metrics, load_from_checkpoint
from comet.download_utils import download_model
from pytorch_lightning.trainer.trainer import Trainer

import json


def score_command() -> None:
    parser = ArgumentParser(
        description='Command for scoring MT systems.'
    )
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-t", "--translations", type=Path_fr, required=True)
    parser.add_argument("-r", "--references", type=Path_fr, required=True)
    parser.add_argument('--to_json', type=Union[bool, str], default=False)
    parser.add_argument(
        '--model', 
        type=Union[str, Path_fr],  
        required=False,
        default="wmt21-large-estimator"
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        default=32
    )
    parser.add_argument(
        '--gpus', 
        type=int,
        default=1
    )
    cfg = parser.parse_args()
    
    model_path = download_model(cfg.model) if cfg.model in available_metrics else cfg.model
    model = load_from_checkpoint(model_path)

    with open(cfg.sources()) as fp:
        sources = fp.readlines()
    
    with open(cfg.translations()) as fp:
        translations = fp.readlines()
    
    with open(cfg.references()) as fp:
        references = fp.readlines()
   
    data = {"src": sources, "mt": translations, "ref": references}
    data = [dict(zip(data, t)) for t in zip(*data.values())]

    dataloader = DataLoader(
        dataset=data, 
        batch_size=cfg.batch_size,
        collate_fn=lambda x: model.prepare_sample(x, inference=True),
        num_workers=multiprocessing.cpu_count(),
    )
    trainer = Trainer(gpus=cfg.gpus, deterministic=True)
    predictions = trainer.predict(
        model, 
        dataloaders=dataloader, 
        return_predictions=True
    )
    predictions = torch.cat(predictions, dim=0).tolist()

    if isinstance(cfg.to_json, str):
        with open(cfg.to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print ("Predictions saved in: {}.".format(cfg.to_json))
    
    for i in range(len(predictions)):
        print ("Segment {} score: {:.3f}".format(i, predictions[i]))

    print (
        "System score: {:.3f}".format(sum(predictions) / len(predictions))
    )
    
    