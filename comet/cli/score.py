import multiprocessing

import torch
from jsonargparse import ArgumentParser
from torch.utils.data import DataLoader


def score_command():
    parser = ArgumentParser(
        description='Command for scoring MT systems.'
    )
    
    cfg = parser.parse_args()
    source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
    hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
    reference = ["They were able to control the fire.", "Schools and kindergartens opened"]

    data = {"src": source, "mt": hypothesis, "ref": reference}
    data = [dict(zip(data, t)) for t in zip(*data.values())]

    dataloader = DataLoader(
        dataset=data, 
        batch_size=cfg.batch_size,
        collate_fn=lambda x: model.prepare_sample(x, inference=True),
        num_workers=multiprocessing.cpu_count(),
    )

    predictions = trainer.predict(
        model, 
        dataloaders=dataloader, 
        return_predictions=True
    )
    predictions = torch.cat(predictions, dim=0)
    