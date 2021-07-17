<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>


## Quick Installation

Detailed usage examples and instructions can be found in the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

Simple installation from PyPI

```bash
pip install unbabel-comet
```

To develop locally install [Poetry](https://python-poetry.org/docs/#installation) and run the following commands:
```bash
git clone https://github.com/Unbabel/COMET
poetry install
```

## Scoring MT outputs:

### Via Bash:

Examples from WMT20:

```bash
echo -e "Dem Feuer konnte Einhalt geboten werden\nSchulen und Kindergärten wurden eröffnet." >> src.de
echo -e "The fire could be stopped\nSchools and kindergartens were open" >> hyp.en
echo -e "They were able to control the fire.\nSchools and kindergartens opened" >> ref.en
```

```bash
comet-score -s src.de -t hyp.en -r ref.en
```

You can select another model/metric with the --model flag and for reference-less (QE-as-a-metric) models you dont need to pass a reference.

```bash
comet-score -s src.de -t hyp.en -r ref.en --model refless-wmt21-large-da-1520
```

Following the work on [Uncertainty-Aware MT Evaluation]() you can use the --mc_dropout flag to get a variance/uncertainty value for each segment score. If this value is high, it means that the metric as less confidence is that prediction.

```bash
comet-score -s src.de -t hyp.en -r ref.en --mc_dropout 100
```

## Languages Covered:

All the above mentioned models are build on top of XLM-R which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

**Thus, results for language pairs containing uncovered languages are unreliable!**

### Scoring within Python:

COMET implements the [Pytorch-Lightning model interface](https://pytorch-lightning.readthedocs.io/en/1.3.8/common/lightning_module.html) which means that you'll need to initialize a trainer in order to run inference.

```python
import torch
from comet import download_model, load_from_checkpoint
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

model = load_from_checkpoint(
  download_model("wmt21-small-da-152012")
)
data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    }
]
data = [dict(zip(data, t)) for t in zip(*data.values())]
dataloader = DataLoader(
  dataset=data,
  batch_size=16,
  collate_fn=lambda x: model.prepare_sample(x, inference=True),
  num_workers=4,
)
trainer = Trainer(gpus=1, deterministic=True, logger=False)
predictions = trainer.predict(
  model, dataloaders=dataloader, return_predictions=True
)
predictions = torch.cat(predictions, dim=0).tolist()
```

**Note:** Using the python interface you will get a list of segment-level scores. You can obtain the corpus-level score by averaging the segment-level scores

## Model Zoo:

:TODO: Update model zoo after the shared task.

| Model              |               Description                        |
| :--------------------- | :------------------------------------------------ |
| ↑`wmt21-large-da-1520` | **RECOMMENDED:** Regression model build on top of XLM-R (large) trained on DA from WMT15, to WMT20 |
| ↑`wmt21-small-da-152012` | Same as the model above but trained on a small version of XLM-R that was distilled from XLM-R large |

#### QE-as-a-metric:

| Model              |               Description                        |
| -------------------- | -------------------------------- |
| `refless-wmt21-large-da-1520` | Reference-less model trained on top of XLM-R large with DAs from WMT15 to WMT20. |

## Train your own Metric: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet-train -cfg configs/models/{your_model_config}.yaml
```

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="lightning_logs/"
```

## unittest:
In order to run the toolkit tests you must run the following command:

```bash
coverage run --source=comet -m unittest discover
coverage report -m
```

## Publications

```
@inproceedings{rei-etal-2020-comet,
    title = "{COMET}: A Neural Framework for {MT} Evaluation",
    author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
    pages = "2685--2702",
}
```

```
@inproceedings{rei-EtAl:2020:WMT,
  author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
  title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
  booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
  month          = {November},
  year           = {2020},
  address        = {Online},
  publisher      = {Association for Computational Linguistics},
  pages     = {909--918},
}
```

```
@inproceedings{stewart-etal-2020-comet,
    title = "{COMET} - Deploying a New State-of-the-art {MT} Evaluation Metric in Production",
    author = "Stewart, Craig  and
      Rei, Ricardo  and
      Farinha, Catarina  and
      Lavie, Alon",
    booktitle = "Proceedings of the 14th Conference of the Association for Machine Translation in the Americas (Volume 2: User Track)",
    month = oct,
    year = "2020",
    address = "Virtual",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://www.aclweb.org/anthology/2020.amta-user.4",
    pages = "78--109",
}
```
