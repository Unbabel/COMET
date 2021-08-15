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

Pre-release of version 1.0:
```bash
pip install unbabel-comet==1.0.0rc3
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

You can select another model/metric with the --model flag and for reference-free (QE-as-a-metric) models you don't need to pass a reference.

```bash
comet-score -s src.de -t hyp.en -r ref.en --model wmt21-comet-qe-da
```

Following the work on [Uncertainty-Aware MT Evaluation]() you can use the --mc_dropout flag to get a variance/uncertainty value for each segment score. If this value is high, it means that the metric is less confident in that prediction.

```bash
comet-score -s src.de -t hyp.en -r ref.en --mc_dropout 30
```

## Languages Covered:

All the above mentioned models are build on top of XLM-R which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

**Thus, results for language pairs containing uncovered languages are unreliable!**

### Scoring within Python:

COMET implements the [Pytorch-Lightning model interface](https://pytorch-lightning.readthedocs.io/en/1.3.8/common/lightning_module.html) which means that you'll need to initialize a trainer in order to run inference.

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("wmt20-comet-da")
model = load_from_checkpoint(model_path)
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
predictions, system_score = model.predict(data, batch_size=8, gpus=1)
```

## Model Zoo:

| Model              |               Description                        |
| :--------------------- | :------------------------------------------------ |
| `wmt20-comet-da` | **RECOMMENDED:** Regression model build on top of XLM-R (large) trained on DA from WMT17, to WMT19. This model was presented at the WMT20 Metrics shared task: [rei et al, 2020](https://aclanthology.org/2020.wmt-1.101.pdf). Same as `wmt-large-da-estimator-1719` from previous versions. |
| `emnlp20-comet-rank` | Translation Ranking model build on top of XLM-R (base) trained with DARR from WMT17 and WMT18. This model was presented at EMNLP20: [rei et al, 2020](https://aclanthology.org/2020.emnlp-main.213.pdf). |

**Note:** Scores between models are not comparable! each model learns its own distribution and the scale might differ.

### QE-as-a-metric:

| Model              |               Description                        |
| :--------------------- | :------------------------------------------------ |
| `wmt20-comet-qe-da` | Reference-free Regression model build on top of XLM-R (large) trained on DA from WMT17, to WMT19. This model was presented at the WMT20 Metrics shared task: [rei et al, 2020](https://aclanthology.org/2020.wmt-1.101.pdf). Same as `wmt-large-qe-estimator-1719` from previous versions. |

## Train your own Metric: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet-train --cfg configs/models/{your_model_config}.yaml
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

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)
