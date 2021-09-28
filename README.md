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

_We are planing to release version 1.0.0 in November. Meanwhile we recommend you to use our Pre-release of version and open issues if you find something unexpected:_

```bash
pip install unbabel-comet==1.0.0rc6
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
comet-score -s src.de -t hyp.en --model wmt20-comet-qe-da
```

Following the work on [Uncertainty-Aware MT Evaluation](https://arxiv.org/abs/2109.06352) you can use the --mc_dropout flag to get a variance/uncertainty value for each segment score. If this value is high, it means that the metric is less confident in that prediction.

```bash
comet-score -s src.de -t hyp.en -r ref.en --mc_dropout 30
```

When comparing two MT systems we encourage you to run the `comet-compare` command to get a **contrastive statistical significance** with bootstrap resampling [(Koehn, et al 2004)](https://aclanthology.org/W04-3250/).

```bash
comet-compare --help
```

For even more detailed MT contrastive evaluation please take a look at our new tool [MT-Telescope](https://github.com/Unbabel/MT-Telescope).

### Scoring within Python:

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
seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)
```

### Languages Covered:

All the above mentioned models are build on top of XLM-R which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

**Thus, results for language pairs containing uncovered languages are unreliable!**

## COMET Models:

We recommend the two following models to evaluate your translations:

- `wmt20-comet-da`: **DEFAULT** Reference-based Regression model build on top of XLM-R (large) and trained of Direct Assessments from WMT17 to WMT19. Same as `wmt-large-da-estimator-1719` from previous versions.
- `wmt20-comet-qe-da`: **Reference-FREE** Regression model build on top of XLM-R (large) and trained of Direct Assessments from WMT17 to WMT19. Same as `wmt-large-qe-estimator-1719` from previous versions.

These two models were developed to participate on the WMT20 Metrics shared task [(Mathur et al. 2020)](https://aclanthology.org/2020.wmt-1.77.pdf) and to the date, these are the best performing metrics at segment-level in the recently released Google MQM data [(Freitag et al. 2020)](https://arxiv.org/pdf/2104.14478.pdf). Also, in a large-scale study performed by Microsoft Research these two metrics are ranked 1st and 2nd in terms of system-level decision accuracy [(Kocmi et al. 2020)](https://arxiv.org/pdf/2107.10821.pdf).

For more information about the available COMET models we invite you to read our metrics descriptions [here](METRICS.md)

## Train your own Metric: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet-train --cfg configs/models/{your_model_config}.yaml
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

- [Uncertainty-Aware Machine Translation Evaluation](https://arxiv.org/pdf/2109.06352.pdf)