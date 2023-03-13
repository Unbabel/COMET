<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>

# Quick Installation

COMET requires python 3.8 or above! 

Simple installation from PyPI

```bash
pip install --upgrade pip  # ensures that pip is current 
pip install unbabel-comet
```

To develop locally install run the following commands:
```bash
git clone https://github.com/Unbabel/COMET
cd COMET
pip install poetry
poetry install
```

For development, you can run the CLI tools directly, e.g.,

```bash
PYTHONPATH=. ./comet/cli/score.py
```

# Scoring MT outputs:

## CLI Usage:

Test examples:

```bash
echo -e "Dem Feuer konnte Einhalt geboten werden\nSchulen und Kindergärten wurden eröffnet." >> src.de
echo -e "The fire could be stopped\nSchools and kindergartens were open" >> hyp1.en
echo -e "The fire could have been stopped\nSchools and pre-school were open" >> hyp2.en
echo -e "They were able to control the fire.\nSchools and kindergartens opened" >> ref.en
```

Basic scoring command:
```bash
comet-score -s src.de -t hyp1.en -r ref.en
```
> you can set the number of gpus using `--gpus` (0 to test on CPU).

Scoring multiple systems:
```bash
comet-score -s src.de -t hyp1.en hyp2.en -r ref.en
```

WMT test sets via [SacreBLEU](https://github.com/mjpost/sacrebleu):

```bash
comet-score -d wmt22:en-de -t PATH/TO/TRANSLATIONS
```

If you are only interested in a system-level score use the following command:

```bash
comet-score -s src.de -t hyp1.en -r ref.en --quiet --only_system
```

### Reference-free evaluation:

```bash
comet-score -s src.de -t hyp1.en --model Unbabel/wmt20-comet-qe-da
```

**Note:** We are currently working on Licensing and releasing `Unbabel/wmt22-cometkiwi-da` but meanwhile that models is not available.

### Comparing multiple systems:

When comparing multiple MT systems we encourage you to run the `comet-compare` command to get **statistical significance** with Paired T-Test and bootstrap resampling [(Koehn, et al 2004)](https://aclanthology.org/W04-3250/).

```bash
comet-compare -s src.de -t hyp1.en hyp2.en hyp3.en -r ref.en
```

### Minimum Bayes Risk Decoding:

The MBR command allows you to rank translations and select the best one according to COMET metrics. For more details you can read our paper on [Quality-Aware Decoding for Neural Machine Translation](https://aclanthology.org/2022.naacl-main.100.pdf).

```bash
comet-mbr -s [SOURCE].txt -t [MT_SAMPLES].txt --num_sample [X] -o [OUTPUT_FILE].txt
```

If working with a very large candidate list you can use `--rerank_top_k` flag to prune the topK most promissing candidates according to a reference-free metric.

Example for a candidate list of 1000 samples:

```bash
comet-mbr -s [SOURCE].txt -t [MT_SAMPLES].txt -o [OUTPUT_FILE].txt --num_sample 1000 --rerank_top_k 100 --gpus 4 --qe_model Unbabel/wmt20-comet-qe-da
```

# COMET Models:

To evaluate your translations, we suggest using one of two models:

- **Default model:** [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) - This model uses a reference-based regression approach and is built on top of XLM-R. It has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1, where 1 represents a perfect translation.
- **Upcoming model:** [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) - This reference-free model uses a regression approach and is built on top of InfoXLM. It has been trained on direct assessments from WMT17 to WMT20, as well as direct assessments from the MLQE-PE corpus. Like the default model, it also provides scores ranging from 0 to 1.

For versions prior to 2.0, you can still use [`Unbabel/wmt20-comet-da`](https://huggingface.co/Unbabel/wmt20-comet-da), which is the primary metric, and [`Unbabel/wmt20-comet-qe-da`](https://huggingface.co/Unbabel/wmt20-comet-qe-da) for the **respective reference-free version**. You can find a list of all other models developed in previous versions on our [MODELS](MODELS.md) page. For more information, please refer to the [model licenses](LICENSE.models.md).

## Interpreting Scores:

When using COMET to evaluate machine translation, it's important to understand how to interpret the scores it produces.

In general, COMET models are trained to predict quality scores for translations. These scores are typically normalized using a [z-score transformation](https://simplypsychology.org/z-score.html) to account for individual differences among annotators. While the raw score itself does not have a direct interpretation, it is useful for ranking translations and systems according to their quality.

However, for the latest COMET models like [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da), we have introduced a new training approach that scales the scores between 0 and 1. This makes it easier to interpret the scores: a score close to 1 indicates a high-quality translation, while a score close to 0 indicates a translation that is no better than random chance.

It's worth noting that when using COMET to compare the performance of two different translation systems, it's important to run the `comet-compare` command to obtain statistical significance measures. This command compares the output of two systems using a statistical hypothesis test, providing an estimate of the probability that the observed difference in scores between the systems is due to chance. This is an important step to ensure that any differences in scores between systems are statistically significant.

Overall, the added interpretability of scores in the latest COMET models, combined with the ability to assess statistical significance between systems using `comet-compare`, make COMET a valuable tool for evaluating machine translation.

## Languages Covered:

All the above mentioned models are build on top of XLM-R which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

**Thus, results for language pairs containing uncovered languages are unreliable!**

## Scoring within Python:

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
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
model_output = model.predict(data, batch_size=8, gpus=1)
print(model_output)
```

# Train your own Metric: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet-train --cfg configs/models/{your_model_config}.yaml
```

You can then use your own metric to score:

```bash
comet-score -s src.de -t hyp1.en -r ref.en --model PATH/TO/CHECKPOINT
```

You can also upload your model to [Hugging Face Hub](https://huggingface.co/docs/hub/index). Use [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) as example. Then you can use your model directly from the hub.

# unittest:
In order to run the toolkit tests you must run the following command:

```bash
coverage run --source=comet -m unittest discover
coverage report -m # Expected coverage 78%
```

**Note:** Testing on CPU takes a long time

# Publications

If you use COMET please cite our work **and don't forget to say which model you used!**

- [CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task](https://aclanthology.org/2022.wmt-1.60/)

- [COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task](https://aclanthology.org/2022.wmt-1.52/)

- [Searching for Cometinho: The Little Metric That Could](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](https://aclanthology.org/2021.wmt-1.111/)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
