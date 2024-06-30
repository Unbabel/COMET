<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>

**NEWS:** 
1) We added a new method to extract free-text explanations from XCOMET outputs! [Check this section](https://github.com/Unbabel/COMET?tab=readme-ov-file#explaining-translation-errors)
2) We now support [DocCOMET](https://statmt.org/wmt22/pdf/2022.wmt-1.6.pdf), a document-level extension of COMET which can utilize contextual information. Using context improves accuracy on discourse phenomena tasks as well as referenceless evaluation of [chat translation quality](https://arxiv.org/pdf/2403.08314).
3) We released our new eXplainable COMET models ([XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) and [-XXL](https://huggingface.co/Unbabel/XCOMET-XXL)) which along with quality scores detects which errors in the translation are minor, major or critical according to MQM typology

Please check all available models [here](https://github.com/Unbabel/COMET/blob/master/MODELS.md)
 
# Quick Installation

COMET requires python 3.8 or above. Simple installation from PyPI

```bash
pip install --upgrade pip  # ensures that pip is current 
pip install unbabel-comet
```

**Note:** To use some COMET models such as `Unbabel/wmt22-cometkiwi-da` you must acknowledge it's license on Hugging Face Hub and [log-in into hugging face hub](https://huggingface.co/docs/huggingface_hub/quick-start#:~:text=Once%20you%20have%20your%20User%20Access%20Token%2C%20run%20the%20following%20command%20in%20your%20terminal%3A).


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

# Table of Contents

1. [Scoring MT outputs](#scoring-mt-outputs)
    1. [CLI Usage](#cli-usage)
        1. [Basic scoring command](#basic-scoring-command)
        2. [Reference-free evaluation](#reference-free-evaluation)
        3. [Comparing multiple systems](#comparing-multiple-systems)
        4. [Minimum Bayes Risk Decoding](#minimum-bayes-risk-decoding)
2. [COMET Models](#comet-models)
    1. [Interpreting Scores](#interpreting-scores)
    2. [Languages Covered](#languages-covered)
    3. [COMET for African Languages](#comet-for-african-languages)
    4. [Scoring within Python](#scoring-within-python)
    5. [Explaining Translation Errors](#explaining-translation-errors)
3. [Train your own Metric](#train-your-own-metric)
4. [Unittest](#unittest)
5. [Publications](#publications)


# Scoring MT outputs:

## CLI Usage:

Test examples:

```bash
echo -e "10 到 15 分钟可以送到吗\nPode ser entregue dentro de 10 a 15 minutos?" >> src.txt
echo -e "Can I receive my food in 10 to 15 minutes?\nCan it be delivered in 10 to 15 minutes?" >> hyp1.txt
echo -e "Can it be delivered within 10 to 15 minutes?\nCan you send it for 10 to 15 minutes?" >> hyp2.txt
echo -e "Can it be delivered between 10 to 15 minutes?\nCan it be delivered between 10 to 15 minutes?" >> ref.txt
```

### Basic scoring command:
```bash
comet-score -s src.txt -t hyp1.txt -r ref.txt
```
> you can set the number of gpus using `--gpus` (0 to test on CPU).

For better error analysis, you can use XCOMET models such as [`Unbabel/XCOMET-XL`](https://huggingface.co/Unbabel/XCOMET-XL), you can export the identified errors using the `--to_json` flag:

```bash
comet-score -s src.txt -t hyp1.txt -r ref.txt --model Unbabel/XCOMET-XL --to_json output.json
```

Scoring multiple systems:
```bash
comet-score -s src.txt -t hyp1.txt hyp2.txt -r ref.txt
```

WMT test sets via [SacreBLEU](https://github.com/mjpost/sacrebleu):

```bash
comet-score -d wmt22:en-de -t PATH/TO/TRANSLATIONS
```

Scoring with context:
```bash
echo -e "Pies made from apples like these. </s> Oh, they do look delicious.\nOh, they do look delicious." >> src.txt
echo -e "Des tartes faites avec des pommes comme celles-ci. </s> Elles ont l’air delicieux.\nElles ont l’air delicieux" >> hyp1.txt
echo -e "Des tartes faites avec des pommes comme celles-ci. </s> Ils ont l’air delicieux.\nIls ont l’air delicieux." >> hyp2.txt
```

where `</s>` is the separator token of the specific tokenizer (here: `xlm-roberta-large`) that the underlying model uses. 

```bash
comet-score -s src.txt -t hyp1.txt hyp2.txt --model Unbabel/wmt20-comet-qe-da --enable-context
```

If you are only interested in a system-level score use the following command:

```bash
comet-score -s src.txt -t hyp1.txt -r ref.txt --quiet --only_system
```

### Reference-free evaluation:

```bash
comet-score -s src.txt -t hyp1.txt --model Unbabel/wmt22-cometkiwi-da
```

**Note:** To use the `Unbabel/wmt23-cometkiwi-da-xl` you first have to acknowledge its license on [Hugging Face Hub](https://huggingface.co/Unbabel/Unbabel/wmt23-cometkiwi-da-xl).

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
comet-mbr -s [SOURCE].txt -t [MT_SAMPLES].txt -o [OUTPUT_FILE].txt --num_sample 1000 --rerank_top_k 100 --gpus 4 --qe_model Unbabel/wmt23-cometkiwi-da-xl
```

Your source and samples file should be [formatted in this way](https://unbabel.github.io/COMET/html/running.html#:~:text=Example%20with%202%20source%20and%203%20samples%3A).

# COMET Models

Within COMET, there are several evaluation models available. You can refer to the [MODELS](MODELS.md) page for a comprehensive list of all available models. Here is a concise list of the main reference-based and reference-free models:

- **Default Model:** [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) - This model employs a reference-based regression approach and is built upon the XLM-R architecture. It has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1, where 1 signifies a perfect translation.
- **Reference-free Model:** [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) - This reference-free model employs a regression approach and is built on top of InfoXLM. It has been trained using direct assessments from WMT17 to WMT20, as well as direct assessments from the MLQE-PE corpus. Similar to other models, it generates scores ranging from 0 to 1. For those interested, we also offer larger versions of this model: [`Unbabel/wmt23-cometkiwi-da-xl`](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl) with 3.5 billion parameters and [`Unbabel/wmt23-cometkiwi-da-xxl`](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl) with 10.7 billion parameters.
- **eXplainable COMET (XCOMET):** [`Unbabel/XCOMET-XXL`](https://huggingface.co/Unbabel/XCOMET-XXL) - Our latest model is trained to identify error spans and assign a final quality score, resulting in an explainable neural metric. We offer this version in XXL with 10.7 billion parameters, as well as the XL variant with 3.5 billion parameters ([`Unbabel/XCOMET-XL`](https://huggingface.co/Unbabel/XCOMET-XL)). These models have demonstrated the highest correlation with MQM and are our best performing evaluation models.

Please be aware that different models may be subject to varying licenses. To learn more, kindly refer to the [LICENSES.models](LICENSE.models.md) and model licenses sections.

If you intend to compare your results with papers published before 2022, it's likely that they used older evaluation models. In such cases, please refer to [`Unbabel/wmt20-comet-da`](https://huggingface.co/Unbabel/wmt20-comet-da) and [`Unbabel/wmt20-comet-qe-da`](https://huggingface.co/Unbabel/wmt20-comet-qe-da), which were the primary checkpoints used in previous versions (<2.0) of COMET.

Also, [UniTE Metric](https://aclanthology.org/2022.acl-long.558/) developed by the NLP2CT Lab at the University of Macau and Alibaba Group can be used directly through COMET check [here for more details](https://huggingface.co/Unbabel/unite-mup).

## Interpreting Scores:

**New:** An excellent reference for learning how to interpret machine translation metrics is the analysis paper by Kocmi et al. (2024), available [at this link.](https://arxiv.org/pdf/2401.06760.pdf)

When using COMET to evaluate machine translation, it's important to understand how to interpret the scores it produces.

In general, COMET models are trained to predict quality scores for translations. These scores are typically normalized using a [z-score transformation](https://simplypsychology.org/z-score.html) to account for individual differences among annotators. While the raw score itself does not have a direct interpretation, it is useful for ranking translations and systems according to their quality.

However, since 2022 we have introduced a new training approach that scales the scores between 0 and 1. This makes it easier to interpret the scores: a score close to 1 indicates a high-quality translation, while a score close to 0 indicates a translation that is no better than random chance. Also, with the introduction of XCOMET models we can now analyse which text spans are part of minor, major or critical errors according to the MQM typology.

It's worth noting that when using COMET to compare the performance of two different translation systems, it's important to run the `comet-compare` command to obtain statistical significance measures. This command compares the output of two systems using a statistical hypothesis test, providing an estimate of the probability that the observed difference in scores between the systems is due to chance. This is an important step to ensure that any differences in scores between systems are statistically significant.

Overall, the added interpretability of scores in the latest COMET models, combined with the ability to assess statistical significance between systems using `comet-compare`, make COMET a valuable tool for evaluating machine translation.

## Languages Covered:

All the above mentioned models are build on top of XLM-R (variants) which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskrit, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

**Thus, results for language pairs containing uncovered languages are unreliable!**

### COMET for African Languages:

If you are interested in COMET metrics for african languages please visit [afriCOMET](https://github.com/masakhane-io/africomet). 

## Scoring within Python:

```python
from comet import download_model, load_from_checkpoint

# Choose your model from Hugging Face Hub
model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
# model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

# Data must be in the following format:
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    }
]
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)
```

As output, we get the following information:
```python
# Sentence-level scores (list)
>>> model_output.scores
[0.9822099208831787, 0.9599897861480713]

# System-level score (float)
>>> model_output.system_score
0.971099853515625

# Detected error spans (list of list of dicts)
>>> model_output.metadata.error_spans
[
  [{'confidence': 0.4160953164100647,
   'end': 21,
   'severity': 'minor',
   'start': 13,
   'text': 'my food'}],
  [{'confidence': 0.40004390478134155,
   'end': 19,
   'severity': 'minor',
   'start': 3,
   'text': 'you send it for'}]
]
```

However, note that not all COMET models return metadata with detected error spans.


## Explaining translation errors:

Check [this notebook](https://gist.github.com/mtreviso/b618b499bc6de0414a3e11157e91cf02) for a minimal example on how you can combine xCOMET with [xTower](https://huggingface.co/sardinelab/xTower13B) to generate a natural language explanation for each error span. 

For the Portuguese-English example above, we would call xTower with the following **prompt**:
> You are provided with a Source, Translation, Translation quality analysis, and Translation quality score (weak, moderate, good, excellent, best). The Translation quality analysis contains a translation with marked error spans with different levels of severity (minor or major). Given this information, generate an explanation for each error and a fully correct translation. <br><br>
> Portuguese source: Pode ser entregue dentro de 10 a 15 minutos? <br>
> English translation: Can you send it for 10 to 15 minutes? <br>
> Translation quality analysis: Can `<error1 severity='minor'>`you send it for`</error1>` 10 to 15 minutes? <br>
> Translation quality score: excellent


And get this as output:
> Explanation for error1: The phrase "Can you send it for 10 to 15 minutes?" is a mistranslation of the original Portuguese sentence. The correct interpretation should focus on the delivery time rather than the duration of sending. The original sentence is asking about the delivery time, not the duration of sending. <br>
> Translation correction: Can it be delivered within 10 to 15 minutes?


For more information, check [xTower documentation](https://huggingface.co/sardinelab/xTower13B).


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
poetry run coverage run --source=comet -m unittest discover
poetry run coverage report -m # Expected coverage 76%
```

**Note:** Testing on CPU takes a long time

# Publications

If you use COMET please cite our work **and don't forget to say which model you used!**

- [xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection](https://arxiv.org/pdf/2310.10482.pdf)

- [Scaling up CometKiwi: Unbabel-IST 2023 Submission for the Quality Estimation Shared Task](https://arxiv.org/pdf/2309.11925.pdf)

- [CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task](https://aclanthology.org/2022.wmt-1.60/)

- [COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task](https://aclanthology.org/2022.wmt-1.52/)

- [Searching for Cometinho: The Little Metric That Could](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](https://aclanthology.org/2021.wmt-1.111/)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
