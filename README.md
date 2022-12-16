<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>

<hr />

> Currently the master is a pre-release of our work for the WMT 2022 shared tasks [Metrics](https://www.statmt.org/wmt22/pdf/2022.wmt-1.52.pdf) and [QE](https://www.statmt.org/wmt22/pdf/2022.wmt-1.60.pdf)! **Use version 1.1.3 if you are looking for a stable version!** 

> We are planning a new release with better models and new-features for January.
<hr />

## Quick Installation

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

## Scoring MT outputs:

### CLI Usage:

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
> you can set `--gpus 0` to test on CPU.

Scoring multiple systems:
```bash
comet-score -s src.de -t hyp1.en hyp2.en -r ref.en
```

WMT test sets via [SacreBLEU](https://github.com/mjpost/sacrebleu):

```bash
comet-score -d wmt20:en-de -t PATH/TO/TRANSLATIONS
```

The default setting of `comet-score` prints the score for each segment individually. If you are only interested in a system-level score, you can use the `--quiet` flag.

```bash
comet-score -s src.de -t hyp1.en -r ref.en --quiet
```

```bash
comet-score -s src.de -t hyp1.en --model wmt22-cometkiwi-da
```

When comparing multiple MT systems we encourage you to run the `comet-compare` command to get **statistical significance** with Paired T-Test and bootstrap resampling [(Koehn, et al 2004)](https://aclanthology.org/W04-3250/).

```bash
comet-compare -s src.de -t hyp1.en hyp2.en hyp3.en -r ref.en
```

## Minimum Bayes Risk Decoding:

The MBR command allows you to rank MT hypotheses and select the best one according to COMET. For more details you can read our paper on [Quality-Aware Decoding for Neural Machine Translation](https://aclanthology.org/2022.naacl-main.100.pdf).

Our implementation is inspired by [Amrhein et al, 2022](https://aclanthology.org/2022.aacl-main.83.pdf) where sentences are cached during inference to avoid quadratic computations while creating the sentence embeddings.

```bash
comet-mbr -s [SOURCE].txt -t [MT_SAMPLES].txt --num_sample [X] -o [OUTPUT_FILE].txt
```

#### Multi-GPU Inference:

COMET is optimized to be used in a single GPU by taking advantage of length batching and embedding caching. When using Multi-GPU since data e spread across GPUs we will typically get fewer cache hits and the length batching samples is replaced by a [DistributedSampler](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#replace-sampler-ddp). Because of that, according to our experiments, using 1 GPU is faster than using 2 GPUs specially when scoring multiple systems for the same source and reference.

Nonetheless, if your data does not have repetitions and you have more than 1 GPU available, you can **run multi-GPU inference with the following command**:

```bash
comet-score -s src.de -t hyp1.en -r ref.en --gpus 2 --quiet
```

**Warning:** Segment-level scores using multigpu will be out of order. This is only useful for system scoring.

### Scoring within Python:

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("wmt22-comet-da")
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
seg_scores, system_score = model_output.scores, model_output.system_score
```

### Languages Covered:

All the above mentioned models are build on top of XLM-R which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

**Thus, results for language pairs containing uncovered languages are unreliable!**

## COMET Models:
We recommend the two following models to evaluate your translations:

- `wmt20-comet-da`: **DEFAULT** Reference-based Regression model build on top of XLM-R (large) and trained of Direct Assessments from WMT17 to WMT19. Same as `wmt-large-da-estimator-1719` from previous versions.
- `wmt21-comet-qe-mqm`: **Reference-FREE** Regression model build on top of XLM-R (large), trained on Direct Assessments and fine-tuned on MQM.
- `eamt22-cometinho-da`: **Lightweight** Reference-based Regression model that was distilled from an ensemble of COMET models similar to `wmt20-comet-da`.

## Train your own Metric: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet-train --cfg configs/models/{your_model_config}.yaml
```

You can then use your own metric to score:

```bash
comet-score -s src.de -t hyp1.en -r ref.en --model PATH/TO/CHECKPOINT
```

**Note:** Please contact ricardo.rei@unbabel.com if you wish to host your own metric within COMET available metrics!

## unittest:
In order to run the toolkit tests you must run the following command:

```bash
coverage run --source=comet -m unittest discover
coverage report -m
```
**Note:** Testing on CPU takes a long time

## Publications
If you use COMET please cite our work and don't forget to say which model you used to evaluate your systems.

- [CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task -- Winning submission](https://arxiv.org/pdf/2209.06243.pdf)

- [COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task](https://www.statmt.org/wmt22/pdf/2022.wmt-1.52.pdf)

- [Searching for Cometinho: The Little Metric That Could -- EAMT22 Best paper award](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
