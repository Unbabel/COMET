<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>

> Version 1.1.2 is out 🥳! whats new?
> 1) Update: `comet-compare` to support multiple system comparison. Thanks to @SamuelLarkin
> 2) Bugfix: Broken link for `wmt21-comet-qe-da` (#78)
> 3) Bugfix: protobuf dependency (#82)
> 4) New models added from Cometinho -- [Best paper award at EAMT 22 paper](https://aclanthology.org/2022.eamt-1.9/) 🥳!

## Quick Installation

Simple installation from PyPI

```bash
pip install --upgrade pip  # ensures that pip is current 
pip install unbabel-comet
```
or
```bash
pip install unbabel-comet==1.1.2 --use-feature=2020-resolver
```

To develop locally install [Poetry](https://python-poetry.org/docs/#installation) and run the following commands:
```bash
git clone https://github.com/Unbabel/COMET
cd COMET
poetry install
```

Alternately, for development, you can run the CLI tools directly, e.g.,

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

The default setting of `comet-score` prints the score for each segment individually. If you are only interested in the score for the whole dataset (computed as the average of the segment scores), you can use the `--quiet` flag.

```bash
comet-score -s src.de -t hyp1.en -r ref.en --quiet
```

You can select another model/metric with the --model flag and for reference-free (QE-as-a-metric) models you don't need to pass a reference.

```bash
comet-score -s src.de -t hyp1.en --model wmt21-comet-qe-mqm
```

Following the work on [Uncertainty-Aware MT Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) you can use the --mc_dropout flag to get a variance/uncertainty value for each segment score. If this value is high, it means that the metric is less confident in that prediction.

```bash
comet-score -s src.de -t hyp1.en -r ref.en --mc_dropout 30
```

When comparing multiple MT systems we encourage you to run the `comet-compare` command to get **statistical significance** with Paired T-Test and bootstrap resampling [(Koehn, et al 2004)](https://aclanthology.org/W04-3250/).

```bash
comet-compare -s src.de -t hyp1.en hyp2.en hyp3.en -r ref.en
```

**New: Minimum Bayes Risk Decoding:**

Inspired by [Amrhein et al, 2022](https://arxiv.org/abs/2202.05148) work, we have developed a command to perform Minimum Bayes Risk decoding. This command receives a text file with source sentences and a text file containing all the MT samples and writes to an output file the best sample according to COMET.

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

#### Changing Embedding Cache Size:
You can change the cache size of COMET using the following env variable:

```bash
export COMET_EMBEDDINGS_CACHE="2048"
```
by default the COMET cache size is 1024.


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
- `wmt21-comet-qe-mqm`: **Reference-FREE** Regression model build on top of XLM-R (large), trained on Direct Assessments and fine-tuned on MQM.
- `eamt22-cometinho-da`: **Lightweight** Reference-based Regression model that was distilled from an ensemble of COMET models similar to `wmt20-comet-da`.

The default model was developed to [participate in the WMT20 Metrics shared task](https://aclanthology.org/2020.wmt-1.101/) [(Mathur et al. 2020)](https://aclanthology.org/2020.wmt-1.77.pdf) and were among the best metrics that year. Also, in a large-scale study performed by Microsoft Research this metrics ranked 1st in terms of system-level decision accuracy [(Kocmi et al. 2020)](https://arxiv.org/pdf/2107.10821.pdf). 

Our recommended QE system was [developed for the WMT21 Metrics shared task](https://aclanthology.org/2021.wmt-1.111/) and was the best performing _QE as a Metric_ that year [(Freitag et al. 2021)](https://aclanthology.org/2021.wmt-1.73/).

**Note:** The range of scores between different models can be totally different. To better understand COMET scores please [take a look at our FAQs](https://unbabel.github.io/COMET/html/faqs.html)

For more information about the available COMET models read our metrics descriptions [here](https://unbabel.github.io/COMET/html/models.html)

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

## Publications
If you use COMET please cite our work! Also, don't forget to say which model you used to evaluate your systems.

- [Searching for Cometinho: The Little Metric That Could -- EAMT22 Best paper award](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)



