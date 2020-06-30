<div style="text-align:center"><img src="resources/LOGO.png" alt="comet_logo"></div>



Currently supported encoders:
- [Learning Joint Multilingual Sentence Representations with Neural Machine Translation](https://arxiv.org/abs/1704.04154)
- [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [XLM-R: Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf)

## Model Architectures:

### Estimator Architectures:
> **CometEstimator:** Uses a pretrained encoder to independently encode the source, MT and Reference and then uses a feed-forward neural network to estimate a MT quality score such as HTER

> **MetricEstimator:** Uses a pretrained encoder to independently encode the reference and MT hypothesis and then uses a feed-forward neural network to estimate a MT quality score such as HTER

### Translation Ranking Architectures:

> **CometRanker:** Uses a pretrained encoder to independently encode the source, a "good" MT hypothesis, a "bad" MT hypothesis and a Reference and then uses the triplet margin loss to minimize the distance between the "good" hypothesis and the anchors (reference/source).

> **MetricRanker:** Uses a pretrained encoder to independently encode the a "good" MT hypothesis, a "bad" MT hypothesis and a Reference and then uses the triplet margin loss to minimize the distance between the "good" hypothesis and the reference.

## Requirements:

This project uses Python >3.6

If you wish to make changes into the code run:
```bash
pip install -r requirements.txt
pip install -e .
```

## Pretrained Models:

| Model              |               Description                        |
| --------------------- | ------------------------------------------------ |
| `da-ranker-v1.0`      | Translation ranking model that uses XLM-R to encode sentences. This model was trained with WMT17 and WMT18 Direct Assessments Relative Ranks (DARR). |
| `hter-estimator-v1.0` | Estimator model that uses XLM-R to encode sentences. This model was trained to regress on HTER with the QT21 corpus. **Note:** For this model, since it regresses on HTER, 0 means a perfect translation a 1 is a very bad one.  |


## Scoring MT outputs:

### Via Bash:
```bash
comet score -s path/to/sources.txt -h path/to/hypothesis.txt -r path/to/references.txt --model da-ranker-v1.0
```

You can export your results to a JSON file using the `--to_json` flag.

```bash
comet score -s path/to/sources.txt -h path/to/hypothesis.txt -r path/to/references.txt --model da-ranker-v1.0 --to_json output.json
```

### Via Python:

```python
from comet.models import download_model
model = download_model("da-ranker-v1.0", "path/where/to/save/models")
data = [
    {
        "src": "Hello world!",
        "mt": "Oi mundo!",
        "ref": "Olá mundo!"
    },
    {
        "src": "This is a sample",
        "mt": "este é um exemplo",
        "ref": "isto é um exemplo!"
    }
]
model.predict(data)
```

### Simple Pythonic way to convert list or segments to model inputs:

```python
source = ["Hello world!", "This is a sample"]
hypothesis = ["Oi mundo!", "este é um exemplo"]
reference = ["Olá mundo!", "isto é um exemplo!"]

data = {"src": source, "mt": hypothesis, "ref": reference}
data = [dict(zip(data, t)) for t in zip(*data.values())]

model.predict(data)
```

## Train Command: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet train -f {config_file_path}.yaml
```

### GPU 16-bit:
Save some memory by using mixed precision training:
1. [Install apex](https://github.com/NVIDIA/apex)
2. Set ``amp_level: 'O1'`` and ``precision: 16`` in your config file.

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```

## Download Command: 

To download public available corpora to train your new models you can use the `download` command. For example to download the WMT17 relative ranks just run the following command:

```bash
comet download -d apequest --saving_path data/
```

### unittest:
```bash
pip install coverage
```

In order to run the toolkit tests you must run the following command:

```bash
coverage run --source=comet -m unittest discover
coverage report -m
```

### Code Style:
To make sure all the code follows the same style we use [Black](https://github.com/psf/black).
