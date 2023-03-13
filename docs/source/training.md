# Train your own Metric

To train your own metric we recommend you to install directly from source:

```bash
   git clone https://github.com/Unbabel/COMET.git
   poetry install
```

After having your repo locally installed you can train your own model/metric with the following command:

```bash
comet-score -s src.de -t hyp1.en -r ref.en --model PATH/TO/CHECKPOINT
```

You can also upload your model to [Hugging Face Hub](https://huggingface.co/docs/hub/index). Use [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) as example. Then you can use your model directly from the hub.

## Config Files

In COMET uses [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) to train models. 
With that said, YAML files will be used to initialize various Lightning objects. 

Config files for Lightning classes:

- [trainer.yaml: ](https://github.com/Unbabel/COMET/blob/master/configs/trainer.yaml) used to initialize [Pytorch-Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api/).
- [model_checkpoint.yaml: ](https://github.com/Unbabel/COMET/blob/master/configs/model_checkpoint.yaml) used to initialize [Pytorch-Lightning ModelCheckpoint](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html?highlight=ModelCheckpoint#modelcheckpoint/).
- [early_stopping.yaml: ](https://github.com/Unbabel/COMET/blob/master/configs/early_stopping.yaml) used to initialize [Pytorch-Lightning EarlyStopping](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping/).

Then after setting up the these Lightning classes you can setup your model architecture. There are 4 different model architectures:

- [RegressionMetric: ](https://github.com/Unbabel/COMET/blob/master/comet/models/regression/regression_metric.py#L32>) used to build metrics that regress on a score given a source, hypothesis and reference.
- [ReferencelessRegression: ](https://github.com/Unbabel/COMET/blob/master/comet/models/regression/referenceless.py#L30) used to build metrics that regress on a score **without a reference translation!** (using only the source, hypothesis).
- [RankingMetric: ](https://github.com/Unbabel/COMET/blob/master/comet/models/ranking/ranking_metric.py#L36>) used to build metrics that learn to rank *good* translations above *bad* transations.
- [UnifiedMetric: ](https://github.com/Unbabel/COMET/blob/master/comet/models/multitask/unified_metric.py#L39>) was proposed in [(Wan et al., ACL 2022)](https://aclanthology.org/2022.acl-long.558.pdf) and it is closely related to [BLEURT](https://aclanthology.org/2020.acl-main.704/) and [OpenKiwi](https://aclanthology.org/P19-3020/) models. This model can be trained with and without references

For each class you can find a config example in [configs/models/](https://github.com/Unbabel/COMET/tree/master/configs/models). 
The `init_args` will then be used to initialize your model/metric.

## Input Data

To train your models you need to pass a train set and a validation set using the `training_data` and `validation_data` arguments respectively.

Depending on the underlying models your data need to be formatted differently. RegressionMetrics expect the following format:

| src | mt | ref | score | 
| :----: | :----: | :----: | :----: |
| isto é um exemplo  | this is a example  | this is an example | 0.2 |

For ReferencelessRegression you can drop the `ref` column but, if passed, it is ignored.

Finally, Ranking Metrics expect two contrastive examples. E.g:

| src | neg | pos | ref |
| :----: | :----: | :----: | :----: |
| isto é um exemplo  | this is a example  | this is an example | this is an example |

where `pos` column contains a postive sample and `neg` a negative sample. 

You can check the available [data from previous WMT editions in here](https://unbabel.github.io/COMET/html/faqs.html#where-can-i-find-the-data-used-to-train-comet-models).

## Available Encoders

All COMET models depend on an underlying encoder. We currently support the following encoders:

- BERT
- XLM-RoBERTa
- MiniLM
- XLM-RoBERTa-XL
- RemBERT

You can change the underlying encoder architecture using the ``encoder_model`` argument in your config file. 
Then, you can select any compatible model from [HuggingFace Transformers](https://huggingface.co/models) using the ``pretrained_model`` argument.
