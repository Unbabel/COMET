# COMET Metrics

Since COMET was released we have been training and releasing different models. In this page we will try to briefly explain the underlying differences and point you to the papers that used them.

## Model Architectures:

All COMET metrics follow one of the following architectures:

[![Model Architectures](/_static/img/architectures.png)](https://raw.githubusercontent.com/Unbabel/COMET/docs-config/docs/source/_static/img/architectures.png)

1) Regression Metric (top-left diagram): This is the architecture that most models use. This model is trained on a regression task using source, MT and reference.
2) Ranking Metric (top-middle diagram): Models that follow this architecture are trained in a Translation Ranking Task using a [Triple Margin Loss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html). This means that the model will learn to optimize the embedding space to encode _good_ translations closer to the anchors (source and reference) while pushing _bad_ translations away.
3) Referenceless Metric (top-right diagram): This architecture resembles architecture 1) but **it does not use the reference translation!** This is purely a Quality Estimation system.
4) Unified Metric (bottom diagram): Unified architecture was proposed in [(Wan et al., ACL 2022)](https://aclanthology.org/2022.acl-long.558.pdf) and it is closely related to [BLEURT](https://aclanthology.org/2020.acl-main.704/) and [OpenKiwi](https://aclanthology.org/P19-3020/) models. This model can be trained with and without references


# Available Evaluation Models

The two main COMET models are:

- **Default model:** [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) - This model uses a reference-based regression approach and is built on top of XLM-R. It has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1, where 1 represents a perfect translation.
- **Upcoming model:** [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) - This reference-free model uses a regression approach and is built on top of InfoXLM. It has been trained on direct assessments from WMT17 to WMT20, as well as direct assessments from the MLQE-PE corpus. Like the default model, it also provides scores ranging from 0 to 1.

These two models were part of the final ensemble used in our WMT22 [Metrics](https://aclanthology.org/2022.wmt-1.52/) and [QE](https://aclanthology.org/2022.wmt-1.60/) shared tasks. 

For versions prior to 2.0, you can still use [`Unbabel/wmt20-comet-da`](https://huggingface.co/Unbabel/wmt20-comet-da), which is the primary metric, and Unbabel/[`Unbabel/wmt20-comet-qe-da`](https://huggingface.co/Unbabel/wmt20-comet-qe-da) for the respective reference-free version.

All other models developed through the years can be accessed through the following links:

| Model | Download Link | Paper | 
| :---: | :-----------: | :---: |
| `emnlp20-comet-rank` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/emnlp20-comet-rank.tar.gz) | [🔗](https://aclanthology.org/2020.emnlp-main.213/) |
| `wmt20-comet-qe-da` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/wmt20-comet-qe-da.tar.gz) | [🔗](https://aclanthology.org/2020.wmt-1.101/) |
| `wmt21-comet-da` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-da.tar.gz) | [🔗](https://aclanthology.org/2021.wmt-1.111/) |
| `wmt21-comet-mqm` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-mqm.tar.gz) | [🔗](https://aclanthology.org/2021.wmt-1.111/) |
| `wmt21-comet-qe-da` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-da.tar.gz) | [🔗](https://aclanthology.org/2021.wmt-1.111/) |
| `wmt21-comet-qe-mqm` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-mqm.tar.gz) | [🔗](https://aclanthology.org/2021.wmt-1.111/) |
| `wmt21-comet-qe-da` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-da.tar.gz) | [🔗](https://aclanthology.org/2021.wmt-1.111/) |
| `wmt21-cometinho-mqm` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-cometinho-mqm.tar.gz) | [🔗](https://aclanthology.org/2021.wmt-1.111/) |
| `wmt21-cometinho-da` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-cometinho-da.tar.gz) | [🔗](https://aclanthology.org/2021.wmt-1.111/) | 
| `eamt22-cometinho-da` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-cometinho-da.tar.gz) | [🔗](https://aclanthology.org/2022.eamt-1.9/) |
| `eamt22-prune-comet-da` | [🔗](https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-prune-comet-da.tar.gz) | [🔗](https://aclanthology.org/2022.eamt-1.9/) |

Example :

```
wget https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-cometinho-da.tar.gz
tar -xf eamt22-cometinho-da.tar.gz
comet-score -s src.de -t hyp1.en -r ref.en --model eamt22-cometinho-da/checkpoints/model.ckpt
```
