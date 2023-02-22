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
