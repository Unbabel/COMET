# Available Evaluation Models

Within COMET, there are several evaluation models available. The primary reference-based and reference-free models are:

- **Default Model:** [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) - This model employs a reference-based regression approach and is built upon the XLM-R architecture. It has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1, where 1 signifies a perfect translation.
- **Reference-free Model:** [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt23-cometkiwi-da) - This reference-free model employs a regression approach and is built on top of InfoXLM. It has been trained using direct assessments from WMT17 to WMT20, as well as direct assessments from the MLQE-PE corpus. Similar to other models, it generates scores ranging from 0 to 1. For those interested, we also offer larger versions of this model: [`Unbabel/wmt23-cometkiwi-da-xl`](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl) with 3.5 billion parameters and [`Unbabel/wmt23-cometkiwi-da-xxl`](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl) with 10.7 billion parameters.
- **eXplainable COMET (XCOMET):** [`Unbabel/XCOMET-XXL`](https://huggingface.co/Unbabel/XCOMET-XXL) - Our latest model is trained to identify error spans and assign a final quality score, resulting in an explainable neural metric. We offer this version in XXL with 10.7 billion parameters, as well as the XL variant with 3.5 billion parameters ([`Unbabel/XCOMET-XL`](https://huggingface.co/Unbabel/XCOMET-XL)). These models have demonstrated the **highest correlation with MQM** and are our best performing evaluation models.

Please be aware that different **models may be subject to varying licenses**. To learn more, kindly refer to the [LICENSES.models](LICENSE.models.md) and model licenses sections.

If you intend to compare your results with papers published before 2022, it's likely that they used older evaluation models. In such cases, please refer to [Unbabel/wmt20-comet-da](https://huggingface.co/Unbabel/wmt20-comet-da) and [Unbabel/wmt20-comet-qe-da](https://huggingface.co/Unbabel/wmt20-comet-qe-da), which were the primary checkpoints used in previous versions (<2.0) of COMET.

## UniTE Models:

[UniTE Metric](https://aclanthology.org/2022.acl-long.558/) was developed by the NLP2CT Lab at the University of Macau and Alibaba Group, and all credits should be attributed to these groups. COMET framework fully supports running UniTE and thus we made the original UniTE-MUP checkpoint available in Hugging Face Hub. Additionally, we also trained our own UniTE model using the same data as `wmt22-comet-da`. You can access both models here:

- [`Unbabel/unite-mup`](https://huggingface.co/Unbabel/unite-mup) - This is the original UniTE Metric proposed in the [UniTE: Unified Translation Evaluation](https://aclanthology.org/2022.acl-long.558/) paper.
- [`Unbabel/wmt22-unite-da`](https://huggingface.co/Unbabel/wmt22-unite-da) - This model was trained for our paper [(Rei et al., ACL 2023)](https://aclanthology.org/2023.acl-short.94/) and it uses the same data as [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) thus, the outputed scores are between 0 and 1.
- [`Unbabel/unite-xxl`](https://huggingface.co/Unbabel/unite-xxl) - xCOMET models [(Guerreiro et al. 2023)](https://arxiv.org/pdf/2310.10482.pdf) are training following a curriculum. The checkpoint resulting from the first phase of that curriculum (before the introduction of a sequence tagging task and MQM data) is a [UniTE Model](https://aclanthology.org/2022.acl-long.558/). An [XL version](https://huggingface.co/Unbabel/unite-xl) is also available.


## Older Models:

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
