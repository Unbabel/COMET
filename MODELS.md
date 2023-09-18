# Available Evaluation Models

The two main COMET models are:

- **Default model:** [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) - This model uses a reference-based regression approach and is built on top of XLM-R. It has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1, where 1 represents a perfect translation.
- **Reference-free:** [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) - This reference-free model uses a regression approach and is built on top of InfoXLM. It has been trained on direct assessments from WMT17 to WMT20, as well as direct assessments from the MLQE-PE corpus. Like the default model, it also provides scores ranging from 0 to 1.

If you intend to compare your results with papers published before 2022, it's likely that they used older evaluation models. In such cases, please refer to Unbabel/wmt20-comet-da and Unbabel/wmt20-comet-qe-da, which were the primary checkpoints used in previous versions (<2.0) of COMET.

## Large-Scale QE Models:

- [`Unbabel/wmt23-cometkiwi-da-xl`](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl) - This model is similar to `wmt22-cometkiwi-da` but utilizes the XLM-R XL encoder, which has 3.5 billion parameters and is multilingual in nature.

- [`Unbabel/wmt23-cometkiwi-da-xxl`](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl) - Similar to `wmt22-cometkiwi-da`, this model uses the XLM-R XXL encoder, boasting 10.7 billion parameters and offering a multilingual encoding capability.

Running these models requires 15GB and 44GB of GPU memory for XL and XXL variants, respectively, when using a batch size of 1.

## UniTE Models:

[UniTE Metric](https://aclanthology.org/2022.acl-long.558/) was developed by the NLP2CT Lab at the University of Macau and Alibaba Group, and all credits should be attributed to these groups. COMET framework fully supports running UniTE and thus we made the original UniTE-MUP checkpoint available in Hugging Face Hub. Additionally, we also trained our own UniTE model using the same data as `wmt22-comet-da`. You can access both models here:

- [`Unbabel/unite-mup`](https://huggingface.co/Unbabel/unite-mup) - This is the original UniTE Metric proposed in the [UniTE: Unified Translation Evaluation](https://aclanthology.org/2022.acl-long.558/) paper.
- [`Unbabel/wmt22-unite-da`](https://huggingface.co/Unbabel/wmt22-unite-da) - This model was trained for our paper [(Rei et al., ACL 2023)](https://aclanthology.org/2023.acl-short.94/) and it uses the same data as [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) thus, the outputed scores are between 0 and 1.

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
