# Publicly available data for Metrics:

# Direct Assessments:

Every year the WMT News Translation task organizers collect thousands of quality annotations in the form of _Direct Assessments_. Most COMET models use this data for training.

In the table below you can find that data in an easy to use format:

| year | data | paper |
|:---: | :--: | :---: |
| 2017 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2017-da.tar.gz) | [Findings of the 2017 Conference on Machine Translation (WMT17)](https://aclanthology.org/W17-4717.pdf) |
| 2018 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2018-da.tar.gz) | [Findings of the 2018 Conference on Machine Translation (WMT18)](https://aclanthology.org/W18-6401.pdf) |
| 2019 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2019-da.tar.gz) | [Findings of the 2019 Conference on Machine Translation (WMT19)](https://aclanthology.org/W19-5301.pdf) |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2020-da.tar.gz) | [Findings of the 2020 Conference on Machine Translation (WMT20)](https://aclanthology.org/2020.wmt-1.1.pdf) |
| 2021 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2021-da.tar.gz) | [Findings of the 2021 Conference on Machine Translation (WMT21)](https://aclanthology.org/2021.wmt-1.1.pdf) |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-da.tar.gz) | [Findings of the 2022 Conference on Machine Translation (WMT22)](https://aclanthology.org/2022.wmt-1.1.pdf) |

Another large source of DA annotations is the [MLQE-PE corpus](https://aclanthology.org/2022.lrec-1.530.pdf) that is typically used for quality estimation shared tasks [(Specia et al. 2020](https://aclanthology.org/2020.wmt-1.79.pdf)[, 2021](https://aclanthology.org/2021.wmt-1.71.pdf)[; Zerva et al. 2022)](https://aclanthology.org/2022.wmt-1.3.pdf).

You can download MLQE-PE by using the following [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/mlqe-pe.tar.gz).

### DA Relative Ranks
Before 2021 the WMT Metrics shared task used relative ranks to evaluate metrics. 

Relative ranks can be created when we have at least two DA scores for translations of the same source input, by converting those DA scores into a relative ranking judgement, if the difference in DA scores allows conclusion that one translation is better than the other (usually atleast 25 points). 

To make it easier to replicate results from previous Metrics shared tasks (2017-2020) you can find the preprocessed DA relative ranks in the table below:

| year | relative ranks | paper |
|:---: | :--: | :---: |
| 2017 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-daRR.csv.tar.gz) | [Results of the WMT17 Metrics Shared Task](https://statmt.org/wmt17/pdf/WMT55.pdf) |
| 2018 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-daRR.csv.tar.gz) | [Results of the WMT18 Metrics Shared Task](https://statmt.org/wmt18/pdf/WMT078.pdf) |
| 2019 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2019-daRR.csv.tar.gz) | [Results of the WMT19 Metrics Shared Task](https://statmt.org/wmt19/pdf/53/WMT02.pdf) |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-daRR.csv.tar.gz) | [Results of the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.77.pdf) |

# Multidimensional Quality Metrics:

Since 2021 the WMT Metrics task decided to perform they own expert-based evaluation based on _Multidimensional Quality Metrics (MQM)_ framework. In the table below you can find MQM annotations from previous years.

| year | data | paper |
|:---: | :--: | :---: |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2020-mqm.tar.gz) | [A Large-Scale Study of Human Evaluation for Machine Translation](https://aclanthology.org/2021.tacl-1.87.pdf) |
| 2021 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2021-mqm.tar.gz) | [Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-mqm.tar.gz) | [Results of the WMT22 Metrics Shared Task](https://aclanthology.org/2022.wmt-1.2.pdf) |

**Note:** You can find the original MQM data [here](https://github.com/google/wmt-mqm-human-evaluation).

# Direct Assessment + Scalar Quality Metric:

In 2022, several changes were made to the annotation procedure used in the WMT Translation task. In contrast to the standard DA (sliding scale from 0-100) used in previous years, in 2022 annotators performed DA+SQM (Direct Assessment + Scalar Quality Metric). In DA+SQM, the annotators still provide a raw score between 0 and 100, but also are presented with seven labeled tick marks. DA+SQM helps to stabilize scores across annotators (as compared to DA).

| year | data | paper |
|:---: | :--: | :---: |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-sqm.tar.gz) | [Findings of the 2022 Conference on Machine Translation (WMT22)](https://aclanthology.org/2022.wmt-1.1.pdf) |
