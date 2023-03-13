# Frequently Asked Questions

Since we released COMET we have received several questions related to interpretabilty of the scores and usage. In this section we try to address these questions the best we can! 

##### Interpreting Scores:

When using COMET to evaluate machine translation, it's important to understand how to interpret the scores it produces.

In general, COMET models are trained to predict quality scores for translations. These scores are typically normalized using a [z-score transformation](https://simplypsychology.org/z-score.html) to account for individual differences among annotators. While the raw score itself does not have a direct interpretation, it is useful for ranking translations and systems according to their quality.

However, for the latest COMET models like [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da), we have introduced a new training approach that scales the scores between 0 and 1. This makes it easier to interpret the scores: a score close to 1 indicates a high-quality translation, while a score close to 0 indicates a translation that is no better than random chance.

It's worth noting that when using COMET to compare the performance of two different translation systems, it's important to run the `comet-compare` command to obtain statistical significance measures. This command compares the output of two systems using a statistical hypothesis test, providing an estimate of the probability that the observed difference in scores between the systems is due to chance. This is an important step to ensure that any differences in scores between systems are statistically significant.

Overall, the added interpretability of scores in the latest COMET models, combined with the ability to assess statistical significance between systems using `comet-compare`, make COMET a valuable tool for evaluating machine translation.

##### Which COMET model should I use?

**For general purpose MT evaluation** we recommend you to use `Unbabel/wmt22-comet-da`. This is the most _stable_ model we have. It is an improved version of our previous model `Unbabel/wmt20-comet-da`. 

**For evaluating models without a reference** we recommend the `Unbabel/wmt20-comet-qe-da` for higher correlations with DA, and to [download `wmt21-comet-qe-mqm`](https://github.com/Unbabel/COMET/blob/master/MODELS.md) for higher correlations with MQM. 

##### Where can I find the data used to train COMET models?

###### Direct Assessments

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

###### Direct Assessments: Relative Ranks
Before 2021 the WMT Metrics shared task used relative ranks to evaluate metrics. 

Relative ranks can be created when we have at least two DA scores for translations of the same source input, by converting those DA scores into a relative ranking judgement, if the difference in DA scores allows conclusion that one translation is better than the other (usually atleast 25 points). 

To make it easier to replicate results from previous Metrics shared tasks (2017-2020) you can find the preprocessed DA relative ranks in the table below:

| year | relative ranks | paper |
|:---: | :--: | :---: |
| 2017 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-daRR.csv.tar.gz) | [Results of the WMT17 Metrics Shared Task](https://statmt.org/wmt17/pdf/WMT55.pdf) |
| 2018 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-daRR.csv.tar.gz) | [Results of the WMT18 Metrics Shared Task](https://statmt.org/wmt18/pdf/WMT078.pdf) |
| 2019 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2019-daRR.csv.tar.gz) | [Results of the WMT19 Metrics Shared Task](https://statmt.org/wmt19/pdf/53/WMT02.pdf) |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-daRR.csv.tar.gz) | [Results of the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.77.pdf) |

###### Direct Assessment + Scalar Quality Metric:

In 2022, several changes were made to the annotation procedure used in the WMT Translation task. In contrast to the standard DA (sliding scale from 0-100) used in previous years, in 2022 annotators performed DA+SQM (Direct Assessment + Scalar Quality Metric). In DA+SQM, the annotators still provide a raw score between 0 and 100, but also are presented with seven labeled tick marks. DA+SQM helps to stabilize scores across annotators (as compared to DA).

| year | data | paper |
|:---: | :--: | :---: |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-sqm.tar.gz) | [Findings of the 2022 Conference on Machine Translation (WMT22)](https://aclanthology.org/2022.wmt-1.1.pdf) |

###### Multidimensional Quality Metrics:

Since 2021 the WMT Metrics task decided to perform they own expert-based evaluation based on _Multidimensional Quality Metrics (MQM)_ framework. In the table below you can find MQM annotations from previous years.

| year | data | paper |
|:---: | :--: | :---: |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2020-mqm.tar.gz) | [A Large-Scale Study of Human Evaluation for Machine Translation](https://aclanthology.org/2021.tacl-1.87.pdf) |
| 2021 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2021-mqm.tar.gz) | [Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-mqm.tar.gz) | [Results of the WMT22 Metrics Shared Task](https://aclanthology.org/2022.wmt-1.2.pdf) |

**Note:** You can find the original MQM data [here](https://github.com/google/wmt-mqm-human-evaluation).