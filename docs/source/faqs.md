# Frequently Asked Questions

Since we released COMET we have received several questions related to interpretabilty of the scores and usage. In this section we try to address these questions the best we can! 

##### Is there a theoretical range of values for the COMET regressor?

Before we dig deeper into details about COMET scores I would like to clarify something: 

_Absolute scores via automatic metrics are meaningless (what does 31 BLEU mean without context? it can be both awesome score for News EN-Finnish or really bad score for EN-French), and pretrained metrics only amplify it by using different scales for different languages and especially different domains._ 

Check [[Kocmi et al. 2021]](https://aclanthology.org/2021.wmt-1.57/) and our discussion here: [#18](https://github.com/Unbabel/COMET/issues/18)

Most COMET models are trained to regress on a specific quality assessment and in most cases we normalize those quality scores to obtain a [z-score](https://www.simplypsychology.org/z-score.html). This means that theoretically our models are unbounded! The score itself has no direct interpretation but they correctly rank translations and systems according to their quality!

Also, depending on the data that they were used to train, different models might have different score ranges. We observed that most scores for our `wmt20-comet-da` fall between -1.5 and 1, while our `wmt21-comet-qe-da` produces scores between -0.2 and 0.2.

![WMT21 Distribution](/_static/img/distributions-WMT21.png)

##### Which COMET model should I use?

**For general purpose MT evaluation** we recommend you to use `wmt20-comet-da`. This is the most _stable_ model we have. It has been studied by several different authors and so far it seems to correlate well with different types of human assessments in different domains and languages.

Nonetheless, for the WMT 2021 shared task we developed several models that predict _Multidimensional Quality Metrics (MQM)_ rather than DA's. The MQM models have similar performance in terms of correlation with _Direct Assessments_ and higher correlation with MQM annotations. Use `wmt21-comet-mqm` if you wish to have a proxy for MQM.

**For evaluating models without a reference** we recommend the models trained for our participation in the WMT21 shared task, namely: `wmt21-comet-qe-da` for higher correlations with DA, and `wmt21-comet-qe-mqm` for higher correlations with MQM.

##### Where can I find the data used to train COMET models?

###### Direct Assessments
Every year the WMT News Translation task organizers collect thousands of quality annotations in the form of _Direct Assessments_. Most COMET models use that data either in the form of z-scores or in the form of relative-ranks.

I'll leave here a table with links for that data.

| year | DA | relative ranks | paper |
|:---: | :--: | :---: | :---: |
| 2017 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-da.csv.tar.gz) | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-daRR.csv.tar.gz) | [Results of the WMT17 Metrics Shared Task](https://statmt.org/wmt17/pdf/WMT55.pdf) |
| 2018 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-daRR.csv.tar.gz) |[Results of the WMT18 Metrics Shared Task](https://statmt.org/wmt18/pdf/WMT078.pdf) |
| 2019 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2019-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2019-daRR.csv.tar.gz) |[Results of the WMT19 Metrics Shared Task](https://statmt.org/wmt19/pdf/53/WMT02.pdf) |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-daRR.csv.tar.gz) |[Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |
| 2021 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2021-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2021-daRR.csv.tar.gz) |[Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |


###### Multidimensional Quality Metrics

In the last editions of the WMT Metrics shared task the organizers decided to run evaluation of MT based on _Multidimensional Quality Metrics (MQM)_ based on findings that crowd-sourced _Direct Assessments_ are noisy and do not correlate well with annotations done by experts [[Freitag, et al. 2021]](https://aclanthology.org/2021.tacl-1.87.pdf).

| year | MQM | paper |
|:---: | :--: | :---:|
| 2020 | [🔗](https://github.com/google/wmt-mqm-human-evaluation) | [A Large-Scale Study of Human Evaluation for Machine Translation](https://aclanthology.org/2021.tacl-1.87.pdf) |
| 2021 | [🔗](https://github.com/google/wmt-mqm-human-evaluation) | [Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |

**Please cite the corresponding papers if you use any of these data!**
