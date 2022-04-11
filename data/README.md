# Publicly available data for Metrics:

## Direct Assessments:

Every year the WMT News Translation task organizers collect thousands of quality annotations in the form of _Direct Assessments_. Most COMET models use that data either in the form of z-scores or in the form of relative-ranks.

I'll leave here a table with links for that data.

| year | DA | relative ranks | paper |
|:---: | :--: | :---: | :---: |
| 2017 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-da.csv.tar.gz) | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-daRR.csv.tar.gz) | [Results of the WMT17 Metrics Shared Task](https://statmt.org/wmt17/pdf/WMT55.pdf) |
| 2018 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-daRR.csv.tar.gz) |[Results of the WMT18 Metrics Shared Task](https://statmt.org/wmt18/pdf/WMT078.pdf) |
| 2019 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2019-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2019-daRR.csv.tar.gz) |[Results of the WMT19 Metrics Shared Task](https://statmt.org/wmt19/pdf/53/WMT02.pdf) |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-daRR.csv.tar.gz) |[Results of the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.77.pdf) |
| 2021 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2021-da.csv.tar.gz) |  [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2021-daRR.csv.tar.gz) |[Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |


## Multidimensional Quality Metrics

In the last editions of the WMT Metrics shared task the organizers decided to run evaluation of MT based on _Multidimensional Quality Metrics (MQM)_ based on findings that crowd-sourced _Direct Assessments_ are noisy and do not correlate well with annotations done by experts [[Freitag, et al. 2021]](https://aclanthology.org/2021.tacl-1.87.pdf).

| year | MQM | paper |
|:---: | :--: | :---:|
| 2020 | [🔗](https://github.com/google/wmt-mqm-human-evaluation) | [A Large-Scale Study of Human Evaluation for Machine Translation](https://aclanthology.org/2021.tacl-1.87.pdf) |
| 2021 | [🔗](https://github.com/google/wmt-mqm-human-evaluation) | [Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |
