## Available Models

COMET models can optimize different types of human judgements and because of that we provide a list of pretrained COMET models.

| Model              |               Description                        |
| --------------------- | ------------------------------------------------ |
| `wmt-large-da-estimator-1719` | **RECOMMENDED:** Estimator model build on top of XLM-R (large) trained on DA from WMT17, WMT18 and WMT19 |
| `wmt-base-da-estimator-1719` | Estimator model build on top of XLM-R (base) trained on DA from WMT17, WMT18 and WMT19 |
| `wmt-large-da-estimator-1718` | Estimator model build on top of XLM-R (large) trained on DA from WMT17 and WMT18 |
| `wmt-base-da-estimator-1718` | Estimator model build on top of XLM-R (base) trained on DA from WMT17 and WMT18 |
| `wmt-large-hter-estimator` | Estimator model build on top of XLM-R (large) trained to regress on HTER. |
| `wmt-base-hter-estimator` | Estimator model build on top of XLM-R (base) trained to regress on HTER. |
| `emnlp-base-da-ranker`      | Translation ranking model that uses XLM-R to encode sentences. This model was trained with WMT17 and WMT18 Direct Assessments Relative Ranks (DARR). |

**NOTE:** Even when regressing on the same Human Judgement scores between models are not comparable! If your system X achieves 0.6 with `wmt-large-da-estimator-1719` and your system Y achieves 0.7 with `wmt-base-da-estimator-1719` it doesn't mean that your system Y is better than X!

Also, for all the HTER models having a low score is better