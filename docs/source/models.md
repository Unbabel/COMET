## COMET Metrics

COMET models can be optimized towards different kinds of human judgements (for example HTER or DA's). Because of we provide a list of different Metrics that you can use to test you systems:

| Model              |               Description                        |
| :--------------------- | :------------------------------------------------ |
| ↑`wmt-large-da-estimator-1719` | **RECOMMENDED:** Estimator model build on top of XLM-R (large) trained on DA from WMT17, WMT18 and WMT19 |
| ↑`wmt-base-da-estimator-1719` | Estimator model build on top of XLM-R (base) trained on DA from WMT17, WMT18 and WMT19 |
| ↓`wmt-large-hter-estimator` | Estimator model build on top of XLM-R (large) trained to regress on HTER. |
| ↓`wmt-base-hter-estimator` | Estimator model build on top of XLM-R (base) trained to regress on HTER. |
| ↑`emnlp-base-da-ranker`    | Translation ranking model that uses XLM-R to encode sentences. This model was trained with WMT17 and WMT18 Direct Assessments Relative Ranks (DARR). |

The first four models (`wmt-*`) where trained and tested for the WMT2020 shared task, thus they were only introduced in our submission to the shared task (paper still under-review)

**NOTE:** Even when regressing on the same Human Judgement scores between metrics are not comparable (e.g. scores from a large and a base model are not comparable even when trained on the same type of judgements)! Please make sure you use the same metric when comparing 2 systems!

Also, since HTER measures the amount of edits we needed to correct an MT hypothesis, models regressing on HTER produce low scores for better systems.