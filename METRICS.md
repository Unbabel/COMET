# WMT20 COMET Metrics:

Our **Primary Metrics** are the models described in our [participation to the WMT20 Metrics Shared task](https://aclanthology.org/2020.wmt-1.101.pdf):
- `wmt20-comet-da`: This model was trained to predict _Direct Assessments_ from WMT17 to WMT19 using source, translation and reference. (Same as `wmt-large-da-estimator-1719` from previous versions.)
- `wmt20-comet-qe-da`: This model was trained to predict _Direct Assessments_ from WMT17 to WMT19 using **source and translation only**! Also, this model is bounded between 0 and 1 which improves interpretability in comparison with the previous model.

These two models were the best performing metrics in the large-scale metrics study performed by Microsoft Research [kocmi et al, 2021](https://arxiv.org/abs/2107.10821) which validates our findings.

# EMNLP20 Metric:

In our [initial COMET release](https://aclanthology.org/2020.emnlp-main.213/) we developed a Translation Ranking Model based on daRR from previous WMT shared tasks. This model achieves **some of the highest Kendall tau-like correlations on the WMT19 daRR benchmark** but does not perform as well on later WMT benchmarks, specially those using MQM annotations.


# WMT21 COMET Metrics:

In our participation to the WMT21 shared task we steer COMET towards higher correlations with MQM. We do so by first pre-training on _Direct Assessments_ and then fine-tuning on z-normalized MQM scores.
- `wmt21-comet-mqm`: This model was pre-trained on _Direct Assessments_ from WMT15 to WMT20 and then fine-tuned on MQM scores from [freitag et al, 2021](https://arxiv.org/pdf/2104.14478.pdf)
- `wmt21-comet-qe-mqm`: Reference-free version of `wmt21-comet-mqm`.

Additionally, we present COMETinho (`wmt21-cometinho-da`), a light-weight COMET model that is 19x faster on CPU than the original model.

**NOTE:** One thing we noticed in the WMT21 Models is that the variance between predicted scores is lower than previous models which make their predictions look very similar to each other even if the overall correlations with human judgments improve and the system rankings is correct.
