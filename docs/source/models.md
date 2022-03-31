# COMET Metrics

Since COMET was released we have been training and releasing different models. In this page we will try to briefly explain the underlying differences and point you to the papers that used them.

## Model Architectures:

All COMET metrics follow one of the following architectures:

[![Model Architectures](/_static/img/architectures.jpg)](https://raw.githubusercontent.com/Unbabel/COMET/docs-config/docs/source/_static/img/architectures.jpg)

1) Regression Metric (left diagram): This is the architecture that most models use. This model is trained on a regression task using source, MT and reference.
2) Ranking Metric (middle diagram): Models that follow this architecture are trained in a Translation Ranking Task using a [Triple Margin Loss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html). This means that the model will learn to optimize the embedding space to encode _good_ translations closer to the anchors (source and reference) while pushing _bad_ translations away.
3) Referenceless Metric (right diagram): This architecture resembles architecture 1) but **it does not use the reference translation!** This is purely a Quality Estimation system.

## Available Metrics:

| Model Name | Architecture | Short Description | 
| :-------- | :----- | --------------- |
|  <td colspan=3> [**WMT20 Metrics** ](#wmt20-comet-metrics)  |
| `wmt20-comet-da` | Regression Metric | Our best performing metric from WMT20. | 
| `wmt20-comet-qe-da` | Referenceless Metric | Referenceless metric trained to predict DA's from WMT17 to WMT19. This was the best performing _QE-as-a-metric_ from WMT20 shared task |
| `wmt20-comet-qe-da-v2` | Referenceless Metric | Reimplementation of the above model without a bounded output. |
| `emnlp20-comet-rank` | Ranking Metric | Translation Ranking model trained with DARR ranging from WMT17 to WMT19. |
|  <td colspan=3>  [**WMT21 Metrics**](#wmt21-comet-metrics)  |
| `wmt21-comet-mqm` | Regression Metric | Our best performing metric from WMT21 MQM benchmark. This metric was pretrained on DA's and adapted to MQM by finetuning on [Freitag et al, 2021](https://aclanthology.org/2021.tacl-1.87/) data. |
| `wmt21-comet-qe-mqm` | Referenceless Metric | Referenceless version of the `wmt21-comet-mqm` metric. This was the best performing _QE-as-a-metric_ from WMT21 shared task |
| `wmt21-cometinho-mqm` | Regression Metric | Trained with the same data as `wmt21-comet-mqm` but with a much smaller encoder model (MiniLMV2) | 
| `wmt21-comet-da` | Regression Metric | Regression metric trained on DA's from WMT15 to WMT20. |
| `wmt21-comet-qe-da` | Referenceless Metric | Referenceless metric trained on DA's from WMT15 to WMT20. |
| `wmt21-cometinho-da` | Regression Metric | Regression metric trained on DA's from WMT15 to WMT20 using a light-weight encoder model. |

Our _default_ metric is the <code>wmt20-comet-da</code>.

## WMT20 COMET Metrics

For [our participation in the WMT20 shared task](https://aclanthology.org/2020.wmt-1.101.pdf) we developed several models. We later release our best regression model, our best refereceless model and our best ranking model.

- <code>wmt20-comet-da</code>: Our best regression metric that year. It is trained to predict _Direct Assessments_ using data ranging 2017 to 2019. (Same as <code>wmt-large-da-estimator-1719</code> from previous versions.)
- <code>wmt20-comet-qe-da</code>: This was the model we used to participate in the QE-as-a-metric subtask. It is trained to predict _Direct Assessments_ using data ranging 2017 to 2019. This is a Referenceless Metric meaning that it uses **source and translation only!** (Same as <code>wmt-large-qe-estimator-1719</code> from previous versions.)
- <code>emnlp20-comet-rank</code>: reimplementation of the ranking model from [Rei et al. 2020](https://aclanthology.org/2020.emnlp-main.213/) with _Direct Assessment Relative Ranks (DARR)_ ranging 2017 to 2019. (Same as <code>wmt-large-da-estimator-1719</code> from previous versions.)

Our **Primary Metric** is <code>wmt20-comet-da</code>. This was one of the best performing metrics in the WMT20 shared task [[Mathur et al, 2020]](https://aclanthology.org/2020.wmt-1.77/) and the best performing metric in the large-scale study on metrics performed by Microsoft Research [[kocmi et al, 2021]](https://arxiv.org/abs/2107.10821).

- <code>wmt20-comet-qe-da-v2</code>: The Referenceless model develop to the WMT20 shared task (<code>wmt20-comet-qe-da</code>) was trained with a sigmoid activation at the end. This was intended to improve interpretability but after the shared task we noted that this model predicts a lot of scores close to 0 (sometimes even with acceptable translations). This does not affect correlations of system-decisions but it **makes it harder to differentiate between low quality translations.** For that reason we decided to retrain the <code>wmt20-comet-qe-da</code> model without the Sigmoid activation. **The <code>wmt20-comet-qe-da-v2</code> is expected to perform as well as the <code>wmt20-comet-qe-da</code> but without producing as many 0's.**


## WMT21 COMET Metrics

### MQM Metrics

In  [our participation to the WMT21 shared task](https://aclanthology.org/2021.wmt-1.111.pdf) we steer COMET towards higher correlations with MQM. We do so by first pre-training on _Direct Assessments_ and then fine-tuning on z-normalized MQM scores.

- <code>wmt21-comet-mqm</code>: This model was pre-trained on _Direct Assessments_ from WMT15 to WMT20 and then fine-tuned on MQM z-scores from [Freitag et al, 2021 (MQM)](https://aclanthology.org/2021.tacl-1.87/). This model was one of the best performing metrics that year [[Freitag et al. 2021 (WMT21)]](https://aclanthology.org/2021.wmt-1.73/).
- <code>wmt21-comet-qe-mqm</code>: Reference-free version of <code>wmt21-comet-mqm</code>. This model was the best performing _QE-as-a-metric_ that year. [[Freitag et al. 2021 (WMT21)]](https://aclanthology.org/2021.wmt-1.73/).
- <code>wmt21-cometinho-mqm</code>: Additionally, we introduced Cometinho, a light-weight COMET model that is built on top of a smaller XLM-R encoder ([MiniLMV2](https://aclanthology.org/2021.findings-acl.188/)). This model is NOT a distilled COMET model... It is simply built on top of a smaller pretrained encoder.

**NOTE:** One thing we noticed in these MQM Models is that the variance between predicted scores is lower than models trained only DA's. The range of scores produced by these models is very narrow.

### DA Metrics
Along with the MQM models we release the checkpoints trained only on DA's data. 

- <code>wmt21-comet-da</code>: Regression Model trained on _Direct Assessments_ from WMT15 to WMT20. 
- <code>wmt21-comet-qe-da</code>: Referenceless Model trained on _Direct Assessments_ from WMT15 to WMT20. 
- <code>wmt21-cometinho-da</code>: Regression Model trained on top of a smaller [MiniLMV2](https://aclanthology.org/2021.findings-acl.188/) encoder using _Direct Assessments_ from WMT15 to WMT20 

## Benchmark

TODO
