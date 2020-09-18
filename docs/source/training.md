# Train your own Metric

To train our models we rely on [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/0.8.4/) Library. This means that all our models are [Lightning Modules](https://pytorch-lightning.readthedocs.io/en/0.8.4/lightning-module.html).

To train a new metric we just need to run 1 command:

```bash
comet train -f {my_configs}.yaml
```

This will setup a [Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/0.8.4/trainer.html) and fit your module accordingly.
## Data Format
To train your metric we expect your data to be a csv with the following columns:
- `src`: The source segment.
- `mt`: The machine translation hypothesis.
- `ref`: The reference segment.
- `score`: The human judgment score.

Example:

| src | mt | ref | score |
| :---------: | :------: | :------: | :------: |
| Hello world! | Oi mundo. | Olá mundo! | 0.5 |
| This is a sample | este é um exemplo | isto é um exemplo! | 0.8 |

## Training flags

### Lightning Trainer Configurations

| Argument | Description | Default |
| :---------: | :------: | :------: |
| `seed` | Training seed. | 3 |
| `deterministic` | If true enables cudnn.deterministic. Might make your system slower, but ensures reproducibility. | True |
| `verbose` | Verbosity mode. | False |
| `overfit_batches` | Uses this much data of the training set. If nonzero, will use the same training set for validation and testing. If the training dataloaders have shuffle=True, Lightning will automatically disable it. | 0 |
| `early_stopping` |  Enables early stopping. | True |
| `save_top_k` | Sets how many checkpoints we want to save (keeping only the best ones). | 1 |
| `monitor` | Metric to monitor during training. | Kendall |
| `metric_mode` | 'min' or 'max' depending if we wish to maximize or minimize the metric. | max |
| `min_delta` | Sensitivity to the metric. | 0 |
| `patience` | Number of epochs without improvement before stopping training | 1 |
| `accumulate_grad_batches` | Gradient accumulation steps | 1 |
| `lr_finder` | Enables the learning rate finder described in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) | False |


### Base Model Configurations

| Argument | Description | Default |
| :--------- | :------ | :------ |
| `model` | Type of metric we want to train. Options: [`CometEstimator`, `CometRanker`, `QualityEstimator`] | None |
| `batch_size` | Batch size used to train the model. | 8 |
| `nr_frozen_epochs` | Number of epochs we keep the encoder frozen. | 0 |
| `keep_embeddings_frozen` | If set to True, keeps the embedding layer frozen during training. Usefull to save some GPU memory. | False |
| `optimizer` |  PyTorch Optimizer class name | Adam |
| `learning_rate` | Learning rate to be used during training. | 1e-05 |
| `scheduler` | Learning Rate scheduler. Options: [`constant`, `linear_warmup`, `warmup_constant`]  | constant |
| `warmup_steps` | Scheduler warmup steps.   | None |
| `encoder_model` | Encoder Model  to be used: Options: [`LASER`, `BERT`, `XLMR`]. | XLMR |
| `pretrained_model` | pretrained model to be used e.g: xlmr.base vs xlmr.large (for LASER this is ignored) | xlmr.base |
| `pool` | Pooling technique to create the sentence embeddings. Options: [`avg`, `avg+cls`, `max`, `cls`, `default`] | avg |
| `load_weights` | Loads compatible weights from another checkpoint file. | False |
| `train_path` | Path to the training csv. | None |
| `val_path` | Path to the validation csv. | None |
| `test_path` | Path to the test csv. (Not used) | None |
| `loader_workers` | Number of workers for loading and preparing the batches. | False |

**Note:** The `Ranker` model requires no further configs.

### Estimator Specific Configurations

| Argument | Description | Default |
| :---------: | :------: | :------: |
| `encoder_learning_rate` | Learning rate used to fine-tune the encoder. Note that this is different from `learning_rate` config that will be used only for the top layer.  | None |
| `layerwise_decay` | Decay for the layer wise learning rates. If 1.0 no decay is applied. | 1.0 |
| `layer` | Layer from the pretrained encoder that we wish to extract the word embeddings. If `mix` uses a layer-wise attention mechanism to combine different layers. | mix |
| `scalar_mix_dropout` | Sets the layer-wise dropout. Ignored if `layer != mix`. | mix |
| `loss` | `mse` for Mean Squared Error or `binary_xent`for Binary Cross Entropy. | mse |
| `hidden_sizes` | Hidden sizes of the different Feed-Forward layers on top. | 1536,768 |
| `activations` | Activation functions for the Feed-Forward on top. | Tanh |
| `dropout` | Dropout used in the Feed-Forward on top. | Tanh |
| `final_activation` | Feed-Forward final activation function. If `False` the model outputs the logits | Sigmoid |


