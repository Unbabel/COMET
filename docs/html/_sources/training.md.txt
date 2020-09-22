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

| Argument | Default | Description |
| :--------- | :------ | :------ |
| `seed` | 3 | Training seed. | 
| `deterministic` | True | If true enables cudnn.deterministic. Might make your system slower, but ensures reproducibility. |
| `verbose` | False | Verbosity mode. |
| `early_stopping` | True | Enables early stopping. | 
| `save_top_k` | 1 | Sets how many checkpoints we want to save (keeping only the best ones). |
| `monitor` | Kendall | Metric to monitor during training. |
| `metric_mode` | max | 'min' or 'max' depending if we wish to maximize or minimize the metric. |
| `min_delta` | 0 | Sensitivity to the metric. |
| `patience` | 1 | Number of epochs without improvement before stopping training |
| `accumulate_grad_batches` | 1 | Gradient accumulation steps |
| `lr_finder` | False | Enables the learning rate finder described in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) |


### Base Model Configurations

| Argument | Default | Description |
| :--------- | :------ | :------ |
| `model` | `required` | Type of metric we want to train. Options: [`CometEstimator`, `CometRanker`, `QualityEstimator`] |
| `batch_size` | 8 | Batch size used to train the model. |
| `nr_frozen_epochs` | 0 | Number of epochs we keep the encoder frozen. |
| `keep_embeddings_frozen` | False | If set to True, keeps the embedding layer frozen during training. Usefull to save some GPU memory. |
| `optimizer` |  Adam | PyTorch Optimizer class name |
| `learning_rate` | 1e-05 | Learning rate to be used during training. |
| `scheduler` | constant | Learning Rate scheduler. Options: [`constant`, `linear_warmup`, `warmup_constant`]  |
| `warmup_steps` | None |Scheduler warmup steps.   | 
| `encoder_model` | XLMR | Encoder Model  to be used: Options: [`LASER`, `BERT`, `XLMR`]. | 
| `pretrained_model` | xlmr.base | pretrained model to be used e.g: xlmr.base vs xlmr.large (for LASER this is ignored) | 
| `pool` | avg | Pooling technique to create the sentence embeddings. Options: [`avg`, `avg+cls`, `max`, `cls`, `default`] |
| `load_weights` | False | Loads compatible weights from another checkpoint file. |
| `train_path` | `required` | Path to the training csv. |
| `val_path` | `required` | Path to the validation csv. |
| `test_path` | None | Path to the test csv. (Not used) |
| `loader_workers` | False | Number of workers for loading and preparing the batches. |

**Note:** The `Ranker` model requires no further configs.

### Estimator Specific Configurations

| Argument | Default | Description |
| :--------- | :------ | :------ |
| `encoder_learning_rate` | `required` | Learning rate used to fine-tune the encoder. Note that this is different from `learning_rate` config that will be used only for the top layer.  |
| `layerwise_decay` | 1.0 | Decay for the layer wise learning rates. If 1.0 no decay is applied. |
| `layer` | mix | Layer from the pretrained encoder that we wish to extract the word embeddings. If `mix` uses a layer-wise attention mechanism to combine different layers. |
| `scalar_mix_dropout` | mix | Sets the layer-wise dropout. Ignored if `layer != mix`. |
| `loss` | mse | `mse` for Mean Squared Error or `binary_xent`for Binary Cross Entropy. |
| `hidden_sizes` | 1536,768 | Hidden sizes of the different Feed-Forward layers on top. |
| `activations` | Tanh | Activation functions for the Feed-Forward on top. |
| `dropout` | 0.1 | Dropout used in the Feed-Forward on top. |
| `final_activation` | Sigmoid | Feed-Forward final activation function. If `False` the model outputs the logits |


