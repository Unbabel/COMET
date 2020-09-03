## WMT Segment-level scripts:

In this folder you can find the scripts and all you need to reproduce our results and the results for the baselines.

## Data:
Please download the WMT data beforehand and save it in the folder `data`.

```bash
comet download -d wmt-metrics --saving_path ./
```

## COMET:

Download a pretrained model:

```bash
comet download -m wmt-large-da-estimator-1718 --saving_path ./
```

Test the model checkpoint inside:

```bash
python segment_level_comet.py --run_wmt{18,19} --checkpoint wmt-large-da-estimator-1718/_ckpt_epoch_1.ckpt --cuda
```

## BERTScores:

### Requirements:
```bash
pip install bert_score==0.3.2
```

### How to run:
To run BERTScores you must use the `segment_level_bertscore.py` script.

```bash
python segment_level_bertscore.py --run_wmt{18,19}
```

Optionally you can specify an encoder model such as XLM-R with the flag `--model_type=xlm-roberta-base` 
and the usage of the inverse-document-frequency with the boolean flag `--idf`

## Sentence BLEU and chrF:

### Requirements:

```bash
pip install sacrebleu
pip install sacremoses
```
### How to run:

```bash
python segment_level_baselines.py --run_wmt{18,19}
```

## BLEURT:

### Requirements:

```bash
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

To run BLEURT you need to download the checkpoint of the model that you want to use and unzip it, for example:

```bash
wget https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip .
unzip bleurt-large-512.zip

wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip .
unzip bleurt-base-128.zip
```

### How to run:

```bash
python segment_level_bleurt.py bleurt/bleurt-large-512
python segment_level_bleurt.py bleurt/bleurt-base-128
```


## Prism:

#### Download model:

```
wget http://data.statmt.org/prism/m39v1.tar
tar xf m39v1.tar
```

### How to run:

```bash
python segment_level_prism.py --run_wmt{18,19} --temperature 0.95
```