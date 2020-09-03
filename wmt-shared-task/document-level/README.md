## WMT Document-level scripts:

## COMET:

Download a pretrained model:

```bash
comet download -d doc-wmt19 --saving_path ./
comet download -m wmt-large-da-estimator-1718 --saving_path ./
```

Test the model checkpoint inside:

```bash
python document_level.py --checkpoint wmt-large-da-estimator-1718/_ckpt_epoch_1.ckpt
```