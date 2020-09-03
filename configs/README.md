### HTER Estimator Training:

From the project root dir: 

```bash
comet download -d qt21 --saving_path data/ 
comet train -f configs/xlmr/base/hter-estimator.yaml
```

### DARR Ranker Training:

From the project root dir: 

```bash
comet download -d wmt-metrics --saving_path data/ 
comet train -f configs/xlmr/base/da-estimator.yaml
```