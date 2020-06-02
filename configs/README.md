### HTER Estimator Training:

From the project root dir: 

```bash
comet download -d qt21 --saving_path data/ 
comet train -f configs/xlmr-hter-estimator.yaml
```

### DARR Ranker Training:

From the project root dir: 

```bash
comet download -d wmt --saving_path data/ 
comet train -f configs/xlmr-da-ranker.yaml
```