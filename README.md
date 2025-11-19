This repository contains the implementation for the paper **Leveraging Foundation Models in Healthcare: A Distillation Approach to Interpretable Clinical Prediction** submitted to the AAAI 2026 Workshop.

All the experimental results can be found in `baseline_results.csv` and `distillation_results.csv`.

## Environment Setup

**Install required packages:**
```bash
pip install -r env.txt
```


## Complete workflow

This section provides a step-by-step guide to reproduce all results. All the distillation and baseline results can be found at `eval_res/`. 

### Step 1: Dataset Preprocessing

All source datasets already reside under `dataset/`. Run `python data_preprocessing.py` to clean and standardize the clinical files before training. The script writes the processed outputs in place, yielding a layout like below:

```
dataset/
├── breastcancer/
│   ├── all_finalb.sas7bdat           # Original data
│   ├── breastcancer.csv              # Preprocessed data
│   └── breastcancer.info             # Metadata
```


### Step 2: Generate Synthetic Data

To generate synthetic data for all datasets:

```bash
python src/synthetic_data_generator.py
```

**Output Structure:**
After running the script, each dataset folder will contain both original and synthetic data:
```
dataset/
├── breastcancer/
│   ├── breastcancer.csv              # Preprocessed data
│   ├── breastcancer_synthetic.csv    # Generated synthetic data
│   └── breastcancer.info             # Metadata
```

### Step 3: Inference

```bash
bash bin/tabpfn.sh

bash bin/tabm.sh
```

**Expected Output Structure:**
After running both scripts, your `eval_res/` directory will contain:
```
eval_res/
├── tabm/
│   ├── breastcancer/
│   │   ├── 4_shot/
│   │   │   └── commandline_args.txt
|   |   |   └── X_synth.npy
|   |   |   └── y_pred.npy
│   │   ├── 16_shot/
│   │   └── ... (other shot configs)
│   └── ... (other datasets)
├── tabpfn/
│   ├── breastcancer/
│   │   ├── 4_shot/
│   │   │   └── commandline_args.txt
```


### Step 4: Distillation

After obtaining the parent models' results, distillation is performed with 5 student interpretable ML models: TTnet, Decision Tree, and Logistic Rule Regression (from the [aix360](https://aix360.readthedocs.io/en/latest/) toolkit).

```bash
python distillation_general.py
```

**Expected Output Structure:**
After running both scripts, your `eval_res/` directory will contain:
```
eval_res/
├── tabpfn/
│   ├── breastcancer/
│   │   ├── 4_shot/
│   |   │   ├── decision_tree/
│   |   │   |   └── results.txt
│   |   │   └── ... (other student models)
│   │   │   └── commandline_args.txt
|   |   |   └── X_synth.npy
|   |   |   └── y_pred.npy
│   │   ├── 16_shot/
│   │   └── ... (other shot configs)
│   └── ... (other datasets)
```


### Step 5: Generate baseline results

Next, we perform few-shot classification of the tabular datasets with the 4 interpretable ML models without distilling the parent models.

```bash
python baseline_general.py
```


### Step 6: Compile results and visualize

```bash
python compile_results.py

python visualize.py
```

Running `compile_results.py` will automatically generate `wilcoxon_test.txt` showing the Wilcoxon signed-rank statistical test mentioned in the paper with the obtained data.