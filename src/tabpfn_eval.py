import argparse
import os
import json
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helper import get_few_shot_from_csv
from baselines.utils_baselines import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--numshot", type=str, required=True)  # Changed to str to handle 'all'
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset_name = args.dataset
    # Convert numshot to int if it's not 'all'
    num_shot = args.numshot if args.numshot.lower() == 'all' else int(args.numshot)
    seed = args.seed
    device = args.device

    set_seed(seed)

    # Output directory
    save_dir = f"eval_res/tabpfn/{dataset_name}/{num_shot}_shot/"
    results_path = os.path.join(save_dir, "eval_metrics.json")
    if os.path.exists(results_path):
        print(f"Output already exists at {save_dir}. Skipping run.")
        return

    # Few-shot sample from train using helper
    X_few, y_few, X_train, y_train, X_test, y_test  = get_few_shot_from_csv(dataset_name, num_shot, seed)
    print(f"Sampled {len(y_few)} few-shot examples: {np.bincount(y_few.to_numpy())}")

    # Load synthetic test set
    synth_path = f"dataset/{dataset_name}/{dataset_name}_synthetic.csv"
    synth_df = pd.read_csv(synth_path)
    X_synth = synth_df.drop(columns=["class"]).values
    y_synth = synth_df["class"].values

    # Train TabPFN
    clf = TabPFNClassifier(device=device)
    clf.fit(X_few.values, y_few.values)

    # Inference on synthetic data
    y_pred = clf.predict(X_synth)

    # Inference on test data
    y_test_pred = clf.predict(X_test.values)
    
    # Evaluate performance on test set
    accuracy = accuracy_score(y_test.values, y_test_pred)
    
    # Check if predictions are all the same class (which makes ROC-AUC undefined)
    if len(np.unique(y_test_pred)) == 1:
        # All predictions are the same class, set AUC to 50%
        auc = 0.5
        print(f"Warning: All test predictions are class {y_test_pred[0]}, setting AUC to 0.5")
    else:
        auc = roc_auc_score(y_test.values, clf.predict_proba(X_test.values)[:, 1])
    
    f1 = f1_score(y_test.values, y_test_pred)
    
    # Calculate TPR and FPR from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test.values, y_test_pred).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity/Recall)
    fpr = fp / (fp + tn)  # False Positive Rate
    
    # Prepare evaluation metrics
    eval_metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'f1': float(f1),
        'tpr': float(tpr),
        'fpr': float(fpr)
    }

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(save_dir, "X_synth.npy"), X_synth)
    np.save(os.path.join(save_dir, "y_test_pred.npy"), y_test_pred)
    
    # Save evaluation metrics
    with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    
    with open(os.path.join(save_dir, "commandline_args.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Saved inference results to {save_dir}")
    print(f"Test set performance: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")

if __name__ == "__main__":
    main()