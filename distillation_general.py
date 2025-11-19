import os
import numpy as np
import torch
import pandas as pd
import warnings
import random
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform

# Suppress sklearn FutureWarning about sparse parameter
warnings.filterwarnings("ignore", message=".*sparse.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*sparse_output.*", category=FutureWarning)

from aix360.algorithms.rbm import FeatureBinarizer, LogisticRuleRegression

from src.utils.helper import get_few_shot_from_csv
from baselines.utils_baselines import set_seed
from src.ttnet.ttnet_wrapper import TTnetStudentModel, TTNetPreprocessor 
from rrl.utils import RRLWrapper

SEEDS = [0, 1, 6, 7, 8]

def main(dataset_name, numshot, parent_model, student_model_name, device='cpu', gpu_id=None):
    # Convert numshot to string if it's 'all' for display purposes
    numshot_str = numshot if isinstance(numshot, str) else str(numshot)
    print(f"Starting {student_model_name} on {dataset_name}, {numshot_str}-shot, parent: {parent_model}")
    info_path = f"dataset/{dataset_name}/{dataset_name}.info"
    
    # If gpu_id is provided, use it for single-GPU models (RRL, TTNet)
    if gpu_id is not None and device.startswith('cuda'):
        device_for_model = f'cuda:{gpu_id}'
    else:
        device_for_model = device

    # Check if results already exist
    result_dir = f"eval_res/{parent_model}/{dataset_name}/{numshot_str}_shot/{student_model_name}"
    results_path = os.path.join(result_dir, "results.txt")
    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}, skipping...")
        return None

    # Check if synthetic data directory exists
    synth_dir = f"eval_res/{parent_model}/{dataset_name}/{numshot_str}_shot"
    if not os.path.exists(synth_dir):
        print(f"Synthetic data directory does not exist: {synth_dir}, skipping...")
        return None
    
    # Check if required synthetic data files exist
    X_synth_path = os.path.join(synth_dir, "X_synth.npy")
    y_synth_path = os.path.join(synth_dir, "y_pred.npy")
    if not os.path.exists(X_synth_path) or not os.path.exists(y_synth_path):
        print(f"Required synthetic data files missing in {synth_dir}, skipping...")
        return None

    results = []
    for seed in SEEDS:
        set_seed(seed)
        # Get few-shot splits
        X_few, y_few, X_train, y_train, X_test, y_test = get_few_shot_from_csv(dataset_name, numshot, seed)
        
        # Combine original training and test sets for fitting the preprocessor
        X_original_full = np.vstack([X_train, X_test])

        # Load synthetic data
        X_synth = np.load(X_synth_path, allow_pickle=True)
        y_synth = np.load(y_synth_path, allow_pickle=True)

        # Check if synthetic data has only one class - if so, skip training
        unique_classes = np.unique(y_synth)
        if len(unique_classes) <= 1:
            print(f"Warning: Synthetic data contains only {len(unique_classes)} class(es): {unique_classes}")
            print("Skipping training and reporting default performance...")
            # Report default performance: AUC=0.5 (random), complexity=-1 (no model)
            results.append({'seed': seed, 'auc': 0.5, 'acc': 0.5, 'tpr': 0.0, 'fpr': 0.0, 'f1': 0.0, 'complexity': -1})
            continue

        # Initialize and fit the preprocessor on the combined original data
        ttnet_preprocessor = TTNetPreprocessor(info_path=info_path)
        ttnet_preprocessor.fit(X_original_full)

        X_synth_proc, synth_index = ttnet_preprocessor.transform(X_synth)
        X_test_proc, test_index = ttnet_preprocessor.transform(X_test)

        # Set up model and param_dist
        model = None
        param_dist = None
        if student_model_name == 'xgboost':
            param_dist = {
                'max_depth': randint(1, 12),
                'n_estimators': randint(100, 6001),
                'gamma': loguniform(1e-8, 7),
                'reg_lambda': loguniform(1, 4),
                'reg_alpha': loguniform(1e-8, 1e2)
            }
            xgb_kwargs = {'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': seed}
            if device.startswith('cuda'):
                # XGBoost will automatically use all visible GPUs
                xgb_kwargs['device'] = 'cuda'
            model = xgb.XGBClassifier(**xgb_kwargs)
        elif student_model_name == 'logistic_regression':
            model = LogisticRegression(random_state=seed)
        elif student_model_name == 'decision_tree':
            param_dist = {
                'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'max_features': ['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'criterion': ['gini', 'entropy']
            }
            model = DecisionTreeClassifier(random_state=seed)
        elif student_model_name == 'logistic_rule_regression':
            # Fit binarizer on the original data (X_train + X_test) preserving column names
            X_fb_all = pd.concat([X_train, X_test], ignore_index=True)
            fb = FeatureBinarizer(negations=True)
            fb.fit(X_fb_all)
            # Convert synthetic data back to DataFrame with proper column names and data types
            X_synth_df = pd.DataFrame(X_synth, columns=X_train.columns)
            # Ensure data types match the original data
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    X_synth_df[col] = X_synth_df[col].astype('object')
                else:
                    X_synth_df[col] = pd.to_numeric(X_synth_df[col], errors='coerce')
            X_synth_fb = fb.transform(X_synth_df)
            X_test_fb = fb.transform(X_test)  # X_test is already a DataFrame
            model = LogisticRuleRegression(lambda0=0.01, lambda1=0.001)
            model.fit(X_synth_fb, y_synth)
        elif student_model_name == 'ttnet':
            features_size = X_synth_proc.shape[1]
            model = TTnetStudentModel(features_size=features_size, index=synth_index, device=device_for_model)
        elif student_model_name == 'rrl':
            features_size = X_synth_proc.shape[1]
            model = RRLWrapper(features_size=features_size, index=synth_index, device=device_for_model)

        # Fit the model
        if student_model_name not in ['logistic_rule_regression']: # LRR is already fitted
            if param_dist is not None:
                search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=2, random_state=seed)
                search.fit(X_synth_proc, y_synth)
                model = search.best_estimator_
            else:
                model.fit(X_synth_proc, y_synth)

        # Evaluate on test set
        if student_model_name == 'logistic_rule_regression':
            y_pred_proba = model.predict_proba(X_test_fb)
            y_pred = (y_pred_proba >= 0.5).astype(int)
        elif student_model_name == 'ttnet':
            # TTNet returns 1D array of probabilities
            y_pred_proba = model.predict_proba(X_test_proc)
            y_pred = (y_pred_proba >= 0.5).astype(int)
        elif student_model_name == 'rrl':
            # RRL also returns 1D array of probabilities
            y_pred_proba = model.predict_proba(X_test_proc)
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            # Other sklearn models return 2D array, we need column 1 for positive class
            y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = f1_score(y_test, y_pred)

        # Calculate model complexity
        complexity = None
        if student_model_name == 'xgboost':
            tree_text = model.get_booster().get_dump()
            complexity = 0
            for tree in tree_text:
                num_leaves = tree.count('leaf')
                num_internal = tree.count('[') - num_leaves
                complexity += num_leaves + num_internal
        elif student_model_name == 'decision_tree':
            num_nodes = model.tree_.node_count
            complexity = num_nodes # node_count is total nodes (leaves + internal)
        elif student_model_name == 'logistic_regression':
            complexity = np.count_nonzero(model.coef_) + model.coef_.shape[1]
        elif student_model_name == 'logistic_rule_regression':
            explanation = model.explain()
            Rule = len(explanation["rule"].to_numpy().tolist()) -1 # Exclude header
            condition = np.sum([x.count("AND") for x in explanation["rule"].to_numpy().tolist() if isinstance(x, str)]) + Rule
            complexity = Rule + condition
        elif student_model_name == 'ttnet':
            rule_info = model.extract_rules() 
            if rule_info is not None and 'complexity' in rule_info:
                complexity = rule_info['complexity']
            else:
                complexity = None
        elif student_model_name == 'rrl':
            rule_info = model.extract_rules() 
            if rule_info is not None and 'complexity' in rule_info:
                complexity = rule_info['complexity']
            else:
                complexity = None
        results.append({'seed': seed, 'auc': auc, 'acc': acc, 'tpr': tpr, 'fpr': fpr, 'f1': f1,
                        'complexity': complexity})

    # Save results
    result_dir = f"eval_res/{parent_model}/{dataset_name}/{numshot}_shot/{student_model_name}"
    os.makedirs(result_dir, exist_ok=True)
    results_path = os.path.join(result_dir, "results.txt")
    with open(results_path, 'w') as f:
        aucs = [r['auc'] for r in results]
        accs = [r['acc'] for r in results]
        tprs = [r['tpr'] for r in results]
        fprs = [r['fpr'] for r in results]
        f1s = [r['f1'] for r in results]
        complexities = [r['complexity'] for r in results]
        f.write(f"AUCs: {aucs}\n")
        f.write(f"Accuracies: {accs}\n")
        f.write(f"TPRs: {tprs}\n")
        f.write(f"FPRs: {fprs}\n")
        f.write(f"F1s: {f1s}\n")
        f.write(f"Complexities: {complexities}\n")
        f.write(f"Mean AUC: {np.mean(aucs):.4f}\n")
        f.write(f"Std AUC: {np.std(aucs):.4f}\n")
        f.write(f"Mean Acc: {np.mean(accs):.4f}\n")
        f.write(f"Std Acc: {np.std(accs):.4f}\n")
        f.write(f"Mean TPR: {np.mean(tprs):.4f}\n")
        f.write(f"Mean FPR: {np.mean(fprs):.4f}\n")
        f.write(f"Mean F1: {np.mean(f1s):.4f}\n")
        # Handle cases where all complexities might be None (e.g., if a model fails for all seeds)
        valid_complexities = [c for c in complexities if c is not None]
        if valid_complexities:
            f.write(f"Mean Complexity: {np.mean(valid_complexities):.2f}\n")
            f.write(f"Std Complexity: {np.std(valid_complexities):.2f}\n")
        else:
            f.write("Mean Complexity: N/A\n")
            f.write("Std Complexity: N/A\n")
    print(f"Results saved to {results_path}")
    return results

if __name__ == '__main__':
    device = 'cuda:0,1,2,3'  # 'cpu', 'cuda:0', 'cuda:0,1,2,3' for multi-GPU
    # numshots = [4, 8, 16, 32, 64, 128, 256]
    numshots = [4, 8, 16, 32, 64, 128, 256, 'all']
    # datasets = ["breastcancer", "breastcancer2", "chemotherapy", "coloncancer", "diabetes", "heart", "respiratory"]
    datasets = ["breastcancer", "breastcancer2", "chemotherapy", "coloncancer", "diabetes", "heart", "respiratory"]
    # parent_models = ['tabpfn', 'tabm']
    parent_models = ['tabpfn', 'tabm']
    # student_models = ['xgboost', 'decision_tree', 'logistic_rule_regression', 'ttnet']
    student_models = ['rrl']

    # Parse GPU IDs for multi-GPU support
    gpu_ids_list = []
    if device.startswith('cuda'):
        if ':' in device:
            gpu_ids_str = device.split(':')[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
            # Parse individual GPU IDs for round-robin assignment
            gpu_ids_list = [int(x.strip()) for x in gpu_ids_str.split(',')]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            gpu_ids_list = [0]
    
    task_idx = 0  # Counter for round-robin GPU assignment
    for dataset_name in datasets:
        for numshot in numshots:
            for parent_model in parent_models:
                for student_model in student_models:
                    # For single-GPU models (RRL, TTNet), distribute across available GPUs
                    gpu_id = None
                    if student_model in ['rrl', 'ttnet'] and gpu_ids_list:
                        # Round-robin assignment: use remapped GPU IDs (0, 1, 2, 3...)
                        gpu_id = task_idx % len(gpu_ids_list)
                        task_idx += 1
                    
                    print(f"Running {student_model} on {dataset_name} with {numshot} shots from {parent_model}" +
                          (f" on GPU {gpu_id}" if gpu_id is not None else ""))
                    try:
                        main(dataset_name, numshot, parent_model, student_model, device=device, gpu_id=gpu_id)
                    except Exception as e:
                        print(f"ERROR during {student_model} on {dataset_name}: {e}")
                        import traceback
                        traceback.print_exc()
    print("All experiments finished.")