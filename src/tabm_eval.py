import argparse
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helper import get_few_shot_from_csv
from src.ttnet.ttnet_wrapper import TTNetPreprocessor
from baselines.utils_baselines import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--numshot", type=str, required=True)  # Changed to str to handle 'all'
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Convert numshot to int if it's not 'all'
    numshot = args.numshot if args.numshot.lower() == 'all' else int(args.numshot)
    
    set_seed(args.seed)

    # Output directory
    save_dir = f"eval_res/tabm/{args.dataset}/{numshot}_shot/"
    results_path = os.path.join(save_dir, "eval_metrics.json")
    if os.path.exists(results_path):
        print(f"Output already exists at {save_dir}. Skipping run.")
        return

    # Load Data
    X_few, y_few, X_train, y_train, X_test, y_test = get_few_shot_from_csv(args.dataset, numshot, args.seed)
    
    synth_path = f"dataset/{args.dataset}/{args.dataset}_synthetic.csv"
    synth_df = pd.read_csv(synth_path)
    X_synth = synth_df.drop(columns=["class"])
    
    # Preprocess data
    info_path = f"dataset/{args.dataset}/{args.dataset}.info"
    preprocessor = TTNetPreprocessor(info_path=info_path)
    
    X_original_full = pd.concat([X_train, X_test], ignore_index=True)
    preprocessor.fit(X_original_full)

    X_few_proc, _ = preprocessor.transform(X_few)
    X_synth_proc, _ = preprocessor.transform(X_synth)

    # Convert to tensors
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X_few_tensor = torch.as_tensor(X_few_proc, device=device, dtype=torch.float32)
    y_few_tensor = torch.as_tensor(y_few.values, device=device, dtype=torch.long)
    X_synth_tensor = torch.as_tensor(X_synth_proc, device=device, dtype=torch.float32)
    
    # Process test data for evaluation
    X_test_proc, _ = preprocessor.transform(X_test)
    X_test_tensor = torch.as_tensor(X_test_proc, device=device, dtype=torch.float32)

    # Model setup
    n_num_features = X_few_proc.shape[1]
    n_classes = len(np.unique(y_few))

    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[],
        d_out=n_classes,
        num_embeddings=None,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=5e-5
    )
    
    # Training
    @torch.inference_mode()
    def apply_model(x_num):
        return model(x_num, None).squeeze(-1).float()

    def loss_fn(y_pred, y_true):
        y_pred = y_pred.flatten(0, 1)
        y_true = y_true.repeat_interleave(model.backbone.k)
        return nn.functional.cross_entropy(y_pred, y_true)

    train_dataset = TensorDataset(X_few_tensor, y_few_tensor)
    batch_size = max(1, min(128, len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    max_epochs = 500
    early_stop_patience = 60
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            y_pred_train = model(batch_X, None).squeeze(-1).float()
            loss = loss_fn(y_pred_train, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_dataset)
        scheduler.step(epoch_loss)

        if epoch_loss + 1e-6 < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}: loss={epoch_loss:.4f}")
            break

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{max_epochs} - loss: {epoch_loss:.4f}")

    # Inference
    model.eval()
    with torch.inference_mode():
        # Inference on synthetic data
        y_pred_synth = apply_model(X_synth_tensor)
        y_pred_synth = torch.softmax(y_pred_synth, dim=-1).mean(1)
        y_pred = y_pred_synth.argmax(1).cpu().numpy()
        
        # Inference on test data
        y_pred_test_logits = apply_model(X_test_tensor)
        y_pred_test_probs = torch.softmax(y_pred_test_logits, dim=-1).mean(1)
        y_test_pred = y_pred_test_probs.argmax(1).cpu().numpy()
        y_test_probs = y_pred_test_probs.cpu().numpy()
    
    # Evaluate performance on test set
    accuracy = accuracy_score(y_test.values, y_test_pred)
    
    # Check if predictions are all the same class (which makes ROC-AUC undefined)
    if len(np.unique(y_test_pred)) == 1:
        # All predictions are the same class, set AUC to 50%
        auc = 0.5
        print(f"Warning: All test predictions are class {y_test_pred[0]}, setting AUC to 0.5")
    else:
        auc = roc_auc_score(y_test.values, y_test_probs[:, 1])
    
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
