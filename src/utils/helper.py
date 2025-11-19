import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

def get_few_shot_from_csv(dataset_name, num_shot, seed):
    np.random.seed(seed)
    # Load original dataset
    orig_path = f"dataset/{dataset_name}/{dataset_name}.csv"
    df = pd.read_csv(orig_path)
    if 'class' not in df.columns:
        raise ValueError("The original dataset must have a 'class' column.")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['class'].astype(str)).astype(bool)
    feature_cols = [col for col in df.columns if col not in ["class", "label"]]
    X = df[feature_cols]
    y = df["label"]

    rng = np.random.RandomState(seed)

    pos_indices = np.where(y.to_numpy() == 1)[0]
    neg_indices = np.where(y.to_numpy() == 0)[0]

    required_per_class = 128
    if len(pos_indices) < required_per_class or len(neg_indices) < required_per_class:
        raise ValueError(
            "Each class must have at least 128 samples to construct the balanced training set."
        )

    pos_train = rng.choice(pos_indices, size=required_per_class, replace=False)
    neg_train = rng.choice(neg_indices, size=required_per_class, replace=False)

    train_idx = np.concatenate([pos_train, neg_train])
    rng.shuffle(train_idx)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    remaining_idx = np.setdiff1d(np.arange(len(df)), train_idx, assume_unique=False)
    X_test = X.iloc[remaining_idx].reset_index(drop=True)
    y_test = y.iloc[remaining_idx].reset_index(drop=True)

    # Handle 'all' shot setting - use entire training set
    if num_shot == 'all' or (isinstance(num_shot, str) and num_shot.lower() == 'all'):
        X_few = X_train.copy()
        y_few = y_train.copy()
        return X_few, y_few, X_train, y_train, X_test, y_test

    if num_shot > len(train_idx):
        raise ValueError("num_shot cannot exceed the size of the balanced training set (256).")
    if num_shot % 2 != 0:
        raise ValueError("num_shot must be an even number to maintain class balance.")

    shots_per_class = num_shot // 2

    pos_in_train = np.where(y_train.to_numpy() == 1)[0]
    neg_in_train = np.where(y_train.to_numpy() == 0)[0]

    if shots_per_class > len(pos_in_train) or shots_per_class > len(neg_in_train):
        raise ValueError("Not enough samples per class in the training set to satisfy num_shot.")

    pos_few_idx = rng.choice(pos_in_train, size=shots_per_class, replace=False)
    neg_few_idx = rng.choice(neg_in_train, size=shots_per_class, replace=False)
    few_idx = np.concatenate([pos_few_idx, neg_few_idx])
    rng.shuffle(few_idx)

    X_few = X_train.iloc[few_idx].reset_index(drop=True)
    y_few = y_train.iloc[few_idx].reset_index(drop=True)

    return X_few, y_few, X_train, y_train, X_test, y_test