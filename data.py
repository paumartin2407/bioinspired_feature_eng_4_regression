from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "California Housing": ("dataset/California.csv", "MedHouseVal"),
    "Diabetes Progression": ("dataset/Diabetes.csv", "target"),
}


def load_dataset(path, target_col):
    frame = pd.read_csv(path)
    if target_col not in frame:
        raise ValueError(f"Target {target_col!r} not found in {path}")
    frame = frame.loc[frame[target_col].notna()]
    return frame.drop(columns=target_col), frame[target_col]


def split_and_preprocess(X, y, seed, test_size=0.20, validation_size=0.25, return_indices=False):
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=validation_size, random_state=seed
    )
    indices = {"train": X_train.index.tolist(), "validation": X_val.index.tolist(), "test": X_test.index.tolist()}
    X_train = pd.get_dummies(X_train, drop_first=True, dtype=float)
    columns = X_train.columns
    X_val = pd.get_dummies(X_val, drop_first=True, dtype=float).reindex(columns=columns, fill_value=0)
    X_test = pd.get_dummies(X_test, drop_first=True, dtype=float).reindex(columns=columns, fill_value=0)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    def transform(frame, fit=False):
        values = imputer.fit_transform(frame) if fit else imputer.transform(frame)
        values = scaler.fit_transform(values) if fit else scaler.transform(values)
        return pd.DataFrame(values, columns=columns, index=frame.index)
    result = (transform(X_train, True), transform(X_val), transform(X_test), y_train, y_val, y_test)
    return (*result, indices) if return_indices else result
