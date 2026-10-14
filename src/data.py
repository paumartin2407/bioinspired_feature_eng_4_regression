from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(spec, root=Path(".")):
    path = Path(root) / spec["path"]
    frame = pd.read_csv(path)
    target = spec["target"]
    if target not in frame:
        raise ValueError(f"Missing target {target!r} in {path}")
    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    y = pd.to_numeric(frame.pop(target), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"Nonfinite target in {path}")
    if len(frame) < 30 or frame.shape[1] == 0:
        raise ValueError(f"Dataset {path} needs at least 30 rows and one predictor")
    return frame, y


def make_outer_split(n_rows, seed):
    return train_test_split(np.arange(n_rows), test_size=0.2, random_state=seed)


def make_inner_splits(n_dev, seed, folds=3):
    return list(KFold(folds, shuffle=True, random_state=seed).split(np.arange(n_dev)))


def fit_preprocessor(train):
    numeric = train.select_dtypes(include=np.number).columns.tolist()
    categorical = [name for name in train if name not in numeric]
    blocks = []
    if numeric:
        blocks.append(("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                             StandardScaler()), numeric))
    if categorical:
        blocks.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                             OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical))
    preprocessor = ColumnTransformer(blocks, sparse_threshold=0).fit(train)
    return preprocessor


def transformed(preprocessor, frame, columns=None):
    values = pd.DataFrame(preprocessor.transform(frame), columns=preprocessor.get_feature_names_out(), index=frame.index)
    return values if columns is None else values.reindex(columns=columns, fill_value=0.0)


def prepared_folds(dev, y_dev, seed, folds=3):
    prepared = []
    for train_idx, val_idx in make_inner_splits(len(dev), seed, folds):
        preprocessor = fit_preprocessor(dev.iloc[train_idx])
        prepared.append((transformed(preprocessor, dev.iloc[train_idx]),
                         transformed(preprocessor, dev.iloc[val_idx]),
                         y_dev[train_idx], y_dev[val_idx], train_idx, val_idx))
    # The union of training-derived names provides a common candidate dimension.
    columns = sorted(set().union(*(set(part[0].columns) for part in prepared)))
    return [(train.reindex(columns=columns, fill_value=0.0),
             val.reindex(columns=columns, fill_value=0.0), yt, yv, ti, vi)
            for train, val, yt, yv, ti, vi in prepared], columns
