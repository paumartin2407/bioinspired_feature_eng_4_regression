"""Fixed continuous search space used by every Milestone 2 optimizer."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


N_CONSTRUCTED = 2
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0


def decode(candidate, n_features):
    """Return the two coefficient rows, rejecting an invalid candidate."""
    weights = np.asarray(candidate, dtype=float).reshape(N_CONSTRUCTED, n_features)
    if not np.all(np.isfinite(weights)) or np.any(weights < LOWER_BOUND) or np.any(weights > UPPER_BOUND):
        raise ValueError("Coefficients must be finite and within [-1, 1]")
    return weights


def construct(X, candidate):
    """Retain every original column and append two squared linear combinations."""
    values = np.asarray(X, dtype=float)
    weights = decode(candidate, values.shape[1])
    constructed = np.square(values @ weights.T)
    return np.column_stack((values, constructed))


def make_fitness(X_train, X_val, y_train, y_val):
    """Validation MAE; the outer test partition is never supplied to this closure."""
    train = np.asarray(X_train, dtype=float)
    val = np.asarray(X_val, dtype=float)
    target = np.asarray(y_train, dtype=float)
    truth = np.asarray(y_val, dtype=float)

    def fitness(candidate):
        with np.errstate(over="ignore", invalid="ignore"):
            train_features = construct(train, candidate)
            val_features = construct(val, candidate)
        if not np.isfinite(train_features).all() or not np.isfinite(val_features).all():
            return float("inf")
        model = LinearRegression().fit(train_features, target)
        prediction = model.predict(val_features)
        return float(mean_absolute_error(truth, prediction)) if np.isfinite(prediction).all() else float("inf")

    return fitness
