import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(X_train, y_train, X_test, y_test, model=None, dataset_name="Dataset"):
    model = LinearRegression() if model is None else model
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_train, pred_train),
        "mse": mean_squared_error(y_train, pred_train),
        "r2": r2_score(y_train, pred_train),
        "mae_test": mean_absolute_error(y_test, pred_test),
        "mse_test": mean_squared_error(y_test, pred_test),
        "r2_test": r2_score(y_test, pred_test),
        "preds_train": pred_train,
        "preds_test": pred_test,
        "model": model,
    }


def regression_metrics(y_true, prediction):
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "R2": float(r2_score(y_true, prediction)),
    }


class EvaluationCounter:
    def __init__(self, limit=1000):
        self.limit = limit
        self.count = 0
        self.best = float("inf")
        self.history = []

    @property
    def remaining(self):
        return self.limit - self.count

    def evaluate(self, fn, candidate):
        if self.count >= self.limit:
            raise RuntimeError("Fitness evaluation budget exhausted")
        score = float(fn(candidate))
        self.count += 1
        self.best = min(self.best, score)
        self.history.append((self.count, self.best))
        return score
