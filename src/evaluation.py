import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from .strategies import transform


class EvaluationCounter:
    def __init__(self, limit):
        self.limit = limit
        self.count = 0
        self.best = float("inf")
        self.history = []

    @property
    def remaining(self):
        return self.limit - self.count

    def evaluate(self, fn, candidate):
        if not self.remaining:
            raise RuntimeError("Fitness evaluation budget exhausted")
        score = float(fn(candidate))
        self.count += 1
        self.best = min(self.best, score)
        self.history.append((self.count, self.best))
        return score


def learner(name, parameter=None, seed=0):
    if name.startswith("linear_panel_"):
        models = {
            "linear_panel_ols": lambda: LinearRegression(),
            "linear_panel_ridge": lambda: Ridge(alpha=1.0),
            "linear_panel_lasso": lambda: Lasso(alpha=0.01, max_iter=10000),
            "linear_panel_elastic": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
            "linear_panel_huber": lambda: HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000),
        }
        if name not in models:
            raise ValueError(f"Unknown learner: {name}")
        # Both scalers fit inside each training fold; predictions are restored
        # to original target units before the validation metric is computed.
        return make_pipeline(StandardScaler(), TransformedTargetRegressor(
            regressor=models[name](), transformer=StandardScaler()))
    if name == "ols":
        return LinearRegression()
    raise ValueError(f"Unknown learner: {name}")


PARAMETERS = {"ols": [None],
              "linear_panel_ols": [None], "linear_panel_ridge": [None],
              "linear_panel_lasso": [None], "linear_panel_elastic": [None], "linear_panel_huber": [None]}


def fit_with_diagnostics(model, X, y, accounting):
    """Count fits with convergence warnings without treating them as successes without notice."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    if accounting is not None and any(issubclass(w.category, ConvergenceWarning) for w in seen):
        accounting["convergence_warning_fits"] = accounting.get("convergence_warning_fits", 0) + 1
    for warning in seen:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return model


def choose_learner(name, folds, seed, accounting=None):
    """Score the fixed learner on original features for the no-FE baseline."""
    scores = []
    fits = 0
    for parameter in PARAMETERS[name]:
        fold_scores = []
        for train, val, yt, yv, *_ in folds:
            model = fit_with_diagnostics(learner(name, parameter, seed), train, yt, accounting)
            fits += 1
            fold_scores.append(mean_absolute_error(yv, model.predict(val)))
        scores.append(float(np.mean(fold_scores)))
    return PARAMETERS[name][int(np.argmin(scores))], fits, scores


def candidate_fitness(folds, strategy, family, k, name, parameter, seed, accounting):
    def score(genes):
        fold_scores = []
        for train, val, yt, yv, *_ in folds:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    transformed_train = transform(train, strategy, family, genes, k)
                    transformed_val = transform(val, strategy, family, genes, k)
                if not np.isfinite(transformed_train).all() or not np.isfinite(transformed_val).all():
                    raise ValueError("Nonfinite feature")
                model = learner(name, parameter, seed)
                accounting["fits"] += 1
                fit_with_diagnostics(model, transformed_train, yt, accounting)
                prediction = model.predict(transformed_val)
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite prediction")
                fold_scores.append(mean_absolute_error(yv, prediction))
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                accounting["failed_folds"] += 1
                accounting["failed_candidates"] = accounting.get("failed_candidates", 0) + 1
                return float("inf")
        return float(np.mean(fold_scores))
    return score


def metrics(y, prediction):
    residual = np.abs(np.asarray(y) - np.asarray(prediction))
    return {"mae": float(np.mean(residual)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction)),
            "p95_abs_residual": float(np.percentile(residual, 95)),
            "p99_abs_residual": float(np.percentile(residual, 99)),
            "max_abs_residual": float(np.max(residual))}
