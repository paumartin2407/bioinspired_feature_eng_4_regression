import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def learner(name, parameter=None, seed=0, forest_jobs=1):
    if name == "ols":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0 if parameter is None else parameter)
    if name == "forest":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=3 if parameter is None else parameter,
                                     random_state=seed, n_jobs=forest_jobs)
    raise ValueError(f"Unknown learner: {name}")


PARAMETERS = {"ols": [None], "ridge": [0.1, 1.0, 10.0, 100.0], "forest": [1, 3, 8]}


def choose_learner(name, folds, seed, forest_jobs=1):
    """Pre-tune on original features, then freeze the choice for all FE arms."""
    scores = []
    fits = 0
    for parameter in PARAMETERS[name]:
        fold_scores = []
        for train, val, yt, yv, *_ in folds:
            model = learner(name, parameter, seed, forest_jobs).fit(train, yt)
            fits += 1
            fold_scores.append(mean_absolute_error(yv, model.predict(val)))
        scores.append(float(np.mean(fold_scores)))
    return PARAMETERS[name][int(np.argmin(scores))], fits, scores


def candidate_fitness(folds, strategy, family, k, name, parameter, seed, accounting,
                      pair_transform=None, forest_jobs=1):
    def score(genes):
        fold_scores = []
        for train, val, yt, yv, *_ in folds:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    if pair_transform is None:
                        transformed_train = transform(train, strategy, family, genes, k)
                        transformed_val = transform(val, strategy, family, genes, k)
                    else:
                        transformed_train, transformed_val = pair_transform(train, val, genes)
                if not np.isfinite(transformed_train).all() or not np.isfinite(transformed_val).all():
                    raise ValueError("Nonfinite feature")
                model = learner(name, parameter, seed, forest_jobs)
                accounting["fits"] += 1
                model.fit(transformed_train, yt)
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
