import numpy as np


FAMILIES = ("quadratic", "rbf", "interactions")


def dimension(family, d, k):
    if family == "quadratic":
        return d * k
    if family == "rbf":
        return (d + 1) * k
    if family == "interactions":
        return 2 * k
    raise ValueError(f"Unknown representation: {family}")


def construct(X, genes, family, k):
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    genes = np.asarray(genes, dtype=float)
    if genes.size != dimension(family, d, k) or not np.isfinite(genes).all() or np.any(np.abs(genes) > 1):
        raise ValueError("Invalid bounded feature genes")
    if k == 0:
        return np.empty((len(X), 0))
    if family == "quadratic":
        return np.square(X @ genes.reshape(k, d).T)
    if family == "rbf":
        rows = genes.reshape(k, d + 1)
        centers = 3 * rows[:, :d]
        gamma = np.exp(2 * rows[:, d]) / max(1, d)
        distances = np.sum(np.square(X[:, None, :] - centers[None, :, :]), axis=2)
        return np.exp(-distances * gamma[None, :])
    if family == "interactions":
        pairs = np.minimum(((genes.reshape(k, 2) + 1) * d / 2).astype(int), d - 1)
        return X[:, pairs[:, 0]] * X[:, pairs[:, 1]]
    raise ValueError(f"Unknown representation: {family}")
