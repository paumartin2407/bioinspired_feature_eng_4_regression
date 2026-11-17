import numpy as np


FAMILIES = ("quadratic", "rbf", "interactions", "hinge", "quadratic_rbf",
            "quadratic_hinge", "quadratic_rbf_hinge", "quadratic_rbf_rbf_heavy",
            "quadratic_rbf_hinge_rbf_heavy", "quadratic_rbf_hinge_quadratic_heavy")


def components(family, k):
    """Return component families and counts for the supported mixtures."""
    if family == "quadratic_rbf":
        # Preserve the existing decoder for previously saved odd-k results.
        return [("quadratic", (k + 1) // 2), ("rbf", k // 2)]
    if family == "quadratic_rbf_rbf_heavy":
        if k not in (0, 4, 8):
            raise ValueError(f"{family} supports k=0,4,8")
        return [("quadratic", k // 4), ("rbf", 3 * k // 4)]
    if family in ("quadratic_rbf_hinge_rbf_heavy", "quadratic_rbf_hinge_quadratic_heavy"):
        if k not in (0, 4, 6):
            raise ValueError(f"{family} supports k=0,4,6")
        if k == 0:
            return [("quadratic", 0), ("rbf", 0), ("hinge", 0)]
        if family == "quadratic_rbf_hinge_rbf_heavy":
            return [("quadratic", 1), ("rbf", k - 2), ("hinge", 1)]
        return [("quadratic", k - 2), ("rbf", 1), ("hinge", 1)]
    if family in ("quadratic_hinge", "quadratic_rbf_hinge"):
        names = ("quadratic", "hinge") if family == "quadratic_hinge" else ("quadratic", "rbf", "hinge")
        if k % len(names):
            raise ValueError(f"{family} requires a feature count divisible by {len(names)}")
        return [(name, k // len(names)) for name in names]
    return None


def dimension(family, d, k):
    if family == "quadratic":
        return d * k
    if family in ("rbf", "hinge"):
        return (d + 1) * k
    if family == "interactions":
        return 2 * k
    parts = components(family, k)
    if parts is not None:
        return sum(dimension(name, d, count) for name, count in parts)
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
    if family == "hinge":
        rows = genes.reshape(k, d + 1)
        return np.maximum(0.0, X @ rows[:, :d].T - 3 * rows[:, d])
    parts = components(family, k)
    if parts is not None:
        features, offset = [], 0
        for name, count in parts:
            end = offset + dimension(name, d, count)
            features.append(construct(X, genes[offset:end], name, count))
            offset = end
        return np.column_stack(features)
    raise ValueError(f"Unknown representation: {family}")
