import numpy as np

from .representations import construct, dimension


STRATEGIES = ("none", "selection", "construction", "combined")


def gene_count(strategy, family, d, k):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    return (d if strategy in ("selection", "combined") else 0) + (
        dimension(family, d, k) if strategy in ("construction", "combined") else 0)


def decode(strategy, family, genes, d, k):
    genes = np.asarray(genes, dtype=float)
    if genes.size != gene_count(strategy, family, d, k):
        raise ValueError("Wrong gene count")
    if not np.isfinite(genes).all() or np.any(np.abs(genes) > 1):
        raise ValueError("Genes must be finite in [-1, 1]")
    mask = np.ones(d, dtype=bool)
    if strategy in ("selection", "combined"):
        mask = genes[:d] >= 0
        if not mask.any():
            mask[np.argmax(genes[:d])] = True
        genes = genes[d:]
    return mask, genes


def transform(X, strategy, family, genes, k):
    X = np.asarray(X, dtype=float)
    mask, construction_genes = decode(strategy, family, genes, X.shape[1], k)
    if strategy in ("none", "selection") or k == 0:
        return X[:, mask]
    # Construction uses all original inputs, including those omitted directly in the combined arm.
    return np.column_stack((X[:, mask], construct(X, construction_genes, family, k)))
