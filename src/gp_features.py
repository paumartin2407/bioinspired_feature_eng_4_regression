"""Bounded symbolic feature sets; GP is a pipeline arm, never an RQ3 optimizer."""

import numpy as np

from .evaluation import EvaluationCounter


OPERATORS = ("+", "-", "*", "/")
MAX_DEPTH = 3


def random_tree(rng, d, depth=MAX_DEPTH):
    if depth == 0 or rng.random() < 0.25:
        return ("x", int(rng.integers(d))) if rng.random() < 0.8 else ("c", float(rng.uniform(-2, 2)))
    return (OPERATORS[int(rng.integers(len(OPERATORS)))],
            random_tree(rng, d, depth - 1), random_tree(rng, d, depth - 1))


def evaluate(tree, X):
    op = tree[0]
    if op == "x":
        return X[:, int(tree[1])]
    if op == "c":
        return np.full(len(X), float(tree[1]))
    left, right = evaluate(tree[1], X), evaluate(tree[2], X)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if op == "+":
            result = left + right
        elif op == "-":
            result = left - right
        elif op == "*":
            result = left * right
        elif op == "/":
            result = np.divide(left, right, out=np.ones_like(left), where=np.abs(right) > 1e-3)
        else:
            raise ValueError(f"Unknown GP operation: {op}")
    return np.clip(result, -1e6, 1e6)


def node_count(tree):
    return 1 if tree[0] in ("x", "c") else 1 + node_count(tree[1]) + node_count(tree[2])


def feature_pair(train, other, trees):
    train, other = np.asarray(train, dtype=float), np.asarray(other, dtype=float)
    first, second = [train], [other]
    for tree in trees:
        values = evaluate(tree, train)
        held = evaluate(tree, other)
        mean, scale = values.mean(), values.std()
        scale = scale if scale > 1e-9 else 1.0
        first.append(((values - mean) / scale)[:, None])
        second.append(((held - mean) / scale)[:, None])
    return np.column_stack(first), np.column_stack(second)


def _replace(tree, path, replacement):
    if not path:
        return replacement
    child = path[0]
    return (tree[0], _replace(tree[1], path[1:], replacement), tree[2]) if child == 1 else (
        tree[0], tree[1], _replace(tree[2], path[1:], replacement))


def _paths(tree, prefix=()):
    paths = [prefix]
    if tree[0] not in ("x", "c"):
        paths += _paths(tree[1], prefix + (1,)) + _paths(tree[2], prefix + (2,))
    return paths


def _subtree(tree, path):
    for child in path:
        tree = tree[child]
    return tree


def optimize_gp(objective, d, k, seed, budget):
    if k < 1 or budget < 20:
        raise ValueError("GP needs positive k and at least 20 evaluations")
    rng = np.random.default_rng(seed)
    counter = EvaluationCounter(budget)
    population = [tuple(random_tree(rng, d) for _ in range(k)) for _ in range(20)]
    scores = [counter.evaluate(objective, item) for item in population]
    best_index = int(np.argmin(scores))
    best, best_score = population[best_index], scores[best_index]
    while counter.remaining:
        children = []
        while len(children) < min(20, counter.remaining):
            parents = [population[min(rng.choice(20, 3), key=lambda i: scores[i])] for _ in range(2)]
            child = list(parents[0])
            slot = int(rng.integers(k))
            source = child[slot]
            donor = parents[1][int(rng.integers(k))]
            path = _paths(source)[int(rng.integers(len(_paths(source))))]
            donor_path = _paths(donor)[int(rng.integers(len(_paths(donor))))]
            proposal = _replace(source, path, _subtree(donor, donor_path))
            if rng.random() < 0.5:
                path = _paths(proposal)[int(rng.integers(len(_paths(proposal))))]
                proposal = _replace(proposal, path, random_tree(rng, d, max(0, MAX_DEPTH - len(path))))
            if node_count(proposal) <= 15 and _depth(proposal) <= MAX_DEPTH:
                child[slot] = proposal
            children.append(tuple(child))
        child_scores = [counter.evaluate(objective, item) for item in children]
        for item, score in zip(children, child_scores):
            if score < best_score:
                best, best_score = item, score
        ranked = sorted(zip(population + children, scores + child_scores), key=lambda pair: pair[1])[:20]
        population, scores = [item for item, _ in ranked], [score for _, score in ranked]
    return best, best_score, counter.count, counter.history


def _depth(tree):
    return 0 if tree[0] in ("x", "c") else 1 + max(_depth(tree[1]), _depth(tree[2]))
