"""Controlled GA, PSO, ES, DE/rand/1/bin and pycma search over one coefficient vector."""

from dataclasses import dataclass

import cma
import numpy as np

from continuous_features import LOWER_BOUND, UPPER_BOUND, decode
from evaluation import EvaluationCounter


METHODS = ("GA", "PSO", "ES", "DE", "CMA-ES")
POPULATION = 40
ES_PARENTS = 20
CMA_POPULATION = 20


@dataclass
class SearchResult:
    weights: np.ndarray
    score: float
    evaluations: int
    history: list


def optimize(method, objective, n_features, seed, budget=1000):
    """Spend exactly ``budget`` objective calls, including initialization."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if budget < max(4, ES_PARENTS if method == "ES" else CMA_POPULATION if method == "CMA-ES" else POPULATION):
        raise ValueError("Budget smaller than the initial population")
    rng = np.random.default_rng(seed)
    dimension = 2 * n_features
    counter = EvaluationCounter(budget)
    best_vector, best_score = None, float("inf")

    def evaluate(vector):
        nonlocal best_vector, best_score
        vector = np.clip(np.asarray(vector, dtype=float), LOWER_BOUND, UPPER_BOUND)
        score = counter.evaluate(objective, decode(vector, n_features))
        if score < best_score or best_vector is None:
            best_vector, best_score = vector.copy(), score
        return score

    if method == "CMA-ES":
        # CMA-ES has its own Gaussian population; its starting mean uses the same
        # uniform region as the other methods. pycma handles boundary transforms.
        initial_mean = rng.uniform(LOWER_BOUND, UPPER_BOUND, dimension)
        strategy = cma.CMAEvolutionStrategy(initial_mean, 0.3, {
            "bounds": [LOWER_BOUND, UPPER_BOUND], "popsize": CMA_POPULATION,
            "seed": int(rng.integers(1, 2**31)), "verbose": -9,
            "verb_log": 0, "tolfun": 0, "tolx": 0, "tolstagnation": 0,
        })
        while counter.remaining:
            candidates = strategy.ask()
            # tell() needs one score per candidate; at a partial final generation,
            # only evaluated candidates enter the result and no update is needed.
            if counter.remaining < len(candidates):
                for candidate in candidates[:counter.remaining]:
                    evaluate(candidate)
                break
            scores = [evaluate(candidate) for candidate in candidates]
            strategy.tell(candidates, scores)
    else:
        initial_size = ES_PARENTS if method == "ES" else POPULATION
        population = rng.uniform(LOWER_BOUND, UPPER_BOUND, (initial_size, dimension))
        scores = np.array([evaluate(vector) for vector in population])
        if method == "PSO":
            velocity = rng.uniform(-0.1, 0.1, population.shape)
            personal = population.copy()
            personal_scores = scores.copy()
        while counter.remaining:
            if method == "GA":
                # Tournament selection, arithmetic crossover, Gaussian mutation,
                # and one carried elite retain the genetic search mechanism.
                elite = int(np.argmin(scores))
                children = []
                while len(children) < min(POPULATION, counter.remaining):
                    parents = []
                    for _ in range(2):
                        pair = rng.integers(0, len(population), size=2)
                        parents.append(population[pair[np.argmin(scores[pair])]])
                    mix = rng.uniform(0, 1, dimension)
                    child = mix * parents[0] + (1 - mix) * parents[1]
                    mask = rng.random(dimension) < 1 / dimension
                    child[mask] += rng.normal(0, 0.15, mask.sum())
                    children.append(np.clip(child, LOWER_BOUND, UPPER_BOUND))
                children = np.asarray(children)
                child_scores = np.array([evaluate(vector) for vector in children])
                if len(children) == POPULATION:
                    combined = np.vstack((population[elite], children))
                    combined_scores = np.r_[scores[elite], child_scores]
                    selected = np.argsort(combined_scores)[:POPULATION]
                    population, scores = combined[selected], combined_scores[selected]
            elif method == "PSO":
                count = min(len(population), counter.remaining)
                leader = personal[np.argmin(personal_scores)]
                velocity[:count] = (0.7 * velocity[:count]
                    + 1.4 * rng.random((count, dimension)) * (personal[:count] - population[:count])
                    + 1.4 * rng.random((count, dimension)) * (leader - population[:count]))
                population[:count] = np.clip(population[:count] + velocity[:count], LOWER_BOUND, UPPER_BOUND)
                new_scores = np.array([evaluate(vector) for vector in population[:count]])
                improved = new_scores < personal_scores[:count]
                personal[:count][improved] = population[:count][improved]
                personal_scores[:count][improved] = new_scores[improved]
            elif method == "ES":
                count = min(POPULATION, counter.remaining)
                parent_ids = rng.integers(0, len(population), size=count)
                children = np.clip(population[parent_ids] + rng.normal(0, 0.15, (count, dimension)),
                                   LOWER_BOUND, UPPER_BOUND)
                child_scores = np.array([evaluate(vector) for vector in children])
                combined = np.vstack((population, children))
                combined_scores = np.r_[scores, child_scores]
                selected = np.argsort(combined_scores)[:ES_PARENTS]
                population, scores = combined[selected], combined_scores[selected]
            else:  # DE/rand/1/bin: three distinct donors excluding the target.
                count = min(len(population), counter.remaining)
                source = population.copy()
                for index in range(count):
                    donors = rng.choice(np.delete(np.arange(len(population)), index), 3, replace=False)
                    mutant = np.clip(source[donors[0]] + 0.8 *
                                     (source[donors[1]] - source[donors[2]]),
                                     LOWER_BOUND, UPPER_BOUND)
                    mask = rng.random(dimension) < 0.9
                    mask[rng.integers(dimension)] = True
                    trial = np.where(mask, mutant, source[index])
                    score = evaluate(trial)
                    if score <= scores[index]:
                        population[index], scores[index] = trial, score
    if counter.count != budget:
        raise AssertionError(f"{method} spent {counter.count} of {budget} evaluations")
    return SearchResult(decode(best_vector, n_features), best_score, counter.count, counter.history)
