"""Classical exact baseline by exhaustive enumeration.

Suitable for tiny instances only: k ** (T * n) candidate schedules.
The instance supplies objective evaluation and feasibility checking.
No quantum libraries or third-party dependencies are required.
"""

from itertools import product


def enumerate_feasible(T, n, k, evaluate, is_feasible):
    """Return (states, (C, W, O, D)) records for feasible schedules.

    states is a tuple ordered by (time, generator). Each entry selects
    exactly one state, so one-hot selection is implicit in this encoding.
    is_feasible(states) must check the instance's remaining constraints.
    """
    if any(not isinstance(v, int) or v < 1 for v in (T, n, k)):
        raise ValueError('T, n, and k must be positive integers')
    return [(states, evaluate(states))
            for states in product(range(k), repeat=T * n)
            if is_feasible(states)]


def weighted_score(terms, weights):
    """Scalar objective with both arguments ordered as C, W, O, D."""
    if len(terms) != 4 or len(weights) != 4:
        raise ValueError('Expected four terms and four weights: C, W, O, D')
    return sum(weight * value for weight, value in zip(weights, terms))


def solve_exact(records, weights):
    """Return (states, terms, score) for a global weighted minimum.

    Pass the complete output of enumerate_feasible for an exact result.
    Ties are resolved by lexicographic state order for reproducibility.
    """
    if not records:
        raise ValueError('No feasible schedules')
    states, terms = min(
        records,
        key=lambda row: (weighted_score(row[1], weights), row[0]),
    )
    return states, terms, weighted_score(terms, weights)
