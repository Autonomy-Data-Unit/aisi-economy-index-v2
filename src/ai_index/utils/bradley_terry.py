"""Bradley-Terry model fitting for pairwise comparison scoring.

Implements the Davidson extension (ties as half-weight observations) and
adaptive pair generation for multi-round scoring.
"""

import numpy as np
from scipy.optimize import minimize


def fit_bradley_terry(
    comparisons: list[tuple[int, int, str]],
    n_items: int,
    reg_lambda: float = 1e-4,
) -> np.ndarray:
    """Fit a Bradley-Terry model from pairwise comparison outcomes.

    Uses the Davidson (1970) extension for ties: each tie is converted to two
    half-weight observations (one A>B, one B>A). The model is fitted by
    minimizing the negative log-likelihood with L2 regularization via L-BFGS-B.

    Args:
        comparisons: List of (item_a_idx, item_b_idx, outcome) where outcome
            is "A" (a wins), "B" (b wins), or "tie".
        n_items: Total number of items.
        reg_lambda: L2 regularization strength. Prevents divergence for
            items with few comparisons.

    Returns:
        Array of latent scores (length n_items). Higher = more exposed.
    """
    # Build observation arrays: (a_idx, b_idx, y, weight)
    # y=1 means a beats b, y=0 means b beats a
    a_list, b_list, y_list, w_list = [], [], [], []

    for a_idx, b_idx, outcome in comparisons:
        if outcome == "A":
            a_list.append(a_idx)
            b_list.append(b_idx)
            y_list.append(1.0)
            w_list.append(1.0)
        elif outcome == "B":
            a_list.append(a_idx)
            b_list.append(b_idx)
            y_list.append(0.0)
            w_list.append(1.0)
        elif outcome == "tie":
            # Davidson: tie -> two half-weight observations
            a_list.append(a_idx)
            b_list.append(b_idx)
            y_list.append(1.0)
            w_list.append(0.5)
            a_list.append(a_idx)
            b_list.append(b_idx)
            y_list.append(0.0)
            w_list.append(0.5)

    a_idx = np.array(a_list, dtype=np.int64)
    b_idx = np.array(b_list, dtype=np.int64)
    y = np.array(y_list, dtype=np.float64)
    w = np.array(w_list, dtype=np.float64)

    eps = 1e-12

    def neg_log_likelihood(theta):
        delta = theta[a_idx] - theta[b_idx]
        p = 1.0 / (1.0 + np.exp(-delta))
        nll = -np.sum(w * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
        nll += 0.5 * reg_lambda * np.dot(theta, theta)
        return nll

    def gradient(theta):
        delta = theta[a_idx] - theta[b_idx]
        p = 1.0 / (1.0 + np.exp(-delta))
        residual = w * (y - p)  # weighted residual

        grad = np.zeros(n_items, dtype=np.float64)
        np.add.at(grad, a_idx, residual)
        np.add.at(grad, b_idx, -residual)
        grad = -grad  # negate for minimization
        grad += reg_lambda * theta
        return grad

    theta0 = np.zeros(n_items, dtype=np.float64)
    result = minimize(
        neg_log_likelihood,
        theta0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-8},
    )

    # Center scores (remove arbitrary constant)
    theta = result.x
    theta -= theta.mean()
    return theta


def generate_random_pairs(
    n_items: int,
    comparisons_per_item: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Generate random pairs ensuring each item appears ~comparisons_per_item times.

    Args:
        n_items: Number of items.
        comparisons_per_item: Target comparisons per item.
        rng: Numpy random generator.

    Returns:
        List of (item_a_idx, item_b_idx) pairs.
    """
    target_total = n_items * comparisons_per_item // 2
    pairs = set()
    indices = np.arange(n_items)

    while len(pairs) < target_total:
        shuffled = rng.permutation(indices)
        # Pair consecutive elements
        for i in range(0, len(shuffled) - 1, 2):
            a, b = int(shuffled[i]), int(shuffled[i + 1])
            pair = (min(a, b), max(a, b))
            pairs.add(pair)
            if len(pairs) >= target_total:
                break

    return [(a, b) for a, b in pairs]


def generate_adaptive_pairs(
    theta: np.ndarray,
    comparisons_per_item: int,
    rng: np.random.Generator,
    threshold: float | None = None,
) -> list[tuple[int, int]]:
    """Generate pairs focused on items with similar latent scores.

    For each item, sample comparison partners from items whose theta is within
    `threshold` of the item's theta. Falls back to random partners if too few
    neighbors are available.

    Args:
        theta: Current latent score estimates (length n_items).
        comparisons_per_item: Target comparisons per item.
        rng: Numpy random generator.
        threshold: Maximum |theta_i - theta_j| for adaptive pairing.
            Defaults to 1 standard deviation of theta.

    Returns:
        List of (item_a_idx, item_b_idx) pairs.
    """
    n_items = len(theta)
    if threshold is None:
        threshold = float(np.std(theta))
        if threshold < 1e-6:
            # Theta is flat (e.g. first round), fall back to random
            return generate_random_pairs(n_items, comparisons_per_item, rng)

    # Sort items by theta for efficient neighbor finding
    sorted_indices = np.argsort(theta)
    sorted_theta = theta[sorted_indices]

    pairs = set()
    item_counts = np.zeros(n_items, dtype=int)
    all_indices = np.arange(n_items)

    # For each item, find neighbors within threshold and sample partners
    for i in range(n_items):
        if item_counts[i] >= comparisons_per_item:
            continue

        needed = comparisons_per_item - item_counts[i]

        # Find items within threshold using sorted array
        pos = np.searchsorted(sorted_theta, theta[i])
        lo = np.searchsorted(sorted_theta, theta[i] - threshold, side="left")
        hi = np.searchsorted(sorted_theta, theta[i] + threshold, side="right")
        neighbors = sorted_indices[lo:hi]
        # Exclude self
        neighbors = neighbors[neighbors != i]

        if len(neighbors) >= needed:
            chosen = rng.choice(neighbors, size=needed, replace=False)
        elif len(neighbors) > 0:
            # Take all neighbors + random fallback
            remaining = needed - len(neighbors)
            others = np.setdiff1d(all_indices, np.append(neighbors, i))
            fallback = rng.choice(others, size=min(remaining, len(others)), replace=False)
            chosen = np.concatenate([neighbors, fallback])
        else:
            # No neighbors, fully random
            others = np.setdiff1d(all_indices, [i])
            chosen = rng.choice(others, size=min(needed, len(others)), replace=False)

        for j in chosen:
            j = int(j)
            pair = (min(i, j), max(i, j))
            if pair not in pairs:
                pairs.add(pair)
                item_counts[i] += 1
                item_counts[j] += 1

    return [(a, b) for a, b in pairs]


def normalize_scores(theta: np.ndarray, floor_cutoff: float | None = None) -> np.ndarray:
    """Normalize BT latent scores to [0, 1].

    If floor_cutoff is provided (on the min-max scale), tasks at or below
    the cutoff are set to 0 and the remaining tasks are rescaled to [0, 1].
    This is used to collapse the physical-task tie cluster to zero.

    If floor_cutoff is None, uses simple min-max scaling.

    Args:
        theta: Raw latent scores from BT fitting.
        floor_cutoff: Cutoff on the [0, 1] min-max scale. Tasks at or
            below this value are set to 0, rest rescaled. If None, no
            cutoff is applied.

    Returns:
        Normalized scores in [0, 1].
    """
    lo = np.min(theta)
    hi = np.max(theta)
    if hi - lo < 1e-10:
        return np.full_like(theta, 0.5)

    # First do min-max to [0, 1]
    minmax = (theta - lo) / (hi - lo)

    if floor_cutoff is None:
        return minmax

    # Apply cutoff: everything <= cutoff -> 0, rest rescaled
    return np.where(minmax <= floor_cutoff, 0.0, (minmax - floor_cutoff) / (1.0 - floor_cutoff))
