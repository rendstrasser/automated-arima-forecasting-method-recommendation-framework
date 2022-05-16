"""Functions for working with the Level-Crossing-Rate (LCR)."""
import numpy as np


def lcr(x: np.ndarray, n_buckets: 50):
    """Performs LCR (Level-Crossing-Rate) transformation (no longer maintained)."""
    assert len(x.shape) == 1  # one-dimensional time-series

    max_y = np.max(x)
    y_step_size = (max_y - np.min(x)) / n_buckets

    lcr = np.zeros(n_buckets)
    total_count = 0

    for i in range(n_buckets):
        level = max_y - i * y_step_size

        prev_x_j = None
        for x_j in x:
            if prev_x_j and (prev_x_j - level) * (x_j - level) < 0:
                lcr[i] += 1
                total_count += 1

            prev_x_j = x_j

    # lcr /= n_buckets - 1
    lcr /= total_count

    return lcr


def randomly_permutate(x):
    """Randomly permutates a time series (no longer maintained)."""
    rng = np.random.default_rng()
    return rng.permutation(x)
