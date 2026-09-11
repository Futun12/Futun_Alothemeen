"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(
    values,
    *,
    n_boot=2000,
    seed=42,
    alpha=0.05,
):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")

    rng = np.random.default_rng(seed)

    point_estimate = float(np.mean(values))

    bootstrap_estimates = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_estimates[i] = np.mean(
            sample
        )

    lower = float(
        np.quantile(
            bootstrap_estimates,
            alpha / 2,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_estimates,
            1 - alpha / 2,
        )
    )

    return point_estimate, lower, upper


def paired_bootstrap_diff(
    a,
    b,
    *,
    n_boot=2000,
    seed=42,
    alpha=0.05,
):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:
        raise ValueError(
            "a and b must not be empty"
        )

    if len(a) != len(b):
        raise ValueError(
            "a and b must have the same length"
        )

    rng = np.random.default_rng(seed)

    differences = a - b

    point_estimate = float(
        np.mean(differences)
    )

    bootstrap_differences = np.empty(
        n_boot,
        dtype=float,
    )

    n = len(a)

    for i in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sampled_diff = (
            a[indices] - b[indices]
        )

        bootstrap_differences[i] = np.mean(
            sampled_diff
        )

    lower = float(
        np.quantile(
            bootstrap_differences,
            alpha / 2,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_differences,
            1 - alpha / 2,
        )
    )

    return point_estimate, lower, upper