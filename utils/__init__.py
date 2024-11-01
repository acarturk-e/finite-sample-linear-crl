"""
Credit (with minor modifications):
https://github.com/acarturk-e/score-based-crl
(used with author's permission)
"""

__all__ = [
    # Submodules
    "dag",
    # Functions
    "cov",
    "gaussian_score_est",
    "powerset",
]

from collections.abc import Iterable
from typing import Any, Callable
from itertools import chain, combinations

import numpy as np
import numpy.typing as npt

from . import dag


def cov(
    x_samples: npt.NDArray[np.floating],
    y_samples: npt.NDArray[np.floating] | None = None,
    center_data: bool = True,
) -> npt.NDArray[np.floating]:
    """Computes batch covariance.

    - Input shapes: `(..., nsamples, n)` and `(..., nsamples, m)`
    - Output shape: `(..., n, m)`

    Second argument is optional; if not provided, computes `x_samples` vs `x_samples`.

    Third argument optionally disables subtracting product of means."""
    if y_samples is None:
        y_samples = x_samples

    assert (
        x_samples.ndim >= 2
        and x_samples.ndim == y_samples.ndim
        and x_samples.shape[:-1] == y_samples.shape[:-1]
    )

    if center_data:
        x_samples -= np.mean(x_samples, axis=-2, keepdims=True)
        y_samples -= np.mean(y_samples, axis=-2, keepdims=True)

    assert y_samples is not None  # pylance...
    return np.mean(x_samples[..., :, None] * y_samples[..., None, :], axis=-3)


def gaussian_score_est(
    x_samples: npt.NDArray[np.floating],
) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Estimates score function of multivariate Gaussian R.V.s using inverse covariance.

    - Input shape: `(nsamples, n, 1)`
    - Output: Function with input & output shapes `(..., n, 1)`."""
    assert x_samples.ndim == 3 and x_samples.shape[-1] == 1

    neg_x_precision_mat = -np.linalg.inv(cov(x_samples[:, :, 0]))

    def score_est(
        x_samples: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        return neg_x_precision_mat @ x_samples

    return score_est


def graph_diff(
    g1: npt.NDArray[np.bool_],
    g2: npt.NDArray[np.bool_],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    edges1 = set(zip(np.where(g1)[0], np.where(g1)[1]))
    edges2 = set(zip(np.where(g2)[0], np.where(g2)[1]))

    g1_reversed = {(j, i) for (i, j) in edges1 if i != j}
    g2_reversed = {(j, i) for (i, j) in edges2 if i != j}

    additions = edges2 - edges1 - g1_reversed
    deletions = edges1 - edges2 - g2_reversed
    reversals = edges1 & g2_reversed

    return additions, deletions, reversals


def graph_diff2(
    g1: npt.NDArray[np.bool_],
    g2: npt.NDArray[np.bool_],
) -> tuple[int, int, int]:
    tp =  g1 &  g2
    fp =  g1 & ~g2
    fn = ~g1 &  g2
    return tp.sum(dtype=int), fp.sum(dtype=int), fn.sum(dtype=int)


def structural_hamming_distance(g1: np.ndarray, g2: np.ndarray) -> int:
    additions, deletions, reversals = graph_diff(g1, g2)
    return len(additions) + len(deletions) + len(reversals)


def powerset(iterable: Iterable[Any]):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
