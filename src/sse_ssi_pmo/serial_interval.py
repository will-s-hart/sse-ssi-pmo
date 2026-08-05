"""Discretisation of a continuous serial-interval distribution.

Adapted from ``temp/_si_discr.py`` (the Cori et al. method, web appendix 11 of
https://doi.org/10.1093/aje/kwt133): the probability mass at integer lag ``x``
is the integral of ``(1 - |x - y|) * pdf(y)`` over ``y in [x - 1, x + 1]``.
"""

from __future__ import annotations

import functools

import numpy as np
import scipy.integrate
import scipy.stats
from numpy.typing import NDArray


def discretise(
    dist_cont: scipy.stats.rv_continuous,
    *,
    max_val: int,
    allow_zero: bool = False,
) -> NDArray[np.float64]:
    """Discretise a continuous distribution by the Cori et al. method.

    Parameters
    ----------
    dist_cont
        A frozen continuous ``scipy.stats`` distribution (i.e. one with
        parameters already supplied), so that ``dist_cont.pdf(y)`` works.
    max_val
        Largest integer lag with its own probability mass; residual mass is
        added to this bin.
    allow_zero
        If ``False`` (default), the mass at lag 0 is folded into lag 1 and the
        returned array starts at lag 1. If ``True``, lag 0 is included.

    Returns
    -------
    Probabilities ``p[0], p[1], ...`` summing to 1.
    """
    if max_val < 0:
        raise ValueError("max_val must be non-negative")
    if not allow_zero and max_val < 1:
        raise ValueError("max_val must be at least 1 when allow_zero is False")

    def _integrand(x: float, y: float) -> float:
        return (1.0 - abs(x - y)) * dist_cont.pdf(y)

    x_vec = np.arange(0, max_val + 1, dtype=int)
    p_vec = np.zeros(len(x_vec), dtype=np.float64)
    for i, x in enumerate(x_vec):
        integrand = functools.partial(_integrand, float(x))
        lower = float(x) - 1.0 if x > 0 else 1e-12
        upper = float(x) + 1.0
        p_vec[i] = scipy.integrate.quad(integrand, lower, upper)[0]

    if not allow_zero:
        p_vec[1] = p_vec[1] + p_vec[0]
        p_vec = p_vec[1:]

    p_vec[-1] = p_vec[-1] + 1.0 - float(np.sum(p_vec))
    return p_vec


def discretise_gamma(
    *,
    mean: float,
    sd: float,
    max_val: int,
    allow_zero: bool = False,
) -> NDArray[np.float64]:
    """Discretise a Gamma serial interval parameterised by mean and SD.

    For a Gamma with mean ``mu`` and standard deviation ``sigma``:
    ``shape = mu**2 / sigma**2`` and ``scale = sigma**2 / mu``.
    """
    if mean <= 0 or sd <= 0:
        raise ValueError("mean and sd must be positive")
    shape = mean**2 / sd**2
    scale = sd**2 / mean
    dist = scipy.stats.gamma(a=shape, scale=scale)
    return discretise(dist, max_val=max_val, allow_zero=allow_zero)


def cumulative(w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative serial-interval distribution ``F_r`` for ``r = 0, 1, ...``.

    Returns an array of length ``len(w) + 1`` with ``F[0] = 0`` and
    ``F[r] = sum(w[:r])``.
    """
    return np.concatenate(([0.0], np.cumsum(w)))


# ---------------------------------------------------------------------------
# Inverse-CDF sampling of a discrete delay distribution (used by the
# onset-anchored and bridge forward simulators). Kept here — a low-level module
# with no package imports — so both simulation.py and simulation_delay.py can
# reuse it without an import cycle.
# ---------------------------------------------------------------------------


def delay_cdf(
    weights: NDArray[np.float64], *, start: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return ``(cdf, support)`` for inverse-CDF sampling of a discrete delay.

    ``weights[i]`` is proportional to the probability of a delay of
    ``start + i`` (``support[i] = start + i``); ``cdf`` is the normalised
    cumulative. ``start=1`` for a delay supported on ``{1, 2, ...}`` (e.g. an
    incubation period that cannot be zero), ``start=0`` to allow a zero delay.
    """
    p = weights / weights.sum()
    cdf = np.cumsum(p)
    support = np.arange(start, start + weights.size, dtype=np.int64)
    return cdf, support


def sample_delay(
    cdf: NDArray[np.float64],
    support: NDArray[np.int64],
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw ``size`` delays by inverse-CDF sampling from ``(cdf, support)``."""
    u = rng.random(size)
    idx = np.searchsorted(cdf, u, side="right")
    np.clip(idx, 0, support.size - 1, out=idx)
    return support[idx]
