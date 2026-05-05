"""Extinction-probability solvers for the SSE and SSI models.

For an offspring distribution :math:`\\mathrm{NB}(\\text{mean}=\\mu,\\text{disp}=\\kappa)`,
branching-process theory says the extinction probability ``q`` is the smallest
non-negative fixed point of the pgf

.. math::

    G(s) = \\bigl(1 + (\\mu / \\kappa)(1 - s)\\bigr)^{-\\kappa}.

If ``mu <= 1`` the only fixed point in ``[0, 1]`` is ``q = 1``; otherwise the
unique fixed point in ``(0, 1)`` is found by ``scipy.optimize.brentq`` on
``f(q) = G(q) - q``.

After conditioning on no cases for ``r`` days, the extinction probability is
``q_r = G_r(q)`` where ``G_r`` is the pgf of the index case's *remaining*
offspring distribution and ``q`` is the standard fixed point above. Each
remaining offspring starts a fresh branching process governed by the original
offspring distribution, so each lineage goes extinct with probability ``q``;
the conditioning does not propagate to later generations. Consequently only
one root-finding call per ``(R0, k)`` pair is needed — every ``q_r`` is then
a closed-form evaluation.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

# Upper bracket for brentq: just below 1 so we don't trivially hit q = 1.
_Q_UPPER = 1.0 - 1e-12


def _nb_extinction_prob(mean: float, disp: float) -> float:
    """Extinction probability for a single ``NB(mean, disp)`` offspring distribution.

    ``disp`` is the NB shape/dispersion parameter (the exponent ``-kappa`` in
    the pgf). When the offspring distribution has ``mean == 0`` (e.g. the
    effective offspring distribution after conditioning has all its mass at
    zero) extinction is certain, so we return 1.
    """
    if mean <= 1.0 or disp <= 0.0:
        return 1.0

    def f(q: float) -> float:
        return (1.0 + (mean / disp) * (1.0 - q)) ** (-disp) - q

    # f(0) = G(0) > 0; f(_Q_UPPER) < 0 because mean > 1 implies G'(1) > 1, so
    # G(q) < q just below 1. Thus brentq finds the unique root in (0, 1).
    return float(scipy.optimize.brentq(f, 0.0, _Q_UPPER))


def q_unconditional(R0: float, k: float) -> float:
    """Extinction probability following 1 case on day 0 (no conditioning)."""
    return _nb_extinction_prob(mean=R0, disp=k)


def q_sse(R0: float, k: float, F_r: float) -> float:
    """SSE extinction probability after ``r`` time-steps with no cases.

    The remaining offspring pgf is ``G_r(s) = G(s)^(1 - F_r)``, so
    ``q_r = G_r(q) = q ** (1 - F_r)``.
    """
    if not 0.0 <= F_r <= 1.0:
        raise ValueError("F_r must lie in [0, 1]")
    q = _nb_extinction_prob(mean=R0, disp=k)
    return q ** (1.0 - F_r)


def q_ssi(R0: float, k: float, F_r: float) -> float:
    """SSI extinction probability after ``r`` time-steps with no cases.

    The remaining offspring distribution (after the Bayesian update on the
    index case's latent infectivity) is
    ``NB(size=k, mean=k R0 (1-F_r) / (k + R0 F_r))``, and
    ``q_r = G_r(q) = (1 + R0 (1-F_r) / (k + R0 F_r) * (1 - q)) ** (-k)``.
    """
    if not 0.0 <= F_r <= 1.0:
        raise ValueError("F_r must lie in [0, 1]")
    q = _nb_extinction_prob(mean=R0, disp=k)
    return (1.0 + R0 * (1.0 - F_r) / (k + R0 * F_r) * (1.0 - q)) ** (-k)


# ---------------------------------------------------------------------------
# vectorised PMO wrappers
# ---------------------------------------------------------------------------


def _q_array(
    R0: ArrayLike, k: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve ``q`` on the broadcast of ``(R0, k)`` only — one root-find per pair.

    Returns ``(q, R0_b, k_b)`` all sharing the broadcast shape, so callers can
    reuse the broadcast ``R0_b`` / ``k_b`` when evaluating closed forms.
    """
    R0_arr = np.atleast_1d(np.asarray(R0, dtype=np.float64))
    k_arr = np.atleast_1d(np.asarray(k, dtype=np.float64))
    R0_b, k_b = np.broadcast_arrays(R0_arr, k_arr)
    q = np.empty(R0_b.shape, dtype=np.float64)
    for idx in np.ndindex(R0_b.shape):
        q[idx] = _nb_extinction_prob(float(R0_b[idx]), float(k_b[idx]))
    return q, R0_b, k_b


def _as_f_r_array(F_r: ArrayLike) -> NDArray[np.float64]:
    arr = np.atleast_1d(np.asarray(F_r, dtype=np.float64))
    if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
        raise ValueError("F_r must lie in [0, 1]")
    return arr


def pmo_unconditional(R0: ArrayLike, k: ArrayLike) -> NDArray[np.float64]:
    """Probability of major outbreak with no conditioning (broadcastable)."""
    q, _, _ = _q_array(R0, k)
    return 1.0 - q


def pmo_sse(R0: ArrayLike, k: ArrayLike, F_r: ArrayLike) -> NDArray[np.float64]:
    """SSE PMO after conditioning on ``r`` steps without cases (broadcastable).

    Uses ``q_r = q ** (1 - F_r)`` — one root-find per ``(R0, k)``, then a
    vectorised closed-form evaluation across ``F_r``.
    """
    F_arr = _as_f_r_array(F_r)
    q, _, _ = _q_array(R0, k)
    q_b, F_b = np.broadcast_arrays(q, F_arr)
    return 1.0 - q_b ** (1.0 - F_b)


def pmo_ssi(R0: ArrayLike, k: ArrayLike, F_r: ArrayLike) -> NDArray[np.float64]:
    """SSI PMO after conditioning on ``r`` steps without cases (broadcastable).

    Uses ``q_r = (1 + R0 (1-F_r) / (k + R0 F_r) * (1 - q)) ** (-k)`` — one
    root-find per ``(R0, k)``, then a vectorised closed-form evaluation.
    """
    F_arr = _as_f_r_array(F_r)
    q, R0_b, k_b = _q_array(R0, k)
    R0_b, k_b, q_b, F_b = np.broadcast_arrays(R0_b, k_b, q, F_arr)
    eff = R0_b * (1.0 - F_b) / (k_b + R0_b * F_b)
    return 1.0 - (1.0 + eff * (1.0 - q_b)) ** (-k_b)
