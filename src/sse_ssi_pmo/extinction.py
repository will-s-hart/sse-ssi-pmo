"""Analytic extinction-probability building blocks for the SSE and SSI models.

For an offspring distribution :math:`\\mathrm{NB}(\\text{mean}=\\mu,\\text{disp}=\\kappa)`,
branching-process theory says the extinction probability ``q`` is the smallest
non-negative fixed point of the pgf

.. math::

    G(s) = \\bigl(1 + (\\mu / \\kappa)(1 - s)\\bigr)^{-\\kappa}.

If ``mu <= 1`` the only fixed point in ``[0, 1]`` is ``q = 1``; otherwise the
unique fixed point in ``(0, 1)`` is found by ``scipy.optimize.brentq`` on
``f(q) = G(q) - q``.

After conditioning on an observed incidence history ``I_0, ..., I_r``, the
extinction probability is ``q_r = G_r(q)`` where ``G_r`` is the pgf of the
remaining offspring distribution of the observed cases. See ``notes/notes.tex``
for the SSE and SSI derivations. The closed forms exposed here are:

* SSE (general history): ``q_r = q ** Lambda`` with
  ``Lambda = sum_{s=0..r} I_s * (1 - F_{r-s})``.
* SSI (special case of cases on day 0 only):
  ``q_r = (1 + R0 (1-F_r)/(k + R0 F_r) * (1 - q)) ** (-k I_0)``.

The functions in this module are private helpers used by ``pmo.py``; they do
not validate user-facing inputs (the dispatcher is responsible for that).
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from sse_ssi_pmo.serial_interval import cumulative

# Upper bracket for brentq: just below 1 so we don't trivially hit q = 1.
_Q_UPPER = 1.0 - 1e-12


def _nb_extinction_prob(mean: float, disp: float) -> float:
    """Extinction probability for a single ``NB(mean, disp)`` offspring distribution.

    ``disp`` is the NB shape/dispersion parameter (the exponent ``-kappa`` in
    the pgf). When the offspring distribution has ``mean <= 1`` extinction is
    certain, so we return 1.
    """
    if mean <= 1.0 or disp <= 0.0:
        return 1.0

    def f(q: float) -> float:
        return (1.0 + (mean / disp) * (1.0 - q)) ** (-disp) - q

    # f(0) = G(0) > 0; f(_Q_UPPER) < 0 because mean > 1 implies G'(1) > 1, so
    # G(q) < q just below 1. Thus brentq finds the unique root in (0, 1).
    return float(scipy.optimize.brentq(f, 0.0, _Q_UPPER))


def _q_array(
    R0: ArrayLike, k: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve ``q`` on the broadcast of ``(R0, k)`` — one root-find per pair.

    Returns ``(q, R0_b, k_b)`` all sharing the broadcast shape, so callers can
    reuse the broadcast ``R0_b`` / ``k_b`` when evaluating closed forms.
    """
    R0_arr = np.asarray(R0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    R0_b, k_b = np.broadcast_arrays(R0_arr, k_arr)
    q = np.empty(R0_b.shape, dtype=np.float64)
    for idx in np.ndindex(R0_b.shape):
        q[idx] = _nb_extinction_prob(float(R0_b[idx]), float(k_b[idx]))
    return q, R0_b, k_b


def _lambda_from_history(
    w: NDArray[np.float64], history: NDArray[np.int64]
) -> float:
    """Pooled effective weight ``Lambda = sum_{s=0..r} I_s (1 - F_{r-s})``.

    Indices ``r - s`` exceeding the support of ``w`` are clipped to
    ``F.size - 1``, where ``F = sum(w)``: the residual weight at lags beyond
    ``len(w)`` is taken to be zero (consistent with how ``cumulative``
    truncates the serial-interval distribution).
    """
    F = cumulative(w)
    r = history.size - 1
    idx = np.minimum(r - np.arange(r + 1), F.size - 1)
    return float((history.astype(np.float64) * (1.0 - F[idx])).sum())


def _pmo_sse_analytic(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSE PMO ``1 - q ** Lambda`` for the general observed history.

    Broadcasts over ``(R0, k)``; ``w`` and ``history`` are single 1-D arrays.
    """
    Lambda = _lambda_from_history(w, history)
    q, _, _ = _q_array(R0, k)
    return 1.0 - q**Lambda


def _pmo_ssi_analytic_special(
    R0: ArrayLike,
    k: ArrayLike,
    I_0: int,
    F_r: float,
) -> NDArray[np.float64]:
    """SSI PMO for the day-0-only special case (cases on day 0, then zeros).

    Broadcasts over ``(R0, k)``; ``I_0`` is the number of cases on day 0 and
    ``F_r`` is the cumulative serial-interval mass at lag ``r``.
    """
    if not 0.0 <= F_r <= 1.0:
        raise ValueError("F_r must lie in [0, 1]")
    if I_0 < 1:
        raise ValueError("I_0 must be at least 1")
    q, R0_b, k_b = _q_array(R0, k)
    eff = R0_b * (1.0 - F_r) / (k_b + R0_b * F_r)
    return 1.0 - (1.0 + eff * (1.0 - q)) ** (-k_b * I_0)
