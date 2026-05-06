"""Probability of major outbreak (PMO) — public dispatcher API.

Two top-level functions, one per model, each taking the observed incidence
``history = (I_0, I_1, ..., I_r)`` together with the serial-interval weights
``w`` and dispatching to either an analytic closed-form or a Monte-Carlo
simulation:

* :func:`pmo_sse` — SSE model; ``method ∈ {"analytic", "simulation"}``.
* :func:`pmo_ssi` — SSI model; ``method ∈ {"analytic", "simulation", "mcmc"}``.
  ``"analytic"`` only supports histories with cases on day 0 alone (followed
  by zeros). ``"mcmc"`` samples the latent infectivities via HMC and averages
  the conditional extinction probability over draws — see ``notes/notes.tex``
  for the derivation.

Both functions broadcast over ``R0`` and ``k`` (scalar or array). Scalar
inputs return a Python ``float``; array inputs return a NumPy array of the
broadcast shape. The simulation and MCMC paths run the inner backend once per
``(R0, k)`` combination and, when ``show_progress=True`` and there is more
than one combination, wrap the parameter-combination loop with a single
``tqdm`` bar (passing ``show_progress=False`` to each inner call).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from sse_ssi_pmo.extinction import (
    _pmo_sse_analytic,
    _pmo_ssi_analytic_special,
    _pmo_ssi_mcmc,
)
from sse_ssi_pmo.serial_interval import cumulative
from sse_ssi_pmo.simulation import _pmo_sse_sim, _pmo_ssi_sim


def _validate_history(history: ArrayLike) -> NDArray[np.int64]:
    arr = np.asarray(history, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("history must be a non-empty 1-D array of integers")
    if arr.min() < 0:
        raise ValueError("history entries must be non-negative")
    if arr[0] < 1:
        raise ValueError("history must have at least one case on day 0 (history[0] >= 1)")
    return arr


def _dispatch(
    fn,
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    label: str,
    fn_kwargs: dict,
) -> NDArray[np.float64]:
    """Broadcast ``(R0, k)`` and call ``fn`` once per combination.

    ``show_progress`` is consumed here: with multiple combinations and
    ``show_progress=True``, wrap the outer loop in a single ``tqdm`` and
    silence the inner per-call bars. With one combination, pass
    ``show_progress`` through unchanged.
    """
    show_progress = fn_kwargs.pop("show_progress", False)
    R0_arr = np.asarray(R0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    R0_b, k_b = np.broadcast_arrays(R0_arr, k_arr)
    out = np.empty(R0_b.shape, dtype=np.float64)

    indices = list(np.ndindex(R0_b.shape))
    n_combos = len(indices)
    if show_progress and n_combos > 1:
        iterable = tqdm(indices, total=n_combos, desc=f"{label} params")
        inner_progress = False
    else:
        iterable = indices
        inner_progress = show_progress

    for idx in iterable:
        out[idx] = fn(
            float(R0_b[idx]),
            float(k_b[idx]),
            w,
            history,
            show_progress=inner_progress,
            **fn_kwargs,
        )
    return out


def pmo_sse(
    *,
    R0: ArrayLike,
    k: ArrayLike,
    w: ArrayLike,
    history: ArrayLike,
    method: Literal["analytic", "simulation"],
    **kwargs,
) -> float | NDArray[np.float64]:
    """Probability of major outbreak under the SSE model.

    Parameters
    ----------
    R0, k
        Reproduction number and dispersion parameter; scalar or array
        (broadcast jointly).
    w
        Discrete serial-interval weights, ``w[s-1] = w_s`` for ``s = 1, 2, ...``;
        treated as a non-negative 1-D array (need not sum to exactly 1, but
        any residual mass beyond ``len(w)`` is taken to be zero).
    history
        Observed incidence ``(I_0, I_1, ..., I_r)`` as a 1-D array of
        non-negative integers with ``history[0] >= 1``.
    method
        ``"analytic"`` for the closed-form ``1 - q ** Lambda``;
        ``"simulation"`` for a Monte-Carlo estimate (forwarded to the SSE
        simulation backend; takes ``n_sims``, ``threshold``, ``t_max``,
        ``rng``, ``show_progress`` as keyword arguments).
    """
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = _validate_history(history)
    scalar_inputs = np.ndim(R0) == 0 and np.ndim(k) == 0

    if method == "analytic":
        if kwargs:
            raise TypeError(
                f"pmo_sse(method='analytic') got unexpected keyword arguments: {sorted(kwargs)}"
            )
        out = _pmo_sse_analytic(R0, k, w_arr, hist_arr)
    elif method == "simulation":
        out = _dispatch(_pmo_sse_sim, R0, k, w_arr, hist_arr, "pmo_sse", kwargs)
    else:
        raise ValueError(f"pmo_sse: method must be 'analytic' or 'simulation', got {method!r}")

    if scalar_inputs:
        return float(np.asarray(out).reshape(()))
    return out


def pmo_ssi(
    *,
    R0: ArrayLike,
    k: ArrayLike,
    w: ArrayLike,
    history: ArrayLike,
    method: Literal["analytic", "simulation", "mcmc"],
    **kwargs,
) -> float | NDArray[np.float64]:
    """Probability of major outbreak under the SSI model.

    Parameters
    ----------
    R0, k
        Reproduction number and dispersion parameter; scalar or array
        (broadcast jointly).
    w
        Discrete serial-interval weights, ``w[s-1] = w_s`` for ``s = 1, 2, ...``.
    history
        Observed incidence ``(I_0, I_1, ..., I_r)`` as a 1-D array of
        non-negative integers with ``history[0] >= 1``.
    method
        ``"analytic"`` for the closed-form day-0-only special case (cases
        on day 0 followed by ``r`` days with no cases); raises
        :class:`ValueError` if the history has a non-zero entry past day 0.
        ``"simulation"`` for a Monte-Carlo estimate (forwarded to the SSI
        simulation backend; takes ``n_sims``, ``threshold``, ``t_max``,
        ``rng``, ``batch_size``, ``max_attempts``, ``show_progress`` as
        keyword arguments).
        ``"mcmc"`` for a Monte-Carlo estimate via MCMC over the latent
        infectivities (see ``notes/notes.tex``); keyword arguments are
        forwarded to ``pm.sample`` (e.g. ``draws``, ``tune``, ``chains``,
        ``progressbar``) plus ``thin`` for posterior thinning.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = _validate_history(history)
    scalar_inputs = np.ndim(R0) == 0 and np.ndim(k) == 0

    if method == "analytic":
        if kwargs:
            raise TypeError(
                f"pmo_ssi(method='analytic') got unexpected keyword arguments: {sorted(kwargs)}"
            )
        nonzero_after_day0 = np.flatnonzero(hist_arr[1:] != 0)
        if nonzero_after_day0.size > 0:
            offending_days = (nonzero_after_day0 + 1).tolist()
            raise ValueError(
                "pmo_ssi(method='analytic') requires a history with cases on "
                "day 0 only (followed by zeros); "
                f"got non-zero cases on day(s) {offending_days}. "
                "Use method='simulation' or method='mcmc'."
            )
        I_0 = int(hist_arr[0])
        r = hist_arr.size - 1
        F = cumulative(w_arr)
        F_r = float(F[min(r, F.size - 1)])
        out = _pmo_ssi_analytic_special(R0, k, I_0, F_r)
    elif method == "simulation":
        out = _dispatch(_pmo_ssi_sim, R0, k, w_arr, hist_arr, "pmo_ssi", kwargs)
    elif method == "mcmc":
        out = _dispatch(_pmo_ssi_mcmc, R0, k, w_arr, hist_arr, "pmo_ssi_mcmc", kwargs)
    else:
        raise ValueError(
            f"pmo_ssi: method must be 'analytic', 'simulation', or 'mcmc', got {method!r}"
        )

    if scalar_inputs:
        return float(np.asarray(out).reshape(()))
    return out


__all__ = ["pmo_sse", "pmo_ssi"]
