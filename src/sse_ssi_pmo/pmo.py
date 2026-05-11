"""Probability of major outbreak (PMO) — public dispatcher API.

Three top-level functions, each taking the observed incidence
``history = (I_0, I_1, ..., I_r)`` together with the serial-interval weights
``w`` and dispatching to either an analytic closed-form or a Monte-Carlo
simulation:

* :func:`pmo_sse` — SSE model; ``method ∈ {"analytic", "simulation"}``.
* :func:`pmo_ssi` — SSI model; ``method ∈ {"analytic", "simulation", "mcmc"}``.
  ``"analytic"`` only supports histories with cases on day 0 alone (followed
  by zeros). ``"mcmc"`` samples the latent infectivities via HMC and averages
  the conditional extinction probability over draws — see ``notes/notes.tex``
  for the derivation.
* :func:`pmo_uncertain` — Bayesian model average across SSE and SSI given a
  prior ``prior_sse`` on the SSE model. ``method ∈ {"analytic",
  "simulation"}``; ``"analytic"`` only supports the day-0-only history.
  Returns a :class:`PmoUncertainResult` NamedTuple with the model-averaged
  PMO, the posterior probability of SSE, and the per-model PMOs.

All functions broadcast over ``R0`` and ``k`` (scalar or array). Scalar
inputs return a Python ``float`` (or float-valued NamedTuple for
``pmo_uncertain``); array inputs return a NumPy array (or array-valued
NamedTuple) of the broadcast shape. The simulation and MCMC paths run the
inner backend once per ``(R0, k)`` combination and, when
``show_progress=True`` and there is more than one combination, wrap the
parameter-combination loop with a single ``tqdm`` bar (passing
``show_progress=False`` to each inner call).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from sse_ssi_pmo._history import classify_history
from sse_ssi_pmo.extinction import (
    _pmo_sse_analytic,
    _pmo_ssi_analytic,
    _pmo_ssi_mcmc,
    _pmo_uncertain_analytic,
    _pmo_uncertain_mcmc,
)
from sse_ssi_pmo.likelihood import SsiEvidenceMethod
from sse_ssi_pmo.simulation import _pmo_sse_sim, _pmo_ssi_sim, _pmo_uncertain_sim


class PmoUncertainResult(NamedTuple):
    """Return type of :func:`pmo_uncertain`.

    ``pmo`` is the model-averaged probability of a major outbreak;
    ``posterior_sse`` is the posterior probability of the SSE model given
    the observed history; ``pmo_sse`` and ``pmo_ssi`` are the per-model
    PMOs. Each field is a Python ``float`` for scalar ``(R0, k)`` inputs
    and a NumPy array of the broadcast shape otherwise.
    """

    pmo: float | NDArray[np.float64]
    posterior_sse: float | NDArray[np.float64]
    pmo_sse: float | NDArray[np.float64]
    pmo_ssi: float | NDArray[np.float64]


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
        ``"analytic"`` for the closed-form solution. Supports histories
        with cases on day 0 only, on day 0 and one later day (case (i)),
        or on day 0 and two later days (case (ii)); raises
        :class:`ValueError` for histories with three or more later non-zero
        days. ``"simulation"`` for a Monte-Carlo estimate (forwarded to the
        SSI simulation backend; takes ``n_sims``, ``threshold``, ``t_max``,
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
        if classify_history(hist_arr)["kind"] == "general":
            offending_days = (np.flatnonzero(hist_arr[1:] != 0) + 1).tolist()
            raise ValueError(
                "pmo_ssi(method='analytic') requires a history with cases on "
                "day 0 and at most two later days; "
                f"got non-zero cases on day(s) {offending_days}. "
                "Use method='simulation' or method='mcmc'."
            )
        out = _pmo_ssi_analytic(R0, k, w_arr, hist_arr)
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


def _dispatch_multi(
    fn,
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    label: str,
    fn_kwargs: dict,
    output_keys: tuple[str, ...],
) -> dict[str, NDArray[np.float64]]:
    """Like :func:`_dispatch` but ``fn`` returns a dict per ``(R0, k)`` call.

    Allocates one broadcast-shaped array per key in ``output_keys`` and
    fills them from the per-call dicts. Progress-bar handling matches
    :func:`_dispatch`.
    """
    show_progress = fn_kwargs.pop("show_progress", False)
    R0_arr = np.asarray(R0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    R0_b, k_b = np.broadcast_arrays(R0_arr, k_arr)
    out = {key: np.empty(R0_b.shape, dtype=np.float64) for key in output_keys}

    indices = list(np.ndindex(R0_b.shape))
    n_combos = len(indices)
    if show_progress and n_combos > 1:
        iterable = tqdm(indices, total=n_combos, desc=f"{label} params")
        inner_progress = False
    else:
        iterable = indices
        inner_progress = show_progress

    for idx in iterable:
        result = fn(
            float(R0_b[idx]),
            float(k_b[idx]),
            w,
            history,
            show_progress=inner_progress,
            **fn_kwargs,
        )
        for key in output_keys:
            out[key][idx] = result[key]
    return out


def pmo_uncertain(
    *,
    R0: ArrayLike,
    k: ArrayLike,
    w: ArrayLike,
    history: ArrayLike,
    method: Literal["analytic", "simulation", "mcmc"],
    prior_sse: float = 0.5,
    ssi_evidence_method: SsiEvidenceMethod = "bridge",
    **kwargs,
) -> PmoUncertainResult:
    """Bayesian model-averaged probability of major outbreak.

    Combines :func:`pmo_sse` and :func:`pmo_ssi` with a prior probability
    ``prior_sse`` on the SSE model (so ``1 - prior_sse`` on SSI), via
    Bayesian model averaging given the observed ``history``. See
    ``notes/notes.tex`` for the derivation.

    Parameters
    ----------
    R0, k
        Reproduction number and dispersion parameter; scalar or array
        (broadcast jointly). The same ``(R0, k)`` is used for both models.
    w
        Discrete serial-interval weights, ``w[s-1] = w_s`` for ``s = 1, 2, ...``.
    history
        Observed incidence ``(I_0, I_1, ..., I_r)`` as a 1-D array of
        non-negative integers with ``history[0] >= 1``.
    method
        ``"analytic"`` for the closed-form Bayes update. Supports
        histories with cases on day 0 only, on day 0 and one later day
        (case (i)), or on day 0 and two later days (case (ii)); raises
        :class:`NotImplementedError` for histories with three or more
        later non-zero days. ``"simulation"`` for the rejection-sampling
        estimator (forwarded to :func:`_pmo_uncertain_sim`; takes
        ``n_sims``, ``threshold``, ``t_max``, ``rng``, ``batch_size``,
        ``max_attempts``, ``show_progress`` as keyword arguments).
        ``"mcmc"`` for an MCMC-based estimator that handles any history:
        SSE stays closed-form; SSI uses ``fit_ssi`` once for the trace and
        reuses it for both the SSI PMO and the SSI marginal log-likelihood.
        Takes ``ssi_evidence_method``, ``rng``, ``n_evidence_samples`` plus
        ``pm.sample`` keyword arguments (``draws``, ``tune``, ``chains``,
        ``thin``, ``progressbar``, ``target_accept``, …).
    prior_sse
        Prior probability of the SSE model in ``[0, 1]``. Default ``0.5``.
    ssi_evidence_method
        Estimator used for ``log L_SSI`` under ``method='mcmc'``: ``"bridge"``
        (default; Meng-Wong bridge sampling), ``"importance_sampling"``, or
        ``"naive"`` (Gamma-prior Monte Carlo, no MCMC trace required for the
        likelihood part — the trace is still run for the SSI PMO). Ignored
        when ``method`` is ``"analytic"`` or ``"simulation"``.

    Returns
    -------
    PmoUncertainResult
        Named tuple ``(pmo, posterior_sse, pmo_sse, pmo_ssi)`` — see
        :class:`PmoUncertainResult`.
    """
    if not 0.0 <= prior_sse <= 1.0:
        raise ValueError("prior_sse must lie in [0, 1]")
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = _validate_history(history)
    scalar_inputs = np.ndim(R0) == 0 and np.ndim(k) == 0

    if method == "analytic":
        if kwargs:
            raise TypeError(
                f"pmo_uncertain(method='analytic') got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )
        if classify_history(hist_arr)["kind"] == "general":
            offending_days = (np.flatnonzero(hist_arr[1:] != 0) + 1).tolist()
            raise NotImplementedError(
                "pmo_uncertain(method='analytic') requires a history with cases on "
                "day 0 and at most two later days; "
                f"got non-zero cases on day(s) {offending_days}. "
                "Use method='simulation' or method='mcmc'."
            )
        result = _pmo_uncertain_analytic(R0, k, w_arr, hist_arr, prior_sse)
    elif method == "simulation":
        kwargs["prior_sse"] = prior_sse
        result = _dispatch_multi(
            _pmo_uncertain_sim,
            R0,
            k,
            w_arr,
            hist_arr,
            "pmo_uncertain",
            kwargs,
            output_keys=("pmo", "posterior_sse", "pmo_sse", "pmo_ssi"),
        )
    elif method == "mcmc":
        kwargs["prior_sse"] = prior_sse
        kwargs["ssi_evidence_method"] = ssi_evidence_method
        result = _dispatch_multi(
            _pmo_uncertain_mcmc,
            R0,
            k,
            w_arr,
            hist_arr,
            "pmo_uncertain_mcmc",
            kwargs,
            output_keys=("pmo", "posterior_sse", "pmo_sse", "pmo_ssi"),
        )
    else:
        raise ValueError(
            f"pmo_uncertain: method must be 'analytic', 'simulation', or 'mcmc', got {method!r}"
        )

    if scalar_inputs:
        return PmoUncertainResult(
            pmo=float(np.asarray(result["pmo"]).reshape(())),
            posterior_sse=float(np.asarray(result["posterior_sse"]).reshape(())),
            pmo_sse=float(np.asarray(result["pmo_sse"]).reshape(())),
            pmo_ssi=float(np.asarray(result["pmo_ssi"]).reshape(())),
        )
    return PmoUncertainResult(
        pmo=result["pmo"],
        posterior_sse=result["posterior_sse"],
        pmo_sse=result["pmo_sse"],
        pmo_ssi=result["pmo_ssi"],
    )


__all__ = ["PmoUncertainResult", "pmo_sse", "pmo_ssi", "pmo_uncertain"]
