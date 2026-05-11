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
NamedTuple) of the broadcast shape.

``history`` may also be a 2-D ``(M, L)`` array (or list/tuple of
equal-length 1-D arrays). The return then gains a *leading* length-``M``
axis (still wrapped in :class:`PmoUncertainResult` for
``pmo_uncertain``). For ``method='simulation'`` of :func:`pmo_ssi` and
:func:`pmo_uncertain`, all rows must share the same ``I_0`` and the
backend runs a single rejection-sampling pass that matches each
simulated trajectory against every history at once — typically far
cheaper than running ``M`` independent passes.

The simulation and MCMC paths run the inner backend once per
``(R0, k)`` combination and, when ``show_progress=True`` and there is
more than one combination, wrap the parameter-combination loop with a
single ``tqdm`` bar (passing ``show_progress=False`` to each inner
call).
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
from sse_ssi_pmo.simulation import (
    _pmo_sse_sim,
    _pmo_ssi_sim_multi,
    _pmo_uncertain_sim_multi,
)


class PmoUncertainResult(NamedTuple):
    """Return type of :func:`pmo_uncertain`.

    ``pmo`` is the model-averaged probability of a major outbreak;
    ``posterior_sse`` is the posterior probability of the SSE model given
    the observed history; ``pmo_sse`` and ``pmo_ssi`` are the per-model
    PMOs. For a 1-D ``history`` and scalar ``(R0, k)``, each field is a
    Python ``float``; an array-valued ``(R0, k)`` adds the broadcast
    shape; a 2-D ``history`` adds a leading length-``M`` axis.
    """

    pmo: float | NDArray[np.float64]
    posterior_sse: float | NDArray[np.float64]
    pmo_sse: float | NDArray[np.float64]
    pmo_ssi: float | NDArray[np.float64]


def _validate_histories(history: ArrayLike) -> tuple[NDArray[np.int64], bool]:
    """Validate ``history`` as a 1-D or 2-D array and return ``(hist_2d, was_1d)``.

    Accepts a 1-D array (single history) or a 2-D array / list of
    equal-length 1-D arrays. Single-history input is reshaped to ``(1, L)``
    with ``was_1d = True``; multi-history input is kept as ``(M, L)`` with
    ``was_1d = False``. Validates that every row is non-negative and starts
    with ``I_0 >= 1``. Raises ``ValueError`` on ragged lists, ``ndim`` outside
    {1, 2}, or empty input.
    """
    try:
        arr = np.asarray(history, dtype=np.int64)
    except (ValueError, TypeError) as err:
        raise ValueError(
            "history must be a 1-D array or an equal-length 2-D / list-of-lists; "
            "for ragged inputs, split by length and call once per group."
        ) from err
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError("history must be a non-empty 1-D array of integers")
        hist_2d = arr.reshape(1, -1)
        was_1d = True
    elif arr.ndim == 2:
        if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("history must be a non-empty 2-D (M, L) array of integers")
        hist_2d = arr
        was_1d = False
    else:
        raise ValueError(
            f"history must be 1-D or 2-D (got {arr.ndim}-D); for ragged inputs, "
            "split by length and call once per group."
        )
    if hist_2d.min() < 0:
        raise ValueError("history entries must be non-negative")
    if (hist_2d[:, 0] < 1).any():
        raise ValueError("every history row must have history[0] >= 1")
    return hist_2d, was_1d


def _broadcast_R0_k(R0: ArrayLike, k: ArrayLike) -> tuple[NDArray, NDArray, tuple[int, ...]]:
    R0_arr = np.asarray(R0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    R0_b, k_b = np.broadcast_arrays(R0_arr, k_arr)
    return R0_b, k_b, R0_b.shape


def _setup_combo_iter(
    indices: list[tuple[int, ...]],
    *,
    show_progress: bool,
    label: str,
):
    n_combos = len(indices)
    if show_progress and n_combos > 1:
        return tqdm(indices, total=n_combos, desc=f"{label} params"), False
    return indices, show_progress


def _dispatch_h_scalar(
    fn,
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    hist_2d: NDArray[np.int64],
    label: str,
    fn_kwargs: dict,
    *,
    native_multi: bool,
) -> NDArray[np.float64]:
    """Drive a scalar-returning backend over ``(R0, k)`` and histories.

    With ``native_multi=True``, ``fn(R0, k, w, hist_2d, ...)`` returns a
    ``(M,)`` array per ``(R0, k)`` combination. Otherwise ``fn`` is
    single-history (``fn(R0, k, w, hist_1d, ...) -> scalar``) and we loop
    over rows. Output shape is ``(M, *broadcast(R0, k).shape)``.
    """
    show_progress = fn_kwargs.pop("show_progress", False)
    R0_b, k_b, shape = _broadcast_R0_k(R0, k)
    M = hist_2d.shape[0]
    out = np.empty((M, *shape), dtype=np.float64)
    indices = list(np.ndindex(shape))
    iterable, inner_progress = _setup_combo_iter(indices, show_progress=show_progress, label=label)
    for combo in iterable:
        if native_multi:
            row = fn(
                float(R0_b[combo]),
                float(k_b[combo]),
                w,
                hist_2d,
                show_progress=inner_progress,
                **fn_kwargs,
            )
            out[(slice(None), *combo)] = row
        else:
            for m in range(M):
                out[(m, *combo)] = fn(
                    float(R0_b[combo]),
                    float(k_b[combo]),
                    w,
                    hist_2d[m],
                    show_progress=inner_progress,
                    **fn_kwargs,
                )
    return out


def _dispatch_h_dict(
    fn,
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    hist_2d: NDArray[np.int64],
    label: str,
    fn_kwargs: dict,
    output_keys: tuple[str, ...],
    *,
    native_multi: bool,
) -> dict[str, NDArray[np.float64]]:
    """Dict-returning analogue of :func:`_dispatch_h_scalar`."""
    show_progress = fn_kwargs.pop("show_progress", False)
    R0_b, k_b, shape = _broadcast_R0_k(R0, k)
    M = hist_2d.shape[0]
    out = {key: np.empty((M, *shape), dtype=np.float64) for key in output_keys}
    indices = list(np.ndindex(shape))
    iterable, inner_progress = _setup_combo_iter(indices, show_progress=show_progress, label=label)
    for combo in iterable:
        if native_multi:
            row = fn(
                float(R0_b[combo]),
                float(k_b[combo]),
                w,
                hist_2d,
                show_progress=inner_progress,
                **fn_kwargs,
            )
            for key in output_keys:
                out[key][(slice(None), *combo)] = row[key]
        else:
            for m in range(M):
                row = fn(
                    float(R0_b[combo]),
                    float(k_b[combo]),
                    w,
                    hist_2d[m],
                    show_progress=inner_progress,
                    **fn_kwargs,
                )
                for key in output_keys:
                    out[key][(m, *combo)] = row[key]
    return out


def _collapse_leading(
    arr: NDArray[np.float64], *, was_1d: bool, scalar_inputs: bool
) -> float | NDArray[np.float64]:
    """Restore the original return shape after a dispatch that always added a leading M axis."""
    if was_1d:
        arr = arr[0]
    if scalar_inputs and was_1d:
        return float(np.asarray(arr).reshape(()))
    return arr


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
        Observed incidence as either a 1-D array
        ``(I_0, I_1, ..., I_r)`` or a 2-D ``(M, L)`` array (or list of
        equal-length 1-D arrays) stacking ``M`` histories. Non-negative
        integers, every row's first entry must be ``>= 1``.
    method
        ``"analytic"`` for the closed-form ``1 - q ** Lambda``;
        ``"simulation"`` for a Monte-Carlo estimate (forwarded to the SSE
        simulation backend; takes ``n_sims``, ``threshold``, ``t_max``,
        ``rng``, ``show_progress`` as keyword arguments). Multi-history
        input loops over rows (SSE has no shared-work payoff).
    """
    w_arr = np.asarray(w, dtype=np.float64)
    hist_2d, was_1d = _validate_histories(history)
    scalar_inputs = np.ndim(R0) == 0 and np.ndim(k) == 0
    M = hist_2d.shape[0]

    if method == "analytic":
        if kwargs:
            raise TypeError(
                f"pmo_sse(method='analytic') got unexpected keyword arguments: {sorted(kwargs)}"
            )
        # _pmo_sse_analytic broadcasts over (R0, k) internally; loop over histories.
        per_row = [_pmo_sse_analytic(R0, k, w_arr, hist_2d[m]) for m in range(M)]
        out = np.stack([np.asarray(r, dtype=np.float64) for r in per_row], axis=0)
    elif method == "simulation":
        out = _dispatch_h_scalar(
            _pmo_sse_sim, R0, k, w_arr, hist_2d, "pmo_sse", kwargs, native_multi=False
        )
    else:
        raise ValueError(f"pmo_sse: method must be 'analytic' or 'simulation', got {method!r}")

    return _collapse_leading(out, was_1d=was_1d, scalar_inputs=scalar_inputs)


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
        Observed incidence as a 1-D array or a 2-D ``(M, L)`` array of
        histories (see :func:`pmo_sse` for the convention).
    method
        ``"analytic"`` for the closed-form solution. Supports histories
        with cases on day 0 only, on day 0 and one later day (case (i)),
        or on day 0 and two later days (case (ii)); raises
        :class:`ValueError` for histories with three or more later non-zero
        days. ``"simulation"`` for a Monte-Carlo estimate (forwarded to the
        SSI simulation backend; takes ``n_sims``, ``threshold``, ``t_max``,
        ``rng``, ``batch_size``, ``max_attempts``, ``show_progress`` as
        keyword arguments). For 2-D ``history`` the simulation path uses
        the shared-rejection-sampling backend, which requires all rows to
        share the same ``I_0`` and produces all ``M`` per-history PMOs in
        one batched pass.
        ``"mcmc"`` for a Monte-Carlo estimate via MCMC over the latent
        infectivities (see ``notes/notes.tex``); keyword arguments are
        forwarded to ``pm.sample`` (e.g. ``draws``, ``tune``, ``chains``,
        ``progressbar``) plus ``thin`` for posterior thinning.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    hist_2d, was_1d = _validate_histories(history)
    scalar_inputs = np.ndim(R0) == 0 and np.ndim(k) == 0
    M = hist_2d.shape[0]

    if method == "analytic":
        if kwargs:
            raise TypeError(
                f"pmo_ssi(method='analytic') got unexpected keyword arguments: {sorted(kwargs)}"
            )
        for m in range(M):
            if classify_history(hist_2d[m])["kind"] == "general":
                offending_days = (np.flatnonzero(hist_2d[m, 1:] != 0) + 1).tolist()
                raise ValueError(
                    "pmo_ssi(method='analytic') requires a history with cases on "
                    "day 0 and at most two later days; "
                    f"row {m} has non-zero cases on day(s) {offending_days}. "
                    "Use method='simulation' or method='mcmc'."
                )
        per_row = [_pmo_ssi_analytic(R0, k, w_arr, hist_2d[m]) for m in range(M)]
        out = np.stack([np.asarray(r, dtype=np.float64) for r in per_row], axis=0)
    elif method == "simulation":
        out = _dispatch_h_scalar(
            _pmo_ssi_sim_multi, R0, k, w_arr, hist_2d, "pmo_ssi", kwargs, native_multi=True
        )
    elif method == "mcmc":
        out = _dispatch_h_scalar(
            _pmo_ssi_mcmc, R0, k, w_arr, hist_2d, "pmo_ssi_mcmc", kwargs, native_multi=False
        )
    else:
        raise ValueError(
            f"pmo_ssi: method must be 'analytic', 'simulation', or 'mcmc', got {method!r}"
        )

    return _collapse_leading(out, was_1d=was_1d, scalar_inputs=scalar_inputs)


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
        Observed incidence as a 1-D array or a 2-D ``(M, L)`` array of
        histories (see :func:`pmo_sse` for the convention).
    method
        ``"analytic"`` for the closed-form Bayes update. Supports
        histories with cases on day 0 only, on day 0 and one later day
        (case (i)), or on day 0 and two later days (case (ii)); raises
        :class:`NotImplementedError` for histories with three or more
        later non-zero days. ``"simulation"`` for the rejection-sampling
        estimator. For 2-D ``history`` the simulation path uses the
        shared-rejection-sampling backend (requires all rows to share the
        same ``I_0``); for 1-D it falls through the same backend with
        ``M = 1``.
        ``"mcmc"`` for an MCMC-based estimator that handles any history:
        SSE stays closed-form; SSI uses ``fit_ssi`` once per row for the
        trace and reuses it for both the SSI PMO and the SSI marginal
        log-likelihood. Takes ``ssi_evidence_method``, ``rng``,
        ``n_evidence_samples`` plus ``pm.sample`` keyword arguments
        (``draws``, ``tune``, ``chains``, ``thin``, ``progressbar``,
        ``target_accept``, …).
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
    hist_2d, was_1d = _validate_histories(history)
    scalar_inputs = np.ndim(R0) == 0 and np.ndim(k) == 0
    M = hist_2d.shape[0]
    output_keys = ("pmo", "posterior_sse", "pmo_sse", "pmo_ssi")

    if method == "analytic":
        if kwargs:
            raise TypeError(
                f"pmo_uncertain(method='analytic') got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )
        for m in range(M):
            if classify_history(hist_2d[m])["kind"] == "general":
                offending_days = (np.flatnonzero(hist_2d[m, 1:] != 0) + 1).tolist()
                raise NotImplementedError(
                    "pmo_uncertain(method='analytic') requires a history with cases on "
                    "day 0 and at most two later days; "
                    f"row {m} has non-zero cases on day(s) {offending_days}. "
                    "Use method='simulation' or method='mcmc'."
                )
        per_row = [_pmo_uncertain_analytic(R0, k, w_arr, hist_2d[m], prior_sse) for m in range(M)]
        result = {
            key: np.stack([np.asarray(r[key], dtype=np.float64) for r in per_row], axis=0)
            for key in output_keys
        }
    elif method == "simulation":
        kwargs["prior_sse"] = prior_sse
        result = _dispatch_h_dict(
            _pmo_uncertain_sim_multi,
            R0,
            k,
            w_arr,
            hist_2d,
            "pmo_uncertain",
            kwargs,
            output_keys=output_keys,
            native_multi=True,
        )
    elif method == "mcmc":
        kwargs["prior_sse"] = prior_sse
        kwargs["ssi_evidence_method"] = ssi_evidence_method
        result = _dispatch_h_dict(
            _pmo_uncertain_mcmc,
            R0,
            k,
            w_arr,
            hist_2d,
            "pmo_uncertain_mcmc",
            kwargs,
            output_keys=output_keys,
            native_multi=False,
        )
    else:
        raise ValueError(
            f"pmo_uncertain: method must be 'analytic', 'simulation', or 'mcmc', got {method!r}"
        )

    return PmoUncertainResult(
        pmo=_collapse_leading(result["pmo"], was_1d=was_1d, scalar_inputs=scalar_inputs),
        posterior_sse=_collapse_leading(
            result["posterior_sse"], was_1d=was_1d, scalar_inputs=scalar_inputs
        ),
        pmo_sse=_collapse_leading(result["pmo_sse"], was_1d=was_1d, scalar_inputs=scalar_inputs),
        pmo_ssi=_collapse_leading(result["pmo_ssi"], was_1d=was_1d, scalar_inputs=scalar_inputs),
    )


__all__ = ["PmoUncertainResult", "pmo_sse", "pmo_ssi", "pmo_uncertain"]
