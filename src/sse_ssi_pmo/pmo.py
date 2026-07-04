"""Probability of major outbreak (PMO) — public dispatcher API.

Five top-level functions, each taking the observed incidence
``history = (I_0, I_1, ..., I_r)`` together with the serial-interval weights
``w`` and dispatching to either an analytic closed-form or a Monte-Carlo
simulation / MCMC backend:

* :func:`pmo_sse` — SSE model; ``method ∈ {"analytic", "simulation", "mcmc"}``.
* :func:`pmo_ssi` — SSI model; ``method ∈ {"analytic", "simulation", "mcmc"}``.
  ``"analytic"`` only supports histories with cases on day 0 alone (followed
  by zeros). ``"mcmc"`` samples the latent infectivities via HMC and averages
  the conditional extinction probability over draws — see ``notes/notes.tex``
  for the derivation.
* :func:`pmo_poisson` — Poisson offspring model (the ``k -> infty`` limit
  of either SSE or SSI); ``method = "analytic"`` only, supports any
  history.
* :func:`pmo_uncertain` — Bayesian model average across SSE and SSI given a
  prior ``prior_sse`` on the SSE model. ``method ∈ {"analytic",
  "simulation", "mcmc"}``; ``"analytic"`` only supports the day-0-only history.
  Returns a :class:`PmoUncertainResult` NamedTuple with the model-averaged
  PMO, the posterior probability of SSE, and the per-model PMOs.
* :func:`pmo_ensemble` — Bayesian model average across an arbitrary list
  of model specs (each a dict identifying SSE, SSI, or Poisson with its
  own scalar parameters) and prior probabilities; ``method ∈ {"analytic",
  "mcmc"}``. Returns a :class:`PmoEnsembleResult` NamedTuple with the
  model-averaged PMO, per-model posteriors, and per-model PMOs.

All functions broadcast over ``R0`` and ``k`` (scalar or array). Scalar
inputs return a Python ``float`` (or float-valued NamedTuple for
``pmo_uncertain``); array inputs return a NumPy array (or array-valued
NamedTuple) of the broadcast shape.

For :func:`pmo_sse`, :func:`pmo_ssi`, and :func:`pmo_uncertain`, either
of ``R0`` and ``k`` may instead be a :class:`~sse_ssi_pmo.priors.Prior`
(Gamma or LogNormal) — the function then integrates over the prior via
MCMC (``method='mcmc'``) or rejection sampling (``method='simulation'``).
``method='analytic'`` rejects Priors. When a Prior is passed, the
broadcast over the other parameter is bypassed: the other parameter must
be scalar. :func:`pmo_ensemble` accepts a Prior on the ``R0`` / ``k``
slot of any SSE or SSI spec (Poisson + Prior is not yet implemented);
under ``method='mcmc'`` the corresponding spec is fitted via
``fit_sse`` / ``fit_ssi`` and the trace drives both the PMO and the
log model evidence. :func:`pmo_poisson` is scalar-only.

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

from typing import Literal, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from sse_ssi_pmo._history import classify_history
from sse_ssi_pmo.evidence import EvidenceMethod
from sse_ssi_pmo.extinction import (
    _pmo_ensemble_analytic,
    _pmo_ensemble_mcmc,
    _pmo_poisson_analytic,
    _pmo_sse_analytic,
    _pmo_sse_mcmc,
    _pmo_ssi_analytic,
    _pmo_ssi_mcmc,
    _pmo_uncertain_analytic,
    _pmo_uncertain_mcmc,
)
from sse_ssi_pmo.priors import Prior
from sse_ssi_pmo.simulation import (
    _pmo_sse_sim,
    _pmo_ssi_sim_multi,
    _pmo_uncertain_sim_multi,
)
from sse_ssi_pmo.simulation_delay import (
    _pmo_sse_delay_sim_multi,
    _pmo_ssi_delay_sim_multi,
)


def _has_prior(R0, k) -> bool:
    """Return ``True`` iff either argument is a :class:`Prior`."""
    return isinstance(R0, Prior) or isinstance(k, Prior)


def _validate_prior_method(R0, k, method: str, fn_label: str) -> None:
    """Raise if a :class:`Prior` was supplied with ``method='analytic'``.

    Also reject the (Prior, array-valued other parameter) combination — the
    broadcast-over-(R0, k) loop is bypassed under priors, so an array-valued
    counterpart has no defined semantics.
    """
    if not _has_prior(R0, k):
        return
    if method == "analytic":
        raise ValueError(
            f"{fn_label}(method='analytic') does not accept a Prior on R0/k; "
            "use method='mcmc' or method='simulation'."
        )
    other = k if isinstance(R0, Prior) else R0
    if not isinstance(other, Prior) and np.ndim(other) > 0:
        raise ValueError(
            f"{fn_label}: when one of R0/k is a Prior, the other must be a scalar "
            f"(got shape {np.shape(other)})."
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


class PmoEnsembleResult(NamedTuple):
    """Return type of :func:`pmo_ensemble`.

    ``pmo`` is the model-averaged probability of a major outbreak;
    ``posteriors`` is the per-model posterior probability vector given the
    observed history; ``pmo_per_model`` is the per-model PMO vector. For
    a 1-D ``history``, ``pmo`` is a Python ``float`` and ``posteriors`` /
    ``pmo_per_model`` are length-``N`` arrays. For a 2-D ``(M, L)``
    ``history``, ``pmo`` is shape ``(M,)`` and the others are ``(M, N)``.
    """

    pmo: float | NDArray[np.float64]
    posteriors: NDArray[np.float64]
    pmo_per_model: NDArray[np.float64]


_ENSEMBLE_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "sse": frozenset({"model", "R0", "k"}),
    "ssi": frozenset({"model", "R0", "k"}),
    "poisson": frozenset({"model", "R0"}),
}


def _validate_model_specs(models: list[dict]) -> None:
    """Validate each ``models[i]`` is a well-formed model spec dict.

    See :func:`pmo_ensemble` for the accepted shapes. SSE / SSI specs may
    carry a :class:`~sse_ssi_pmo.priors.Prior` on ``R0`` and/or ``k``;
    Poisson specs currently require a scalar ``R0`` (Prior on Poisson R0
    is not yet implemented). Raises ``ValueError`` on any irregularity
    and ``NotImplementedError`` on a Prior in a Poisson spec.
    """
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("models must be a non-empty list of dicts")
    for i, spec in enumerate(models):
        if not isinstance(spec, dict):
            raise ValueError(f"models[{i}] must be a dict; got {type(spec).__name__}")
        if "model" not in spec:
            raise ValueError(f"models[{i}] missing required key 'model'")
        kind = spec["model"]
        if kind not in _ENSEMBLE_REQUIRED_KEYS:
            raise ValueError(
                f"models[{i}]['model'] = {kind!r}; expected one of "
                f"{sorted(_ENSEMBLE_REQUIRED_KEYS)}"
            )
        required = _ENSEMBLE_REQUIRED_KEYS[kind]
        missing = required - set(spec)
        extra = set(spec) - required
        if missing:
            raise ValueError(f"models[{i}] (model={kind!r}) missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"models[{i}] (model={kind!r}) has unexpected keys: {sorted(extra)}")
        if kind == "poisson" and isinstance(spec["R0"], Prior):
            raise NotImplementedError(
                f"models[{i}] (model='poisson') carries a Prior on R0; "
                "Poisson + Prior is not yet implemented (see pmo_ensemble docstring)."
            )


def _spec_has_prior(spec: dict) -> bool:
    """Return ``True`` iff any of the spec's parameter values is a :class:`Prior`."""
    return any(isinstance(spec.get(key), Prior) for key in ("R0", "k"))


def _models_have_prior(models: list[dict]) -> bool:
    """Return ``True`` iff any model spec in ``models`` carries a Prior."""
    return any(_spec_has_prior(spec) for spec in models)


def _validate_priors(priors: ArrayLike, n_models: int) -> NDArray[np.float64]:
    """Validate ``priors`` is a length-``n_models`` non-negative simplex point."""
    arr = np.asarray(priors, dtype=np.float64)
    if arr.ndim != 1 or arr.size != n_models:
        raise ValueError(f"priors must be a 1-D array of length {n_models} (matching len(models))")
    if (arr < 0).any():
        raise ValueError("priors must be non-negative")
    s = float(arr.sum())
    if not np.isclose(s, 1.0, atol=1e-9):
        raise ValueError(f"priors must sum to 1 (got {s})")
    return arr


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
    R0: ArrayLike | Prior,
    k: ArrayLike | Prior,
    w: ArrayLike,
    history: ArrayLike,
    method: Literal["analytic", "simulation", "mcmc"],
    **kwargs,
) -> float | NDArray[np.float64]:
    """Probability of major outbreak under the SSE model.

    Parameters
    ----------
    R0, k
        Reproduction number and dispersion parameter; scalar or array
        (broadcast jointly), or a :class:`~sse_ssi_pmo.priors.Prior` instance
        to integrate over a Gamma / LogNormal prior on that parameter via
        MCMC or rejection sampling. ``method='analytic'`` rejects Priors.
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
        ``"analytic"`` for the closed-form ``1 - q ** Lambda`` at fixed
        ``(R0, k)``; ``"simulation"`` for a Monte-Carlo estimate (forwarded
        to the SSE simulation backend; takes ``n_sims``, ``threshold``,
        ``t_max``, ``rng``, ``show_progress`` as keyword arguments).
        ``"mcmc"`` is only meaningful when a :class:`Prior` is supplied:
        ``fit_sse`` runs over the prior on ``(R0, k)`` and the closed-form
        analytic PMO is averaged over the posterior draws. With both
        scalars ``"mcmc"`` short-circuits to the closed form.
        Pass ``datatree=`` to reuse a pre-existing ``fit_sse`` trace.
        Multi-history input loops over rows (SSE has no shared-work payoff).
    """
    w_arr = np.asarray(w, dtype=np.float64)
    hist_2d, was_1d = _validate_histories(history)
    _validate_prior_method(R0, k, method, "pmo_sse")
    M = hist_2d.shape[0]

    if _has_prior(R0, k):
        # Bypass the (R0, k) broadcast loop — _validate_prior_method guarantees
        # the non-prior parameter is a scalar, so narrowing to float | Prior is safe.
        R0_p = cast("float | Prior", R0)
        k_p = cast("float | Prior", k)
        if method == "simulation":
            per_row = [_pmo_sse_sim(R0_p, k_p, w_arr, hist_2d[m], **kwargs) for m in range(M)]
        elif method == "mcmc":
            per_row = [_pmo_sse_mcmc(R0_p, k_p, w_arr, hist_2d[m], **kwargs) for m in range(M)]
        else:
            raise ValueError(
                f"pmo_sse: method must be 'mcmc' or 'simulation' under priors, got {method!r}"
            )
        out = np.asarray(per_row, dtype=np.float64)
        return float(out[0]) if was_1d else out

    R0_a = cast("ArrayLike", R0)
    k_a = cast("ArrayLike", k)
    scalar_inputs = np.ndim(R0_a) == 0 and np.ndim(k_a) == 0
    if method == "analytic":
        if kwargs:
            raise TypeError(
                f"pmo_sse(method='analytic') got unexpected keyword arguments: {sorted(kwargs)}"
            )
        per_row = [_pmo_sse_analytic(R0_a, k_a, w_arr, hist_2d[m]) for m in range(M)]
        out = np.stack([np.asarray(r, dtype=np.float64) for r in per_row], axis=0)
    elif method == "simulation":
        out = _dispatch_h_scalar(
            _pmo_sse_sim, R0_a, k_a, w_arr, hist_2d, "pmo_sse", kwargs, native_multi=False
        )
    elif method == "mcmc":
        # Fixed (R0, k) shortcut: SSE PMO is closed-form, _pmo_sse_mcmc returns it directly.
        out = _dispatch_h_scalar(
            _pmo_sse_mcmc, R0_a, k_a, w_arr, hist_2d, "pmo_sse_mcmc", kwargs, native_multi=False
        )
    else:
        raise ValueError(
            f"pmo_sse: method must be 'analytic', 'simulation', or 'mcmc', got {method!r}"
        )

    return _collapse_leading(out, was_1d=was_1d, scalar_inputs=scalar_inputs)


def pmo_ssi(
    *,
    R0: ArrayLike | Prior,
    k: ArrayLike | Prior,
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
        (broadcast jointly), or a :class:`~sse_ssi_pmo.priors.Prior` instance
        to integrate over a Gamma / LogNormal prior (only ``method='mcmc'``
        or ``method='simulation'`` accept Priors).
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
        ``progressbar``) plus ``thin`` for posterior thinning. Pass
        ``datatree=`` to reuse a pre-existing ``fit_ssi`` trace.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    hist_2d, was_1d = _validate_histories(history)
    _validate_prior_method(R0, k, method, "pmo_ssi")
    M = hist_2d.shape[0]

    if _has_prior(R0, k):
        R0_p = cast("float | Prior", R0)
        k_p = cast("float | Prior", k)
        if method == "simulation":
            out_arr = _pmo_ssi_sim_multi(R0_p, k_p, w_arr, hist_2d, **kwargs)
            return float(out_arr[0]) if was_1d else out_arr
        if method == "mcmc":
            per_row = [_pmo_ssi_mcmc(R0_p, k_p, w_arr, hist_2d[m], **kwargs) for m in range(M)]
            out = np.asarray(per_row, dtype=np.float64)
            return float(out[0]) if was_1d else out
        raise ValueError(
            f"pmo_ssi: method must be 'mcmc' or 'simulation' under priors, got {method!r}"
        )

    R0_a = cast("ArrayLike", R0)
    k_a = cast("ArrayLike", k)
    scalar_inputs = np.ndim(R0_a) == 0 and np.ndim(k_a) == 0
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
        per_row = [_pmo_ssi_analytic(R0_a, k_a, w_arr, hist_2d[m]) for m in range(M)]
        out = np.stack([np.asarray(r, dtype=np.float64) for r in per_row], axis=0)
    elif method == "simulation":
        out = _dispatch_h_scalar(
            _pmo_ssi_sim_multi, R0_a, k_a, w_arr, hist_2d, "pmo_ssi", kwargs, native_multi=True
        )
    elif method == "mcmc":
        out = _dispatch_h_scalar(
            _pmo_ssi_mcmc, R0_a, k_a, w_arr, hist_2d, "pmo_ssi_mcmc", kwargs, native_multi=False
        )
    else:
        raise ValueError(
            f"pmo_ssi: method must be 'analytic', 'simulation', or 'mcmc', got {method!r}"
        )

    return _collapse_leading(out, was_1d=was_1d, scalar_inputs=scalar_inputs)


def pmo_uncertain(
    *,
    R0: ArrayLike | Prior,
    k: ArrayLike | Prior,
    w: ArrayLike,
    history: ArrayLike,
    method: Literal["analytic", "simulation", "mcmc"],
    prior_sse: float = 0.5,
    evidence_method: EvidenceMethod = "bridge",
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
        (broadcast jointly), or a :class:`~sse_ssi_pmo.priors.Prior` instance
        to integrate over a prior. The same ``(R0, k)`` (or prior) is used
        for both models. Priors are accepted only by ``method='mcmc'`` and
        ``method='simulation'``.
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
        SSE stays closed-form at fixed ``(R0, k)``, but is averaged over
        the fit_sse trace when a :class:`Prior` is supplied; SSI uses
        ``fit_ssi`` once per row for the trace and reuses it for both the
        SSI PMO and the SSI log model evidence. Pass ``datatree_sse=`` /
        ``datatree_ssi=`` to reuse pre-existing traces. Takes
        ``evidence_method``, ``rng``, ``n_evidence_samples`` plus
        ``pm.sample`` keyword arguments (``draws``, ``tune``, ``chains``,
        ``thin``, ``progressbar``, ``target_accept``, …).
    prior_sse
        Prior probability of the SSE model in ``[0, 1]``. Default ``0.5``.
    evidence_method
        Estimator used for the log model evidences under ``method='mcmc'``:
        ``"bridge"`` (default; Meng-Wong bridge sampling),
        ``"importance_sampling"``, or ``"naive"`` (no MCMC trace required
        for the SSI evidence). With fixed ``(R0, k)`` only the SSI side
        needs an MCMC estimator (SSE is closed-form); under priors on
        ``R0``/``k`` the same knob also selects the estimator for the SSE
        side. Ignored when ``method`` is ``"analytic"`` or ``"simulation"``.

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
    _validate_prior_method(R0, k, method, "pmo_uncertain")
    M = hist_2d.shape[0]
    output_keys = ("pmo", "posterior_sse", "pmo_sse", "pmo_ssi")

    if _has_prior(R0, k):
        R0_p = cast("float | Prior", R0)
        k_p = cast("float | Prior", k)
        if method == "simulation":
            kwargs["prior_sse"] = prior_sse
            out_dict = _pmo_uncertain_sim_multi(R0_p, k_p, w_arr, hist_2d, **kwargs)
        elif method == "mcmc":
            kwargs["prior_sse"] = prior_sse
            kwargs["evidence_method"] = evidence_method
            per_row = [
                _pmo_uncertain_mcmc(R0_p, k_p, w_arr, hist_2d[m], **kwargs) for m in range(M)
            ]
            out_dict = {
                key: np.asarray([r[key] for r in per_row], dtype=np.float64) for key in output_keys
            }
        else:
            raise ValueError(
                f"pmo_uncertain: method must be 'mcmc' or 'simulation' under priors, got {method!r}"
            )

        def _maybe_scalar(arr):
            return float(arr[0]) if was_1d else arr

        return PmoUncertainResult(
            pmo=_maybe_scalar(out_dict["pmo"]),
            posterior_sse=_maybe_scalar(out_dict["posterior_sse"]),
            pmo_sse=_maybe_scalar(out_dict["pmo_sse"]),
            pmo_ssi=_maybe_scalar(out_dict["pmo_ssi"]),
        )

    R0_a = cast("ArrayLike", R0)
    k_a = cast("ArrayLike", k)
    scalar_inputs = np.ndim(R0_a) == 0 and np.ndim(k_a) == 0
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
        per_row = [
            _pmo_uncertain_analytic(R0_a, k_a, w_arr, hist_2d[m], prior_sse) for m in range(M)
        ]
        result = {
            key: np.stack([np.asarray(r[key], dtype=np.float64) for r in per_row], axis=0)
            for key in output_keys
        }
    elif method == "simulation":
        kwargs["prior_sse"] = prior_sse
        result = _dispatch_h_dict(
            _pmo_uncertain_sim_multi,
            R0_a,
            k_a,
            w_arr,
            hist_2d,
            "pmo_uncertain",
            kwargs,
            output_keys=output_keys,
            native_multi=True,
        )
    elif method == "mcmc":
        kwargs["prior_sse"] = prior_sse
        kwargs["evidence_method"] = evidence_method
        result = _dispatch_h_dict(
            _pmo_uncertain_mcmc,
            R0_a,
            k_a,
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


def pmo_poisson(
    *,
    R0: ArrayLike,
    w: ArrayLike,
    history: ArrayLike,
    method: Literal["analytic"] = "analytic",
) -> float | NDArray[np.float64]:
    """Probability of major outbreak under the Poisson offspring model.

    The Poisson offspring model is the ``k -> infty`` limit of either SSE
    or SSI; offspring at each lag follow ``Poisson(R0 w_s)`` independently,
    and the closed-form PMO ``1 - q ** Lambda`` matches the SSE form but
    with ``q`` solved from the Poisson pgf.

    Parameters
    ----------
    R0
        Reproduction number; scalar or array (broadcast over ``R0``).
    w
        Discrete serial-interval weights, ``w[s-1] = w_s`` for ``s = 1, 2, ...``.
    history
        Observed incidence as a 1-D array or a 2-D ``(M, L)`` array of
        histories (see :func:`pmo_sse` for the convention).
    method
        Only ``"analytic"`` is supported. ``"simulation"`` and ``"mcmc"``
        raise :class:`NotImplementedError` — for forward Monte-Carlo use
        :func:`~sse_ssi_pmo.simulate_poisson` directly.
    """
    if method != "analytic":
        raise NotImplementedError(
            "pmo_poisson currently supports method='analytic' only; "
            "for forward simulation use sse_ssi_pmo.simulate_poisson directly."
        )
    w_arr = np.asarray(w, dtype=np.float64)
    hist_2d, was_1d = _validate_histories(history)
    scalar_inputs = np.ndim(R0) == 0
    M = hist_2d.shape[0]
    per_row = [_pmo_poisson_analytic(R0, w_arr, hist_2d[m]) for m in range(M)]
    out = np.stack([np.asarray(r, dtype=np.float64) for r in per_row], axis=0)
    return _collapse_leading(out, was_1d=was_1d, scalar_inputs=scalar_inputs)


def pmo_ensemble(
    *,
    models: list[dict],
    priors: ArrayLike,
    w: ArrayLike,
    history: ArrayLike,
    method: Literal["analytic", "mcmc"],
    evidence_method: EvidenceMethod = "bridge",
    **kwargs,
) -> PmoEnsembleResult:
    """Bayesian model-averaged probability of major outbreak across an N-model ensemble.

    Generalises :func:`pmo_uncertain` to an arbitrary list of model specs
    and prior probabilities. Each ``models[i]`` is a dict identifying one
    candidate offspring model and carrying its own parameters (``R0``,
    plus ``k`` for SSE/SSI):

    - ``{"model": "sse", "R0": float | Prior, "k": float | Prior}``
    - ``{"model": "ssi", "R0": float | Prior, "k": float | Prior}``
    - ``{"model": "poisson", "R0": float}``

    ``priors[i]`` is the prior probability assigned to ``models[i]``;
    ``priors`` must be a 1-D non-negative array of length ``len(models)``
    summing to 1.

    SSE / SSI specs may carry a :class:`~sse_ssi_pmo.priors.Prior` on
    ``R0`` and/or ``k``, in which case the per-spec evidence and PMO are
    integrated over the prior via MCMC under ``method='mcmc'``. Poisson
    specs currently require a scalar ``R0``; a Prior on a Poisson spec
    raises :class:`NotImplementedError` (the corresponding evidence
    estimator has not yet been added — see ``notes/notes.tex``
    §\\ref{sec:param_uncertain} for the general theory).

    Parameters
    ----------
    method
        ``"analytic"`` for closed-form per-model PMOs and evidences.
        SSI specs require a history with cases on day 0 and at most two
        later days; otherwise raises :class:`NotImplementedError`. Specs
        carrying a Prior are rejected (raises :class:`ValueError`).
        ``"mcmc"`` runs ``fit_ssi`` once per SSI spec per history and
        reuses the trace for both the PMO and the log model evidence
        (SSE and Poisson remain closed-form at fixed parameters; an SSE
        spec carrying a Prior is fitted via ``fit_sse``). ``"simulation"``
        raises :class:`NotImplementedError`.
    evidence_method
        Estimator used for the log model evidence under ``method='mcmc'``:
        ``"bridge"`` (default), ``"importance_sampling"``, or ``"naive"``.
        Applies to any SSI spec and to any SSE spec that carries a Prior.
        Ignored when ``method='analytic'``.
    **kwargs
        Forwarded to the MCMC backends as needed (``rng``,
        ``n_evidence_samples``, ``pm.sample`` kwargs such as ``draws``,
        ``tune``, ``chains``, ``thin``, ``progressbar``, ``target_accept``,
        …). ``show_progress`` controls the per-row tqdm bar.
    """
    _validate_model_specs(models)
    priors_arr = _validate_priors(priors, len(models))
    has_prior = _models_have_prior(models)

    w_arr = np.asarray(w, dtype=np.float64)
    hist_2d, was_1d = _validate_histories(history)
    M = hist_2d.shape[0]
    N = len(models)

    pmo_out = np.empty(M, dtype=np.float64)
    posteriors_out = np.empty((M, N), dtype=np.float64)
    pmo_per_model_out = np.empty((M, N), dtype=np.float64)

    if method == "analytic":
        if has_prior:
            raise ValueError(
                "pmo_ensemble(method='analytic') does not accept Priors on R0/k; use method='mcmc'."
            )
        if kwargs:
            raise TypeError(
                f"pmo_ensemble(method='analytic') got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )
        has_ssi = any(spec["model"] == "ssi" for spec in models)
        if has_ssi:
            for m in range(M):
                if classify_history(hist_2d[m])["kind"] == "general":
                    offending_days = (np.flatnonzero(hist_2d[m, 1:] != 0) + 1).tolist()
                    raise NotImplementedError(
                        "pmo_ensemble(method='analytic') with an SSI spec requires "
                        "a history with cases on day 0 and at most two later days; "
                        f"row {m} has non-zero cases on day(s) {offending_days}. "
                        "Use method='mcmc'."
                    )
        for m in range(M):
            result = _pmo_ensemble_analytic(models, priors_arr, w_arr, hist_2d[m])
            pmo_out[m] = float(np.asarray(result["pmo"]).reshape(()))
            posteriors_out[m] = result["posteriors"]
            pmo_per_model_out[m] = result["pmo_per_model"]
    elif method == "mcmc":
        show_progress = kwargs.pop("show_progress", False)
        if show_progress and M > 1:
            row_iter: range | tqdm = tqdm(range(M), desc="pmo_ensemble", leave=False)
            inner_progress = False
        else:
            row_iter = range(M)
            inner_progress = show_progress
        for m in row_iter:
            result = _pmo_ensemble_mcmc(
                models,
                priors_arr,
                w_arr,
                hist_2d[m],
                evidence_method=evidence_method,
                show_progress=inner_progress,
                **kwargs,
            )
            pmo_out[m] = float(np.asarray(result["pmo"]).reshape(()))
            posteriors_out[m] = result["posteriors"]
            pmo_per_model_out[m] = result["pmo_per_model"]
    elif method == "simulation":
        raise NotImplementedError(
            "pmo_ensemble(method='simulation') is not yet supported; "
            "use method='analytic' or 'mcmc'."
        )
    else:
        raise ValueError(f"pmo_ensemble: method must be 'analytic' or 'mcmc', got {method!r}")

    if was_1d:
        return PmoEnsembleResult(
            pmo=float(pmo_out[0]),
            posteriors=posteriors_out[0],
            pmo_per_model=pmo_per_model_out[0],
        )
    return PmoEnsembleResult(
        pmo=pmo_out,
        posteriors=posteriors_out,
        pmo_per_model=pmo_per_model_out,
    )


def _validate_delay_weights(tost: NDArray[np.float64], inc: NDArray[np.float64]) -> None:
    """Validate the TOST and incubation weight arrays for the onset-anchored models."""
    for name, arr in (("tost", tost), ("inc", inc)):
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty 1-D array")
        if (arr < 0.0).any():
            raise ValueError(f"{name} must be non-negative")
        if not arr.sum() > 0.0:
            raise ValueError(f"{name} must have positive total mass")


def _pmo_delay_dispatch(
    model: str,
    *,
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    history: ArrayLike,
    method: str,
    kwargs: dict,
) -> float | NDArray[np.float64]:
    """Shared body for :func:`pmo_sse_delay` / :func:`pmo_ssi_delay`."""
    if method != "simulation":
        raise NotImplementedError(
            f"pmo_{model}_delay only supports method='simulation' "
            f"(the onset-anchored models are simulation-only), got {method!r}."
        )
    if np.ndim(R0) != 0 or np.ndim(k) != 0:
        raise ValueError(f"pmo_{model}_delay: R0 and k must be scalars (no broadcasting yet)")
    if float(R0) <= 0.0 or float(k) <= 0.0:
        raise ValueError("R0 and k must be positive")
    tost_arr = np.asarray(tost, dtype=np.float64)
    inc_arr = np.asarray(inc, dtype=np.float64)
    _validate_delay_weights(tost_arr, inc_arr)
    hist_2d, was_1d = _validate_histories(history)
    fn = _pmo_sse_delay_sim_multi if model == "sse" else _pmo_ssi_delay_sim_multi
    out = fn(float(R0), float(k), tost_arr, inc_arr, hist_2d, **kwargs)
    return float(out[0]) if was_1d else out


def pmo_sse_delay(
    *,
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    history: ArrayLike,
    method: Literal["simulation"] = "simulation",
    **kwargs,
) -> float | NDArray[np.float64]:
    """Probability of major outbreak under the onset-anchored SSE model.

    Symptom-onset-anchored variant of :func:`pmo_sse`: the observed ``history``
    is symptom-onset incidence ``(D_0, D_1, ..., D_r)``, transmission is driven
    by the TOST weights ``tost`` (indexed from lag 0), and infections are mapped
    forward to onsets by the incubation weights ``inc`` (indexed from lag 1).
    See ``notes/notes.tex`` §"Symptom-onset data and delayed transmission".

    Estimated by rejection sampling only — the observed onsets do not determine
    the latent incubation pipeline, so there is no closed form or seed-and-
    continue shortcut. ``R0`` and ``k`` must be scalars.

    Parameters
    ----------
    R0, k
        Reproduction number and dispersion parameter (positive scalars).
    tost
        TOST weights ``tost[s]`` for ``s = 0, 1, ...`` (from lag 0); non-negative
        1-D array, need not sum to exactly 1.
    inc
        Incubation-period weights ``inc[a-1]`` for ``a = 1, 2, ...`` (from lag 1);
        non-negative 1-D array.
    history
        Observed onset incidence as a 1-D array or a 2-D ``(M, L)`` array of
        histories (see :func:`pmo_sse` for the convention). For 2-D input all
        rows must share the same ``D_0`` (shared rejection-sampling pass).
    method
        Only ``"simulation"`` is supported. Keyword arguments (``n_sims``,
        ``threshold``, ``t_max``, ``rng``, ``batch_size``, ``max_attempts``,
        ``show_progress``) are forwarded to the simulation backend.
    """
    return _pmo_delay_dispatch(
        "sse", R0=R0, k=k, tost=tost, inc=inc, history=history, method=method, kwargs=kwargs
    )


def pmo_ssi_delay(
    *,
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    history: ArrayLike,
    method: Literal["simulation"] = "simulation",
    **kwargs,
) -> float | NDArray[np.float64]:
    """Probability of major outbreak under the onset-anchored SSI model.

    Symptom-onset-anchored variant of :func:`pmo_ssi` (see
    :func:`pmo_sse_delay` for the shared parameter conventions). Each onset
    cohort carries a latent aggregate infectivity ``Y_t ~ Gamma(k*D_t, k)`` and
    new infections are ``Poisson(R0 * sum_s tost_s Y_{t-s})``. Simulation-only.
    """
    return _pmo_delay_dispatch(
        "ssi", R0=R0, k=k, tost=tost, inc=inc, history=history, method=method, kwargs=kwargs
    )


__all__ = [
    "PmoEnsembleResult",
    "PmoUncertainResult",
    "pmo_ensemble",
    "pmo_poisson",
    "pmo_sse",
    "pmo_sse_delay",
    "pmo_ssi",
    "pmo_ssi_delay",
    "pmo_uncertain",
]
