"""Forward simulation of the SSE, SSI, and Poisson models.

Public API:

* :func:`simulate_sse`, :func:`simulate_ssi`, :func:`simulate_poisson` —
  single trajectories, terminating early on extinction (last ``len(w)``
  time-steps all zero) or on single-step incidence reaching ``threshold``
  (taken as the operational definition of "major outbreak").

Private API (used by :mod:`sse_ssi_pmo.pmo`):

* :func:`_pmo_sse_sim`, :func:`_pmo_ssi_sim` — Monte-Carlo estimates of the
  probability of major outbreak after a given observed incidence ``history``.
  SSE seeds forward simulation with the history; SSI simulates from ``t = 0``
  and rejects sims whose first ``len(history)`` steps don't match (with
  vectorised early termination on first mismatch).
* :func:`_pmo_uncertain_sim` — model-averaged Monte-Carlo PMO. Each batch is
  split between SSE and SSI via a Binomial draw with probability
  ``prior_sse``; both sub-batches use rejection sampling against the
  history. The acceptance-rate ratio implicitly realises the Bayesian
  posterior over models.

The batched simulators (:func:`_batch_sse`, :func:`_batch_ssi`) accept
``R0`` and ``k`` as either scalars or per-sim ``(n_sims,)`` arrays; the
array case is used by the parameter-uncertain code path, which draws fresh
``(R0, k)`` per batch from the user-supplied :class:`Prior`.

The simulation parameterisation matches the offspring distributions used in
``extinction.py`` and described in ``notes/notes.tex``:

* SSE step: ``I_t | foi ~ NB(mean=R0 * foi, disp=k * foi)`` where
  ``foi = sum_{s=1..L} w_s * I_{t-s}``. With ``p = k / (k + R0)`` (independent
  of ``foi``) this is ``NB(n = k * foi, p)`` in numpy's parameterisation.
* SSI step: ``I_t | foi ~ Poisson(foi)`` with ``foi = R0 * sum_s w_s Y_{t-s}``
  and ``Y_t | I_t ~ Gamma(shape=k * I_t, scale=1/k)`` (so ``Y_t = 0`` whenever
  ``I_t = 0``).
* Poisson step: ``I_t | foi ~ Poisson(R0 * foi)`` with
  ``foi = sum_{s=1..L} w_s * I_{t-s}`` — the ``k -> infty`` limit of either
  the SSE or SSI model.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from sse_ssi_pmo.priors import Prior
from sse_ssi_pmo.serial_interval import delay_cdf, sample_delay

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_inputs(
    R0: float | NDArray[np.float64],
    k: float | NDArray[np.float64],
    w: NDArray[np.float64],
    threshold: int,
    t_max: int,
) -> None:
    R0_arr = np.asarray(R0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    if (R0_arr <= 0.0).any() or (k_arr <= 0.0).any():
        raise ValueError("R0 and k must be positive")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("w must be a non-empty 1-D array")
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")


def _check_inputs_poisson(R0: float, w: NDArray[np.float64], threshold: int, t_max: int) -> None:
    """Variant of :func:`_check_inputs` without the ``k > 0`` check (no ``k``)."""
    if R0 <= 0.0:
        raise ValueError("R0 must be positive")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("w must be a non-empty 1-D array")
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")


def _expand_param(x: float | NDArray[np.float64], n_sims: int, label: str) -> NDArray[np.float64]:
    """Broadcast a scalar to ``(n_sims,)``; pass through arrays of that length.

    Raises if ``x`` is an array of any other length.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_sims, float(arr), dtype=np.float64)
    if arr.ndim == 1 and arr.size == n_sims:
        return arr.astype(np.float64, copy=False)
    raise ValueError(
        f"{label}: expected a scalar or 1-D array of length {n_sims}, got shape {arr.shape}"
    )


def _maybe_draw(x: float | Prior, n_sims: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Per-sim parameter array: draw from a :class:`Prior` or broadcast a scalar."""
    if isinstance(x, Prior):
        return x.sample(n_sims, rng)
    return np.full(n_sims, float(x), dtype=np.float64)


def _check_param_spec(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    threshold: int,
    t_max: int,
) -> None:
    """Validate R0/k (scalar-or-:class:`Prior`) + ``(w, threshold, t_max)`` jointly."""
    if not isinstance(R0, Prior) and float(R0) <= 0.0:
        raise ValueError("R0 must be positive")
    if not isinstance(k, Prior) and float(k) <= 0.0:
        raise ValueError("k must be positive")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("w must be a non-empty 1-D array")
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")


def _rejection_loop(
    *,
    n_sims: int,
    batch_size: int,
    max_attempts: int,
    M: int,
    is_done: Callable[[], bool],
    progress_count: Callable[[], int],
    run_batch: Callable[[int], None],
    show_progress: bool,
    desc: str,
) -> int:
    """Drive a rejection-sampling loop, calling ``run_batch(b)`` per batch.

    Continues until ``is_done()`` returns True or ``max_attempts`` total
    trajectories have been tried. ``progress_count`` returns the current
    count to display on the optional progress bar (clipped at ``n_sims * M``).
    Returns the total number of trajectories attempted.
    """
    n_attempted = 0
    pbar = tqdm(total=n_sims * M, desc=desc, leave=False) if show_progress else None
    while not is_done() and n_attempted < max_attempts:
        b = min(batch_size, max_attempts - n_attempted)
        run_batch(b)
        n_attempted += b
        if pbar is not None:
            pbar.update(progress_count() - pbar.n)
    if pbar is not None:
        pbar.close()
    return n_attempted


# ---------------------------------------------------------------------------
# Single-trajectory simulators (public)
# ---------------------------------------------------------------------------


def simulate_sse(
    R0: float,
    k: float,
    w: ArrayLike,
    *,
    init_incidence: ArrayLike = (1,),
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Simulate one SSE trajectory; return the incidence vector.

    The trajectory ends as soon as either a single-step incidence reaches
    ``threshold`` (counted as a major outbreak) or the most recent
    ``len(w)`` entries are all zero (extinction); otherwise it runs to
    length ``t_max``.

    ``init_incidence`` is the seeded history (default ``(1,)`` — a single index
    case at ``t = 0``); the simulator extends from ``t = len(init_incidence)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    init = np.asarray(init_incidence, dtype=np.int64)
    _check_inputs(R0, k, w_arr, threshold, t_max)
    if init.ndim != 1 or init.size == 0:
        raise ValueError("init_incidence must be a non-empty 1-D array of integers")

    _major, _extinct, _matched, incidence = _batch_sse(
        R0, k, w_arr, init, n_sims=1, threshold=threshold, t_max=t_max, rng=rng
    )
    end = _resolution_index(incidence[0], len(w_arr), threshold)
    return incidence[0, : end + 1].copy()


def simulate_ssi(
    R0: float,
    k: float,
    w: ArrayLike,
    *,
    init_incidence: int = 1,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Simulate one SSI trajectory from ``t = 0``; return the incidence vector.

    The latent infectivity at ``t = 0`` is drawn from
    ``Gamma(shape=k * init_incidence, scale=1/k)``. The trajectory ends as
    soon as either a single-step incidence reaches ``threshold`` (major
    outbreak) or the most recent ``len(w)`` entries are all zero (extinction);
    otherwise it runs to length ``t_max``.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    _check_inputs(R0, k, w_arr, threshold, t_max)
    if init_incidence < 1:
        raise ValueError("init_incidence must be at least 1")

    _major, _extinct, _matched, incidence = _batch_ssi(
        R0,
        k,
        w_arr,
        init_incidence=init_incidence,
        n_sims=1,
        threshold=threshold,
        t_max=t_max,
        rng=rng,
    )
    end = _resolution_index(incidence[0], len(w_arr), threshold)
    return incidence[0, : end + 1].copy()


def simulate_poisson(
    R0: float,
    w: ArrayLike,
    *,
    init_incidence: ArrayLike = (1,),
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Simulate one Poisson-offspring trajectory; return the incidence vector.

    Step: ``I_t ~ Poisson(R0 * sum_{s=1..L} w_s * I_{t-s})`` — the
    ``k -> infty`` limit of :func:`simulate_sse` / :func:`simulate_ssi`.
    The trajectory ends as soon as either a single-step incidence reaches
    ``threshold`` (major outbreak) or the most recent ``len(w)`` entries
    are all zero (extinction); otherwise it runs to length ``t_max``.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    init = np.asarray(init_incidence, dtype=np.int64)
    _check_inputs_poisson(R0, w_arr, threshold, t_max)
    if init.ndim != 1 or init.size == 0:
        raise ValueError("init_incidence must be a non-empty 1-D array of integers")
    history_len = init.size
    if history_len > t_max:
        raise ValueError("init_incidence longer than t_max")

    L_w = w_arr.size
    incidence = np.zeros(t_max, dtype=np.int64)
    incidence[:history_len] = init

    for t in range(history_len, t_max):
        L_use = min(t, L_w)
        foi = float(incidence[t - L_use : t] @ w_arr[:L_use][::-1])
        if foi > 0.0:
            incidence[t] = int(rng.poisson(R0 * foi))
        if incidence[t] >= threshold:
            break
        ext_start = max(0, t + 1 - L_w)
        if incidence[ext_start : t + 1].sum() == 0:
            break

    end = _resolution_index(incidence, L_w, threshold)
    return incidence[: end + 1].copy()


def _resolution_index(traj: NDArray[np.int64], L: int, threshold: int) -> int:
    """Index of the last entry to keep: first time the trajectory resolved."""
    major_mask = traj >= threshold
    if major_mask.any():
        return int(np.argmax(major_mask))
    # extinction = last L entries all zero
    if traj.size >= L:
        windowed = np.lib.stride_tricks.sliding_window_view(traj, L)
        zero_window = windowed.sum(axis=1) == 0  # length traj.size - L + 1
        if zero_window.any():
            return int(np.argmax(zero_window) + L - 1)
    return traj.size - 1


# ---------------------------------------------------------------------------
# Batched simulators (private)
# ---------------------------------------------------------------------------


def _batch_sse(
    R0: float | NDArray[np.float64],
    k: float | NDArray[np.float64],
    w: NDArray[np.float64],
    init_incidence: NDArray[np.int64],
    *,
    n_sims: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator,
    match_histories: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]]:
    """Vectorised SSE forward sim, all sims seeded with ``init_incidence``.

    Returns ``(major, extinct, matched_history, incidence)``.
    ``matched_history`` is an ``(n_sims,)`` int array indicating which row of
    ``match_histories`` each surviving sim matches; ``-1`` for sims that
    diverged from every history. Without ``match_histories`` every sim is
    treated as matching an implicit single history (``matched_history`` is
    all-zero) and the indeterminate-at-``t_max`` mask reduces to
    ``~(major | extinct)``.

    ``R0`` and ``k`` may be scalars (broadcast to every sim) or per-sim
    ``(n_sims,)`` arrays. The per-sim case is used by the
    parameter-uncertain code path, where fresh ``(R0, k)`` are drawn from a
    :class:`~sse_ssi_pmo.priors.Prior` once per batch.

    With ``match_histories`` provided (shape ``(M, L)``), the
    ``len(init_incidence)`` prefix of every row must equal ``init_incidence``
    (so the seeded incidence is consistent with all histories); sims are
    progressively pruned via an ``(n_sims, M)`` ``alive`` mask, and abandoned
    once their row of ``alive`` is all-False. Duplicate rows in
    ``match_histories`` are not allowed (each surviving sim must lock onto a
    unique matched-history index).
    """
    R0_arr = _expand_param(R0, n_sims, "_batch_sse: R0")
    k_arr = _expand_param(k, n_sims, "_batch_sse: k")
    L_w = len(w)
    history_len = len(init_incidence)
    if history_len > t_max:
        raise ValueError("init_incidence longer than t_max")
    if match_histories is not None:
        if match_histories.ndim != 2:
            raise ValueError("match_histories must be a 2-D (M, L) int array")
        M, match_len = match_histories.shape
        if M < 1:
            raise ValueError("match_histories must have at least one row")
        if match_len < history_len:
            raise ValueError("match_histories must extend init_incidence")
        if match_len > t_max:
            raise ValueError("match_histories longer than t_max")
        if history_len > 0 and not np.array_equal(
            match_histories[:, :history_len],
            np.broadcast_to(init_incidence, (M, history_len)),
        ):
            raise ValueError("each row of match_histories must start with init_incidence")
        if M > 1 and np.unique(match_histories, axis=0).shape[0] != M:
            raise ValueError("match_histories rows must be distinct")
    else:
        M = 1
        match_len = 0

    incidence = np.zeros((n_sims, t_max), dtype=np.int64)
    incidence[:, :history_len] = init_incidence

    alive = np.ones((n_sims, M), dtype=bool)
    major = (
        incidence[:, :history_len].max(axis=1) >= threshold
        if history_len > 0
        else np.zeros(n_sims, dtype=bool)
    )
    if history_len >= L_w:
        extinct = incidence[:, history_len - L_w : history_len].sum(axis=1) == 0
    else:
        extinct = np.zeros(n_sims, dtype=bool)
    extinct &= ~major
    p_nb_full = k_arr / (k_arr + R0_arr)
    w_rev_cache: dict[int, NDArray[np.float64]] = {}

    for t in range(history_len, t_max):
        live_mask = alive.any(axis=1) & ~(major | extinct)
        if not live_mask.any():
            break
        idx = np.where(live_mask)[0]
        L_use = min(t, L_w)
        if L_use not in w_rev_cache:
            w_rev_cache[L_use] = w[:L_use][::-1].copy()
        recent = incidence[idx, t - L_use : t]
        foi = recent @ w_rev_cache[L_use]
        new_inc = np.zeros(idx.size, dtype=np.int64)
        nz = foi > 0.0
        if nz.any():
            sel = idx[nz]
            new_inc[nz] = rng.negative_binomial(k_arr[sel] * foi[nz], p_nb_full[sel])
        incidence[idx, t] = new_inc

        if t < match_len:
            assert match_histories is not None  # narrows type for the checker
            match_t = new_inc[:, None] == match_histories[None, :, t]  # (k, M)
            alive[idx] &= match_t
            keep = alive[idx].any(axis=1)
            if not keep.all():
                idx = idx[keep]
                new_inc = new_inc[keep]
                if idx.size == 0:
                    continue

        new_major = new_inc >= threshold
        ext_start = max(0, t + 1 - L_w)
        new_extinct = incidence[idx, ext_start : t + 1].sum(axis=1) == 0
        new_extinct &= ~new_major
        major[idx[new_major]] = True
        extinct[idx[new_extinct]] = True

    matched_history = np.where(alive.any(axis=1), alive.argmax(axis=1), -1).astype(np.int64)
    return major, extinct, matched_history, incidence


def _batch_ssi(
    R0: float | NDArray[np.float64],
    k: float | NDArray[np.float64],
    w: NDArray[np.float64],
    *,
    init_incidence: int,
    n_sims: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator,
    match_histories: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]]:
    """Vectorised SSI forward sim from ``t = 0``, seeded with ``init_incidence``.

    Returns ``(major, extinct, matched_history, incidence)``. With
    ``match_histories`` provided (shape ``(M, L)``, ``L >= 1``), every row
    must start with ``init_incidence`` and rows must be distinct; sims are
    progressively pruned via an ``(n_sims, M)`` ``alive`` mask and abandoned
    when no history can still match. ``matched_history[i]`` is the row index
    each surviving sim matches, or ``-1`` for abandoned sims. Without
    ``match_histories`` no post-day-0 matching is applied and
    ``matched_history`` is all-zero (the implicit single-history case).

    ``R0`` and ``k`` may be scalars (broadcast to every sim) or per-sim
    ``(n_sims,)`` arrays (parameter-uncertain code path).
    """
    if init_incidence < 1:
        raise ValueError("init_incidence must be at least 1 (I_0 >= 1)")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")
    R0_arr = _expand_param(R0, n_sims, "_batch_ssi: R0")
    k_arr = _expand_param(k, n_sims, "_batch_ssi: k")
    L_w = len(w)
    if match_histories is not None:
        if match_histories.ndim != 2:
            raise ValueError("match_histories must be a 2-D (M, L) int array")
        M, match_len = match_histories.shape
        if M < 1 or match_len < 1:
            raise ValueError("match_histories must have shape (M >= 1, L >= 1)")
        if match_len > t_max:
            raise ValueError("match_histories longer than t_max")
        if not (match_histories[:, 0] == init_incidence).all():
            raise ValueError("every row of match_histories must start with init_incidence")
        if M > 1 and np.unique(match_histories, axis=0).shape[0] != M:
            raise ValueError("match_histories rows must be distinct")
    else:
        M = 1
        match_len = 0

    incidence = np.zeros((n_sims, t_max), dtype=np.int64)
    incidence[:, 0] = init_incidence
    Y = np.zeros((n_sims, t_max), dtype=np.float64)
    Y[:, 0] = rng.gamma(k_arr * init_incidence, 1.0 / k_arr)

    alive = np.ones((n_sims, M), dtype=bool)
    major = incidence[:, 0] >= threshold
    extinct = np.zeros(n_sims, dtype=bool)
    w_rev_cache: dict[int, NDArray[np.float64]] = {}

    for t in range(1, t_max):
        live_mask = alive.any(axis=1) & ~(major | extinct)
        if not live_mask.any():
            break
        idx = np.where(live_mask)[0]
        L_use = min(t, L_w)
        if L_use not in w_rev_cache:
            w_rev_cache[L_use] = w[:L_use][::-1].copy()
        recent_Y = Y[idx, t - L_use : t]
        foi = R0_arr[idx] * (recent_Y @ w_rev_cache[L_use])
        new_inc = np.zeros(idx.size, dtype=np.int64)
        nz = foi > 0.0
        if nz.any():
            new_inc[nz] = rng.poisson(foi[nz])
        incidence[idx, t] = new_inc
        new_Y = np.zeros(idx.size, dtype=np.float64)
        nz_inc = new_inc > 0
        if nz_inc.any():
            k_sel = k_arr[idx[nz_inc]]
            new_Y[nz_inc] = rng.gamma(k_sel * new_inc[nz_inc], 1.0 / k_sel)
        Y[idx, t] = new_Y

        if t < match_len:
            assert match_histories is not None  # narrows type for the checker
            match_t = new_inc[:, None] == match_histories[None, :, t]  # (k, M)
            alive[idx] &= match_t
            keep = alive[idx].any(axis=1)
            if not keep.all():
                idx = idx[keep]
                new_inc = new_inc[keep]
                if idx.size == 0:
                    continue

        new_major = new_inc >= threshold
        ext_start = max(0, t + 1 - L_w)
        new_extinct = incidence[idx, ext_start : t + 1].sum(axis=1) == 0
        new_extinct &= ~new_major
        major[idx[new_major]] = True
        extinct[idx[new_extinct]] = True

    matched_history = np.where(alive.any(axis=1), alive.argmax(axis=1), -1).astype(np.int64)
    return major, extinct, matched_history, incidence


# ---------------------------------------------------------------------------
# Real-time PMO via a bootstrap particle filter (private)
#
# Two infection-anchored variants:
#   * ``_pmo_infection_realtime`` -- observed data are infections themselves
#     ("naive": case dates taken as infection times);
#   * ``_pmo_bridge_realtime`` -- infections follow the same generation-time
#     renewal, but are observed only through symptom onsets obtained by an
#     independent incubation delay (the "bridge" to the onset-anchored models).
# Both define a major outbreak on the infection process, so their forward
# resolution is shared (:func:`_resolve_forward_infection`).
# ---------------------------------------------------------------------------


def _infection_step_inplace(
    model: str,
    inf: NDArray[np.int64],
    Y: NDArray[np.float64],
    d: int,
    idx: NDArray[np.int64],
    *,
    R0: float,
    k: float,
    w: NDArray[np.float64],
    p_nb: float,
    w_rev_cache: dict[int, NDArray[np.float64]],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate day-``d+1`` infections for particles ``idx``, in place.

    ``inf[idx, :d+1]`` (and, for SSI, ``Y``) must be filled. Writes
    ``inf[idx, d+1]`` and, for SSI, its cohort infectivity ``Y[idx, d+1]``.
    Returns the new day-``d+1`` infection counts aligned with ``idx``. Same
    numerics as :func:`_batch_sse`/:func:`_batch_ssi` for scalar ``(R0, k)``.
    """
    t = d + 1
    L_w = w.size
    L_use = min(t, L_w)
    if L_use not in w_rev_cache:
        w_rev_cache[L_use] = w[:L_use][::-1].copy()
    w_rev = w_rev_cache[L_use]
    new = np.zeros(idx.size, dtype=np.int64)
    if model == "sse":
        foi = inf[idx, t - L_use : t] @ w_rev
        nz = foi > 0.0
        if nz.any():
            new[nz] = rng.negative_binomial(k * foi[nz], p_nb)
    else:  # ssi
        foi = R0 * (Y[idx, t - L_use : t] @ w_rev)
        nz = foi > 0.0
        if nz.any():
            new[nz] = rng.poisson(foi[nz])
    inf[idx, t] = new
    if model == "ssi":
        nz_inc = new > 0
        if nz_inc.any():
            sel = idx[nz_inc]
            Y[sel, t] = rng.gamma(k * new[nz_inc], 1.0 / k)
    return new


def _resolve_forward_infection(
    model: str,
    inf: NDArray[np.int64],
    Y: NDArray[np.float64],
    d_start: int,
    *,
    R0: float,
    k: float,
    w: NDArray[np.float64],
    p_nb: float,
    threshold: int,
    t_max: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Free-forward each particle's infection process from day ``d_start``.

    Steps every particle forward (no history matching) until it reaches the
    major-outbreak threshold (single-day infections ``>= threshold``) or goes
    extinct (last ``len(w)`` days of infections all zero). Mutates
    ``inf``/``Y`` -- pass copies. Returns ``(major, extinct)`` boolean arrays.
    """
    n = inf.shape[0]
    L_w = w.size
    major = np.zeros(n, dtype=bool)
    extinct = np.zeros(n, dtype=bool)
    w_rev_cache: dict[int, NDArray[np.float64]] = {}
    for t in range(d_start, t_max):
        live = ~(major | extinct)
        new_major = live & (inf[:, t] >= threshold)
        major[new_major] = True
        live &= ~new_major
        if not live.any():
            break
        ext_start = max(0, t + 1 - L_w)
        new_extinct = live & (inf[:, ext_start : t + 1].sum(axis=1) == 0)
        extinct[new_extinct] = True
        live &= ~new_extinct
        if not live.any() or t + 1 >= t_max:
            break
        _infection_step_inplace(
            model,
            inf,
            Y,
            t,
            np.where(live)[0],
            R0=R0,
            k=k,
            w=w,
            p_nb=p_nb,
            w_rev_cache=w_rev_cache,
            rng=rng,
        )
    return major, extinct


def _pmo_infection_realtime(
    model: str,
    R0: float,
    k: float,
    w: ArrayLike,
    observed: ArrayLike,
    *,
    n_particles: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> dict[str, NDArray]:
    """Real-time PMO across the weeks of an observed *infection* history.

    Bootstrap particle filter for the infection-anchored SSE/SSI models when the
    observed history is infection incidence itself (case dates taken as infection
    times). ``n_particles`` particles are advanced one week at a time; those
    whose freshly generated incidence matches the observation are kept and
    resampled. The week-``w`` PMO is the fraction of the resampled population that
    reaches a major outbreak when simulated forward freely. Returns a dict with
    ``"pmo"``, ``"n_matches"``, ``"n_distinct"``, ``"n_indet"``, ``"log_evidence"``
    (each length ``L = len(observed)``); see :class:`PmoRealtimeResult`.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.int64)
    L = obs.size
    N = n_particles
    p_nb = k / (k + R0)

    inf = np.zeros((N, t_max), dtype=np.int64)
    inf[:, 0] = int(obs[0])
    Y = np.zeros((N, t_max), dtype=np.float64)
    if model == "ssi":
        Y[:, 0] = rng.gamma(k * int(obs[0]), 1.0 / k, size=N)

    pmo = np.full(L, np.nan, dtype=np.float64)
    n_matches = np.zeros(L, dtype=np.int64)
    n_distinct = np.zeros(L, dtype=np.int64)
    n_indet = np.zeros(L, dtype=np.int64)
    log_evidence = np.full(L, -np.inf, dtype=np.float64)
    cum_log_ev = 0.0
    all_idx = np.arange(N)
    w_rev_cache: dict[int, NDArray[np.float64]] = {}

    for wk in range(L):
        if wk == 0:
            n_matches[0] = N
            n_distinct[0] = N
            log_evidence[0] = 0.0
        else:
            match_idx = np.where(inf[:, wk] == obs[wk])[0]
            n_matches[wk] = match_idx.size
            if match_idx.size == 0:
                warnings.warn(
                    f"real-time PMO particle filter collapsed at week {wk} "
                    f"(no particle matched incidence={obs[wk]}); PMO is NaN from week {wk}. "
                    "Increase n_particles.",
                    stacklevel=2,
                )
                break
            cum_log_ev += float(np.log(match_idx.size / N))
            log_evidence[wk] = cum_log_ev
            chosen = rng.choice(match_idx, size=N, replace=True)
            inf = inf[chosen]
            Y = Y[chosen]
            n_distinct[wk] = int(np.unique(chosen).size)

        major, extinct = _resolve_forward_infection(
            model,
            inf.copy(),
            Y.copy(),
            wk,
            R0=R0,
            k=k,
            w=w_arr,
            p_nb=p_nb,
            threshold=threshold,
            t_max=t_max,
            rng=rng,
        )
        resolved = major | extinct
        n_indet[wk] = int((~resolved).sum())
        n_res = int(resolved.sum())
        pmo[wk] = float(major.sum()) / n_res if n_res else np.nan

        if wk < L - 1:
            _infection_step_inplace(
                model,
                inf,
                Y,
                wk,
                all_idx,
                R0=R0,
                k=k,
                w=w_arr,
                p_nb=p_nb,
                w_rev_cache=w_rev_cache,
                rng=rng,
            )

    if int(n_indet.sum()) > 0:
        warnings.warn(
            f"{int(n_indet.sum())} forward sims across weeks hit t_max={t_max} "
            "unresolved; raise t_max.",
            stacklevel=2,
        )
    return {
        "pmo": pmo,
        "n_matches": n_matches,
        "n_distinct": n_distinct,
        "n_indet": n_indet,
        "log_evidence": log_evidence,
    }


def _pmo_bridge_realtime(
    model: str,
    R0: float,
    k: float,
    w: ArrayLike,
    inc: ArrayLike,
    observed: ArrayLike,
    *,
    seed_lead: int = 1,
    n_particles: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> dict[str, NDArray]:
    """Real-time PMO over an observed *onset* history under the bridge model.

    Infections follow the same generation-time renewal as
    :func:`_pmo_infection_realtime` (weights ``w``, lag >= 1), but are observed
    only through symptom onsets: each infection is scattered forward by an
    independent incubation delay ``inc`` (indexed from lag 0). A single index
    infection is seeded ``seed_lead`` weeks *before* the first observed onset, so
    its descendants can populate the early observed weeks (with a lag-1 renewal
    an index seeded at the first observed week could not). The particle filter
    matches the observed onset cohorts; a major outbreak is defined on the
    infection process (as in the naive variant), so forward resolution is shared.

    Returns a dict with ``"pmo"``, ``"n_matches"``, ``"n_distinct"``,
    ``"n_indet"``, ``"log_evidence"`` (each length ``L = len(observed)``).
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    inc_arr = np.asarray(inc, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.int64)
    n = obs.size
    N = n_particles
    p_nb = k / (k + R0)
    inc_cdf, inc_support = delay_cdf(inc_arr, start=0)  # incubation allowed from lag 0
    d_end = seed_lead + n - 1
    if d_end >= t_max:
        raise ValueError("t_max too small for seed_lead + len(observed)")

    inf = np.zeros((N, t_max), dtype=np.int64)
    inf[:, 0] = 1  # single index infection, seed_lead weeks before the first onset
    Y = np.zeros((N, t_max), dtype=np.float64)
    if model == "ssi":
        Y[:, 0] = rng.gamma(k * 1.0, 1.0 / k, size=N)
    onset = np.zeros((N, t_max), dtype=np.int64)
    # scatter the index case's own onset (incubation from lag 0)
    incs0 = sample_delay(inc_cdf, inc_support, N, rng)
    in_range0 = incs0 < t_max
    np.add.at(onset, (np.arange(N)[in_range0], incs0[in_range0]), 1)

    pmo = np.full(n, np.nan, dtype=np.float64)
    n_matches = np.zeros(n, dtype=np.int64)
    n_distinct = np.zeros(n, dtype=np.int64)
    n_indet = np.zeros(n, dtype=np.int64)
    log_evidence = np.full(n, -np.inf, dtype=np.float64)
    cum_log_ev = 0.0
    all_idx = np.arange(N)
    w_rev_cache: dict[int, NDArray[np.float64]] = {}

    for d in range(d_end + 1):
        if d >= seed_lead:
            j = d - seed_lead
            match_idx = np.where(onset[:, d] == obs[j])[0]
            n_matches[j] = match_idx.size
            if match_idx.size == 0:
                warnings.warn(
                    f"real-time PMO particle filter collapsed at week {j} "
                    f"(no particle matched onsets={obs[j]}); PMO is NaN from week {j}. "
                    "Increase n_particles or seed_lead.",
                    stacklevel=2,
                )
                break
            cum_log_ev += float(np.log(match_idx.size / N))
            log_evidence[j] = cum_log_ev
            chosen = rng.choice(match_idx, size=N, replace=True)
            inf = inf[chosen]
            Y = Y[chosen]
            onset = onset[chosen]
            n_distinct[j] = int(np.unique(chosen).size)

            major, extinct = _resolve_forward_infection(
                model,
                inf.copy(),
                Y.copy(),
                d,
                R0=R0,
                k=k,
                w=w_arr,
                p_nb=p_nb,
                threshold=threshold,
                t_max=t_max,
                rng=rng,
            )
            resolved = major | extinct
            n_indet[j] = int((~resolved).sum())
            n_res = int(resolved.sum())
            pmo[j] = float(major.sum()) / n_res if n_res else np.nan

        if d < d_end:
            new = _infection_step_inplace(
                model,
                inf,
                Y,
                d,
                all_idx,
                R0=R0,
                k=k,
                w=w_arr,
                p_nb=p_nb,
                w_rev_cache=w_rev_cache,
                rng=rng,
            )
            total = int(new.sum())
            if total > 0:
                sim_of_each = np.repeat(all_idx, new)
                incs = sample_delay(inc_cdf, inc_support, total, rng)
                target = (d + 1) + incs
                in_range = target < t_max
                if in_range.any():
                    np.add.at(onset, (sim_of_each[in_range], target[in_range]), 1)

    if int(n_indet.sum()) > 0:
        warnings.warn(
            f"{int(n_indet.sum())} forward sims across weeks hit t_max={t_max} "
            "unresolved; raise t_max.",
            stacklevel=2,
        )
    return {
        "pmo": pmo,
        "n_matches": n_matches,
        "n_distinct": n_distinct,
        "n_indet": n_indet,
        "log_evidence": log_evidence,
    }


# ---------------------------------------------------------------------------
# Public PMO-via-simulation
# ---------------------------------------------------------------------------


def _pmo_sse_sim(
    R0: float | Prior,
    k: float | Prior,
    w: ArrayLike,
    history: ArrayLike,
    *,
    n_sims: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
    show_progress: bool = False,
) -> float:
    """SSE PMO estimate after the observed ``history`` (Monte-Carlo).

    Forward-simulates ``n_sims`` SSE trajectories seeded with ``history`` and
    returns the fraction whose single-step incidence reaches ``threshold``
    (the operational definition of a "major outbreak"); the rest go extinct
    (last ``len(w)`` steps all zero).

    ``R0``/``k`` may be :class:`~sse_ssi_pmo.priors.Prior` instances; in that
    case a fresh per-sim draw is made from the prior before forward
    simulation, so the returned PMO is the marginal under the prior.

    ``show_progress`` is accepted for signature symmetry with ``_pmo_ssi_sim``
    but has no effect: the SSE batch runs in a single vectorised call.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = np.asarray(history, dtype=np.int64)
    _check_param_spec(R0, k, w_arr, threshold, t_max)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    R0_arr = _maybe_draw(R0, n_sims, rng)
    k_arr = _maybe_draw(k, n_sims, rng)

    major, extinct, _matched, _ = _batch_sse(
        R0_arr, k_arr, w_arr, hist_arr, n_sims=n_sims, threshold=threshold, t_max=t_max, rng=rng
    )
    n_indet = int((~(major | extinct)).sum())
    if n_indet:
        warnings.warn(
            f"{n_indet}/{n_sims} SSE sims hit t_max={t_max} unresolved; raise t_max.",
            stacklevel=2,
        )
    n_resolved = int(major.sum() + extinct.sum())
    if n_resolved == 0:
        return float("nan")
    return float(major.sum()) / n_resolved


def _validate_histories_for_sim(
    histories: NDArray[np.int64], *, t_max: int, label: str
) -> tuple[int, int, int]:
    """Common validation for the multi-history simulation backends.

    Returns ``(M, L, I_0)``. Raises ``ValueError`` if ``histories`` isn't 2-D,
    has a non-shared ``I_0``, has ``I_0 < 1``, exceeds ``t_max`` in length, or
    contains duplicate rows.
    """
    if histories.ndim != 2:
        raise ValueError(f"{label}: histories must be a 2-D (M, L) int array")
    M, L = histories.shape
    if M < 1 or L < 1:
        raise ValueError(f"{label}: histories must have shape (M >= 1, L >= 1)")
    if t_max < L:
        raise ValueError(f"{label}: histories longer than t_max")
    if not (histories[:, 0] == histories[0, 0]).all():
        raise ValueError(
            f"{label}: every history must share the same I_0; "
            "split your input into shared-I_0 groups and call per group."
        )
    if histories[0, 0] < 1:
        raise ValueError(f"{label}: histories[:, 0] (I_0) must be >= 1")
    if M > 1 and np.unique(histories, axis=0).shape[0] != M:
        raise ValueError(f"{label}: histories rows must be distinct")
    return M, L, int(histories[0, 0])


def _pmo_ssi_sim_multi(
    R0: float | Prior,
    k: float | Prior,
    w: ArrayLike,
    histories: ArrayLike,
    *,
    n_sims: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
    batch_size: int = 2000,
    max_attempts: int | None = None,
    show_progress: bool = False,
) -> NDArray[np.float64]:
    """Per-history SSI PMO estimates via shared rejection sampling.

    Simulates SSI trajectories in batches seeded with the shared
    ``I_0 = histories[0, 0]`` and routes each accepted trajectory to the
    unique history it matches. Acceptance rate is the union rate across all
    ``M`` histories, so one batched pass effectively replaces ``M``
    independent rejection-sampling loops. Continues until every history has
    accumulated ``n_sims`` resolved matches (or ``max_attempts`` total
    trajectories have been tried). Returns ``(M,)`` float array of
    per-history PMO estimates; ``NaN`` for histories with no resolved
    matches.

    ``R0``/``k`` may be :class:`~sse_ssi_pmo.priors.Prior` instances; in that
    case fresh per-sim draws are made from the prior at the start of every
    batch, so the returned PMOs are marginal under the prior.

    ``max_attempts`` defaults to ``200 * n_sims``.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = np.asarray(histories, dtype=np.int64)
    _check_param_spec(R0, k, w_arr, threshold, t_max)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_attempts is None:
        max_attempts = 200 * n_sims
    M, _L, I_0 = _validate_histories_for_sim(hist_arr, t_max=t_max, label="_pmo_ssi_sim_multi")

    n_major = np.zeros(M, dtype=np.int64)
    n_extinct = np.zeros(M, dtype=np.int64)
    n_indet = np.zeros(M, dtype=np.int64)

    def run_batch(b: int) -> None:
        R0_batch = _maybe_draw(R0, b, rng)
        k_batch = _maybe_draw(k, b, rng)
        major, extinct, matched, _ = _batch_ssi(
            R0_batch,
            k_batch,
            w_arr,
            init_incidence=I_0,
            n_sims=b,
            threshold=threshold,
            t_max=t_max,
            rng=rng,
            match_histories=hist_arr,
        )
        ok = matched >= 0
        np.add.at(n_major, matched[ok & major], 1)
        np.add.at(n_extinct, matched[ok & extinct], 1)
        np.add.at(n_indet, matched[ok & ~(major | extinct)], 1)

    n_attempted = _rejection_loop(
        n_sims=n_sims,
        batch_size=batch_size,
        max_attempts=max_attempts,
        M=M,
        is_done=lambda: bool((n_major + n_extinct).min() >= n_sims),
        progress_count=lambda: int(np.minimum(n_major + n_extinct, n_sims).sum()),
        run_batch=run_batch,
        show_progress=show_progress,
        desc="SSI matching sims",
    )

    n_indet_total = int(n_indet.sum())
    if n_indet_total:
        warnings.warn(
            f"{n_indet_total} matching SSI sims hit t_max={t_max} unresolved; raise t_max.",
            stacklevel=2,
        )
    n_resolved = n_major + n_extinct
    under = int((n_resolved < n_sims).sum())
    if under:
        zeros = int((n_resolved == 0).sum())
        warnings.warn(
            f"{under}/{M} histories under-resolved after {n_attempted} attempts "
            f"(max_attempts={max_attempts}); {zeros} returned NaN. "
            "Increase max_attempts or check history feasibility.",
            stacklevel=2,
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n_resolved > 0, n_major / n_resolved, np.nan)
    return out.astype(np.float64)


def _pmo_ssi_sim(
    R0: float | Prior,
    k: float | Prior,
    w: ArrayLike,
    history: ArrayLike,
    *,
    n_sims: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
    batch_size: int = 2000,
    max_attempts: int | None = None,
    show_progress: bool = False,
) -> float:
    """SSI PMO estimate for a single history (thin wrapper around the multi version)."""
    hist_2d = np.atleast_2d(np.asarray(history, dtype=np.int64))
    out = _pmo_ssi_sim_multi(
        R0,
        k,
        w,
        hist_2d,
        n_sims=n_sims,
        threshold=threshold,
        t_max=t_max,
        rng=rng,
        batch_size=batch_size,
        max_attempts=max_attempts,
        show_progress=show_progress,
    )
    return float(out[0])


def _pmo_uncertain_sim_multi(
    R0: float | Prior,
    k: float | Prior,
    w: ArrayLike,
    histories: ArrayLike,
    *,
    n_sims: int,
    threshold: int,
    t_max: int,
    prior_sse: float,
    rng: np.random.Generator | None = None,
    batch_size: int = 2000,
    max_attempts: int | None = None,
    show_progress: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Per-history model-averaged PMO via shared rejection sampling.

    Mirrors :func:`_pmo_uncertain_sim` but operates on a stack of histories
    sharing ``I_0``. Each batch is split between SSE and SSI by drawing
    ``b_sse ~ Bin(batch_size, prior_sse)``; both sub-batches use the
    multi-history matching path. Returns a dict with keys ``pmo``,
    ``posterior_sse``, ``pmo_sse``, ``pmo_ssi`` — each a ``(M,)`` array.

    ``R0``/``k`` may be :class:`~sse_ssi_pmo.priors.Prior` instances; the same
    fresh per-sim ``(R0, k)`` draws are used for both SSE and SSI sub-batches
    so the implicit posterior model split is sampled correctly under the
    parameter prior.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = np.asarray(histories, dtype=np.int64)
    _check_param_spec(R0, k, w_arr, threshold, t_max)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= prior_sse <= 1.0:
        raise ValueError("prior_sse must lie in [0, 1]")
    if max_attempts is None:
        max_attempts = 200 * n_sims
    M, _L, I_0 = _validate_histories_for_sim(
        hist_arr, t_max=t_max, label="_pmo_uncertain_sim_multi"
    )
    seed = hist_arr[:1, 0]  # length-1 init_incidence prefix shared across rows

    n_major_sse = np.zeros(M, dtype=np.int64)
    n_extinct_sse = np.zeros(M, dtype=np.int64)
    n_indet_sse = np.zeros(M, dtype=np.int64)
    n_major_ssi = np.zeros(M, dtype=np.int64)
    n_extinct_ssi = np.zeros(M, dtype=np.int64)
    n_indet_ssi = np.zeros(M, dtype=np.int64)

    def _resolved_total() -> NDArray[np.int64]:
        return n_major_sse + n_extinct_sse + n_major_ssi + n_extinct_ssi

    def run_batch(b: int) -> None:
        b_sse = int(rng.binomial(b, prior_sse))
        b_ssi = b - b_sse
        if b_sse > 0:
            R0_sse = _maybe_draw(R0, b_sse, rng)
            k_sse = _maybe_draw(k, b_sse, rng)
            major, extinct, matched, _ = _batch_sse(
                R0_sse,
                k_sse,
                w_arr,
                seed,
                n_sims=b_sse,
                threshold=threshold,
                t_max=t_max,
                rng=rng,
                match_histories=hist_arr,
            )
            ok = matched >= 0
            np.add.at(n_major_sse, matched[ok & major], 1)
            np.add.at(n_extinct_sse, matched[ok & extinct], 1)
            np.add.at(n_indet_sse, matched[ok & ~(major | extinct)], 1)
        if b_ssi > 0:
            R0_ssi = _maybe_draw(R0, b_ssi, rng)
            k_ssi = _maybe_draw(k, b_ssi, rng)
            major, extinct, matched, _ = _batch_ssi(
                R0_ssi,
                k_ssi,
                w_arr,
                init_incidence=I_0,
                n_sims=b_ssi,
                threshold=threshold,
                t_max=t_max,
                rng=rng,
                match_histories=hist_arr,
            )
            ok = matched >= 0
            np.add.at(n_major_ssi, matched[ok & major], 1)
            np.add.at(n_extinct_ssi, matched[ok & extinct], 1)
            np.add.at(n_indet_ssi, matched[ok & ~(major | extinct)], 1)

    n_attempted = _rejection_loop(
        n_sims=n_sims,
        batch_size=batch_size,
        max_attempts=max_attempts,
        M=M,
        is_done=lambda: bool(_resolved_total().min() >= n_sims),
        progress_count=lambda: int(np.minimum(_resolved_total(), n_sims).sum()),
        run_batch=run_batch,
        show_progress=show_progress,
        desc="uncertain matching sims",
    )

    n_indet_total = int(n_indet_sse.sum() + n_indet_ssi.sum())
    if n_indet_total:
        warnings.warn(
            f"{n_indet_total} matching uncertain sims hit t_max={t_max} unresolved; raise t_max.",
            stacklevel=2,
        )

    n_acc_sse = n_major_sse + n_extinct_sse
    n_acc_ssi = n_major_ssi + n_extinct_ssi
    n_acc_total = n_acc_sse + n_acc_ssi
    under = int((n_acc_total < n_sims).sum())
    if under:
        zeros = int((n_acc_total == 0).sum())
        warnings.warn(
            f"{under}/{M} histories under-resolved after {n_attempted} attempts "
            f"(max_attempts={max_attempts}); {zeros} returned NaN. "
            "Increase max_attempts or check history feasibility.",
            stacklevel=2,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        pmo = np.where(n_acc_total > 0, (n_major_sse + n_major_ssi) / n_acc_total, np.nan)
        posterior_sse = np.where(n_acc_total > 0, n_acc_sse / n_acc_total, np.nan)
        pmo_sse = np.where(n_acc_sse > 0, n_major_sse / n_acc_sse, np.nan)
        pmo_ssi = np.where(n_acc_ssi > 0, n_major_ssi / n_acc_ssi, np.nan)
    return {
        "pmo": pmo.astype(np.float64),
        "posterior_sse": posterior_sse.astype(np.float64),
        "pmo_sse": pmo_sse.astype(np.float64),
        "pmo_ssi": pmo_ssi.astype(np.float64),
    }


def _pmo_uncertain_sim(
    R0: float | Prior,
    k: float | Prior,
    w: ArrayLike,
    history: ArrayLike,
    *,
    n_sims: int,
    threshold: int,
    t_max: int,
    prior_sse: float,
    rng: np.random.Generator | None = None,
    batch_size: int = 2000,
    max_attempts: int | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    """Single-history model-averaged PMO (thin wrapper around the multi version)."""
    hist_2d = np.atleast_2d(np.asarray(history, dtype=np.int64))
    out = _pmo_uncertain_sim_multi(
        R0,
        k,
        w,
        hist_2d,
        n_sims=n_sims,
        threshold=threshold,
        t_max=t_max,
        prior_sse=prior_sse,
        rng=rng,
        batch_size=batch_size,
        max_attempts=max_attempts,
        show_progress=show_progress,
    )
    return {key: float(val[0]) for key, val in out.items()}


__all__ = [
    "simulate_poisson",
    "simulate_sse",
    "simulate_ssi",
]
