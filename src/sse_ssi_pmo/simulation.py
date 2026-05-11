"""Forward simulation of the SSE and SSI models.

Public API:

* :func:`simulate_sse`, :func:`simulate_ssi` — single trajectories, terminating
  early on extinction (last ``len(w)`` time-steps all zero) or on single-step
  incidence reaching ``threshold`` (taken as the operational definition of
  "major outbreak").

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

The simulation parameterisation matches the offspring distributions used in
``extinction.py`` and described in ``notes/notes.tex``:

* SSE step: ``I_t | foi ~ NB(mean=R0 * foi, disp=k * foi)`` where
  ``foi = sum_{s=1..L} w_s * I_{t-s}``. With ``p = k / (k + R0)`` (independent
  of ``foi``) this is ``NB(n = k * foi, p)`` in numpy's parameterisation.
* SSI step: ``I_t | foi ~ Poisson(foi)`` with ``foi = R0 * sum_s w_s Y_{t-s}``
  and ``Y_t | I_t ~ Gamma(shape=k * I_t, scale=1/k)`` (so ``Y_t = 0`` whenever
  ``I_t = 0``).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_inputs(R0: float, k: float, w: NDArray[np.float64], threshold: int, t_max: int) -> None:
    if R0 <= 0.0 or k <= 0.0:
        raise ValueError("R0 and k must be positive")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("w must be a non-empty 1-D array")
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")


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
    R0: float,
    k: float,
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

    With ``match_histories`` provided (shape ``(M, L)``), the
    ``len(init_incidence)`` prefix of every row must equal ``init_incidence``
    (so the seeded incidence is consistent with all histories); sims are
    progressively pruned via an ``(n_sims, M)`` ``alive`` mask, and abandoned
    once their row of ``alive`` is all-False. Duplicate rows in
    ``match_histories`` are not allowed (each surviving sim must lock onto a
    unique matched-history index).
    """
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
    p_nb = k / (k + R0)
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
            new_inc[nz] = rng.negative_binomial(k * foi[nz], p_nb)
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
    R0: float,
    k: float,
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
    """
    if init_incidence < 1:
        raise ValueError("init_incidence must be at least 1 (I_0 >= 1)")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")
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
    Y[:, 0] = rng.gamma(k * init_incidence, 1.0 / k, size=n_sims)

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
        foi = R0 * (recent_Y @ w_rev_cache[L_use])
        new_inc = np.zeros(idx.size, dtype=np.int64)
        nz = foi > 0.0
        if nz.any():
            new_inc[nz] = rng.poisson(foi[nz])
        incidence[idx, t] = new_inc
        new_Y = np.zeros(idx.size, dtype=np.float64)
        nz_inc = new_inc > 0
        if nz_inc.any():
            new_Y[nz_inc] = rng.gamma(k * new_inc[nz_inc], 1.0 / k)
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
# Public PMO-via-simulation
# ---------------------------------------------------------------------------


def _pmo_sse_sim(
    R0: float,
    k: float,
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

    ``show_progress`` is accepted for signature symmetry with ``_pmo_ssi_sim``
    but has no effect: the SSE batch runs in a single vectorised call.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = np.asarray(history, dtype=np.int64)
    _check_inputs(R0, k, w_arr, threshold, t_max)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")

    major, extinct, _matched, _ = _batch_sse(
        R0, k, w_arr, hist_arr, n_sims=n_sims, threshold=threshold, t_max=t_max, rng=rng
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
    R0: float,
    k: float,
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

    ``max_attempts`` defaults to ``200 * n_sims``.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = np.asarray(histories, dtype=np.int64)
    _check_inputs(R0, k, w_arr, threshold, t_max)
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
    n_attempted = 0
    pbar = tqdm(total=n_sims * M, desc="SSI matching sims", leave=False) if show_progress else None

    while (n_major + n_extinct).min() < n_sims and n_attempted < max_attempts:
        b = min(batch_size, max_attempts - n_attempted)
        major, extinct, matched, _ = _batch_ssi(
            R0,
            k,
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
        n_attempted += b
        if pbar is not None:
            resolved = int((np.minimum(n_major + n_extinct, n_sims)).sum())
            pbar.update(resolved - pbar.n)

    if pbar is not None:
        pbar.close()

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
    R0: float,
    k: float,
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
    R0: float,
    k: float,
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
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = np.asarray(histories, dtype=np.int64)
    _check_inputs(R0, k, w_arr, threshold, t_max)
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
    n_attempted = 0

    pbar = (
        tqdm(total=n_sims * M, desc="uncertain matching sims", leave=False)
        if show_progress
        else None
    )

    def _resolved_total() -> NDArray[np.int64]:
        return n_major_sse + n_extinct_sse + n_major_ssi + n_extinct_ssi

    while _resolved_total().min() < n_sims and n_attempted < max_attempts:
        b = min(batch_size, max_attempts - n_attempted)
        b_sse = int(rng.binomial(b, prior_sse))
        b_ssi = b - b_sse

        if b_sse > 0:
            major, extinct, matched, _ = _batch_sse(
                R0,
                k,
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
            major, extinct, matched, _ = _batch_ssi(
                R0,
                k,
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

        n_attempted += b
        if pbar is not None:
            resolved = int(np.minimum(_resolved_total(), n_sims).sum())
            pbar.update(resolved - pbar.n)

    if pbar is not None:
        pbar.close()

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
    R0: float,
    k: float,
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
    "simulate_sse",
    "simulate_ssi",
]
