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

    _major, _extinct, _indet, incidence = _batch_sse(
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

    history = np.array([init_incidence], dtype=np.int64)
    _major, _extinct, _valid, incidence = _batch_ssi(
        R0, k, w_arr, history, n_sims=1, threshold=threshold, t_max=t_max, rng=rng
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
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int64]]:
    """Vectorised SSE forward sim, all sims seeded with ``init_incidence``.

    Returns ``(major, extinct, indeterminate, incidence)``: three disjoint bool
    arrays of length ``n_sims`` and the full incidence array of shape
    ``(n_sims, t_max)``.
    """
    L = len(w)
    history_len = len(init_incidence)
    if history_len > t_max:
        raise ValueError("init_incidence longer than t_max")

    incidence = np.zeros((n_sims, t_max), dtype=np.int64)
    incidence[:, :history_len] = init_incidence

    major = (
        incidence[:, :history_len].max(axis=1) >= threshold
        if history_len > 0
        else np.zeros(n_sims, dtype=bool)
    )
    if history_len >= L:
        extinct = incidence[:, history_len - L : history_len].sum(axis=1) == 0
    else:
        extinct = np.zeros(n_sims, dtype=bool)
    extinct &= ~major
    p_nb = k / (k + R0)
    w_rev_cache: dict[int, NDArray[np.float64]] = {}

    for t in range(history_len, t_max):
        live_mask = ~(major | extinct)
        if not live_mask.any():
            break
        idx = np.where(live_mask)[0]
        L_use = min(t, L)
        if L_use not in w_rev_cache:
            w_rev_cache[L_use] = w[:L_use][::-1].copy()
        recent = incidence[idx, t - L_use : t]
        foi = recent @ w_rev_cache[L_use]
        new_inc = np.zeros(idx.size, dtype=np.int64)
        nz = foi > 0.0
        if nz.any():
            new_inc[nz] = rng.negative_binomial(k * foi[nz], p_nb)
        incidence[idx, t] = new_inc
        new_major = new_inc >= threshold
        ext_start = max(0, t + 1 - L)
        new_extinct = incidence[idx, ext_start : t + 1].sum(axis=1) == 0
        new_extinct &= ~new_major
        major[idx[new_major]] = True
        extinct[idx[new_extinct]] = True

    indeterminate = ~(major | extinct)
    return major, extinct, indeterminate, incidence


def _batch_ssi(
    R0: float,
    k: float,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    n_sims: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int64]]:
    """Vectorised SSI forward sim from ``t = 0``, with optional history matching.

    Sims whose ``incidence[t]`` differs from ``history[t]`` for any
    ``1 <= t < len(history)`` are marked invalid and abandoned. Only sims that
    match the entire history are eligible for ``major`` / ``extinct``; invalid
    sims are reported via the ``indeterminate`` flag implicitly (they will be
    neither major nor extinct).

    Returns ``(major, extinct, valid, incidence)``: ``valid`` is True for sims
    that matched the full history.
    """
    L = len(w)
    history_len = len(history)
    if history_len < 1 or history[0] < 1:
        raise ValueError("history must start with I_0 >= 1")
    if history_len > t_max:
        raise ValueError("history longer than t_max")

    incidence = np.zeros((n_sims, t_max), dtype=np.int64)
    incidence[:, 0] = history[0]
    Y = np.zeros((n_sims, t_max), dtype=np.float64)
    Y[:, 0] = rng.gamma(k * history[0], 1.0 / k, size=n_sims)

    valid = np.ones(n_sims, dtype=bool)
    major = incidence[:, 0] >= threshold
    extinct = np.zeros(n_sims, dtype=bool)
    w_rev_cache: dict[int, NDArray[np.float64]] = {}

    for t in range(1, t_max):
        live_mask = valid & ~(major | extinct)
        if not live_mask.any():
            break
        idx = np.where(live_mask)[0]
        L_use = min(t, L)
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

        if t < history_len:
            mismatch = new_inc != history[t]
            if mismatch.any():
                valid[idx[mismatch]] = False
                keep = ~mismatch
                idx = idx[keep]
                new_inc = new_inc[keep]
                if idx.size == 0:
                    continue

        new_major = new_inc >= threshold
        ext_start = max(0, t + 1 - L)
        new_extinct = incidence[idx, ext_start : t + 1].sum(axis=1) == 0
        new_extinct &= ~new_major
        major[idx[new_major]] = True
        extinct[idx[new_extinct]] = True

    return major, extinct, valid, incidence


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

    major, extinct, indet, _ = _batch_sse(
        R0, k, w_arr, hist_arr, n_sims=n_sims, threshold=threshold, t_max=t_max, rng=rng
    )
    n_indet = int(indet.sum())
    if n_indet:
        warnings.warn(
            f"{n_indet}/{n_sims} SSE sims hit t_max={t_max} unresolved; raise t_max.",
            stacklevel=2,
        )
    n_resolved = int(major.sum() + extinct.sum())
    if n_resolved == 0:
        return float("nan")
    return float(major.sum()) / n_resolved


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
    """SSI PMO estimate, with rejection sampling against ``history``.

    Simulates SSI trajectories from ``t = 0`` in batches of ``batch_size``,
    keeping only those whose first ``len(history)`` entries match ``history``,
    until ``n_sims`` matching trajectories have been resolved (or
    ``max_attempts`` total trajectories tried). Returns the fraction of
    resolved matching sims whose single-step incidence reaches ``threshold``
    (the operational definition of a "major outbreak").

    ``max_attempts`` defaults to ``200 * n_sims``.
    """
    rng = np.random.default_rng() if rng is None else rng
    w_arr = np.asarray(w, dtype=np.float64)
    hist_arr = np.asarray(history, dtype=np.int64)
    _check_inputs(R0, k, w_arr, threshold, t_max)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_attempts is None:
        max_attempts = 200 * n_sims

    n_major = 0
    n_extinct = 0
    n_indet = 0
    n_attempted = 0
    pbar = tqdm(total=n_sims, desc="SSI matching sims", leave=False) if show_progress else None

    while n_major + n_extinct < n_sims and n_attempted < max_attempts:
        b = min(batch_size, max_attempts - n_attempted)
        major, extinct, valid, _ = _batch_ssi(
            R0, k, w_arr, hist_arr, n_sims=b, threshold=threshold, t_max=t_max, rng=rng
        )
        b_major = int((major & valid).sum())
        b_extinct = int((extinct & valid).sum())
        b_indet = int((valid & ~(major | extinct)).sum())
        n_major += b_major
        n_extinct += b_extinct
        n_indet += b_indet
        n_attempted += b
        if pbar is not None:
            pbar.update(b_major + b_extinct)

    if pbar is not None:
        pbar.close()

    if n_indet:
        warnings.warn(
            f"{n_indet} matching SSI sims hit t_max={t_max} unresolved; raise t_max.",
            stacklevel=2,
        )
    n_resolved = n_major + n_extinct
    if n_resolved == 0:
        warnings.warn(
            f"No matching SSI sims after {n_attempted} attempts; returning NaN. "
            "Increase max_attempts or check history feasibility.",
            stacklevel=2,
        )
        return float("nan")
    if n_resolved < n_sims:
        warnings.warn(
            f"Only {n_resolved}/{n_sims} matching SSI sims resolved after "
            f"{n_attempted} attempts (max_attempts={max_attempts}).",
            stacklevel=2,
        )
    return n_major / n_resolved


__all__ = [
    "simulate_sse",
    "simulate_ssi",
]
