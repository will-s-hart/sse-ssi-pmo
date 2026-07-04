"""Forward simulation of the symptom-onset-anchored SSE and SSI models.

These models track **symptom-onset** incidence ``D_t`` (the observed quantity)
rather than infections. Transmission is anchored to a case's own onset via the
time-from-symptom-onset-to-transmission (TOST) distribution ``tost`` (indexed
from lag 0), and each new infection is mapped forward to its own onset via the
incubation-period distribution ``inc`` (indexed from lag 1). See
``notes/notes.tex`` §"Symptom-onset data and delayed transmission".

Public API:

* :func:`simulate_sse_delay`, :func:`simulate_ssi_delay` — single onset
  trajectories from a lone index onset on day 0, terminating early on a major
  outbreak (single-day onsets reaching ``threshold``) or extinction.

Private API (used by :mod:`sse_ssi_pmo.pmo`):

* :func:`_pmo_sse_delay_sim`, :func:`_pmo_ssi_delay_sim` (and their
  ``_multi`` forms) — Monte-Carlo PMO estimates after an observed onset
  ``history``. Unlike the infection-anchored SSE model, the observed onsets do
  not pin down the latent incubation pipeline, so **both** models use shared
  rejection sampling: simulate from day 0 seeded with the shared ``D_0`` and
  retain trajectories whose onset cohorts on days ``1..r`` match the history.

Model steps (matching ``notes/notes.tex``):

* SSE-delay: ``J_t | foi ~ NB(mean=R0*foi, disp=k*foi)`` with
  ``foi = sum_{s>=0} tost_s * D_{t-s}`` — ``J_t`` new infections on day ``t``.
* SSI-delay: each onset cohort carries ``Y_t | D_t ~ Gamma(shape=k*D_t,
  rate=k)``; ``J_t | foi ~ Poisson(foi)`` with
  ``foi = R0 * sum_{s>=0} tost_s * Y_{t-s}``.
* Each of the ``J_t`` infections draws an incubation period ``a >= 1`` from
  ``inc`` and contributes an onset on day ``t + a``.

**Extinction** is pipeline-aware: a trajectory is extinct once no
infected-but-not-yet-symptomatic individuals remain (``pending == 0``) *and* no
onsets have occurred within the TOST support of recent days (the last
``tost_max = len(tost) - 1`` days). A run of zero onsets alone does not suffice,
since onsets may already be scheduled from earlier infections.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sse_ssi_pmo.simulation import _rejection_loop, _validate_histories_for_sim

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_inputs_delay(
    R0: float,
    k: float,
    tost: NDArray[np.float64],
    inc: NDArray[np.float64],
    threshold: int,
    t_max: int,
) -> None:
    if R0 <= 0.0 or k <= 0.0:
        raise ValueError("R0 and k must be positive")
    if tost.ndim != 1 or tost.size == 0:
        raise ValueError("tost must be a non-empty 1-D array")
    if inc.ndim != 1 or inc.size == 0:
        raise ValueError("inc must be a non-empty 1-D array")
    if (tost < 0.0).any() or (inc < 0.0).any():
        raise ValueError("tost and inc must be non-negative")
    if not (inc.sum() > 0.0) or not (tost.sum() > 0.0):
        raise ValueError("tost and inc must each have positive total mass")
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")


def _inc_sampler(inc: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return ``(cdf, support)`` for inverse-CDF sampling of the incubation period.

    ``support[i] = i + 1`` (incubation indexed from 1); ``cdf`` is the
    normalised cumulative of ``inc``.
    """
    p = inc / inc.sum()
    cdf = np.cumsum(p)
    support = np.arange(1, inc.size + 1, dtype=np.int64)
    return cdf, support


def _sample_incubation(
    cdf: NDArray[np.float64],
    support: NDArray[np.int64],
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw ``size`` incubation periods (>= 1) by inverse-CDF sampling."""
    u = rng.random(size)
    idx = np.searchsorted(cdf, u, side="right")
    np.clip(idx, 0, support.size - 1, out=idx)
    return support[idx]


def _batch_delay(
    model: str,
    R0: float | NDArray[np.float64],
    k: float | NDArray[np.float64],
    tost: NDArray[np.float64],
    inc: NDArray[np.float64],
    *,
    init_incidence: int,
    n_sims: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator,
    match_histories: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]]:
    """Vectorised onset-anchored forward sim from day 0, seeded with ``init_incidence``.

    ``model`` is ``"sse"`` or ``"ssi"``. Returns
    ``(major, extinct, matched_history, onset)``. With ``match_histories``
    (shape ``(M, L)``, ``L >= 1``, every row starting with ``init_incidence``,
    rows distinct), sims are progressively pruned via an ``(n_sims, M)``
    ``alive`` mask matched against the onset cohorts on days ``1..L-1``, and
    abandoned when no history can still match; ``matched_history[i]`` is the
    matched row index or ``-1``. Without ``match_histories`` no matching is
    applied and ``matched_history`` is all-zero.

    ``R0`` and ``k`` may be scalars (broadcast) or per-sim ``(n_sims,)`` arrays.
    """
    if model not in ("sse", "ssi"):
        raise ValueError("model must be 'sse' or 'ssi'")
    if init_incidence < 1:
        raise ValueError("init_incidence must be at least 1 (D_0 >= 1)")
    if t_max < 1:
        raise ValueError("t_max must be at least 1")

    R0_arr = np.asarray(R0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    R0_arr = np.full(n_sims, float(R0_arr)) if R0_arr.ndim == 0 else R0_arr
    k_arr = np.full(n_sims, float(k_arr)) if k_arr.ndim == 0 else k_arr

    L_tost = tost.size
    tost_max = L_tost - 1
    inc_cdf, inc_support = _inc_sampler(inc)

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

    onset = np.zeros((n_sims, t_max), dtype=np.int64)
    onset[:, 0] = init_incidence
    if model == "ssi":
        Y = np.zeros((n_sims, t_max), dtype=np.float64)
        Y[:, 0] = rng.gamma(k_arr * init_incidence, 1.0 / k_arr)

    pending = np.zeros(n_sims, dtype=np.int64)  # infected but not yet symptomatic
    alive = np.ones((n_sims, M), dtype=bool)
    major = np.zeros(n_sims, dtype=bool)
    extinct = np.zeros(n_sims, dtype=bool)
    p_nb_full = k_arr / (k_arr + R0_arr)  # NB success prob (SSE), constant in foi
    tost_rev_cache: dict[int, NDArray[np.float64]] = {}

    for t in range(t_max):
        live_mask = alive.any(axis=1) & ~(major | extinct)
        if not live_mask.any():
            break
        idx = np.where(live_mask)[0]

        # Realise day-t onsets (fully accumulated, since incubation >= 1). The
        # index seed on day 0 was never counted as pending.
        if t >= 1:
            pending[idx] -= onset[idx, t]

        # Match the observed onset cohort D_t and prune (day 0 is the shared seed).
        if 1 <= t < match_len:
            assert match_histories is not None  # narrows type for the checker
            match_t = onset[idx, t][:, None] == match_histories[None, :, t]  # (len(idx), M)
            alive[idx] &= match_t
            keep = alive[idx].any(axis=1)
            idx = idx[keep]
            if idx.size == 0:
                continue

        new_major = onset[idx, t] >= threshold

        L_use = min(t + 1, L_tost)
        if L_use not in tost_rev_cache:
            tost_rev_cache[L_use] = tost[:L_use][::-1].copy()
        if model == "sse":
            recent = onset[idx, t + 1 - L_use : t + 1]
            foi = recent @ tost_rev_cache[L_use]
        else:  # ssi
            if t >= 1:
                nz_onset = onset[idx, t] > 0
                if nz_onset.any():
                    sel = idx[nz_onset]
                    Y[sel, t] = rng.gamma(k_arr[sel] * onset[sel, t], 1.0 / k_arr[sel])
            recent_Y = Y[idx, t + 1 - L_use : t + 1]
            foi = R0_arr[idx] * (recent_Y @ tost_rev_cache[L_use])

        new_inf = np.zeros(idx.size, dtype=np.int64)
        nz = foi > 0.0
        if nz.any():
            sel = idx[nz]
            if model == "sse":
                new_inf[nz] = rng.negative_binomial(k_arr[sel] * foi[nz], p_nb_full[sel])
            else:
                new_inf[nz] = rng.poisson(foi[nz])
        pending[idx] += new_inf

        # Scatter each infection forward to its onset day t + a, a ~ inc.
        total = int(new_inf.sum())
        if total > 0:
            sim_of_each = np.repeat(idx, new_inf)
            incs = _sample_incubation(inc_cdf, inc_support, total, rng)
            target = t + incs
            in_range = target < t_max
            if in_range.any():
                np.add.at(onset, (sim_of_each[in_range], target[in_range]), 1)

        # Extinction: no pending infections and no recent onsets that could still
        # transmit (onset on day d transmits on days d..d+tost_max).
        ext_start = max(0, t - tost_max + 1)
        window_zero = onset[idx, ext_start : t + 1].sum(axis=1) == 0
        new_extinct = (pending[idx] == 0) & window_zero & ~new_major

        major[idx[new_major]] = True
        extinct[idx[new_extinct]] = True

    matched_history = np.where(alive.any(axis=1), alive.argmax(axis=1), -1).astype(np.int64)
    return major, extinct, matched_history, onset


def _resolution_index_delay(onset: NDArray[np.int64], threshold: int, major: bool) -> int:
    """Index of the last onset entry to keep for a single trajectory.

    Major → first day the onset count reaches ``threshold``; otherwise the last
    non-zero onset day (trailing zeros are uninformative), or 0 if none.
    """
    if major:
        return int(np.argmax(onset >= threshold))
    nz = np.flatnonzero(onset)
    return int(nz[-1]) if nz.size else 0


# ---------------------------------------------------------------------------
# Real-time PMO via a bootstrap particle filter (private)
# ---------------------------------------------------------------------------


def _delay_step_inplace(
    model: str,
    onset: NDArray[np.int64],
    Y: NDArray[np.float64],
    pending: NDArray[np.int64],
    w: int,
    *,
    R0: float,
    k: float,
    tost: NDArray[np.float64],
    inc_cdf: NDArray[np.float64],
    inc_support: NDArray[np.int64],
    p_nb: float,
    t_max: int,
    rng: np.random.Generator,
    live_idx: NDArray[np.int64],
) -> None:
    """Process week ``w`` (one transmission step) for particles ``live_idx`` in place.

    Assumes ``onset[:, w]`` is already scheduled/known. Realises the week-``w``
    onsets against ``pending``, (SSI) draws ``Y[:, w]`` from the cohort size,
    computes the force of infection, draws new infections, and scatters them
    forward to future onset weeks. Same numerics as :func:`_batch_delay` but for
    a single week and scalar ``(R0, k)``.
    """
    if live_idx.size == 0:
        return
    if w >= 1:
        pending[live_idx] -= onset[live_idx, w]
    L_use = min(w + 1, tost.size)
    tost_rev = tost[:L_use][::-1]
    if model == "ssi":
        if w >= 1:
            nz_onset = onset[live_idx, w] > 0
            if nz_onset.any():
                sel = live_idx[nz_onset]
                Y[sel, w] = rng.gamma(k * onset[sel, w], 1.0 / k)
        foi = R0 * (Y[live_idx, w + 1 - L_use : w + 1] @ tost_rev)
    else:
        foi = onset[live_idx, w + 1 - L_use : w + 1] @ tost_rev
    new_inf = np.zeros(live_idx.size, dtype=np.int64)
    nz = foi > 0.0
    if nz.any():
        if model == "ssi":
            new_inf[nz] = rng.poisson(foi[nz])
        else:
            new_inf[nz] = rng.negative_binomial(k * foi[nz], p_nb)
    pending[live_idx] += new_inf
    total = int(new_inf.sum())
    if total > 0:
        sim_of_each = np.repeat(live_idx, new_inf)
        incs = _sample_incubation(inc_cdf, inc_support, total, rng)
        target = w + incs
        in_range = target < t_max
        if in_range.any():
            np.add.at(onset, (sim_of_each[in_range], target[in_range]), 1)


def _resolve_forward(
    model: str,
    onset: NDArray[np.int64],
    Y: NDArray[np.float64],
    pending: NDArray[np.int64],
    w_start: int,
    *,
    R0: float,
    k: float,
    tost: NDArray[np.float64],
    inc_cdf: NDArray[np.float64],
    inc_support: NDArray[np.int64],
    p_nb: float,
    threshold: int,
    t_max: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Free-forward each particle from week ``w_start`` until it resolves.

    Steps every particle forward (no history matching) until it reaches the
    major-outbreak threshold or goes extinct, mirroring :func:`_batch_delay`'s
    resolution. Mutates ``onset``/``Y``/``pending`` — pass copies. Returns
    ``(major, extinct)`` boolean arrays.
    """
    n = onset.shape[0]
    tost_max = tost.size - 1
    major = np.zeros(n, dtype=bool)
    extinct = np.zeros(n, dtype=bool)
    for w in range(w_start, t_max):
        live = ~(major | extinct)
        if not live.any():
            break
        new_major = live & (onset[:, w] >= threshold)
        major[new_major] = True
        live &= ~new_major
        if not live.any():
            break
        live_idx = np.where(live)[0]
        _delay_step_inplace(
            model,
            onset,
            Y,
            pending,
            w,
            R0=R0,
            k=k,
            tost=tost,
            inc_cdf=inc_cdf,
            inc_support=inc_support,
            p_nb=p_nb,
            t_max=t_max,
            rng=rng,
            live_idx=live_idx,
        )
        ext_start = max(0, w - tost_max + 1)
        window_zero = onset[live_idx, ext_start : w + 1].sum(axis=1) == 0
        new_extinct = (pending[live_idx] == 0) & window_zero
        extinct[live_idx[new_extinct]] = True
    return major, extinct


def _pmo_delay_realtime(
    model: str,
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    observed: ArrayLike,
    *,
    n_particles: int,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> dict[str, NDArray]:
    """Real-time PMO across the weeks of an observed onset history.

    A bootstrap particle filter over the latent state (the incubation pipeline
    and, for SSI, the infectivities). ``n_particles`` particles are advanced one
    week at a time; at each week the particles whose scheduled onset count
    matches the observation are kept and resampled (with replacement) back up to
    ``n_particles``. The week-``w`` PMO is the fraction of the resampled
    week-``w`` population that reaches a major outbreak when simulated forward
    freely — i.e. the probability of a major outbreak conditional on the onset
    history observed up to and including week ``w``.

    Far more efficient than per-week rejection sampling for long histories: the
    expensive early-history match is reused via resampling instead of being
    re-simulated from scratch every attempt.

    Returns a dict with ``"pmo"``, ``"n_matches"`` (particles matching before
    resampling), ``"n_distinct"`` (distinct particles after resampling), and
    ``"n_indet"`` (forward sims unresolved at ``t_max``) — each length ``L``.
    ``pmo`` is ``NaN`` from the week the filter collapses (no matching particle).
    """
    rng = np.random.default_rng() if rng is None else rng
    tost_arr = np.asarray(tost, dtype=np.float64)
    inc_arr = np.asarray(inc, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.int64)
    _check_inputs_delay(R0, k, tost_arr, inc_arr, threshold, t_max)
    if obs.ndim != 1 or obs.size == 0:
        raise ValueError("observed must be a non-empty 1-D onset history")
    if obs[0] < 1:
        raise ValueError("observed[0] (index onset) must be >= 1")
    if obs.size > t_max:
        raise ValueError("observed history longer than t_max")
    if n_particles < 1:
        raise ValueError("n_particles must be at least 1")

    L = obs.size
    N = n_particles
    inc_cdf, inc_support = _inc_sampler(inc_arr)
    p_nb = k / (k + R0)

    onset = np.zeros((N, t_max), dtype=np.int64)
    onset[:, 0] = int(obs[0])
    Y = np.zeros((N, t_max), dtype=np.float64)
    if model == "ssi":
        Y[:, 0] = rng.gamma(k * int(obs[0]), 1.0 / k, size=N)
    pending = np.zeros(N, dtype=np.int64)

    pmo = np.full(L, np.nan, dtype=np.float64)
    n_matches = np.zeros(L, dtype=np.int64)
    n_distinct = np.zeros(L, dtype=np.int64)
    n_indet = np.zeros(L, dtype=np.int64)
    all_idx = np.arange(N)

    for w in range(L):
        # onset[:, w] is known: the seed for w == 0, else scheduled by prior steps.
        if w == 0:
            n_matches[0] = N
            n_distinct[0] = N
        else:
            match_idx = np.where(onset[:, w] == obs[w])[0]
            n_matches[w] = match_idx.size
            if match_idx.size == 0:
                warnings.warn(
                    f"real-time PMO particle filter collapsed at week {w} "
                    f"(no particle matched onset={obs[w]}); PMO is NaN from week {w}. "
                    "Increase n_particles.",
                    stacklevel=2,
                )
                break
            chosen = rng.choice(match_idx, size=N, replace=True)
            onset = onset[chosen]
            Y = Y[chosen]
            pending = pending[chosen]
            n_distinct[w] = int(np.unique(chosen).size)

        major, extinct = _resolve_forward(
            model,
            onset.copy(),
            Y.copy(),
            pending.copy(),
            w,
            R0=R0,
            k=k,
            tost=tost_arr,
            inc_cdf=inc_cdf,
            inc_support=inc_support,
            p_nb=p_nb,
            threshold=threshold,
            t_max=t_max,
            rng=rng,
        )
        resolved = major | extinct
        n_indet[w] = int((~resolved).sum())
        n_res = int(resolved.sum())
        pmo[w] = float(major.sum()) / n_res if n_res else np.nan

        # Advance the filter one week (schedules onset[:, w + 1]) for the next match.
        if w < L - 1:
            _delay_step_inplace(
                model,
                onset,
                Y,
                pending,
                w,
                R0=R0,
                k=k,
                tost=tost_arr,
                inc_cdf=inc_cdf,
                inc_support=inc_support,
                p_nb=p_nb,
                t_max=t_max,
                rng=rng,
                live_idx=all_idx,
            )

    if int(n_indet.sum()) > 0:
        warnings.warn(
            f"{int(n_indet.sum())} forward sims across weeks hit t_max={t_max} "
            "unresolved; raise t_max.",
            stacklevel=2,
        )
    return {"pmo": pmo, "n_matches": n_matches, "n_distinct": n_distinct, "n_indet": n_indet}


# ---------------------------------------------------------------------------
# Single-trajectory simulators (public)
# ---------------------------------------------------------------------------


def simulate_sse_delay(
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    *,
    init_incidence: int = 1,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Simulate one onset-anchored SSE trajectory; return the onset-incidence vector.

    Starts from ``init_incidence`` index onsets on day 0 and runs forward until
    a single-day onset count reaches ``threshold`` (major outbreak) or the
    trajectory goes extinct (no pending infections and no recent onsets),
    otherwise to length ``t_max``. ``tost`` is indexed from lag 0, ``inc`` from
    lag 1 (see module docstring).
    """
    rng = np.random.default_rng() if rng is None else rng
    tost_arr = np.asarray(tost, dtype=np.float64)
    inc_arr = np.asarray(inc, dtype=np.float64)
    _check_inputs_delay(R0, k, tost_arr, inc_arr, threshold, t_max)
    if init_incidence < 1:
        raise ValueError("init_incidence must be at least 1")

    major, _extinct, _matched, onset = _batch_delay(
        "sse",
        R0,
        k,
        tost_arr,
        inc_arr,
        init_incidence=init_incidence,
        n_sims=1,
        threshold=threshold,
        t_max=t_max,
        rng=rng,
    )
    end = _resolution_index_delay(onset[0], threshold, bool(major[0]))
    return onset[0, : end + 1].copy()


def simulate_ssi_delay(
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    *,
    init_incidence: int = 1,
    threshold: int,
    t_max: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Simulate one onset-anchored SSI trajectory; return the onset-incidence vector.

    As :func:`simulate_sse_delay` but with SSI dynamics: each onset cohort's
    aggregate infectivity is ``Gamma(shape=k*D_t, rate=k)`` and new infections
    are ``Poisson(R0 * sum_s tost_s Y_{t-s})``.
    """
    rng = np.random.default_rng() if rng is None else rng
    tost_arr = np.asarray(tost, dtype=np.float64)
    inc_arr = np.asarray(inc, dtype=np.float64)
    _check_inputs_delay(R0, k, tost_arr, inc_arr, threshold, t_max)
    if init_incidence < 1:
        raise ValueError("init_incidence must be at least 1")

    major, _extinct, _matched, onset = _batch_delay(
        "ssi",
        R0,
        k,
        tost_arr,
        inc_arr,
        init_incidence=init_incidence,
        n_sims=1,
        threshold=threshold,
        t_max=t_max,
        rng=rng,
    )
    end = _resolution_index_delay(onset[0], threshold, bool(major[0]))
    return onset[0, : end + 1].copy()


# ---------------------------------------------------------------------------
# PMO-via-rejection-sampling (private)
# ---------------------------------------------------------------------------


def _pmo_delay_sim_multi(
    model: str,
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
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
    """Per-history onset-anchored PMO estimates via shared rejection sampling.

    Simulates onset trajectories seeded with the shared ``D_0 = histories[0, 0]``
    and routes each accepted trajectory to the unique history it matches,
    exactly like :func:`sse_ssi_pmo.simulation._pmo_ssi_sim_multi` but on the
    onset process. ``model`` is ``"sse"`` or ``"ssi"``. Continues until every
    history has ``n_sims`` resolved matches (or ``max_attempts`` trajectories
    tried). Returns ``(M,)`` PMO estimates; ``NaN`` where unresolved.
    """
    rng = np.random.default_rng() if rng is None else rng
    tost_arr = np.asarray(tost, dtype=np.float64)
    inc_arr = np.asarray(inc, dtype=np.float64)
    hist_arr = np.asarray(histories, dtype=np.int64)
    _check_inputs_delay(R0, k, tost_arr, inc_arr, threshold, t_max)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_attempts is None:
        max_attempts = 200 * n_sims
    label = f"_pmo_{model}_delay_sim_multi"
    M, _L, I_0 = _validate_histories_for_sim(hist_arr, t_max=t_max, label=label)

    n_major = np.zeros(M, dtype=np.int64)
    n_extinct = np.zeros(M, dtype=np.int64)
    n_indet = np.zeros(M, dtype=np.int64)

    def run_batch(b: int) -> None:
        major, extinct, matched, _ = _batch_delay(
            model,
            R0,
            k,
            tost_arr,
            inc_arr,
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
        desc=f"{model.upper()}-delay matching sims",
    )

    n_indet_total = int(n_indet.sum())
    if n_indet_total:
        warnings.warn(
            f"{n_indet_total} matching {model.upper()}-delay sims hit t_max={t_max} "
            "unresolved; raise t_max.",
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


def _pmo_sse_delay_sim(
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    history: ArrayLike,
    **kwargs,
) -> float:
    """SSE-delay PMO estimate for a single onset history (wraps the multi version)."""
    hist_2d = np.atleast_2d(np.asarray(history, dtype=np.int64))
    out = _pmo_delay_sim_multi("sse", R0, k, tost, inc, hist_2d, **kwargs)
    return float(out[0])


def _pmo_ssi_delay_sim(
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    history: ArrayLike,
    **kwargs,
) -> float:
    """SSI-delay PMO estimate for a single onset history (wraps the multi version)."""
    hist_2d = np.atleast_2d(np.asarray(history, dtype=np.int64))
    out = _pmo_delay_sim_multi("ssi", R0, k, tost, inc, hist_2d, **kwargs)
    return float(out[0])


def _pmo_sse_delay_sim_multi(
    R0: float, k: float, tost: ArrayLike, inc: ArrayLike, histories: ArrayLike, **kwargs
) -> NDArray[np.float64]:
    """SSE-delay per-history PMO estimates (wraps :func:`_pmo_delay_sim_multi`)."""
    return _pmo_delay_sim_multi("sse", R0, k, tost, inc, histories, **kwargs)


def _pmo_ssi_delay_sim_multi(
    R0: float, k: float, tost: ArrayLike, inc: ArrayLike, histories: ArrayLike, **kwargs
) -> NDArray[np.float64]:
    """SSI-delay per-history PMO estimates (wraps :func:`_pmo_delay_sim_multi`)."""
    return _pmo_delay_sim_multi("ssi", R0, k, tost, inc, histories, **kwargs)


__all__ = [
    "simulate_sse_delay",
    "simulate_ssi_delay",
]
