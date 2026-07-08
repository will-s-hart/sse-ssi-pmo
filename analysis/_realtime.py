"""Shared helpers for the real-time (particle-filter) onset-anchored figures.

Both the simulated-outbreak (fig 13) and the real EVD outbreak (fig 14) results
scripts run the SSE and SSI real-time particle filters over an observed weekly
onset history and Bayesian-average them; this module holds that shared logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sse_ssi_pmo import pmo_sse_delay_realtime, pmo_ssi_delay_realtime


def bin_calendar_weeks(dates: Iterable[date]) -> tuple[NDArray[np.int64], list[str]]:
    """Bin dates into calendar (Mon-Sun) weeks from the first date's Monday.

    Returns ``(counts, week_starts)`` where ``counts[w]`` is the number of dates
    in week ``w`` (week 0 = the calendar week of the earliest date) and
    ``week_starts[w]`` is that week's Monday formatted ``dd/mm``.
    """
    sorted_dates = sorted(dates)
    mondays = [d - timedelta(days=d.weekday()) for d in sorted_dates]
    first_monday = mondays[0]
    week_idx = [(m - first_monday).days // 7 for m in mondays]
    n_weeks = week_idx[-1] + 1
    counts = np.bincount(week_idx, minlength=n_weeks).astype(np.int64)
    week_starts = [(first_monday + timedelta(weeks=w)).strftime("%d/%m") for w in range(n_weeks)]
    return counts, week_starts


def model_average(
    log_ev_sse: NDArray[np.float64],
    log_ev_ssi: NDArray[np.float64],
    pmo_sse: NDArray[np.float64],
    pmo_ssi: NDArray[np.float64],
    prior_sse: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Posterior SSE probability and model-averaged PMO from per-model log evidence.

    ``posterior_sse = prior_sse L_sse / (prior_sse L_sse + prior_ssi L_ssi)``,
    computed in log space; the ensemble PMO is the posterior-weighted mean of the
    per-model PMOs.
    """
    log_w_sse = np.log(prior_sse) + log_ev_sse
    log_w_ssi = np.log(1.0 - prior_sse) + log_ev_ssi
    m = np.maximum(log_w_sse, log_w_ssi)
    w_sse = np.exp(log_w_sse - m)
    w_ssi = np.exp(log_w_ssi - m)
    post_sse = w_sse / (w_sse + w_ssi)
    ensemble_pmo = post_sse * pmo_sse + (1.0 - post_sse) * pmo_ssi
    return post_sse, ensemble_pmo


def realtime_ensemble(
    counts: ArrayLike,
    *,
    R0: float,
    k: float,
    tost: ArrayLike,
    inc: ArrayLike,
    n_particles: int,
    threshold: int,
    t_max: int,
    prior_sse: float,
    rng: np.random.Generator,
) -> dict[str, NDArray]:
    """Run the SSE and SSI real-time particle filters and Bayesian-average them.

    Returns a dict of per-week columns (``week``, ``cases``, per-model and
    ensemble PMO, posterior model probabilities, and particle-diversity
    diagnostics) ready to build a DataFrame.
    """
    counts_arr = np.asarray(counts, dtype=np.int64)
    sse = pmo_sse_delay_realtime(
        R0=R0,
        k=k,
        tost=tost,
        inc=inc,
        history=counts_arr,
        n_particles=n_particles,
        threshold=threshold,
        t_max=t_max,
        rng=rng,
    )
    ssi = pmo_ssi_delay_realtime(
        R0=R0,
        k=k,
        tost=tost,
        inc=inc,
        history=counts_arr,
        n_particles=n_particles,
        threshold=threshold,
        t_max=t_max,
        rng=rng,
    )
    post_sse, ensemble_pmo = model_average(
        sse.log_evidence, ssi.log_evidence, sse.pmo, ssi.pmo, prior_sse
    )
    return {
        "week": np.arange(counts_arr.size),
        "cases": counts_arr,
        "sse_delay_pmo": sse.pmo,
        "ssi_delay_pmo": ssi.pmo,
        "ensemble_pmo": ensemble_pmo,
        "post_sse": post_sse,
        "post_ssi": 1.0 - post_sse,
        "sse_n_distinct": sse.n_distinct,
        "ssi_n_distinct": ssi.n_distinct,
    }
