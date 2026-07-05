"""Shared helpers for the real-time (particle-filter) onset-anchored figures.

Both the simulated-outbreak (fig 13) and the real EVD outbreak (fig 14) results
scripts run the SSE and SSI real-time particle filters over an observed weekly
onset history and Bayesian-average them; this module holds that shared logic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sse_ssi_pmo import pmo_sse_delay_realtime, pmo_ssi_delay_realtime


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
