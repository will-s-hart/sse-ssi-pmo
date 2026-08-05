"""Compute and save results for Figure 11: naive real-time PMO, 2018 Equateur.

The daily disease-incidence series for the 2018 Equateur (DRC) EVD outbreak is
binned into calendar (Mon-Sun) weeks and, under the *naive* assumption that the
case dates are infection times, the weekly counts are treated as infection
incidence. For each week we estimate the probability of a major outbreak (and the
posterior SSE/SSI model probabilities) conditional on the history so far, two
ways:

* a bootstrap particle filter (``pf_*`` columns), and
* the established analytic / MCMC machinery (``mcmc_*`` columns) --
  :func:`pmo_uncertain` with ``method="analytic"`` where the truncated history
  has a closed form and ``method="mcmc"`` otherwise.

Overlaying the two validates the particle filter against the established method
before it is used, in the companion figures, for models with no closed form.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from analysis._shared.defaults import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_R0,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from analysis._shared.realtime import bin_calendar_weeks, realtime_ensemble_infection
from analysis.base.defaults import (
    EVD2018_DATA_FILE,
    FIG11_MCMC_CHAINS,
    FIG11_MCMC_DRAWS,
    FIG11_MCMC_TUNE,
    FIG11_N_PARTICLES,
    FIG11_PRIOR_SSE,
    RESULTS_DIR,
)
from sse_ssi_pmo import discretise_gamma, pmo_uncertain

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def _mcmc_arm(w: np.ndarray, counts: np.ndarray) -> dict[str, list[float]]:
    """Per-week model-averaged PMO via analytic (special histories) or MCMC."""
    cols: dict[str, list[float]] = {
        k: [] for k in ("sse_pmo", "ssi_pmo", "ensemble_pmo", "post_sse")
    }
    for wk in range(counts.size):
        hist = counts[: wk + 1]
        try:
            res = pmo_uncertain(
                R0=R0, k=K, w=w, history=hist, method="analytic", prior_sse=FIG11_PRIOR_SSE
            )
        except (ValueError, NotImplementedError):
            res = pmo_uncertain(
                R0=R0,
                k=K,
                w=w,
                history=hist,
                method="mcmc",
                prior_sse=FIG11_PRIOR_SSE,
                draws=FIG11_MCMC_DRAWS,
                tune=FIG11_MCMC_TUNE,
                chains=FIG11_MCMC_CHAINS,
                target_accept=0.99,
                progressbar=False,
            )
        cols["sse_pmo"].append(float(res.pmo_sse))
        cols["ssi_pmo"].append(float(res.pmo_ssi))
        cols["ensemble_pmo"].append(float(res.pmo))
        cols["post_sse"].append(float(res.posterior_sse))
    return cols


def main() -> None:
    w = discretise_gamma(mean=DEFAULT_SI_MEAN, sd=DEFAULT_SI_SD, max_val=DEFAULT_SI_MAX)

    raw = pd.read_csv(DATA_DIR / EVD2018_DATA_FILE)
    dates = [
        date.fromisoformat(d)
        for d, c in zip(raw["date"], raw["incidence"], strict=True)
        for _ in range(c)
    ]
    counts, week_starts = bin_calendar_weeks(dates)
    print(f"weekly counts (naive infections): {counts.tolist()} (total {int(counts.sum())})")

    pf = realtime_ensemble_infection(
        counts,
        R0=R0,
        k=K,
        w=w,
        n_particles=FIG11_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        prior_sse=FIG11_PRIOR_SSE,
        rng=np.random.default_rng(SIM_SEED),
    )
    mcmc = _mcmc_arm(w, counts)

    df = pd.DataFrame(
        {
            "week": pf["week"],
            "week_start": week_starts,
            "cases": pf["cases"],
            "pf_sse_pmo": pf["sse_pmo"],
            "pf_ssi_pmo": pf["ssi_pmo"],
            "pf_ensemble_pmo": pf["ensemble_pmo"],
            "pf_post_sse": pf["post_sse"],
            "pf_post_ssi": pf["post_ssi"],
            "mcmc_sse_pmo": mcmc["sse_pmo"],
            "mcmc_ssi_pmo": mcmc["ssi_pmo"],
            "mcmc_ensemble_pmo": mcmc["ensemble_pmo"],
            "mcmc_post_sse": mcmc["post_sse"],
            "mcmc_post_ssi": [1.0 - p for p in mcmc["post_sse"]],
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig11_pmo_realtime_naive_evd.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
