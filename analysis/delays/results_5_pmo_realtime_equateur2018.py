"""Compute and save results for Figure 5: real-time PMO, 2018 Equateur outbreak.

Daily disease-incidence series for the 2018 Equateur (DRC) EVD outbreak
(committed CSV, one row per day from the first case on 5 Apr 2018) is converted
to calendar (Mon-Sun) weeks. For each week we estimate, via a bootstrap particle
filter, the probability of a major outbreak and the posterior SSE/SSI model
probabilities given the history observed up to that week, plus their Bayesian
model average (prior 0.5 each).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from analysis._shared.defaults import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_R0,
    DELAY_INC_MAX,
    DELAY_INC_MEAN,
    DELAY_INC_SD,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from analysis._shared.realtime import bin_calendar_weeks, realtime_ensemble
from analysis.delays.defaults import (
    DELAY_TOST_MAX,
    DELAY_TOST_MEAN,
    DELAY_TOST_SD,
    FIG5_DATA_FILE,
    FIG5_N_PARTICLES,
    FIG5_PRIOR_SSE,
    RESULTS_DIR,
)
from sse_ssi_pmo import discretise_gamma

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    tost = discretise_gamma(
        mean=DELAY_TOST_MEAN, sd=DELAY_TOST_SD, max_val=DELAY_TOST_MAX, allow_zero=True
    )
    inc = discretise_gamma(
        mean=DELAY_INC_MEAN, sd=DELAY_INC_SD, max_val=DELAY_INC_MAX, allow_zero=False
    )

    raw = pd.read_csv(DATA_DIR / FIG5_DATA_FILE)
    # Expand the daily incidence into one date per case, then bin by calendar week.
    dates = [
        date.fromisoformat(d)
        for d, c in zip(raw["date"], raw["incidence"], strict=True)
        for _ in range(c)
    ]
    counts, week_starts = bin_calendar_weeks(dates)
    print(f"weekly counts: {counts.tolist()} (total {int(counts.sum())} cases)")

    cols = realtime_ensemble(
        counts,
        R0=R0,
        k=K,
        tost=tost,
        inc=inc,
        n_particles=FIG5_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        prior_sse=FIG5_PRIOR_SSE,
        rng=np.random.default_rng(SIM_SEED),
    )
    df = pd.DataFrame({**cols, "week_start": week_starts})
    df = df[["week", "week_start", *[c for c in cols if c != "week"]]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig5_pmo_realtime_equateur2018.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
