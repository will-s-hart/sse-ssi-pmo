"""Compute and save results for Figure 4: real-time PMO, 2017 Likati EVD outbreak.

Symptom-onset dates for the 2017 Likati (DRC) EVD outbreak are binned into
calendar (Mon-Sun) weeks. For each week we estimate, via a bootstrap particle
filter, the probability of a major outbreak and the posterior SSE/SSI model
probabilities given the onset history observed up to that week, plus their
Bayesian model average (prior model probabilities 0.5 each).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from analysis._shared.defaults import (
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
    FIG4_N_PARTICLES,
    FIG4_ONSET_DATES,
    FIG4_PRIOR_SSE,
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

    dates = [datetime.strptime(s, "%d/%m/%Y").date() for s in FIG4_ONSET_DATES]
    counts, week_starts = bin_calendar_weeks(dates)
    print(f"weekly onset counts: {counts.tolist()} (total {int(counts.sum())} cases)")

    cols = realtime_ensemble(
        counts,
        R0=R0,
        k=K,
        tost=tost,
        inc=inc,
        n_particles=FIG4_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        prior_sse=FIG4_PRIOR_SSE,
        rng=np.random.default_rng(SIM_SEED),
    )
    df = pd.DataFrame({**cols, "week_start": week_starts})
    df = df[["week", "week_start", *[c for c in cols if c != "week"]]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig4_pmo_realtime_likati.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
