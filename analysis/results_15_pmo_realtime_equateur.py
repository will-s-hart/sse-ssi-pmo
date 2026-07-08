"""Compute and save results for Figure 15: real-time PMO, 2020 Equateur outbreak.

Case line list for the 2020 Equateur (DRC) EVD outbreak (committed CSV of
reported dates in analysis/data/) is binned into calendar (Mon-Sun) weeks. For
each week we estimate, via a bootstrap particle filter, the probability of a
major outbreak and the posterior SSE/SSI model probabilities given the history
observed up to that week, plus their Bayesian model average (prior 0.5 each).

NB the input records are *reporting* dates, used here as a proxy for the
symptom-onset timeline the model assumes (a reporting delay is not modelled).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from _realtime import bin_calendar_weeks, realtime_ensemble
from analysis_defaults import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_R0,
    DELAY_INC_MAX,
    DELAY_INC_MEAN,
    DELAY_INC_SD,
    DELAY_TOST_MAX,
    DELAY_TOST_MEAN,
    DELAY_TOST_SD,
    FIG15_DATA_FILE,
    FIG15_N_PARTICLES,
    FIG15_PRIOR_SSE,
    RESULTS_DIR,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
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

    raw = pd.read_csv(DATA_DIR / FIG15_DATA_FILE)
    dates = [date.fromisoformat(s) for s in raw["date_reported"]]
    counts, week_starts = bin_calendar_weeks(dates)
    print(f"weekly reported counts: {counts.tolist()} (total {int(counts.sum())} cases)")

    cols = realtime_ensemble(
        counts,
        R0=R0,
        k=K,
        tost=tost,
        inc=inc,
        n_particles=FIG15_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        prior_sse=FIG15_PRIOR_SSE,
        rng=np.random.default_rng(SIM_SEED),
    )
    df = pd.DataFrame({**cols, "week_start": week_starts})
    df = df[["week", "week_start", *[c for c in cols if c != "week"]]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig15_pmo_realtime_equateur.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
