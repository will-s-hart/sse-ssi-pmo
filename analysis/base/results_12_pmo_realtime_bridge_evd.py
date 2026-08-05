"""Compute and save results for Figure 12: bridge real-time PMO, 2018 Equateur.

The 2018 Equateur (DRC) EVD weekly counts are treated as *symptom-onset*
incidence. Transmission still follows the generation-time renewal of the base
paper (weights ``w``), but infections are observed only through onsets obtained
by an independent incubation delay (``inc``, allowed from lag 0). A bootstrap
particle filter estimates the real-time probability of a major outbreak and the
posterior SSE/SSI model probabilities. This "bridge" figure accounts for the
infection-to-onset delay while keeping the base paper's renewal, motivating the
fully onset-anchored models of the companion (delays) paper.
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
    DELAY_INC_MAX,
    DELAY_INC_MEAN,
    DELAY_INC_SD,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from analysis._shared.realtime import bin_calendar_weeks, realtime_ensemble_bridge
from analysis.base.defaults import (
    EVD2018_DATA_FILE,
    FIG12_N_PARTICLES,
    FIG12_PRIOR_SSE,
    FIG12_SEED_LEAD,
    RESULTS_DIR,
)
from sse_ssi_pmo import discretise_gamma

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    w = discretise_gamma(mean=DEFAULT_SI_MEAN, sd=DEFAULT_SI_SD, max_val=DEFAULT_SI_MAX)
    inc = discretise_gamma(
        mean=DELAY_INC_MEAN, sd=DELAY_INC_SD, max_val=DELAY_INC_MAX, allow_zero=True
    )

    raw = pd.read_csv(DATA_DIR / EVD2018_DATA_FILE)
    dates = [
        date.fromisoformat(d)
        for d, c in zip(raw["date"], raw["incidence"], strict=True)
        for _ in range(c)
    ]
    counts, week_starts = bin_calendar_weeks(dates)
    print(f"weekly counts (onsets): {counts.tolist()} (total {int(counts.sum())})")

    cols = realtime_ensemble_bridge(
        counts,
        R0=R0,
        k=K,
        w=w,
        inc=inc,
        seed_lead=FIG12_SEED_LEAD,
        n_particles=FIG12_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        prior_sse=FIG12_PRIOR_SSE,
        rng=np.random.default_rng(SIM_SEED),
    )
    df = pd.DataFrame({**cols, "week_start": week_starts})
    df = df[["week", "week_start", *[c for c in cols if c != "week"]]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig12_pmo_realtime_bridge_evd.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
