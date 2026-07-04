"""Compute and save results for Figure 13: real-time onset-anchored PMO.

Symptom-onset dates for the 2017 Likati (DRC) EVD outbreak are binned into
calendar (Mon-Sun) weeks. For each week we estimate the probability of a major
outbreak given the onset history observed up to that week, under the
onset-anchored SSE and SSI models, via a bootstrap particle filter
(``pmo_*_delay_realtime``).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    DELAY_INC_MAX,
    DELAY_INC_MEAN,
    DELAY_INC_SD,
    DELAY_TOST_MAX,
    DELAY_TOST_MEAN,
    DELAY_TOST_SD,
    FIG13_N_PARTICLES,
    FIG13_ONSET_DATES,
    RESULTS_DIR,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)

from sse_ssi_pmo import (
    discretise_gamma,
    pmo_sse_delay_realtime,
    pmo_ssi_delay_realtime,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def weekly_onsets(date_strings: list[str]) -> tuple[np.ndarray, list[str]]:
    """Bin dd/mm/yyyy onset dates into calendar (Mon-Sun) weeks.

    Returns ``(counts, week_starts)`` where ``counts[w]`` is the number of
    onsets in week ``w`` (week 0 = the calendar week of the first onset) and
    ``week_starts[w]`` is that week's Monday as ``dd/mm``.
    """
    dates = sorted(datetime.strptime(s, "%d/%m/%Y").date() for s in date_strings)
    mondays = [d - timedelta(days=d.weekday()) for d in dates]  # Monday of each week
    first_monday = mondays[0]
    week_idx = [(m - first_monday).days // 7 for m in mondays]
    n_weeks = week_idx[-1] + 1
    counts = np.bincount(week_idx, minlength=n_weeks).astype(np.int64)
    week_starts = [(first_monday + timedelta(weeks=w)).strftime("%d/%m") for w in range(n_weeks)]
    return counts, week_starts


def main() -> None:
    tost = discretise_gamma(
        mean=DELAY_TOST_MEAN, sd=DELAY_TOST_SD, max_val=DELAY_TOST_MAX, allow_zero=True
    )
    inc = discretise_gamma(
        mean=DELAY_INC_MEAN, sd=DELAY_INC_SD, max_val=DELAY_INC_MAX, allow_zero=False
    )

    counts, week_starts = weekly_onsets(FIG13_ONSET_DATES)
    print(f"weekly onset counts: {counts.tolist()} (total {int(counts.sum())} cases)")

    rng = np.random.default_rng(SIM_SEED)
    sse = pmo_sse_delay_realtime(
        R0=R0,
        k=K,
        tost=tost,
        inc=inc,
        history=counts,
        n_particles=FIG13_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
    )
    ssi = pmo_ssi_delay_realtime(
        R0=R0,
        k=K,
        tost=tost,
        inc=inc,
        history=counts,
        n_particles=FIG13_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig13_pmo_realtime_delay.csv"
    df = pd.DataFrame(
        {
            "week": np.arange(counts.size),
            "week_start": week_starts,
            "cases": counts,
            "sse_delay_pmo": sse.pmo,
            "ssi_delay_pmo": ssi.pmo,
            "sse_n_distinct": sse.n_distinct,
            "ssi_n_distinct": ssi.n_distinct,
        }
    )
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
