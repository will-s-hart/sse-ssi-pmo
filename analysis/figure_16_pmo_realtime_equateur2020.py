"""Figure 16: real-time onset-anchored PMO over the 2020 Equateur EVD outbreak.

Two stacked panels sharing the week axis (see figure_14 for the layout): top is
the real-time PMO of the SSE/SSI models and their Bayesian model average with
the weekly epi curve on a twin right axis; bottom is the posterior model
probabilities. The week of the 1 Jun 2020 outbreak declaration (the response /
decision point) is marked with a vertical line. Input records are reporting
dates used as a proxy for symptom onset (a reporting delay is not modelled).

Loads pre-computed results from results/fig16_pmo_realtime_equateur2020.csv (run
results_16_pmo_realtime_equateur2020.py first).
"""

from __future__ import annotations

import pandas as pd
from _plotting import realtime_two_panel, save, set_style
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    FIG16_PRIOR_SSE,
    FIG16_RESPONSE_WEEK,
    OUT_DIR,
    RESULTS_DIR,
    SIM_THRESHOLD,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    set_style()
    df = pd.read_csv(RESULTS_DIR / "fig16_pmo_realtime_equateur2020.csv")

    # 19 weeks: show the date only every 3rd week so the tick labels don't collide.
    xtick_labels = [
        f"{w}\n{d}" if w % 3 == 0 else str(w)
        for w, d in zip(df["week"], df["week_start"], strict=True)
    ]
    title = (
        rf"2020 Equateur EVD outbreak (reported dates): real-time PMO "
        rf"($R_0 = {R0}$, $k = {K}$, threshold: peak weekly $\geq {SIM_THRESHOLD}$)"
    )
    fig = realtime_two_panel(
        df,
        title=title,
        xtick_labels=xtick_labels,
        prior_sse=FIG16_PRIOR_SSE,
        highlight_week=FIG16_RESPONSE_WEEK,
        highlight_label="declaration (~1 Jun)",
        pmo_legend_loc="lower right",
    )
    save(fig, "fig16_pmo_realtime_equateur2020", OUT_DIR)


if __name__ == "__main__":
    main()
