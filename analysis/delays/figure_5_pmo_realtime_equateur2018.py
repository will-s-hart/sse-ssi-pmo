"""Figure 5: real-time onset-anchored PMO over the 2018 Equateur EVD outbreak.

Two stacked panels sharing the week axis (see figure_14 for the layout): top is
the real-time PMO of the SSE/SSI models and their Bayesian model average with
the weekly epi curve on a twin right axis; bottom is the posterior model
probabilities. The week the response team arrived (8 May 2018) is marked with a
vertical line.

Loads pre-computed results from results/fig5_pmo_realtime_equateur2018.csv (run
results_15_pmo_realtime_equateur2018.py first).
"""

from __future__ import annotations

import pandas as pd

from analysis._shared.defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    SIM_THRESHOLD,
)
from analysis._shared.plotting import realtime_two_panel, save, set_style
from analysis.delays.defaults import (
    FIG5_PRIOR_SSE,
    FIG5_RESPONSE_WEEK,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    set_style()
    df = pd.read_csv(RESULTS_DIR / "fig5_pmo_realtime_equateur2018.csv")

    xtick_labels = [f"{w}\n{d}" for w, d in zip(df["week"], df["week_start"], strict=True)]
    title = (
        rf"2018 Equateur EVD outbreak: real-time PMO ($R_0 = {R0}$, $k = {K}$, "
        rf"threshold: peak weekly $\geq {SIM_THRESHOLD}$)"
    )
    fig = realtime_two_panel(
        df,
        title=title,
        xtick_labels=xtick_labels,
        prior_sse=FIG5_PRIOR_SSE,
        highlight_week=FIG5_RESPONSE_WEEK,
        highlight_label="response (8 May)",
        pmo_legend_loc="lower right",
    )
    save(fig, "fig5_pmo_realtime_equateur2018", OUT_DIR)


if __name__ == "__main__":
    main()
