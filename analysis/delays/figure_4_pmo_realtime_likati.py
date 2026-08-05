"""Figure 4: real-time onset-anchored PMO over the 2017 Likati EVD outbreak.

Two stacked panels sharing the week axis: the top shows the real-time PMO for
the onset-anchored SSE and SSI models and their Bayesian model average (prior
0.5 each); the bottom shows the posterior model probabilities over the weeks.
Both are read off the same bootstrap particle filter and condition on the onset
history observed up to and including each week. The weekly onset counts are
drawn as bars on a secondary (right) axis in both panels.

Loads pre-computed results from results/fig4_pmo_realtime_likati.csv (run
results_14_pmo_realtime_likati.py first).
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
    FIG4_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    set_style()
    df = pd.read_csv(RESULTS_DIR / "fig4_pmo_realtime_likati.csv")

    xtick_labels = [f"{w}\n{d}" for w, d in zip(df["week"], df["week_start"], strict=True)]
    title = (
        rf"2017 Likati EVD outbreak: real-time PMO ($R_0 = {R0}$, $k = {K}$, "
        rf"threshold: peak weekly onsets $\geq {SIM_THRESHOLD}$)"
    )
    fig = realtime_two_panel(df, title=title, xtick_labels=xtick_labels, prior_sse=FIG4_PRIOR_SSE)
    save(fig, "fig4_pmo_realtime_likati", OUT_DIR)


if __name__ == "__main__":
    main()
