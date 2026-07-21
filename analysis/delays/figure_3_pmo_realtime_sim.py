"""Figure 3: real-time onset-anchored PMO over a simulated SSI outbreak.

Companion to Figure 4 (the real Likati outbreak) on a weekly onset history
simulated from the *SSI* model (the true model). Two stacked panels sharing the
week axis: the top shows the real-time PMO for the SSE and SSI models and their
Bayesian model average (prior 0.5 each); the bottom shows the posterior model
probabilities, which should drift toward SSI as the weeks accumulate. Weekly
onset counts are drawn as bars on a secondary (right) axis in both panels.

Loads pre-computed results from results/fig3_pmo_realtime_sim.csv (run
results_13_pmo_realtime_sim.py first).
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
    FIG3_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    set_style()
    df = pd.read_csv(RESULTS_DIR / "fig3_pmo_realtime_sim.csv")

    xtick_labels = [str(w) for w in df["week"]]
    title = (
        rf"Simulated SSI outbreak (true model: SSI): real-time PMO "
        rf"($R_0 = {R0}$, $k = {K}$, threshold: peak weekly onsets $\geq {SIM_THRESHOLD}$)"
    )
    fig = realtime_two_panel(df, title=title, xtick_labels=xtick_labels, prior_sse=FIG3_PRIOR_SSE)
    save(fig, "fig3_pmo_realtime_sim", OUT_DIR)


if __name__ == "__main__":
    main()
