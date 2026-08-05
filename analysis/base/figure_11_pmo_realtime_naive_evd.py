"""Figure 11: naive real-time PMO over the 2018 Equateur EVD outbreak.

Case dates are taken as infection times; the real-time probability of a major
outbreak and posterior SSE/SSI model probabilities are shown as estimated by a
bootstrap particle filter (lines) and by the established analytic / MCMC
machinery (markers). Agreement validates the particle filter.
"""

from __future__ import annotations

import pandas as pd

from analysis._shared.plotting import realtime_two_panel_compare, save, set_style
from analysis.base.defaults import (
    EVD2018_RESPONSE_WEEK,
    FIG11_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
)


def main() -> None:
    set_style()
    df = pd.read_csv(RESULTS_DIR / "fig11_pmo_realtime_naive_evd.csv")
    fig = realtime_two_panel_compare(
        df,
        title="2018 Equateur outbreak: real-time PMO (case dates as infection times)",
        xtick_labels=df["week_start"].tolist(),
        prior_sse=FIG11_PRIOR_SSE,
        highlight_week=EVD2018_RESPONSE_WEEK,
        highlight_label="response (8 May)",
    )
    save(fig, "fig11_pmo_realtime_naive_evd", OUT_DIR)


if __name__ == "__main__":
    main()
