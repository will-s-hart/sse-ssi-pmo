"""Figure 12: bridge real-time PMO over the 2018 Equateur EVD outbreak.

Case dates are taken as symptom onsets; transmission follows the base paper's
generation-time renewal, observed through an independent incubation delay. Shows
the real-time SSE/SSI/model-average PMO and the posterior model probabilities.
"""

from __future__ import annotations

import pandas as pd

from analysis._shared.plotting import realtime_two_panel, save, set_style
from analysis.base.defaults import (
    EVD2018_RESPONSE_WEEK,
    FIG12_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
)


def main() -> None:
    set_style()
    df = pd.read_csv(RESULTS_DIR / "fig12_pmo_realtime_bridge_evd.csv")
    fig = realtime_two_panel(
        df,
        title="2018 Equateur outbreak: bridge real-time PMO (onsets, generation-time renewal)",
        xtick_labels=df["week_start"].tolist(),
        prior_sse=FIG12_PRIOR_SSE,
        sse_col="sse_pmo",
        ssi_col="ssi_pmo",
        highlight_week=EVD2018_RESPONSE_WEEK,
        highlight_label="response (8 May)",
        pmo_legend_loc="lower right",
    )
    save(fig, "fig12_pmo_realtime_bridge_evd", OUT_DIR)


if __name__ == "__main__":
    main()
