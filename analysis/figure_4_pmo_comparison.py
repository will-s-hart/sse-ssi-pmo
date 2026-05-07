"""Figure 4: PMO comparison across incidence histories.

Grouped bar chart showing SSE (analytic) and SSI (analytic when possible,
MCMC otherwise) PMO estimates as bars, with simulation x markers overlaid, for
a configurable list of incidence histories.  Confirms agreement between methods
across a variety of observed histories, including cases with non-zero incidence
after day 0 where SSI requires MCMC.

Loads pre-computed results from results/fig4_pmo_comparison.csv (run
results_4_pmo_comparison.py first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _plotting import (
    SIM_LABEL_SUFFIX,
    SIM_MARKERSIZE,
    SSE_COLOUR,
    SSE_LABEL,
    SSI_COLOUR,
    SSI_LABEL,
    save,
    set_style,
)
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig4_pmo_comparison.csv")
    n = len(df)
    bar_width = 0.35
    x = np.arange(n)

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2 * default_w, default_h))

    ax.bar(
        x - bar_width / 2,
        df["sse_analytic"],
        width=bar_width,
        color=SSE_COLOUR,
        alpha=0.85,
        label=SSE_LABEL,
    )
    ax.bar(
        x + bar_width / 2,
        df["ssi_best"],
        width=bar_width,
        color=SSI_COLOUR,
        alpha=0.85,
        label=SSI_LABEL + " (analytic / MCMC)",
    )

    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2
    ax.scatter(
        x - bar_width / 2,
        df["sse_sim"],
        color=SSE_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        x + bar_width / 2,
        df["ssi_sim"],
        color=SSI_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["history"])
    ax.set_xlabel("Incidence history")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(rf"$R_0 = {R0}$, $k = {K}$")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig4_pmo_comparison", OUT_DIR)


if __name__ == "__main__":
    main()
