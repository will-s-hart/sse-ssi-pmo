"""Figure 1: PMO vs r (weeks without cases since the index case).

Analytic curves (one per model) overlaid with Monte-Carlo simulation points at
each integer r.

Loads pre-computed results from results/fig1_pmo_vs_r.csv (run
results_1_pmo_vs_r.py first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from _plotting import (
    SIM_LABEL_SUFFIX,
    SIM_MARKERSIZE,
    SSE_COLOUR,
    SSE_LABEL,
    SSE_LINESTYLE,
    SSE_SIM_MARKER,
    SSI_COLOUR,
    SSI_LABEL,
    SSI_LINESTYLE,
    SSI_SIM_MARKER,
    save,
    set_style,
)
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    FIG1_R_MAX,
    OUT_DIR,
    RESULTS_DIR,
    SIM_THRESHOLD,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
R_MAX: int = FIG1_R_MAX


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig1_pmo_vs_r.csv")

    fig, ax = plt.subplots()
    ax.plot(df["r"], df["pmo_sse"], color=SSE_COLOUR, label=SSE_LABEL, linestyle=SSE_LINESTYLE)
    ax.plot(df["r"], df["pmo_ssi"], color=SSI_COLOUR, label=SSI_LABEL, linestyle=SSI_LINESTYLE)
    ax.scatter(
        df["r"],
        df["sse_sim"],
        color=SSE_COLOUR,
        marker=SSE_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        df["r"],
        df["ssi_sim"],
        color=SSI_COLOUR,
        marker=SSI_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.set_xlabel("Weeks without cases since index case ($r$)")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(rf"$R_0 = {R0}$, $k = {K}$ (sim threshold: peak weekly $\geq {SIM_THRESHOLD}$)")
    ax.set_xlim(0, R_MAX)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig1_pmo_vs_r", OUT_DIR)


if __name__ == "__main__":
    main()
