"""Figure 2: PMO vs R_0 after R_WEEKS weeks without cases.

Analytic curves (one per model) overlaid with Monte-Carlo simulation points at
a subset of R_0 values.

Loads pre-computed results from results/fig2_pmo_vs_R0.csv (run
results_2_pmo_vs_R0.py first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analysis._shared.defaults import (
    DEFAULT_K,
    DEFAULT_R_WEEKS,
    SIM_THRESHOLD,
)
from analysis._shared.plotting import (
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
from analysis.base.defaults import (
    FIG2_R0_MAX,
    FIG2_R0_MIN,
    OUT_DIR,
    RESULTS_DIR,
)

K: float = DEFAULT_K
R_WEEKS: int = DEFAULT_R_WEEKS
R0_MIN: float = FIG2_R0_MIN
R0_MAX: float = FIG2_R0_MAX


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig2_pmo_vs_R0.csv")
    df_sim = df.dropna(subset=["sse_sim", "ssi_sim"])
    F_r = float(df["F_r"].iloc[0])

    fig, ax = plt.subplots()
    ax.plot(df["R0"], df["pmo_sse"], color=SSE_COLOUR, label=SSE_LABEL, linestyle=SSE_LINESTYLE)
    ax.plot(df["R0"], df["pmo_ssi"], color=SSI_COLOUR, label=SSI_LABEL, linestyle=SSI_LINESTYLE)
    ax.scatter(
        df_sim["R0"],
        df_sim["sse_sim"],
        color=SSE_COLOUR,
        marker=SSE_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        df_sim["R0"],
        df_sim["ssi_sim"],
        color=SSI_COLOUR,
        marker=SSI_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.set_xlabel(r"Reproduction number $R_0$")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(
        rf"After {R_WEEKS} weeks without cases ($k = {K}$, $F_r = {F_r:.3f}$, "
        rf"sim threshold $\geq {SIM_THRESHOLD}$)"
    )
    ax.set_xlim(R0_MIN, R0_MAX)
    ax.set_ylim(0, 1)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig2_pmo_vs_R0", OUT_DIR)


if __name__ == "__main__":
    main()
