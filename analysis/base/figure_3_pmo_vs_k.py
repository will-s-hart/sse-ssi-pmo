"""Figure 3: PMO vs k (dispersion) after R_WEEKS weeks without cases.

Analytic curves (one per model) overlaid with Monte-Carlo simulation points at
a subset of k values.

Loads pre-computed results from results/fig3_pmo_vs_k.csv (run
results_3_pmo_vs_k.py first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analysis._shared.defaults import (
    DEFAULT_R0,
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
    FIG3_K_MAX,
    FIG3_K_MIN,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
R_WEEKS: int = DEFAULT_R_WEEKS
K_MIN: float = FIG3_K_MIN
K_MAX: float = FIG3_K_MAX


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig3_pmo_vs_k.csv")
    df_sim = df.dropna(subset=["sse_sim", "ssi_sim"])
    F_r = float(df["F_r"].iloc[0])

    fig, ax = plt.subplots()
    ax.plot(df["k"], df["pmo_sse"], color=SSE_COLOUR, label=SSE_LABEL, linestyle=SSE_LINESTYLE)
    ax.plot(df["k"], df["pmo_ssi"], color=SSI_COLOUR, label=SSI_LABEL, linestyle=SSI_LINESTYLE)
    ax.scatter(
        df_sim["k"],
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
        df_sim["k"],
        df_sim["ssi_sim"],
        color=SSI_COLOUR,
        marker=SSI_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Dispersion parameter $k$")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(
        rf"After {R_WEEKS} weeks without cases ($R_0 = {R0}$, $F_r = {F_r:.3f}$, "
        rf"sim threshold $\geq {SIM_THRESHOLD}$)"
    )
    ax.set_xlim(K_MIN, K_MAX)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    save(fig, "fig3_pmo_vs_k", OUT_DIR)


if __name__ == "__main__":
    main()
