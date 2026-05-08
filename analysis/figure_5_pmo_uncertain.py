"""Figure 5: Model-averaged PMO across incidence histories.

Grouped bar chart with three analytic bars per history — SSE, SSI, and the
Bayesian model-averaged PMO from ``pmo_uncertain``.  Simulation x markers
are overlaid for all three bars as one cross-check; open-circle MCMC
markers are overlaid at the SSI bar position as a second cross-check.
The posterior probability of SSE (from ``pmo_uncertain``) is annotated
below each cluster, illustrating how the data shifts the prior.

Loads pre-computed results from results/fig5_pmo_uncertain.csv (run
results_5_pmo_uncertain.py first).
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
    FIG5_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
PRIOR_SSE: float = FIG5_PRIOR_SSE

UNCERTAIN_COLOUR = "#009E73"  # Wong palette green; distinct from SSE/SSI
UNCERTAIN_LABEL = f"Model-averaged ($\\pi_\\mathrm{{SSE}}={PRIOR_SSE}$)"


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig5_pmo_uncertain.csv")
    n = len(df)
    bar_width = 0.27
    x = np.arange(n)

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2 * default_w, 1.15 * default_h))

    ax.bar(
        x - bar_width,
        df["sse_analytic"],
        width=bar_width,
        color=SSE_COLOUR,
        alpha=0.85,
        label=SSE_LABEL,
    )
    ax.bar(x, df["ssi_analytic"], width=bar_width, color=SSI_COLOUR, alpha=0.85, label=SSI_LABEL)
    ax.bar(
        x + bar_width,
        df["uncertain_analytic"],
        width=bar_width,
        color=UNCERTAIN_COLOUR,
        alpha=0.9,
        label=UNCERTAIN_LABEL,
    )

    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2
    ax.scatter(
        x - bar_width,
        df["sse_sim"],
        color=SSE_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        x,
        df["ssi_sim"],
        color=SSI_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        x,
        df["ssi_mcmc"],
        facecolors="none",
        edgecolors=SSI_COLOUR,
        marker="o",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSI_LABEL + " (MCMC)",
    )
    ax.scatter(
        x + bar_width,
        df["uncertain_sim"],
        color=UNCERTAIN_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label="Model-averaged" + SIM_LABEL_SUFFIX,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["history"])
    ax.set_xlabel("Incidence history")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(rf"$R_0 = {R0}$, $k = {K}$, $\pi_\mathrm{{SSE}} = {PRIOR_SSE}$")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for xi, p in zip(x, df["posterior_sse"], strict=True):
        ax.text(
            xi,
            -0.06,
            rf"$\pi^\star_\mathrm{{SSE}}={p:.2f}$",
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    save(fig, "fig5_pmo_uncertain", OUT_DIR)


if __name__ == "__main__":
    main()
