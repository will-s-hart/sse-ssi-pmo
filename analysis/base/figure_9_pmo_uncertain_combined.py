"""Figure 9: model-averaged PMO across incidence histories with uncertain R0.

Mirror of fig5 but with ``R0`` integrated over a Gamma prior (``k`` fixed).
Three grouped bars per history — SSE, SSI, and the Bayesian model-averaged
PMO from ``pmo_uncertain`` — computed via ``method='mcmc'``. Simulation
cross-checks (``method='simulation'`` over the same R0 prior) are overlaid
as open-circle markers. The posterior probability of SSE (from
``pmo_uncertain``) is annotated below each cluster.

Loads pre-computed results from ``results/fig9_pmo_uncertain_combined.csv``
(run ``results_9_pmo_uncertain_combined.py`` first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._shared.defaults import (
    DEFAULT_K,
)
from analysis._shared.plotting import (
    SIM_LABEL_SUFFIX,
    SIM_MARKERSIZE,
    SSE_COLOUR,
    SSE_LABEL,
    SSI_COLOUR,
    SSI_LABEL,
    save,
    set_style,
)
from analysis.base.defaults import (
    FIG9_PRIOR_SSE,
    FIG9_R0_PRIOR_MEAN,
    FIG9_R0_PRIOR_SD,
    OUT_DIR,
    RESULTS_DIR,
)

K: float = DEFAULT_K
PRIOR_SSE: float = FIG9_PRIOR_SSE
R0_PRIOR_MEAN: float = FIG9_R0_PRIOR_MEAN
R0_PRIOR_SD: float = FIG9_R0_PRIOR_SD

UNCERTAIN_COLOUR = "#009E73"  # match fig5
UNCERTAIN_LABEL = f"Model-averaged ($\\pi_\\mathrm{{SSE}}={PRIOR_SSE}$)"


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig9_pmo_uncertain_combined.csv")
    n = len(df)
    bar_width = 0.27
    x = np.arange(n)
    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2 * default_w, 1.15 * default_h))

    ax.bar(
        x - bar_width,
        df["pmo_sse_mcmc"],
        width=bar_width,
        color=SSE_COLOUR,
        alpha=0.85,
        label=SSE_LABEL,
    )
    ax.bar(x, df["pmo_ssi_mcmc"], width=bar_width, color=SSI_COLOUR, alpha=0.85, label=SSI_LABEL)
    ax.bar(
        x + bar_width,
        df["uncertain_mcmc"],
        width=bar_width,
        color=UNCERTAIN_COLOUR,
        alpha=0.9,
        label=UNCERTAIN_LABEL,
    )

    ax.scatter(
        x - bar_width,
        df["pmo_sse_sim"],
        color=SSE_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        x,
        df["pmo_ssi_sim"],
        color=SSI_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
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
    ax.set_title(
        rf"$R_0\sim\mathrm{{Gamma}}(\mu={R0_PRIOR_MEAN},\sigma={R0_PRIOR_SD})$, "
        rf"$k={K}$, $\pi_\mathrm{{SSE}} = {PRIOR_SSE}$"
    )
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

    save(fig, "fig9_pmo_uncertain_combined", OUT_DIR)


if __name__ == "__main__":
    main()
