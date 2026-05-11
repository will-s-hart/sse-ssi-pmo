"""Figure 6: PMO error under model misspecification vs. model averaging.

Two-column figure with one column per history length L in
``FIG6_HISTORY_LENGTHS`` and two rows: the top row plots the mean
*absolute* error and the bottom row the mean *signed* error in the PMO
estimate, both averaged across simulations whose true model was drawn
with probability ``FIG6_PRIOR_SSE`` (SSE) or ``1 - FIG6_PRIOR_SSE`` (SSI).
The three bars per panel correspond to assuming SSE, assuming SSI, and
the model-averaged ``pmo_uncertain`` estimator. The bias (signed-error)
row is the headline result: the wrong-model bars sit consistently away
from zero, while the model-averaged bar collapses toward it.

Loads pre-computed results from results/fig6_pmo_error.csv (run
results_6_pmo_error.py first). Set ``USE_PERCENT_ERROR = True`` to switch
both metrics to percentages (sims with ``pmo_true == 0`` are dropped from
the percentage averages).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _plotting import (
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
    FIG6_HISTORY_LENGTHS,
    FIG6_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
PRIOR_SSE: float = FIG6_PRIOR_SSE
HISTORY_LENGTHS: list[int] = FIG6_HISTORY_LENGTHS
USE_PERCENT_ERROR: bool = False

UNCERTAIN_COLOUR = "#009E73"
UNCERTAIN_LABEL = f"Model-averaged ($\\pi_\\mathrm{{SSE}}={PRIOR_SSE}$)"

BAR_KEYS: list[tuple[str, str, str, str]] = [
    ("pmo_sse", "pmo_sse_sim", SSE_LABEL, SSE_COLOUR),
    ("pmo_ssi", "pmo_ssi_sim", SSI_LABEL, SSI_COLOUR),
    ("pmo_uncertain", "pmo_uncertain_sim", UNCERTAIN_LABEL, UNCERTAIN_COLOUR),
]


def _aggregate(panel: pd.DataFrame, est_col: str, *, signed: bool) -> float:
    err = panel[est_col] - panel["pmo_true"]
    if not signed:
        err = err.abs()
    if USE_PERCENT_ERROR:
        mask = panel["pmo_true"] > 0
        return float((err[mask] / panel["pmo_true"][mask]).mean() * 100.0)
    return float(err.mean())


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig6_pmo_error.csv")

    default_w, default_h = plt.rcParams["figure.figsize"]
    n_cols = len(HISTORY_LENGTHS)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(1.2 * default_w * n_cols, 1.9 * default_h),
        sharey="row",
        squeeze=False,
    )

    if USE_PERCENT_ERROR:
        ylabels = ("Mean absolute % error in PMO", "Mean signed % error in PMO")
    else:
        ylabels = ("Mean absolute error in PMO", "Mean signed error in PMO")

    labels = [lbl for _, _, lbl, _ in BAR_KEYS]
    colours = [col for _, _, _, col in BAR_KEYS]
    x = np.arange(len(BAR_KEYS))
    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2

    for j, length in enumerate(HISTORY_LENGTHS):
        panel = df[df["history_length"] == length]
        for row, signed in enumerate((False, True)):
            ax = axes[row, j]
            analytic = [_aggregate(panel, ac, signed=signed) for ac, _, _, _ in BAR_KEYS]
            sim = [_aggregate(panel, sc, signed=signed) for _, sc, _, _ in BAR_KEYS]
            ax.bar(x, analytic, color=colours, alpha=0.9)
            for xi, h in zip(x, analytic, strict=True):
                va = "bottom" if h >= 0 else "top"
                ax.text(xi, h, f"{h:.3f}", ha="center", va=va, fontsize=9)
            ax.scatter(
                x,
                sim,
                color="black",
                marker="x",
                s=sim_marker_size,
                linewidths=1.8,
                zorder=3,
                label="Simulation estimate",
            )
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xticks(x)
            if signed:
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.set_xticklabels(labels, rotation=15, ha="right")
            else:
                ax.set_xticklabels([])
                ax.set_title(f"{length}-week history (N = {len(panel)})")

    axes[0, 0].set_ylabel(ylabels[0])
    axes[1, 0].set_ylabel(ylabels[1])
    axes[0, -1].legend(loc="upper right", fontsize=9)
    fig.suptitle(rf"$R_0 = {R0}$, $k = {K}$, true-model prior $\pi_\mathrm{{SSE}} = {PRIOR_SSE}$")
    fig.tight_layout()

    save(fig, "fig6_pmo_error", OUT_DIR)


if __name__ == "__main__":
    main()
