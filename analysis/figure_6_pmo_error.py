"""Figure 6: PMO error under model misspecification vs. model averaging.

Two-column figure with one column per history length L in
``FIG6_HISTORY_LENGTHS`` and two rows: the top row plots the RMS
error (default; toggle with ``USE_RMSE``) and the bottom row the mean
*signed* error in the PMO estimate, both averaged across simulations
whose true model was drawn with probability ``FIG6_PRIOR_SSE`` (SSE) or
``1 - FIG6_PRIOR_SSE`` (SSI). The three bars per panel correspond to
assuming SSE, assuming SSI, and the model-averaged ``pmo_uncertain``
estimator. The bias (signed-error) row is the headline result: the
wrong-model bars sit consistently away from zero, while the
model-averaged bar collapses toward it.

Per panel the bar source is chosen by what the results CSV contains:
analytic columns when the panel supports them, MCMC columns when the
``USE_MCMC=True`` run was used, otherwise simulation columns. When the
bars come from analytic or MCMC, the simulation values are overlaid as
cross-check markers.

Loads pre-computed results from results/fig6_pmo_error.csv (run
results_6_pmo_error.py first). Set ``USE_RMSE = False`` to switch the
top row back to mean absolute error.
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
    FIG6_HISTORY_LENGTHS,
    FIG6_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
PRIOR_SSE: float = FIG6_PRIOR_SSE
HISTORY_LENGTHS: list[int] = FIG6_HISTORY_LENGTHS
USE_RMSE: bool = True

UNCERTAIN_COLOUR = "#009E73"
UNCERTAIN_LABEL = f"Model-averaged ($\\pi_\\mathrm{{SSE}}={PRIOR_SSE}$)"

# (sim_col, analytic_col, mcmc_col, label, colour) per estimator.
ESTIMATORS: list[tuple[str, str, str, str, str]] = [
    ("pmo_sse_sim", "pmo_sse", "pmo_sse_mcmc", SSE_LABEL, SSE_COLOUR),
    ("pmo_ssi_sim", "pmo_ssi", "pmo_ssi_mcmc", SSI_LABEL, SSI_COLOUR),
    (
        "pmo_uncertain_sim",
        "pmo_uncertain",
        "pmo_uncertain_mcmc",
        UNCERTAIN_LABEL,
        UNCERTAIN_COLOUR,
    ),
]


def _aggregate(panel: pd.DataFrame, est_col: str, *, signed: bool) -> float:
    err = panel[est_col] - panel["pmo_true"]
    if signed:
        return float(err.mean())
    if USE_RMSE:
        return float(np.sqrt((err**2).mean()))
    return float(err.abs().mean())


def _cols_present(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns and df[c].notna().any() for c in cols)


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig6_pmo_error.csv")

    analytic_cols = [a for _, a, _, _, _ in ESTIMATORS]
    mcmc_cols = [m for _, _, m, _, _ in ESTIMATORS]
    sim_cols = [s for s, _, _, _, _ in ESTIMATORS]

    default_w, default_h = plt.rcParams["figure.figsize"]
    n_cols = len(HISTORY_LENGTHS)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(1.2 * default_w * n_cols, 1.9 * default_h),
        sharey="row",
        squeeze=False,
    )

    ylabels = (
        "RMS error in PMO" if USE_RMSE else "Mean absolute error in PMO",
        "Mean signed error in PMO",
    )

    labels = [lbl for _, _, _, lbl, _ in ESTIMATORS]
    colours = [col for _, _, _, _, col in ESTIMATORS]
    x = np.arange(len(ESTIMATORS))
    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2
    legend_ax = None

    for j, length in enumerate(HISTORY_LENGTHS):
        panel = df[df["history_length"] == length]
        if _cols_present(panel, analytic_cols):
            bars_keys, overlay_keys = analytic_cols, sim_cols
        elif _cols_present(panel, mcmc_cols):
            bars_keys, overlay_keys = mcmc_cols, sim_cols
        else:
            bars_keys, overlay_keys = sim_cols, None
        plot_cols = set(bars_keys)
        if overlay_keys is not None:
            plot_cols.update(overlay_keys)
        panel = panel.dropna(subset=list(plot_cols))
        for row, signed in enumerate((False, True)):
            ax = axes[row, j]
            bars = [_aggregate(panel, c, signed=signed) for c in bars_keys]
            ax.bar(x, bars, color=colours, alpha=0.9)
            for xi, h in zip(x, bars, strict=True):
                va = "bottom" if h >= 0 else "top"
                ax.text(xi, h, f"{h:.3f}", ha="center", va=va, fontsize=9)
            if overlay_keys is not None:
                overlay = [_aggregate(panel, c, signed=signed) for c in overlay_keys]
                ax.scatter(
                    x,
                    overlay,
                    color="black",
                    marker="x",
                    s=sim_marker_size,
                    linewidths=1.8,
                    zorder=3,
                    label="Estimate" + SIM_LABEL_SUFFIX,
                )
                if row == 0:
                    legend_ax = ax
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
    if legend_ax is not None:
        legend_ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(rf"$R_0 = {R0}$, $k = {K}$, true-model prior $\pi_\mathrm{{SSE}} = {PRIOR_SSE}$")
    fig.tight_layout()

    save(fig, "fig6_pmo_error", OUT_DIR)


if __name__ == "__main__":
    main()
