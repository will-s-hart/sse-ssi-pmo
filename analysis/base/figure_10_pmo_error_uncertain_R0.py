"""Figure 10: PMO error under model misspecification with uncertain R0.

Two-column figure (one column per history length L in
``FIG10_HISTORY_LENGTHS``) with two rows: RMS error (top; default,
toggle with ``USE_RMSE``) and mean *signed* error (bottom) in the PMO
estimate, both averaged across simulations whose true model was drawn
with probability ``FIG10_PRIOR_SSE`` (SSE) or ``1 - FIG10_PRIOR_SSE``
(SSI) and whose true R0 was drawn from the Gamma prior the estimators
integrate over. The three bars per panel correspond to assuming SSE,
assuming SSI, and the model-averaged ``pmo_uncertain`` estimator.

Bars come from the MCMC columns if they are present in the CSV (the
``USE_MCMC=True`` run of the results script); otherwise they come from
the simulation columns. When MCMC bars are shown, the simulation
estimates are overlaid as cross-check markers.

Loads ``results/fig10_pmo_error_uncertain_R0.csv`` (run
``results_10_pmo_error_uncertain_R0.py`` first). Set ``USE_RMSE = False``
to switch the top row back to mean absolute error.
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
    FIG10_HISTORY_LENGTHS,
    FIG10_PRIOR_SSE,
    FIG10_R0_PRIOR_MEAN,
    FIG10_R0_PRIOR_SD,
    OUT_DIR,
    RESULTS_DIR,
)

K: float = DEFAULT_K
PRIOR_SSE: float = FIG10_PRIOR_SSE
R0_PRIOR_MEAN: float = FIG10_R0_PRIOR_MEAN
R0_PRIOR_SD: float = FIG10_R0_PRIOR_SD
HISTORY_LENGTHS: list[int] = FIG10_HISTORY_LENGTHS
USE_RMSE: bool = True

UNCERTAIN_COLOUR = "#009E73"  # match figs 6/9
UNCERTAIN_LABEL = f"Model-averaged ($\\pi_\\mathrm{{SSE}}={PRIOR_SSE}$)"

# (sim_col, mcmc_col, label, colour) per estimator.
ESTIMATORS: list[tuple[str, str, str, str]] = [
    ("pmo_sse_sim", "pmo_sse_mcmc", SSE_LABEL, SSE_COLOUR),
    ("pmo_ssi_sim", "pmo_ssi_mcmc", SSI_LABEL, SSI_COLOUR),
    ("pmo_uncertain_sim", "pmo_uncertain_mcmc", UNCERTAIN_LABEL, UNCERTAIN_COLOUR),
]


def _aggregate(panel: pd.DataFrame, est_col: str, *, signed: bool) -> float:
    err = panel[est_col] - panel["pmo_true"]
    if signed:
        return float(err.mean())
    if USE_RMSE:
        return float(np.sqrt((err**2).mean()))
    return float(err.abs().mean())


def _mcmc_present(df: pd.DataFrame) -> bool:
    cols = [mcmc_col for _, mcmc_col, _, _ in ESTIMATORS]
    return all(col in df.columns and df[col].notna().any() for col in cols)


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig10_pmo_error_uncertain_R0.csv")
    mcmc_present = _mcmc_present(df)
    # bar source / overlay source per estimator.
    bars_keys = [(mcmc_col if mcmc_present else sim_col) for sim_col, mcmc_col, _, _ in ESTIMATORS]
    overlay_keys = [sim_col for sim_col, _, _, _ in ESTIMATORS] if mcmc_present else None

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

    labels = [lbl for _, _, lbl, _ in ESTIMATORS]
    colours = [col for _, _, _, col in ESTIMATORS]
    x = np.arange(len(ESTIMATORS))
    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2

    for j, length in enumerate(HISTORY_LENGTHS):
        panel = df[df["history_length"] == length]
        # Drop rows where any plotted estimator is NaN (rejection-sampling
        # under-resolution on rare histories) so the reported N matches the
        # row count behind every bar.
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
    if overlay_keys is not None:
        axes[0, -1].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        rf"$R_0\sim\mathrm{{Gamma}}(\mu={R0_PRIOR_MEAN},\sigma={R0_PRIOR_SD})$, "
        rf"$k = {K}$, true-model prior $\pi_\mathrm{{SSE}} = {PRIOR_SSE}$"
    )
    fig.tight_layout()

    save(fig, "fig10_pmo_error_uncertain_R0", OUT_DIR)


if __name__ == "__main__":
    main()
