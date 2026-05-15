"""Figure 7: PMO error across a 5-model ensemble vs. ensemble averaging.

Same two-row layout as figure 6 (abs error top, signed error bottom) but
with six bars per panel:

* SSE (k = DEFAULT_K), SSE (k = FIG7_K_LARGE)
* SSI (k = DEFAULT_K), SSI (k = FIG7_K_LARGE)
* Poisson (k -> infty)
* Model-averaged ``pmo_ensemble`` with equal priors (1/5 each)

Loads pre-computed results from results/fig7_pmo_ensemble_error.csv (run
``results_7_pmo_ensemble_error.py`` first). Set ``USE_RMSE = False`` to
switch the top row from RMS error to mean absolute error.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _plotting import (
    SSE_COLOUR,
    SSI_COLOUR,
    save,
    set_style,
)
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    FIG7_HISTORY_LENGTHS,
    FIG7_K_LARGE,
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
K_LARGE: float = FIG7_K_LARGE
HISTORY_LENGTHS: list[int] = FIG7_HISTORY_LENGTHS
USE_RMSE: bool = True

# Reuse fig6's ensemble colour for the model-averaged bar.
ENSEMBLE_COLOUR = "#009E73"
POISSON_COLOUR = "#9467BD"  # tab purple — distinct from SSE / SSI / ensemble

# Lighter variant for the k = K_LARGE bars (k is closer to Poisson; lighter
# colour cues the link). The hex values below are matplotlib's
# alpha-blended approximations.
SSE_LIGHT_COLOUR = "#66A6CE"  # lighter SSE blue
SSI_LIGHT_COLOUR = "#E69E66"  # lighter SSI vermillion

BAR_KEYS: list[tuple[str, str, str]] = [
    ("pmo_sse_k", f"SSE ($k={K}$)", SSE_COLOUR),
    ("pmo_sse_k_large", f"SSE ($k={K_LARGE}$)", SSE_LIGHT_COLOUR),
    ("pmo_ssi_k", f"SSI ($k={K}$)", SSI_COLOUR),
    ("pmo_ssi_k_large", f"SSI ($k={K_LARGE}$)", SSI_LIGHT_COLOUR),
    ("pmo_poisson", r"Poisson ($k \to \infty$)", POISSON_COLOUR),
    ("pmo_ensemble", "Ensemble-averaged", ENSEMBLE_COLOUR),
]


def _aggregate(panel: pd.DataFrame, est_col: str, *, signed: bool) -> float:
    err = panel[est_col] - panel["pmo_true"]
    if signed:
        return float(err.mean())
    if USE_RMSE:
        return float(np.sqrt((err**2).mean()))
    return float(err.abs().mean())


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig7_pmo_ensemble_error.csv")

    default_w, default_h = plt.rcParams["figure.figsize"]
    n_cols = len(HISTORY_LENGTHS)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(1.6 * default_w * n_cols, 1.9 * default_h),
        sharey="row",
        squeeze=False,
    )

    top = "RMS error in PMO" if USE_RMSE else "Mean absolute error in PMO"
    ylabels = (top, "Mean signed error in PMO")

    labels = [lbl for _, lbl, _ in BAR_KEYS]
    colours = [col for _, _, col in BAR_KEYS]
    x = np.arange(len(BAR_KEYS))

    for j, length in enumerate(HISTORY_LENGTHS):
        panel = df[df["history_length"] == length]
        for row, signed in enumerate((False, True)):
            ax = axes[row, j]
            heights = [_aggregate(panel, ac, signed=signed) for ac, _, _ in BAR_KEYS]
            ax.bar(x, heights, color=colours, alpha=0.9)
            for xi, h in zip(x, heights, strict=True):
                va = "bottom" if h >= 0 else "top"
                ax.text(xi, h, f"{h:.3f}", ha="center", va=va, fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xticks(x)
            if signed:
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
            else:
                ax.set_xticklabels([])
                ax.set_title(f"{length}-week history (N = {len(panel)})")

    axes[0, 0].set_ylabel(ylabels[0])
    axes[1, 0].set_ylabel(ylabels[1])
    fig.suptitle(
        rf"$R_0 = {R0}$ — five-model ensemble with equal priors "
        rf"($\pi = 1/5$ each)"
    )
    fig.tight_layout()

    save(fig, "fig7_pmo_ensemble_error", OUT_DIR)


if __name__ == "__main__":
    main()
