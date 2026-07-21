"""Figure 2: onset-anchored PMO comparison across symptom-onset histories.

Symptom-onset-anchored analogue of Figure 4. Grouped bar chart of the
simulation-based PMO for the onset-anchored SSE and SSI models (no analytic or
MCMC overlay is available for these models).

Loads pre-computed results from results/fig2_pmo_comparison_delay.csv (run
results_12_pmo_comparison_delay.py first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._shared.defaults import (
    DEFAULT_K,
    DEFAULT_R0,
)
from analysis._shared.plotting import (
    SSE_COLOUR,
    SSE_LABEL,
    SSI_COLOUR,
    SSI_LABEL,
    save,
    set_style,
)
from analysis.delays.defaults import (
    OUT_DIR,
    RESULTS_DIR,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig2_pmo_comparison_delay.csv")
    n = len(df)
    bar_width = 0.35
    x = np.arange(n)

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2 * default_w, default_h))

    ax.bar(
        x - bar_width / 2,
        df["sse_delay_sim"],
        width=bar_width,
        color=SSE_COLOUR,
        alpha=0.85,
        label=SSE_LABEL,
    )
    ax.bar(
        x + bar_width / 2,
        df["ssi_delay_sim"],
        width=bar_width,
        color=SSI_COLOUR,
        alpha=0.85,
        label=SSI_LABEL,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["history"])
    ax.set_xlabel("Symptom-onset history")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(rf"Onset-anchored models: $R_0 = {R0}$, $k = {K}$")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig2_pmo_comparison_delay", OUT_DIR)


if __name__ == "__main__":
    main()
