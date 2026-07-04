"""Figure 11: onset-anchored PMO vs r (weeks without onsets since index onset).

Symptom-onset-anchored analogue of Figure 1. Simulation-based PMO for the
onset-anchored SSE and SSI models (no analytic curve is available), shown as
points joined by lines at each integer r.

Loads pre-computed results from results/fig11_pmo_vs_r_delay.csv (run
results_11_pmo_vs_r_delay.py first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from _plotting import (
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
    FIG11_R_MAX,
    OUT_DIR,
    RESULTS_DIR,
    SIM_THRESHOLD,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
R_MAX: int = FIG11_R_MAX


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig11_pmo_vs_r_delay.csv")

    fig, ax = plt.subplots()
    ax.plot(
        df["r"],
        df["sse_delay_sim"],
        color=SSE_COLOUR,
        linestyle=SSE_LINESTYLE,
        marker=SSE_SIM_MARKER,
        label=SSE_LABEL,
    )
    ax.plot(
        df["r"],
        df["ssi_delay_sim"],
        color=SSI_COLOUR,
        linestyle=SSI_LINESTYLE,
        marker=SSI_SIM_MARKER,
        label=SSI_LABEL,
    )
    ax.set_xlabel("Weeks without onsets since index onset ($r$)")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(
        rf"Onset-anchored models: $R_0 = {R0}$, $k = {K}$ "
        rf"(sim threshold: peak weekly onsets $\geq {SIM_THRESHOLD}$)"
    )
    ax.set_xlim(0, R_MAX)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig11_pmo_vs_r_delay", OUT_DIR)


if __name__ == "__main__":
    main()
