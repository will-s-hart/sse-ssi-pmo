"""Figure 13: real-time onset-anchored PMO over a real EVD outbreak.

Weekly symptom-onset counts (bars, left axis) for the 2017 Likati (DRC) EVD
outbreak, with the probability of a major outbreak estimated in real time at
each week (lines, right axis) for the onset-anchored SSE and SSI models. The
week-$t$ PMO conditions on the onset history observed up to and including week
$t$.

Loads pre-computed results from results/fig13_pmo_realtime_delay.csv (run
results_13_pmo_realtime_delay.py first).
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
    OUT_DIR,
    RESULTS_DIR,
    SIM_THRESHOLD,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
BAR_COLOUR = "#BBBBBB"


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig13_pmo_realtime_delay.csv")
    weeks = df["week"].to_numpy()

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax_cases = plt.subplots(figsize=(1.5 * default_w, default_h))

    # Left axis: weekly onset counts (epi curve).
    ax_cases.bar(
        weeks,
        df["cases"],
        width=0.9,
        color=BAR_COLOUR,
        edgecolor="white",
        zorder=1,
        label="Weekly onsets",
    )
    ax_cases.set_xlabel("Week of outbreak")
    ax_cases.set_ylabel("Symptom onsets per week")
    ax_cases.set_ylim(0, max(4, int(df["cases"].max()) + 1))

    # Right axis: real-time PMO for each model.
    ax_pmo = ax_cases.twinx()
    ax_pmo.plot(
        weeks,
        df["sse_delay_pmo"],
        color=SSE_COLOUR,
        linestyle=SSE_LINESTYLE,
        marker=SSE_SIM_MARKER,
        zorder=3,
        label=SSE_LABEL,
    )
    ax_pmo.plot(
        weeks,
        df["ssi_delay_pmo"],
        color=SSI_COLOUR,
        linestyle=SSI_LINESTYLE,
        marker=SSI_SIM_MARKER,
        zorder=3,
        label=SSI_LABEL,
    )
    ax_pmo.set_ylabel("Probability of major outbreak")
    ax_pmo.set_ylim(0, 1)

    # x tick labels: week index with the week-start (Monday) date underneath.
    ax_cases.set_xticks(weeks)
    ax_cases.set_xticklabels([f"{w}\n{d}" for w, d in zip(weeks, df["week_start"], strict=True)])

    # Combined legend from both axes.
    h1, l1 = ax_cases.get_legend_handles_labels()
    h2, l2 = ax_pmo.get_legend_handles_labels()
    ax_pmo.legend(h1 + h2, l1 + l2, loc="upper left")

    ax_cases.set_title(
        rf"2017 Likati EVD outbreak: real-time PMO ($R_0 = {R0}$, $k = {K}$, "
        rf"threshold: peak weekly onsets $\geq {SIM_THRESHOLD}$)"
    )
    ax_cases.grid(True, axis="y", alpha=0.3)

    save(fig, "fig13_pmo_realtime_delay", OUT_DIR)


if __name__ == "__main__":
    main()
