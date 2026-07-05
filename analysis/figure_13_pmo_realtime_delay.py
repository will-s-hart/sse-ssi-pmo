"""Figure 13: real-time onset-anchored PMO over a real EVD outbreak.

Two stacked panels sharing the week axis for the 2017 Likati (DRC) EVD outbreak,
each with the weekly symptom-onset counts drawn as bars on a secondary (right)
axis:

* top — the probability of a major outbreak estimated in real time (left axis)
  for the onset-anchored SSE and SSI models and their Bayesian model average
  (prior model probabilities 0.5 each);
* bottom — the posterior model probabilities over the weeks of the outbreak.

Both are read off the same bootstrap particle filter: the per-model PMO from the
forward-resolved particles, the posterior model probabilities from the filter's
marginal-likelihood (evidence) estimate. The week-$t$ quantities condition on
the onset history observed up to and including week $t$.

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
    FIG13_PRIOR_SSE,
    OUT_DIR,
    RESULTS_DIR,
    SIM_THRESHOLD,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
PRIOR_SSE: float = FIG13_PRIOR_SSE
BAR_COLOUR = "#DDDDDD"
ENSEMBLE_COLOUR = "#009E73"  # Wong palette green; matches the model-average figures


def _cases_axis(ax_left, weeks, cases):
    """Twin a right axis onto ``ax_left`` and draw the weekly onset bars there."""
    ax_cases = ax_left.twinx()
    ax_cases.bar(
        weeks,
        cases,
        width=0.9,
        color=BAR_COLOUR,
        edgecolor="white",
        zorder=0,
        label="Weekly onsets",
    )
    ax_cases.set_ylabel("Symptom onsets per week")
    ax_cases.set_ylim(0, max(4, int(cases.max()) + 1))
    ax_cases.set_zorder(ax_left.get_zorder() - 1)  # bars behind the lines
    ax_left.patch.set_visible(False)
    return ax_cases


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig13_pmo_realtime_delay.csv")
    weeks = df["week"].to_numpy()
    cases = df["cases"].to_numpy()

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, (ax_pmo, ax_post) = plt.subplots(
        2, 1, sharex=True, figsize=(1.5 * default_w, 1.7 * default_h)
    )

    # --- Top panel: real-time PMO (left) + epi curve (right) ---
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
    ax_pmo.plot(
        weeks,
        df["ensemble_pmo"],
        color=ENSEMBLE_COLOUR,
        linestyle="-",
        marker="D",
        zorder=4,
        label=rf"Model average ($\pi_\mathrm{{SSE}}={PRIOR_SSE}$)",
    )
    ax_pmo.set_ylabel("Probability of major outbreak")
    ax_pmo.set_ylim(0, 1)
    ax_pmo.grid(True, axis="y", alpha=0.3)
    ax_pmo_cases = _cases_axis(ax_pmo, weeks, cases)

    h1, l1 = ax_pmo.get_legend_handles_labels()
    h2, l2 = ax_pmo_cases.get_legend_handles_labels()
    ax_pmo.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    ax_pmo.set_title(
        rf"2017 Likati EVD outbreak: real-time PMO ($R_0 = {R0}$, $k = {K}$, "
        rf"threshold: peak weekly onsets $\geq {SIM_THRESHOLD}$)"
    )

    # --- Bottom panel: posterior model probabilities (left) + epi curve (right) ---
    ax_post.plot(
        weeks,
        df["post_sse"],
        color=SSE_COLOUR,
        linestyle=SSE_LINESTYLE,
        marker=SSE_SIM_MARKER,
        zorder=3,
        label=SSE_LABEL,
    )
    ax_post.plot(
        weeks,
        df["post_ssi"],
        color=SSI_COLOUR,
        linestyle=SSI_LINESTYLE,
        marker=SSI_SIM_MARKER,
        zorder=3,
        label=SSI_LABEL,
    )
    ax_post.axhline(PRIOR_SSE, color="0.5", linestyle=":", linewidth=1, zorder=1)
    ax_post.set_ylabel("Posterior model probability")
    ax_post.set_ylim(0, 1)
    ax_post.grid(True, axis="y", alpha=0.3)
    _cases_axis(ax_post, weeks, cases)
    ax_post.legend(loc="upper left", ncol=2)

    ax_post.set_xlabel("Week of outbreak")
    ax_post.set_xticks(weeks)
    ax_post.set_xticklabels([f"{w}\n{d}" for w, d in zip(weeks, df["week_start"], strict=True)])

    save(fig, "fig13_pmo_realtime_delay", OUT_DIR)


if __name__ == "__main__":
    main()
