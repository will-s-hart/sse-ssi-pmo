"""Figure 8: PMO vs r with uncertain R0.

Two-panel figure.

* Top panel: posterior median of ``R0`` (line) and 95% credible interval
  (shaded band) versus weeks without cases since the index case, one
  curve per model (SSE + SSI overlaid). A dashed horizontal at the
  prior mean marks the no-data baseline.
* Bottom panel: PMO versus weeks without cases, mirroring fig1 but with
  the analytic curves replaced by the prior-marginal estimates from
  ``method='mcmc'`` and simulation cross-check markers.

Loads pre-computed results from ``results/fig8_pmo_uncertain_R0.csv``
(run ``results_8_pmo_uncertain_R0.py`` first).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from _plotting import (
    SIM_LABEL_SUFFIX,
    SIM_MARKERSIZE,
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
    FIG8_CRI_LEVEL,
    FIG8_R0_PRIOR_MEAN,
    FIG8_R0_PRIOR_SD,
    FIG8_R_MAX,
    OUT_DIR,
    RESULTS_DIR,
    SIM_THRESHOLD,
)

K: float = DEFAULT_K
R0_PRIOR_MEAN: float = FIG8_R0_PRIOR_MEAN
R0_PRIOR_SD: float = FIG8_R0_PRIOR_SD
R_MAX: int = FIG8_R_MAX
CRI: float = FIG8_CRI_LEVEL


def main() -> None:
    set_style()

    df = pd.read_csv(RESULTS_DIR / "fig8_pmo_uncertain_R0.csv")

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(default_w, 1.85 * default_h), sharex=True)

    ax_top.fill_between(df["r"], df["R0_lo_sse"], df["R0_hi_sse"], color=SSE_COLOUR, alpha=0.2)
    ax_top.fill_between(df["r"], df["R0_lo_ssi"], df["R0_hi_ssi"], color=SSI_COLOUR, alpha=0.2)
    ax_top.plot(
        df["r"],
        df["R0_med_sse"],
        color=SSE_COLOUR,
        linestyle=SSE_LINESTYLE,
        label=f"{SSE_LABEL} (median)",
    )
    ax_top.plot(
        df["r"],
        df["R0_med_ssi"],
        color=SSI_COLOUR,
        linestyle=SSI_LINESTYLE,
        label=f"{SSI_LABEL} (median)",
    )
    ax_top.axhline(
        R0_PRIOR_MEAN,
        color="0.4",
        linestyle=":",
        linewidth=1.5,
        label=rf"Prior mean $\langle R_0\rangle={R0_PRIOR_MEAN}$",
    )
    ax_top.set_ylabel(rf"Posterior $R_0$ ({int(CRI * 100)}% CrI)")
    ax_top.set_title(
        rf"$R_0\sim\mathrm{{Gamma}}(\mu={R0_PRIOR_MEAN},\sigma={R0_PRIOR_SD})$, $k={K}$"
    )
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(
        df["r"], df["pmo_sse_mcmc"], color=SSE_COLOUR, linestyle=SSE_LINESTYLE, label=SSE_LABEL
    )
    ax_bot.plot(
        df["r"], df["pmo_ssi_mcmc"], color=SSI_COLOUR, linestyle=SSI_LINESTYLE, label=SSI_LABEL
    )
    ax_bot.scatter(
        df["r"],
        df["pmo_sse_sim"],
        color=SSE_COLOUR,
        marker=SSE_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax_bot.scatter(
        df["r"],
        df["pmo_ssi_sim"],
        color=SSI_COLOUR,
        marker=SSI_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax_bot.set_xlabel("Weeks without cases since index case ($r$)")
    ax_bot.set_ylabel("Probability of major outbreak")
    ax_bot.set_xlim(0, R_MAX)
    ax_bot.set_ylim(0, 1)
    ax_bot.set_title(rf"PMO (sim threshold: peak weekly $\geq {SIM_THRESHOLD}$)")
    ax_bot.legend(loc="upper right", fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    save(fig, "fig8_pmo_uncertain_R0", OUT_DIR)


if __name__ == "__main__":
    main()
