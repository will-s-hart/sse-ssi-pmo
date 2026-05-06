"""Figure 4: PMO comparison across incidence histories.

Grouped bar chart showing SSE (analytic) and SSI (analytic when possible,
MCMC otherwise) PMO estimates as bars, with simulation × markers overlaid, for
a configurable list of incidence histories.  Confirms agreement between methods
across a variety of observed histories, including cases with non-zero incidence
after day 0 where SSI requires MCMC.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG4_HISTORIES,
    FIG4_MCMC_CHAINS,
    FIG4_MCMC_DRAWS,
    FIG4_MCMC_TUNE,
    OUT_DIR,
    SIM_BATCH_SSI,
    SIM_MAX_ATTEMPTS_SSI,
    SIM_N_SSE,
    SIM_N_SSI,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from tqdm.auto import tqdm

from sse_ssi_pmo import (
    discretise_gamma,
    pmo_sse,
    pmo_ssi,
)

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
R0: float = DEFAULT_R0
K: float = DEFAULT_K
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
HISTORIES: list[list[int]] = FIG4_HISTORIES
MCMC_DRAWS: int = FIG4_MCMC_DRAWS
MCMC_TUNE: int = FIG4_MCMC_TUNE
MCMC_CHAINS: int = FIG4_MCMC_CHAINS


def _ssi_analytic_applicable(history: list[int]) -> bool:
    return all(x == 0 for x in history[1:])


def main() -> None:
    set_style()

    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)

    n = len(HISTORIES)
    sse_analytic = np.empty(n)
    ssi_best = np.empty(n)
    sse_sim = np.empty(n)
    ssi_sim = np.empty(n)
    ssi_method_labels: list[str] = []

    for i, hist in enumerate(tqdm(HISTORIES, desc="fig4 histories")):
        history = np.array(hist, dtype=np.int64)

        sse_analytic[i] = pmo_sse(R0=R0, k=K, w=w, history=history, method="analytic")

        if _ssi_analytic_applicable(hist):
            ssi_best[i] = pmo_ssi(R0=R0, k=K, w=w, history=history, method="analytic")
            ssi_method_labels.append("analytic")
        else:
            ssi_best[i] = pmo_ssi(
                R0=R0,
                k=K,
                w=w,
                history=history,
                method="mcmc",
                draws=MCMC_DRAWS,
                tune=MCMC_TUNE,
                chains=MCMC_CHAINS,
                target_accept=0.99,
                progressbar=False,
            )
            ssi_method_labels.append("mcmc")

        sse_sim[i] = pmo_sse(
            R0=R0,
            k=K,
            w=w,
            history=history,
            method="simulation",
            n_sims=SIM_N_SSE,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
        )
        ssi_sim[i] = pmo_ssi(
            R0=R0,
            k=K,
            w=w,
            history=history,
            method="simulation",
            n_sims=SIM_N_SSI,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
            batch_size=SIM_BATCH_SSI,
            max_attempts=SIM_MAX_ATTEMPTS_SSI,
        )

    bar_width = 0.35
    x = np.arange(n)

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2 * default_w, default_h))

    ax.bar(
        x - bar_width / 2,
        sse_analytic,
        width=bar_width,
        color=SSE_COLOUR,
        alpha=0.85,
        label=SSE_LABEL,
    )
    ax.bar(
        x + bar_width / 2,
        ssi_best,
        width=bar_width,
        color=SSI_COLOUR,
        alpha=0.85,
        label=SSI_LABEL + " (analytic / MCMC)",
    )

    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2
    ax.scatter(
        x - bar_width / 2,
        sse_sim,
        color=SSE_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        x + bar_width / 2,
        ssi_sim,
        color=SSI_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )

    tick_labels = [str(h) for h in HISTORIES]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Incidence history")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(rf"$R_0 = {R0}$, $k = {K}$")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig4_pmo_comparison", OUT_DIR)


if __name__ == "__main__":
    main()
