"""Figure 5: Model-averaged PMO across incidence histories.

Grouped bar chart with three bars per history — SSE, SSI, and the Bayesian
model-averaged PMO from ``pmo_uncertain``.  Each bar shows the
analytic/MCMC value, with simulation x markers overlaid as a cross-check.
The posterior probability of SSE (from ``pmo_uncertain``) is annotated
below each cluster, illustrating how the data shifts the prior.

Day-0-only histories use the closed-form ``pmo_uncertain(method="analytic")``;
general histories use ``pmo_uncertain(method="simulation")`` for the bar
height, with the per-model bars drawn from analytic SSE and MCMC SSI.
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
    FIG4_MCMC_CHAINS,
    FIG4_MCMC_DRAWS,
    FIG4_MCMC_TUNE,
    FIG5_HISTORIES,
    FIG5_PRIOR_SSE,
    FIG5_SIM_BATCH,
    FIG5_SIM_MAX_ATTEMPTS,
    FIG5_SIM_N,
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
    pmo_uncertain,
)

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
R0: float = DEFAULT_R0
K: float = DEFAULT_K
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
PRIOR_SSE: float = FIG5_PRIOR_SSE
HISTORIES: list[list[int]] = FIG5_HISTORIES
MCMC_DRAWS: int = FIG4_MCMC_DRAWS
MCMC_TUNE: int = FIG4_MCMC_TUNE
MCMC_CHAINS: int = FIG4_MCMC_CHAINS

UNCERTAIN_COLOUR = "#009E73"  # Wong palette green; distinct from SSE/SSI
UNCERTAIN_LABEL = f"Model-averaged ($\\pi_\\mathrm{{SSE}}={PRIOR_SSE}$)"


def _ssi_analytic_applicable(history: list[int]) -> bool:
    return all(x == 0 for x in history[1:])


def main() -> None:
    set_style()

    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)

    n = len(HISTORIES)
    sse_best = np.empty(n)
    ssi_best = np.empty(n)
    uncertain_best = np.empty(n)
    sse_sim = np.empty(n)
    ssi_sim = np.empty(n)
    uncertain_sim = np.empty(n)
    posterior_sse = np.empty(n)

    for i, hist in enumerate(tqdm(HISTORIES, desc="fig5 histories")):
        history = np.array(hist, dtype=np.int64)
        day0_only = _ssi_analytic_applicable(hist)

        sse_best[i] = pmo_sse(R0=R0, k=K, w=w, history=history, method="analytic")

        if day0_only:
            ssi_best[i] = pmo_ssi(R0=R0, k=K, w=w, history=history, method="analytic")
            analytic_result = pmo_uncertain(
                R0=R0, k=K, w=w, history=history, method="analytic", prior_sse=PRIOR_SSE
            )
            uncertain_best[i] = analytic_result.pmo
            posterior_sse[i] = analytic_result.posterior_sse
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
            # Use simulation as the headline value for general histories.
            sim_result = pmo_uncertain(
                R0=R0,
                k=K,
                w=w,
                history=history,
                method="simulation",
                prior_sse=PRIOR_SSE,
                n_sims=FIG5_SIM_N,
                threshold=SIM_THRESHOLD,
                t_max=SIM_T_MAX,
                rng=rng,
                batch_size=FIG5_SIM_BATCH,
                max_attempts=FIG5_SIM_MAX_ATTEMPTS,
            )
            uncertain_best[i] = sim_result.pmo
            posterior_sse[i] = sim_result.posterior_sse

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
        uncertain_sim[i] = pmo_uncertain(
            R0=R0,
            k=K,
            w=w,
            history=history,
            method="simulation",
            prior_sse=PRIOR_SSE,
            n_sims=FIG5_SIM_N,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
            batch_size=FIG5_SIM_BATCH,
            max_attempts=FIG5_SIM_MAX_ATTEMPTS,
        ).pmo

    bar_width = 0.27
    x = np.arange(n)

    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2 * default_w, 1.15 * default_h))

    ax.bar(x - bar_width, sse_best, width=bar_width, color=SSE_COLOUR, alpha=0.85, label=SSE_LABEL)
    ax.bar(x, ssi_best, width=bar_width, color=SSI_COLOUR, alpha=0.85, label=SSI_LABEL)
    ax.bar(
        x + bar_width,
        uncertain_best,
        width=bar_width,
        color=UNCERTAIN_COLOUR,
        alpha=0.9,
        label=UNCERTAIN_LABEL,
    )

    sim_marker_size = (SIM_MARKERSIZE * 2) ** 2
    ax.scatter(
        x - bar_width,
        sse_sim,
        color=SSE_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        x,
        ssi_sim,
        color=SSI_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        x + bar_width,
        uncertain_sim,
        color=UNCERTAIN_COLOUR,
        marker="x",
        s=sim_marker_size,
        linewidths=1.5,
        zorder=3,
        label="Model-averaged" + SIM_LABEL_SUFFIX,
    )

    tick_labels = [str(h) for h in HISTORIES]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Incidence history")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(rf"$R_0 = {R0}$, $k = {K}$, $\pi_\mathrm{{SSE}} = {PRIOR_SSE}$")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for xi, p in zip(x, posterior_sse, strict=True):
        ax.text(
            xi,
            -0.06,
            rf"$\pi^\star_\mathrm{{SSE}}={p:.2f}$",
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    save(fig, "fig5_pmo_uncertain", OUT_DIR)


if __name__ == "__main__":
    main()
