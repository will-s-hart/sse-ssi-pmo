"""Figure 2: PMO vs R_0 after R_WEEKS weeks without cases.

Analytic curves (one per model) overlaid with Monte-Carlo simulation points at
a subset of R_0 values.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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
    DEFAULT_N_POINTS,
    DEFAULT_R_WEEKS,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG2_R0_MAX,
    FIG2_R0_MIN,
    OUT_DIR,
    SIM_BATCH_SSI,
    SIM_MAX_ATTEMPTS_SSI,
    SIM_N_SSE,
    SIM_N_SSI,
    SIM_SEED,
    SIM_SUBSET_POINTS,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from tqdm.auto import tqdm

from sse_ssi_pmo import (
    cumulative,
    discretise_gamma,
    pmo_sse,
    pmo_ssi,
)

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
K: float = DEFAULT_K
R_WEEKS: int = DEFAULT_R_WEEKS  # conditioning window (weeks of zero cases)
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
R0_MIN: float = FIG2_R0_MIN
R0_MAX: float = FIG2_R0_MAX
N_POINTS: int = DEFAULT_N_POINTS


def main() -> None:
    set_style()

    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    F_r = float(cumulative(w)[R_WEEKS])  # for the title only

    R0_vals = np.linspace(R0_MIN, R0_MAX, N_POINTS)
    history = np.array([1] + [0] * R_WEEKS, dtype=np.int64)
    pmo_sse_vals = pmo_sse(R0=R0_vals, k=K, w=w, history=history, method="analytic")
    pmo_ssi_vals = pmo_ssi(R0=R0_vals, k=K, w=w, history=history, method="analytic")

    sim_idx = np.linspace(0, N_POINTS - 1, SIM_SUBSET_POINTS, dtype=int)
    R0_sim = R0_vals[sim_idx]
    rng = np.random.default_rng(SIM_SEED)
    sse_sim_vals = np.empty(R0_sim.size, dtype=np.float64)
    ssi_sim_vals = np.empty(R0_sim.size, dtype=np.float64)
    for i, R0 in enumerate(tqdm(R0_sim, desc="fig2 sims")):
        sse_sim_vals[i] = pmo_sse(
            R0=float(R0),
            k=K,
            w=w,
            history=history,
            method="simulation",
            n_sims=SIM_N_SSE,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
        )
        ssi_sim_vals[i] = pmo_ssi(
            R0=float(R0),
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

    fig, ax = plt.subplots()
    ax.plot(R0_vals, pmo_sse_vals, color=SSE_COLOUR, label=SSE_LABEL, linestyle=SSE_LINESTYLE)
    ax.plot(R0_vals, pmo_ssi_vals, color=SSI_COLOUR, label=SSI_LABEL, linestyle=SSI_LINESTYLE)
    ax.scatter(
        R0_sim,
        sse_sim_vals,
        color=SSE_COLOUR,
        marker=SSE_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        R0_sim,
        ssi_sim_vals,
        color=SSI_COLOUR,
        marker=SSI_SIM_MARKER,
        s=SIM_MARKERSIZE**2,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
        label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.set_xlabel(r"Reproduction number $R_0$")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(
        rf"After {R_WEEKS} weeks without cases ($k = {K}$, $F_r = {F_r:.3f}$, "
        rf"sim threshold $\geq {SIM_THRESHOLD}$)"
    )
    ax.set_xlim(R0_MIN, R0_MAX)
    ax.set_ylim(0, 1)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig2_pmo_vs_R0", OUT_DIR)


if __name__ == "__main__":
    main()
