"""Figure 3: PMO vs k (dispersion) after R_WEEKS weeks without cases.

Analytic curves (one per model) overlaid with Monte-Carlo simulation points at
a subset of k values.
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
    DEFAULT_N_POINTS,
    DEFAULT_R0,
    DEFAULT_R_WEEKS,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG3_K_MAX,
    FIG3_K_MIN,
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
    pmo_sse_sim,
    pmo_ssi,
    pmo_ssi_sim,
)

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
R0: float = DEFAULT_R0
R_WEEKS: int = DEFAULT_R_WEEKS  # conditioning window (weeks of zero cases)
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
K_MIN: float = FIG3_K_MIN
K_MAX: float = FIG3_K_MAX
N_POINTS: int = DEFAULT_N_POINTS


def main() -> None:
    set_style()

    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    F = cumulative(w)
    F_r = float(F[R_WEEKS])

    k_vals = np.geomspace(K_MIN, K_MAX, N_POINTS)
    pmo_sse_vals = pmo_sse(R0, k_vals, F_r)
    pmo_ssi_vals = pmo_ssi(R0, k_vals, F_r)

    sim_idx = np.linspace(0, N_POINTS - 1, SIM_SUBSET_POINTS, dtype=int)
    k_sim = k_vals[sim_idx]
    history = np.array([1] + [0] * R_WEEKS, dtype=np.int64)
    rng = np.random.default_rng(SIM_SEED)
    sse_sim_vals = np.empty(k_sim.size, dtype=np.float64)
    ssi_sim_vals = np.empty(k_sim.size, dtype=np.float64)
    for i, k in enumerate(tqdm(k_sim, desc="fig3 sims")):
        sse_sim_vals[i] = pmo_sse_sim(
            R0, float(k), w, history,
            n_sims=SIM_N_SSE, threshold=SIM_THRESHOLD, t_max=SIM_T_MAX, rng=rng,
        )
        ssi_sim_vals[i] = pmo_ssi_sim(
            R0, float(k), w, history,
            n_sims=SIM_N_SSI, threshold=SIM_THRESHOLD, t_max=SIM_T_MAX, rng=rng,
            batch_size=SIM_BATCH_SSI, max_attempts=SIM_MAX_ATTEMPTS_SSI,
        )

    fig, ax = plt.subplots()
    ax.plot(k_vals, pmo_sse_vals, color=SSE_COLOUR, label=SSE_LABEL, linestyle=SSE_LINESTYLE)
    ax.plot(k_vals, pmo_ssi_vals, color=SSI_COLOUR, label=SSI_LABEL, linestyle=SSI_LINESTYLE)
    ax.scatter(
        k_sim, sse_sim_vals,
        color=SSE_COLOUR, marker=SSE_SIM_MARKER, s=SIM_MARKERSIZE**2, zorder=3,
        edgecolors="white", linewidths=0.6, label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        k_sim, ssi_sim_vals,
        color=SSI_COLOUR, marker=SSI_SIM_MARKER, s=SIM_MARKERSIZE**2, zorder=3,
        edgecolors="white", linewidths=0.6, label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Dispersion parameter $k$")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(
        rf"After {R_WEEKS} weeks without cases ($R_0 = {R0}$, $F_r = {F_r:.3f}$, "
        rf"sim threshold $\geq {SIM_THRESHOLD}$)"
    )
    ax.set_xlim(K_MIN, K_MAX)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    save(fig, "fig3_pmo_vs_k", OUT_DIR)


if __name__ == "__main__":
    main()
