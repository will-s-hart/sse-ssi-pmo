"""Figure 1: PMO vs r (weeks without cases since the index case).

Analytic curves (one per model) overlaid with Monte-Carlo simulation points at
each integer r.
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
    DEFAULT_R0,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG1_R_MAX,
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
R_MAX: int = FIG1_R_MAX


def main() -> None:
    set_style()

    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)

    r_vals = np.arange(0, R_MAX + 1)
    pmo_sse_vals = np.empty(r_vals.size, dtype=np.float64)
    pmo_ssi_vals = np.empty(r_vals.size, dtype=np.float64)
    sse_sim_vals = np.empty(r_vals.size, dtype=np.float64)
    ssi_sim_vals = np.empty(r_vals.size, dtype=np.float64)
    rng = np.random.default_rng(SIM_SEED)
    for i, r in enumerate(tqdm(r_vals, desc="fig1 sims")):
        history = np.array([1] + [0] * int(r), dtype=np.int64)
        pmo_sse_vals[i] = pmo_sse(
            R0=R0, k=K, w=w, history=history, method="analytic",
        )
        pmo_ssi_vals[i] = pmo_ssi(
            R0=R0, k=K, w=w, history=history, method="analytic",
        )
        sse_sim_vals[i] = pmo_sse(
            R0=R0, k=K, w=w, history=history, method="simulation",
            n_sims=SIM_N_SSE, threshold=SIM_THRESHOLD, t_max=SIM_T_MAX, rng=rng,
        )
        ssi_sim_vals[i] = pmo_ssi(
            R0=R0, k=K, w=w, history=history, method="simulation",
            n_sims=SIM_N_SSI, threshold=SIM_THRESHOLD, t_max=SIM_T_MAX, rng=rng,
            batch_size=SIM_BATCH_SSI, max_attempts=SIM_MAX_ATTEMPTS_SSI,
        )

    fig, ax = plt.subplots()
    ax.plot(r_vals, pmo_sse_vals, color=SSE_COLOUR, label=SSE_LABEL, linestyle=SSE_LINESTYLE)
    ax.plot(r_vals, pmo_ssi_vals, color=SSI_COLOUR, label=SSI_LABEL, linestyle=SSI_LINESTYLE)
    ax.scatter(
        r_vals, sse_sim_vals,
        color=SSE_COLOUR, marker=SSE_SIM_MARKER, s=SIM_MARKERSIZE**2, zorder=3,
        edgecolors="white", linewidths=0.6, label=SSE_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.scatter(
        r_vals, ssi_sim_vals,
        color=SSI_COLOUR, marker=SSI_SIM_MARKER, s=SIM_MARKERSIZE**2, zorder=3,
        edgecolors="white", linewidths=0.6, label=SSI_LABEL + SIM_LABEL_SUFFIX,
    )
    ax.set_xlabel("Weeks without cases since index case ($r$)")
    ax.set_ylabel("Probability of major outbreak")
    ax.set_title(rf"$R_0 = {R0}$, $k = {K}$ (sim threshold: peak weekly $\geq {SIM_THRESHOLD}$)")
    ax.set_xlim(0, R_MAX)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "fig1_pmo_vs_r", OUT_DIR)


if __name__ == "__main__":
    main()
