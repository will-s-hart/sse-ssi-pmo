"""Compute and save results for Figure 3: PMO vs k (dispersion)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis._shared.defaults import (
    DEFAULT_R0,
    DEFAULT_R_WEEKS,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    SIM_BATCH_SSI,
    SIM_MAX_ATTEMPTS_SSI,
    SIM_N_SSE,
    SIM_N_SSI,
    SIM_SEED,
    SIM_SUBSET_POINTS,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from analysis.base.defaults import (
    DEFAULT_N_POINTS,
    FIG3_K_MAX,
    FIG3_K_MIN,
    RESULTS_DIR,
)
from sse_ssi_pmo import (
    cumulative,
    discretise_gamma,
    pmo_sse,
    pmo_ssi,
)

R0: float = DEFAULT_R0
R_WEEKS: int = DEFAULT_R_WEEKS
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
K_MIN: float = FIG3_K_MIN
K_MAX: float = FIG3_K_MAX
N_POINTS: int = DEFAULT_N_POINTS


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    F_r = float(cumulative(w)[R_WEEKS])

    k_vals = np.geomspace(K_MIN, K_MAX, N_POINTS)
    history = np.array([1] + [0] * R_WEEKS, dtype=np.int64)
    pmo_sse_vals = pmo_sse(R0=R0, k=k_vals, w=w, history=history, method="analytic")
    pmo_ssi_vals = pmo_ssi(R0=R0, k=k_vals, w=w, history=history, method="analytic")

    sim_idx = np.linspace(0, N_POINTS - 1, SIM_SUBSET_POINTS, dtype=int)
    k_sim = k_vals[sim_idx]
    rng = np.random.default_rng(SIM_SEED)
    sse_sim_vals = np.empty(k_sim.size, dtype=np.float64)
    ssi_sim_vals = np.empty(k_sim.size, dtype=np.float64)
    for i, k in enumerate(tqdm(k_sim, desc="fig3 sims")):
        sse_sim_vals[i] = pmo_sse(
            R0=R0,
            k=float(k),
            w=w,
            history=history,
            method="simulation",
            n_sims=SIM_N_SSE,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
        )
        ssi_sim_vals[i] = pmo_ssi(
            R0=R0,
            k=float(k),
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

    # Analytic values at all k points; sim values only at the subset (NaN elsewhere).
    sse_sim_col = np.full(N_POINTS, np.nan)
    ssi_sim_col = np.full(N_POINTS, np.nan)
    sse_sim_col[sim_idx] = sse_sim_vals
    ssi_sim_col[sim_idx] = ssi_sim_vals

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig3_pmo_vs_k.csv"
    df = pd.DataFrame(
        {
            "k": k_vals,
            "pmo_sse": pmo_sse_vals,
            "pmo_ssi": pmo_ssi_vals,
            "sse_sim": sse_sim_col,
            "ssi_sim": ssi_sim_col,
            "F_r": F_r,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
