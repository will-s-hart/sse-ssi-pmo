"""Compute and save results for Figure 2: PMO vs R_0."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis._shared.defaults import (
    DEFAULT_K,
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
    FIG2_R0_MAX,
    FIG2_R0_MIN,
    RESULTS_DIR,
)
from sse_ssi_pmo import (
    cumulative,
    discretise_gamma,
    pmo_sse,
    pmo_ssi,
)

K: float = DEFAULT_K
R_WEEKS: int = DEFAULT_R_WEEKS
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
R0_MIN: float = FIG2_R0_MIN
R0_MAX: float = FIG2_R0_MAX
N_POINTS: int = DEFAULT_N_POINTS


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    F_r = float(cumulative(w)[R_WEEKS])

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

    # Analytic values at all R0 points; sim values only at the subset (NaN elsewhere).
    sse_sim_col = np.full(N_POINTS, np.nan)
    ssi_sim_col = np.full(N_POINTS, np.nan)
    sse_sim_col[sim_idx] = sse_sim_vals
    ssi_sim_col[sim_idx] = ssi_sim_vals

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig2_pmo_vs_R0.csv"
    df = pd.DataFrame(
        {
            "R0": R0_vals,
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
