"""Compute and save results for Figure 1: PMO vs r."""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG1_R_MAX,
    RESULTS_DIR,
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

R0: float = DEFAULT_R0
K: float = DEFAULT_K
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
R_MAX: int = FIG1_R_MAX


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)

    r_vals = np.arange(0, R_MAX + 1)
    pmo_sse_vals = np.empty(r_vals.size, dtype=np.float64)
    pmo_ssi_vals = np.empty(r_vals.size, dtype=np.float64)
    sse_sim_vals = np.empty(r_vals.size, dtype=np.float64)
    ssi_sim_vals = np.empty(r_vals.size, dtype=np.float64)
    rng = np.random.default_rng(SIM_SEED)
    for i, r in enumerate(tqdm(r_vals, desc="fig1 sims")):
        history = np.array([1] + [0] * int(r), dtype=np.int64)
        pmo_sse_vals[i] = pmo_sse(R0=R0, k=K, w=w, history=history, method="analytic")
        pmo_ssi_vals[i] = pmo_ssi(R0=R0, k=K, w=w, history=history, method="analytic")
        sse_sim_vals[i] = pmo_sse(
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
        ssi_sim_vals[i] = pmo_ssi(
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig1_pmo_vs_r.csv"
    df = pd.DataFrame(
        {
            "r": r_vals,
            "pmo_sse": pmo_sse_vals,
            "pmo_ssi": pmo_ssi_vals,
            "sse_sim": sse_sim_vals,
            "ssi_sim": ssi_sim_vals,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
