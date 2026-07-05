"""Compute and save results for Figure 13: real-time PMO, simulated SSI outbreak.

A weekly onset history is simulated from the *SSI* model (the true model) with
the default parameters. For each week we estimate, via a bootstrap particle
filter, the probability of a major outbreak and the posterior SSE/SSI model
probabilities given the onsets observed up to that week, plus their Bayesian
model average (prior model probabilities 0.5 each). As the weeks accumulate the
model averaging should recover SSI as the more probable model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _realtime import realtime_ensemble
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    DELAY_INC_MAX,
    DELAY_INC_MEAN,
    DELAY_INC_SD,
    DELAY_TOST_MAX,
    DELAY_TOST_MEAN,
    DELAY_TOST_SD,
    FIG13_N_PARTICLES,
    FIG13_PRIOR_SSE,
    FIG13_SIM_SEED,
    RESULTS_DIR,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)

from sse_ssi_pmo import discretise_gamma, simulate_ssi_delay

R0: float = DEFAULT_R0
K: float = DEFAULT_K


def main() -> None:
    tost = discretise_gamma(
        mean=DELAY_TOST_MEAN, sd=DELAY_TOST_SD, max_val=DELAY_TOST_MAX, allow_zero=True
    )
    inc = discretise_gamma(
        mean=DELAY_INC_MEAN, sd=DELAY_INC_SD, max_val=DELAY_INC_MAX, allow_zero=False
    )

    # Simulate the observed weekly onset history from the true (SSI) model.
    counts = simulate_ssi_delay(
        R0,
        K,
        tost,
        inc,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=np.random.default_rng(FIG13_SIM_SEED),
    )
    print(
        f"simulated SSI onset counts (seed {FIG13_SIM_SEED}): "
        f"{counts.tolist()} (total {int(counts.sum())} cases)"
    )

    cols = realtime_ensemble(
        counts,
        R0=R0,
        k=K,
        tost=tost,
        inc=inc,
        n_particles=FIG13_N_PARTICLES,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        prior_sse=FIG13_PRIOR_SSE,
        rng=np.random.default_rng(SIM_SEED),
    )
    df = pd.DataFrame(cols)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig13_pmo_realtime_sim.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
