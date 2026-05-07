"""Compute and save results for Figure 4: PMO comparison across incidence histories."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
HISTORIES: list[list[int]] = FIG4_HISTORIES
MCMC_DRAWS: int = FIG4_MCMC_DRAWS
MCMC_TUNE: int = FIG4_MCMC_TUNE
MCMC_CHAINS: int = FIG4_MCMC_CHAINS


def _ssi_analytic_applicable(history: list[int]) -> bool:
    return all(x == 0 for x in history[1:])


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)

    n = len(HISTORIES)
    sse_analytic = np.empty(n)
    ssi_best = np.empty(n)
    sse_sim = np.empty(n)
    ssi_sim = np.empty(n)

    for i, hist in enumerate(tqdm(HISTORIES, desc="fig4 histories")):
        history = np.array(hist, dtype=np.int64)

        sse_analytic[i] = pmo_sse(R0=R0, k=K, w=w, history=history, method="analytic")

        if _ssi_analytic_applicable(hist):
            ssi_best[i] = pmo_ssi(R0=R0, k=K, w=w, history=history, method="analytic")
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig4_pmo_comparison.csv"
    df = pd.DataFrame(
        {
            "history": [str(h) for h in HISTORIES],
            "sse_analytic": sse_analytic,
            "ssi_best": ssi_best,
            "sse_sim": sse_sim,
            "ssi_sim": ssi_sim,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
