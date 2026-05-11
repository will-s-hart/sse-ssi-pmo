"""Compute and save results for Figure 5: Model-averaged PMO across incidence histories."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    pmo_uncertain,
)

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


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)

    n = len(HISTORIES)
    sse_analytic = np.empty(n)
    ssi_analytic = np.empty(n)
    ssi_mcmc = np.empty(n)
    uncertain_analytic = np.empty(n)
    uncertain_mcmc = np.empty(n)
    sse_sim = np.empty(n)
    ssi_sim = np.empty(n)
    uncertain_sim = np.empty(n)
    posterior_sse = np.empty(n)

    for i, hist in enumerate(tqdm(HISTORIES, desc="fig5 histories")):
        history = np.array(hist, dtype=np.int64)

        sse_analytic[i] = pmo_sse(R0=R0, k=K, w=w, history=history, method="analytic")
        ssi_analytic[i] = pmo_ssi(R0=R0, k=K, w=w, history=history, method="analytic")
        ssi_mcmc[i] = pmo_ssi(
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
        analytic_result = pmo_uncertain(
            R0=R0, k=K, w=w, history=history, method="analytic", prior_sse=PRIOR_SSE
        )
        uncertain_analytic[i] = analytic_result.pmo
        posterior_sse[i] = analytic_result.posterior_sse

        uncertain_mcmc[i] = pmo_uncertain(
            R0=R0,
            k=K,
            w=w,
            history=history,
            method="mcmc",
            prior_sse=PRIOR_SSE,
            draws=MCMC_DRAWS,
            tune=MCMC_TUNE,
            chains=MCMC_CHAINS,
            target_accept=0.99,
            progressbar=False,
        ).pmo

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig5_pmo_uncertain.csv"
    df = pd.DataFrame(
        {
            "history": [str(h) for h in HISTORIES],
            "sse_analytic": sse_analytic,
            "ssi_analytic": ssi_analytic,
            "ssi_mcmc": ssi_mcmc,
            "uncertain_analytic": uncertain_analytic,
            "uncertain_mcmc": uncertain_mcmc,
            "sse_sim": sse_sim,
            "ssi_sim": ssi_sim,
            "uncertain_sim": uncertain_sim,
            "posterior_sse": posterior_sse,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
