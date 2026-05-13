"""Compute and save results for Figure 9: model-averaged PMO with uncertain R0.

Mirror of :mod:`results_5_pmo_uncertain` but with ``R0`` integrated over a
Gamma prior (``k`` fixed). ``method='analytic'`` cannot accept Priors, so
the comparison is between ``method='mcmc'`` and ``method='simulation'``.

Outputs ``results/fig9_pmo_uncertain_combined.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG9_HISTORIES,
    FIG9_MCMC_CHAINS,
    FIG9_MCMC_DRAWS,
    FIG9_MCMC_TUNE,
    FIG9_PRIOR_SSE,
    FIG9_R0_PRIOR_MEAN,
    FIG9_R0_PRIOR_SD,
    FIG9_SIM_BATCH,
    FIG9_SIM_MAX_ATTEMPTS,
    FIG9_SIM_N,
    RESULTS_DIR,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from tqdm.auto import tqdm

from sse_ssi_pmo import (
    Prior,
    discretise_gamma,
    pmo_sse,
    pmo_ssi,
    pmo_uncertain,
)

K: float = DEFAULT_K
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
HISTORIES: list[list[int]] = FIG9_HISTORIES
PRIOR_SSE: float = FIG9_PRIOR_SSE


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)
    R0_prior = Prior.gamma(mean=FIG9_R0_PRIOR_MEAN, sd=FIG9_R0_PRIOR_SD)

    n = len(HISTORIES)
    pmo_sse_mcmc = np.empty(n)
    pmo_ssi_mcmc = np.empty(n)
    pmo_sse_sim = np.empty(n)
    pmo_ssi_sim = np.empty(n)
    uncertain_mcmc = np.empty(n)
    uncertain_sim = np.empty(n)
    posterior_sse = np.empty(n)

    for i, hist in enumerate(tqdm(HISTORIES, desc="fig9 histories")):
        history = np.array(hist, dtype=np.int64)

        pmo_sse_mcmc[i] = pmo_sse(
            R0=R0_prior,
            k=K,
            w=w,
            history=history,
            method="mcmc",
            draws=FIG9_MCMC_DRAWS,
            tune=FIG9_MCMC_TUNE,
            chains=FIG9_MCMC_CHAINS,
            target_accept=0.95,
            progressbar=False,
        )
        pmo_ssi_mcmc[i] = pmo_ssi(
            R0=R0_prior,
            k=K,
            w=w,
            history=history,
            method="mcmc",
            draws=FIG9_MCMC_DRAWS,
            tune=FIG9_MCMC_TUNE,
            chains=FIG9_MCMC_CHAINS,
            target_accept=0.99,
            progressbar=False,
        )
        result = pmo_uncertain(
            R0=R0_prior,
            k=K,
            w=w,
            history=history,
            method="mcmc",
            prior_sse=PRIOR_SSE,
            draws=FIG9_MCMC_DRAWS,
            tune=FIG9_MCMC_TUNE,
            chains=FIG9_MCMC_CHAINS,
            target_accept=0.99,
            progressbar=False,
        )
        uncertain_mcmc[i] = result.pmo
        posterior_sse[i] = result.posterior_sse

        pmo_sse_sim[i] = pmo_sse(
            R0=R0_prior,
            k=K,
            w=w,
            history=history,
            method="simulation",
            n_sims=FIG9_SIM_N,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
        )
        pmo_ssi_sim[i] = pmo_ssi(
            R0=R0_prior,
            k=K,
            w=w,
            history=history,
            method="simulation",
            n_sims=FIG9_SIM_N,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
            batch_size=FIG9_SIM_BATCH,
            max_attempts=FIG9_SIM_MAX_ATTEMPTS,
        )
        uncertain_sim[i] = pmo_uncertain(
            R0=R0_prior,
            k=K,
            w=w,
            history=history,
            method="simulation",
            prior_sse=PRIOR_SSE,
            n_sims=FIG9_SIM_N,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
            batch_size=FIG9_SIM_BATCH,
            max_attempts=FIG9_SIM_MAX_ATTEMPTS,
        ).pmo

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig9_pmo_uncertain_combined.csv"
    df = pd.DataFrame(
        {
            "history": [str(h) for h in HISTORIES],
            "pmo_sse_mcmc": pmo_sse_mcmc,
            "pmo_ssi_mcmc": pmo_ssi_mcmc,
            "uncertain_mcmc": uncertain_mcmc,
            "pmo_sse_sim": pmo_sse_sim,
            "pmo_ssi_sim": pmo_ssi_sim,
            "uncertain_sim": uncertain_sim,
            "posterior_sse": posterior_sse,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
