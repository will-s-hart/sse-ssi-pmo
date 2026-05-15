"""Compute and save results for Figure 8: PMO vs r with uncertain R0.

For each ``r in {0, ..., R_MAX}`` with the history ``[1] + [0]*r``:

* Fit ``fit_sse`` and ``fit_ssi`` once each with ``R0=None`` (so the
  ``rep_no`` prior is fitted from the supplied Gamma prior). Record the
  posterior median and 95% credible interval of ``R0`` from each.
* Reuse the same traces to compute ``pmo_sse(method='mcmc', datatree=...)``
  and ``pmo_ssi(method='mcmc', datatree=...)`` — so the panel-A posterior
  and the panel-B PMO are produced from a single MCMC fit per model.
* Cross-check with ``method='simulation'`` (rejection sampling over the
  same R0 prior; ``k`` stays fixed throughout the figure).

Outputs ``results/fig8_pmo_uncertain_R0.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG8_CRI_LEVEL,
    FIG8_MCMC_CHAINS,
    FIG8_MCMC_DRAWS,
    FIG8_MCMC_TUNE,
    FIG8_R0_PRIOR_MEAN,
    FIG8_R0_PRIOR_SD,
    FIG8_R_MAX,
    FIG8_SIM_BATCH,
    FIG8_SIM_MAX_ATTEMPTS,
    FIG8_SIM_N,
    RESULTS_DIR,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from tqdm.auto import tqdm

from sse_ssi_pmo import (
    Prior,
    discretise_gamma,
    fit_sse,
    fit_ssi,
    pmo_sse,
    pmo_ssi,
)

K: float = DEFAULT_K
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
R_MAX: int = FIG8_R_MAX
R0_PRIOR_MEAN: float = FIG8_R0_PRIOR_MEAN
R0_PRIOR_SD: float = FIG8_R0_PRIOR_SD
CRI: float = FIG8_CRI_LEVEL


def _cri_bounds(samples: np.ndarray, level: float) -> tuple[float, float]:
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(samples, lo)), float(np.quantile(samples, hi))


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)
    R0_prior = Prior.gamma(mean=R0_PRIOR_MEAN, sd=R0_PRIOR_SD)
    # Build matching PyMC prior dict so fit_sse/fit_ssi use the same Gamma
    # rather than the library DEFAULT_PRIORS LogNormal.
    pymc_priors = {
        "rep_no": (
            __import__("pymc").Gamma,
            dict(R0_prior.pymc_params),
        ),
    }

    r_vals = np.arange(0, R_MAX + 1)
    n = r_vals.size
    sse_med = np.empty(n)
    sse_lo = np.empty(n)
    sse_hi = np.empty(n)
    ssi_med = np.empty(n)
    ssi_lo = np.empty(n)
    ssi_hi = np.empty(n)
    pmo_sse_mcmc = np.empty(n)
    pmo_ssi_mcmc = np.empty(n)
    pmo_sse_sim = np.empty(n)
    pmo_ssi_sim = np.empty(n)

    for i, r in enumerate(tqdm(r_vals, desc="fig8 r values")):
        history = np.array([1] + [0] * int(r), dtype=np.int64)

        # Fit once per model with R0 as a free parameter under the Gamma prior.
        trace_sse = fit_sse(
            history,
            w,
            R0=None,
            k=K,
            priors=pymc_priors,
            draws=FIG8_MCMC_DRAWS,
            tune=FIG8_MCMC_TUNE,
            chains=FIG8_MCMC_CHAINS,
            target_accept=0.95,
            progressbar=False,
        )
        trace_ssi = fit_ssi(
            history,
            w,
            R0=None,
            k=K,
            priors=pymc_priors,
            draws=FIG8_MCMC_DRAWS,
            tune=FIG8_MCMC_TUNE,
            chains=FIG8_MCMC_CHAINS,
            target_accept=0.99,
            progressbar=False,
        )
        sse_samples = trace_sse["posterior"].ds["rep_no"].values.reshape(-1)
        ssi_samples = trace_ssi["posterior"].ds["rep_no"].values.reshape(-1)
        sse_med[i] = float(np.median(sse_samples))
        sse_lo[i], sse_hi[i] = _cri_bounds(sse_samples, CRI)
        ssi_med[i] = float(np.median(ssi_samples))
        ssi_lo[i], ssi_hi[i] = _cri_bounds(ssi_samples, CRI)

        # PMO via MCMC, reusing the trace from the posterior extraction above.
        pmo_sse_mcmc[i] = pmo_sse(
            R0=R0_prior, k=K, w=w, history=history, method="mcmc", datatree=trace_sse
        )
        pmo_ssi_mcmc[i] = pmo_ssi(
            R0=R0_prior, k=K, w=w, history=history, method="mcmc", datatree=trace_ssi
        )

        # Simulation cross-check (independent of MCMC trace).
        pmo_sse_sim[i] = pmo_sse(
            R0=R0_prior,
            k=K,
            w=w,
            history=history,
            method="simulation",
            n_sims=FIG8_SIM_N,
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
            n_sims=FIG8_SIM_N,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
            batch_size=FIG8_SIM_BATCH,
            max_attempts=FIG8_SIM_MAX_ATTEMPTS,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig8_pmo_uncertain_R0.csv"
    df = pd.DataFrame(
        {
            "r": r_vals,
            "R0_med_sse": sse_med,
            "R0_lo_sse": sse_lo,
            "R0_hi_sse": sse_hi,
            "R0_med_ssi": ssi_med,
            "R0_lo_ssi": ssi_lo,
            "R0_hi_ssi": ssi_hi,
            "pmo_sse_mcmc": pmo_sse_mcmc,
            "pmo_ssi_mcmc": pmo_ssi_mcmc,
            "pmo_sse_sim": pmo_sse_sim,
            "pmo_ssi_sim": pmo_ssi_sim,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
