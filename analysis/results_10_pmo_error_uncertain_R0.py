"""Compute and save results for Figure 10: PMO error with uncertain R0.

Mirror of :mod:`results_6_pmo_error` but with ``R0`` integrated over a
Gamma prior (``k`` fixed). For each panel (history length L in
``FIG10_HISTORY_LENGTHS``), draw ``FIG10_N_SIM`` simulations. Each sim:

* Draws a true ``R0`` from the same Gamma prior the estimators use.
* Picks the true model SSE with prob ``FIG10_PRIOR_SSE`` (else SSI).
* Seeds with a single index case at ``t = 0`` and forward-simulates
  ``L`` weeks under the true ``(R0, model)``.
* Computes the simulation-based PMO under SSE, under SSI, and via
  Bayesian model averaging (``pmo_uncertain``), all with ``R0`` set to
  the Gamma prior. The "true" PMO is the analytic value at the per-sim
  scalar ``true_R0`` under the model that generated the history.

Setting ``USE_MCMC = True`` additionally computes ``pmo_*_mcmc`` for
every unique history per panel (one PyMC fit per estimator per unique
history; expensive). Recommended workflow when enabling: reduce
``N_SIM`` and ``HISTORY_LENGTHS`` locally so the unique-history count
stays small.

Unique observed histories are deduplicated and stacked by length, then
fed in one shot to the simulation-based ``pmo_*`` APIs: SSI and
uncertain simulation share a single rejection-sampling pass across all
histories in the panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG10_HISTORY_LENGTHS,
    FIG10_MCMC_CHAINS,
    FIG10_MCMC_DRAWS,
    FIG10_MCMC_TUNE,
    FIG10_N_SIM,
    FIG10_PRIOR_SSE,
    FIG10_R0_PRIOR_MEAN,
    FIG10_R0_PRIOR_SD,
    FIG10_SIM_BATCH,
    FIG10_SIM_MAX_ATTEMPTS,
    FIG10_SIM_N,
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
    simulate_sse,
    simulate_ssi,
)

# When True, also compute pmo_*_mcmc per unique history and add the
# matching columns to the CSV. One PyMC fit per estimator per unique
# history — expensive. Verify by toggling on with a reduced N_SIM /
# HISTORY_LENGTHS first.
USE_MCMC: bool = False

K: float = DEFAULT_K
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
PRIOR_SSE: float = FIG10_PRIOR_SSE
HISTORY_LENGTHS: list[int] = FIG10_HISTORY_LENGTHS
N_SIM: int = FIG10_N_SIM


def _simulate_history(
    model: str,
    R0: float,
    length: int,
    rng: np.random.Generator,
    w: np.ndarray,
) -> np.ndarray:
    """Return a length-``length`` history simulated under ``model`` at scalar R0.

    Mirror of ``results_6._simulate_history`` but with ``R0`` a per-sim
    argument rather than the module-level ``DEFAULT_R0``.
    """
    if model == "sse":
        traj = simulate_sse(
            R0=R0,
            k=K,
            w=w,
            init_incidence=(1,),
            threshold=SIM_THRESHOLD,
            t_max=length,
            rng=rng,
        )
    else:
        traj = simulate_ssi(
            R0=R0,
            k=K,
            w=w,
            init_incidence=1,
            threshold=SIM_THRESHOLD,
            t_max=length,
            rng=rng,
        )
    if traj.size < length:
        out = np.zeros(length, dtype=np.int64)
        out[: traj.size] = traj
        return out
    return traj


def _sim_pmos_for_panel(
    unique_histories: np.ndarray,
    R0_prior: Prior,
    w: np.ndarray,
    rng: np.random.Generator,
) -> dict[tuple[int, ...], dict[str, float]]:
    """Compute the three simulation-based PMOs for every row of ``unique_histories``.

    SSE simulation loops per row internally; SSI- and uncertain-simulation
    use the multi-history backend that matches each simulated trajectory
    against every history in one pass.
    """
    sse_s = pmo_sse(
        R0=R0_prior,
        k=K,
        w=w,
        history=unique_histories,
        method="simulation",
        n_sims=FIG10_SIM_N,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
    )
    ssi_s = pmo_ssi(
        R0=R0_prior,
        k=K,
        w=w,
        history=unique_histories,
        method="simulation",
        n_sims=FIG10_SIM_N,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
        batch_size=FIG10_SIM_BATCH,
        max_attempts=FIG10_SIM_MAX_ATTEMPTS,
    )
    unc_s = pmo_uncertain(
        R0=R0_prior,
        k=K,
        w=w,
        history=unique_histories,
        method="simulation",
        prior_sse=PRIOR_SSE,
        n_sims=FIG10_SIM_N,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
        batch_size=FIG10_SIM_BATCH,
        max_attempts=FIG10_SIM_MAX_ATTEMPTS,
    )

    columns = {
        "pmo_sse_sim": np.asarray(sse_s),
        "pmo_ssi_sim": np.asarray(ssi_s),
        "pmo_uncertain_sim": np.asarray(unc_s.pmo),
    }
    cache: dict[tuple[int, ...], dict[str, float]] = {}
    for m in range(unique_histories.shape[0]):
        key = tuple(int(x) for x in unique_histories[m])
        cache[key] = {col: float(arr[m]) for col, arr in columns.items()}
    return cache


def _mcmc_pmos_for_panel(
    unique_histories: np.ndarray,
    R0_prior: Prior,
    w: np.ndarray,
    length: int,
) -> dict[tuple[int, ...], dict[str, float]]:
    """One MCMC fit per estimator per unique history; expensive — opt-in."""
    cache: dict[tuple[int, ...], dict[str, float]] = {}
    M = unique_histories.shape[0]
    for m in tqdm(range(M), desc=f"fig10 MCMC L={length}", leave=False):
        hist = unique_histories[m]
        sse_m = pmo_sse(
            R0=R0_prior,
            k=K,
            w=w,
            history=hist,
            method="mcmc",
            draws=FIG10_MCMC_DRAWS,
            tune=FIG10_MCMC_TUNE,
            chains=FIG10_MCMC_CHAINS,
            target_accept=0.95,
            progressbar=False,
        )
        ssi_m = pmo_ssi(
            R0=R0_prior,
            k=K,
            w=w,
            history=hist,
            method="mcmc",
            draws=FIG10_MCMC_DRAWS,
            tune=FIG10_MCMC_TUNE,
            chains=FIG10_MCMC_CHAINS,
            target_accept=0.99,
            progressbar=False,
        )
        unc_m = pmo_uncertain(
            R0=R0_prior,
            k=K,
            w=w,
            history=hist,
            method="mcmc",
            prior_sse=PRIOR_SSE,
            draws=FIG10_MCMC_DRAWS,
            tune=FIG10_MCMC_TUNE,
            chains=FIG10_MCMC_CHAINS,
            target_accept=0.99,
            progressbar=False,
        )
        key = tuple(int(x) for x in hist)
        cache[key] = {
            "pmo_sse_mcmc": float(np.asarray(sse_m)),
            "pmo_ssi_mcmc": float(np.asarray(ssi_m)),
            "pmo_uncertain_mcmc": float(unc_m.pmo),
        }
    return cache


def _pmo_true(true_model: str, R0_true: float, w: np.ndarray, hist: np.ndarray) -> float:
    """Analytic PMO at the per-sim scalar ``R0_true`` under the true model."""
    if true_model == "sse":
        val = pmo_sse(R0=R0_true, k=K, w=w, history=hist, method="analytic")
    else:
        val = pmo_ssi(R0=R0_true, k=K, w=w, history=hist, method="analytic")
    return float(np.asarray(val))


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)
    R0_prior = Prior.gamma(mean=FIG10_R0_PRIOR_MEAN, sd=FIG10_R0_PRIOR_SD)

    # 1. Per panel: draw (true_R0, true_model) and forward-simulate the history.
    panel_sims: list[tuple[int, list[tuple[str, float, np.ndarray]]]] = []
    for length in HISTORY_LENGTHS:
        true_R0_arr = R0_prior.sample(N_SIM, rng)
        true_sse = rng.random(N_SIM) < PRIOR_SSE
        pairs: list[tuple[str, float, np.ndarray]] = []
        for is_sse, R0_true in zip(
            tqdm(true_sse, desc=f"fig10 simulate L={length}", leave=False),
            true_R0_arr,
            strict=True,
        ):
            tm = "sse" if is_sse else "ssi"
            hist = _simulate_history(tm, float(R0_true), length, rng, w)
            pairs.append((tm, float(R0_true), hist))
        panel_sims.append((length, pairs))

    # 2. Per panel: dedupe histories; run simulation estimators on the stack;
    #    optionally run MCMC once per unique history.
    sim_cache: dict[tuple[int, ...], dict[str, float]] = {}
    mcmc_cache: dict[tuple[int, ...], dict[str, float]] = {}
    for length, pairs in panel_sims:
        seen: set[tuple[int, ...]] = set()
        rows: list[np.ndarray] = []
        for _, _, hist in pairs:
            key = tuple(int(x) for x in hist)
            if key not in seen:
                seen.add(key)
                rows.append(hist)
        unique_histories = np.stack(rows, axis=0)
        print(f"fig10 L={length}: {len(rows)} unique histories of length {length}")
        sim_cache.update(_sim_pmos_for_panel(unique_histories, R0_prior, w, rng))
        if USE_MCMC:
            mcmc_cache.update(_mcmc_pmos_for_panel(unique_histories, R0_prior, w, length))

    # 3. Per-sim records: pmo_true depends on the per-sim scalar true_R0,
    #    so it can't be deduped by history alone.
    records: list[dict] = []
    for length, pairs in panel_sims:
        for sim_idx, (true_model, R0_true, hist) in enumerate(pairs):
            key = tuple(int(x) for x in hist)
            row: dict = {
                "history_length": length,
                "sim_index": sim_idx,
                "true_model": true_model,
                "true_R0": R0_true,
                "history": " ".join(str(int(x)) for x in hist),
                "pmo_true": _pmo_true(true_model, R0_true, w, hist),
                **sim_cache[key],
            }
            if USE_MCMC:
                row.update(mcmc_cache[key])
            records.append(row)

    df = pd.DataFrame.from_records(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig10_pmo_error_uncertain_R0.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
