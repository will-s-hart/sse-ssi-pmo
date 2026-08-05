"""Compute and save results for Figure 6: PMO error under model misspecification.

For each panel (history length L in ``FIG6_HISTORY_LENGTHS``), draw
``FIG6_N_SIM`` simulations. Each sim:

* Picks the true model SSE with prob ``FIG6_PRIOR_SSE`` (else SSI).
* Seeds with a single index case at ``t = 0`` and forward-simulates ``L``
  weeks under the chosen model.
* Computes the simulation-based PMO under SSE, under SSI, and via Bayesian
  model averaging (``pmo_uncertain``). When every observed history in the
  panel is non-"general" (so the SSI/uncertain closed-forms apply), the
  analytic PMOs are also computed. The "true" PMO is the cached value
  under the model that generated the history — analytic when the panel
  supports it, simulation otherwise.

Setting ``USE_MCMC = True`` additionally computes ``pmo_*_mcmc`` for
every unique history in panels where the analytic closed-form is not
available (one PyMC fit per estimator per unique history; expensive).

Unique observed histories are deduplicated and stacked by length, then
fed in one shot to the multi-history ``pmo_*`` APIs: SSI and uncertain
simulation share a single rejection-sampling pass across all histories
in the panel (each matching trajectory contributes to whichever history
it happens to match), so the per-panel cost scales with the *union*
acceptance rate rather than per-history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis._shared.defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    SIM_BATCH_SSI,
    SIM_MAX_ATTEMPTS_SSI,
    SIM_N_SSE,
    SIM_N_SSI,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from analysis.base.defaults import (
    FIG4_MCMC_CHAINS,
    FIG4_MCMC_DRAWS,
    FIG4_MCMC_TUNE,
    FIG5_SIM_BATCH,
    FIG5_SIM_MAX_ATTEMPTS,
    FIG5_SIM_N,
    FIG6_HISTORY_LENGTHS,
    FIG6_N_SIM,
    FIG6_PRIOR_SSE,
    RESULTS_DIR,
)
from sse_ssi_pmo import (
    discretise_gamma,
    pmo_sse,
    pmo_ssi,
    pmo_uncertain,
    simulate_sse,
    simulate_ssi,
)
from sse_ssi_pmo._history import classify_history

# When True, also compute pmo_*_mcmc for every unique history in panels
# where the analytic closed-form isn't available. One PyMC fit per
# estimator per unique history — expensive. Verify by toggling on with a
# reduced N_SIM / HISTORY_LENGTHS first.
USE_MCMC: bool = False

R0: float = DEFAULT_R0
K: float = DEFAULT_K
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
PRIOR_SSE: float = FIG6_PRIOR_SSE
HISTORY_LENGTHS: list[int] = FIG6_HISTORY_LENGTHS
N_SIM: int = FIG6_N_SIM


def _simulate_history(
    model: str, length: int, rng: np.random.Generator, w: np.ndarray
) -> np.ndarray:
    """Return a length-``length`` history simulated under ``model`` from I_0 = 1.

    Uses ``SIM_THRESHOLD`` and ``t_max = length``; the public simulators may
    terminate early on major outbreak, so we right-pad with zeros to keep a
    fixed-length record. Extinction can't trigger inside this window because
    I_0 = 1 keeps the L-step rolling sum positive.
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


def _analytic_possible(unique_histories: np.ndarray) -> bool:
    """True iff every row of ``unique_histories`` admits the SSI closed-form.

    The SSI / uncertain analytic backends only cover the three "special"
    history shapes (``day0_only`` / ``one_later`` / ``two_later``). Any row
    classified ``general`` would raise inside ``pmo_ssi(method="analytic")``,
    so the whole panel falls back to simulation.
    """
    return all(classify_history(h)["kind"] != "general" for h in unique_histories)


def _pmos_for_panel(
    unique_histories: np.ndarray,
    w: np.ndarray,
    rng: np.random.Generator,
    *,
    include_analytic: bool,
) -> dict[tuple[int, ...], dict[str, float]]:
    """Compute simulation PMOs for every row; optionally also analytic PMOs.

    Sim block always runs. Analytic block runs only when ``include_analytic``
    is True (i.e., every row in the panel is non-"general" so SSI/uncertain
    closed-forms apply).
    """
    columns: dict[str, np.ndarray] = {}
    if include_analytic:
        columns["pmo_sse"] = np.asarray(
            pmo_sse(R0=R0, k=K, w=w, history=unique_histories, method="analytic")
        )
        columns["pmo_ssi"] = np.asarray(
            pmo_ssi(R0=R0, k=K, w=w, history=unique_histories, method="analytic")
        )
        columns["pmo_uncertain"] = np.asarray(
            pmo_uncertain(
                R0=R0,
                k=K,
                w=w,
                history=unique_histories,
                method="analytic",
                prior_sse=PRIOR_SSE,
            ).pmo
        )
    sse_s = pmo_sse(
        R0=R0,
        k=K,
        w=w,
        history=unique_histories,
        method="simulation",
        n_sims=SIM_N_SSE,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
    )
    ssi_s = pmo_ssi(
        R0=R0,
        k=K,
        w=w,
        history=unique_histories,
        method="simulation",
        n_sims=SIM_N_SSI,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
        batch_size=SIM_BATCH_SSI,
        max_attempts=SIM_MAX_ATTEMPTS_SSI,
    )
    unc_s = pmo_uncertain(
        R0=R0,
        k=K,
        w=w,
        history=unique_histories,
        method="simulation",
        prior_sse=PRIOR_SSE,
        n_sims=FIG5_SIM_N,
        threshold=SIM_THRESHOLD,
        t_max=SIM_T_MAX,
        rng=rng,
        batch_size=FIG5_SIM_BATCH,
        max_attempts=FIG5_SIM_MAX_ATTEMPTS,
    )
    columns["pmo_sse_sim"] = np.asarray(sse_s)
    columns["pmo_ssi_sim"] = np.asarray(ssi_s)
    columns["pmo_uncertain_sim"] = np.asarray(unc_s.pmo)

    cache: dict[tuple[int, ...], dict[str, float]] = {}
    for m in range(unique_histories.shape[0]):
        key = tuple(int(x) for x in unique_histories[m])
        cache[key] = {col: float(arr[m]) for col, arr in columns.items()}
    return cache


def _mcmc_pmos_for_panel(
    unique_histories: np.ndarray,
    w: np.ndarray,
    length: int,
) -> dict[tuple[int, ...], dict[str, float]]:
    """One MCMC fit per estimator per unique history; expensive — opt-in."""
    cache: dict[tuple[int, ...], dict[str, float]] = {}
    M = unique_histories.shape[0]
    for m in tqdm(range(M), desc=f"fig6 MCMC L={length}", leave=False):
        hist = unique_histories[m]
        sse_m = pmo_sse(
            R0=R0,
            k=K,
            w=w,
            history=hist,
            method="mcmc",
            draws=FIG4_MCMC_DRAWS,
            tune=FIG4_MCMC_TUNE,
            chains=FIG4_MCMC_CHAINS,
            target_accept=0.95,
            progressbar=False,
        )
        ssi_m = pmo_ssi(
            R0=R0,
            k=K,
            w=w,
            history=hist,
            method="mcmc",
            draws=FIG4_MCMC_DRAWS,
            tune=FIG4_MCMC_TUNE,
            chains=FIG4_MCMC_CHAINS,
            target_accept=0.99,
            progressbar=False,
        )
        unc_m = pmo_uncertain(
            R0=R0,
            k=K,
            w=w,
            history=hist,
            method="mcmc",
            prior_sse=PRIOR_SSE,
            draws=FIG4_MCMC_DRAWS,
            tune=FIG4_MCMC_TUNE,
            chains=FIG4_MCMC_CHAINS,
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


def _pmo_true(true_model: str, cached: dict[str, float]) -> float:
    """Cached PMO under the true model.

    Analytic when the panel supports it (``pmo_sse`` / ``pmo_ssi`` keys
    populated); simulation otherwise. Picking from the panel cache means a
    sim-only panel's true-model residual is 0 by construction — the
    headline result is the wrong-model bars sitting away from zero, not
    sim-vs-analytic noise.
    """
    if true_model == "sse":
        return cached["pmo_sse"] if "pmo_sse" in cached else cached["pmo_sse_sim"]
    return cached["pmo_ssi"] if "pmo_ssi" in cached else cached["pmo_ssi_sim"]


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)

    # 1. Generate per-sim (true model, history) pairs for each panel.
    panel_sims: list[tuple[int, list[tuple[str, np.ndarray]]]] = []
    for length in HISTORY_LENGTHS:
        true_models = rng.random(N_SIM) < PRIOR_SSE
        pairs: list[tuple[str, np.ndarray]] = []
        for is_sse in tqdm(true_models, desc=f"fig6 simulate L={length}", leave=False):
            tm = "sse" if is_sse else "ssi"
            pairs.append((tm, _simulate_history(tm, length, rng, w)))
        panel_sims.append((length, pairs))

    # 2. Dedupe histories per panel; run sim (+ analytic when possible,
    #    + MCMC when opted in and analytic isn't).
    sim_cache: dict[tuple[int, ...], dict[str, float]] = {}
    mcmc_cache: dict[tuple[int, ...], dict[str, float]] = {}
    for length, pairs in panel_sims:
        seen: set[tuple[int, ...]] = set()
        rows: list[np.ndarray] = []
        for _, hist in pairs:
            key = tuple(int(x) for x in hist)
            if key not in seen:
                seen.add(key)
                rows.append(hist)
        unique_histories = np.stack(rows, axis=0)
        analytic_ok = _analytic_possible(unique_histories)
        if analytic_ok:
            tag = "analytic+sim"
        elif USE_MCMC:
            tag = "sim+mcmc"
        else:
            tag = "sim-only"
        print(f"fig6 L={length}: {len(rows)} unique histories ({tag})")
        sim_cache.update(_pmos_for_panel(unique_histories, w, rng, include_analytic=analytic_ok))
        if not analytic_ok and USE_MCMC:
            mcmc_cache.update(_mcmc_pmos_for_panel(unique_histories, w, length))

    # 3. Assemble per-sim records by looking up the cached PMOs.
    records: list[dict] = []
    for length, pairs in panel_sims:
        for sim_idx, (true_model, hist) in enumerate(pairs):
            key = tuple(int(x) for x in hist)
            cached = sim_cache[key]
            row: dict = {
                "history_length": length,
                "sim_index": sim_idx,
                "true_model": true_model,
                "history": " ".join(str(int(x)) for x in hist),
                "pmo_true": _pmo_true(true_model, cached),
                **cached,
            }
            if key in mcmc_cache:
                row.update(mcmc_cache[key])
            records.append(row)

    df = pd.DataFrame.from_records(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig6_pmo_error.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
