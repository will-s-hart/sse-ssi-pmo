"""Compute and save results for Figure 6: PMO error under model misspecification.

For each panel (history length L in ``FIG6_HISTORY_LENGTHS``), draw
``FIG6_N_SIM`` simulations. Each sim:

* Picks the true model SSE with prob ``FIG6_PRIOR_SSE`` (else SSI).
* Seeds with a single index case at ``t = 0`` and forward-simulates ``L``
  weeks under the chosen model.
* Computes the analytic PMO under SSE, under SSI, and via Bayesian
  model averaging (``pmo_uncertain``); also the simulation-based PMO for
  each of the three estimators. The "true" PMO is the analytic value
  under the model that generated the history.

Unique observed histories are deduplicated and stacked by length, then
fed in one shot to the multi-history `pmo_*` APIs: SSI and uncertain
simulation share a single rejection-sampling pass across all histories
in the panel (each matching trajectory contributes to whichever history
it happens to match), so the per-panel cost scales with the *union*
acceptance rate rather than per-history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis_defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    DEFAULT_SI_MAX,
    DEFAULT_SI_MEAN,
    DEFAULT_SI_SD,
    FIG5_SIM_BATCH,
    FIG5_SIM_MAX_ATTEMPTS,
    FIG5_SIM_N,
    FIG6_HISTORY_LENGTHS,
    FIG6_N_SIM,
    FIG6_PRIOR_SSE,
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
    simulate_sse,
    simulate_ssi,
)

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


def _pmos_for_panel(
    unique_histories: np.ndarray, w: np.ndarray, rng: np.random.Generator
) -> dict[tuple[int, ...], dict[str, float]]:
    """Compute all six PMO values for every row of ``unique_histories``.

    Returns a dict keyed by the row's tuple-of-ints, with the six PMO
    floats per row. SSE simulation loops per row internally (no
    shared-work payoff); SSI- and uncertain-simulation use the
    multi-history backend that matches each simulated trajectory against
    every history in one pass.
    """
    sse_a = pmo_sse(R0=R0, k=K, w=w, history=unique_histories, method="analytic")
    ssi_a = pmo_ssi(R0=R0, k=K, w=w, history=unique_histories, method="analytic")
    unc_a = pmo_uncertain(
        R0=R0,
        k=K,
        w=w,
        history=unique_histories,
        method="analytic",
        prior_sse=PRIOR_SSE,
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

    # 2-D ``history`` input ⇒ each PMO call returns an (M,) array; np.asarray
    # narrows the public ``float | NDArray`` return type for the type-checker.
    columns = {
        "pmo_sse": np.asarray(sse_a),
        "pmo_ssi": np.asarray(ssi_a),
        "pmo_uncertain": np.asarray(unc_a.pmo),
        "pmo_sse_sim": np.asarray(sse_s),
        "pmo_ssi_sim": np.asarray(ssi_s),
        "pmo_uncertain_sim": np.asarray(unc_s.pmo),
    }
    cache: dict[tuple[int, ...], dict[str, float]] = {}
    for m in range(unique_histories.shape[0]):
        key = tuple(int(x) for x in unique_histories[m])
        cache[key] = {col: float(arr[m]) for col, arr in columns.items()}
    return cache


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

    # 2. Dedupe histories per panel and run the multi-history PMOs in one shot.
    pmo_cache: dict[tuple[int, ...], dict[str, float]] = {}
    for length, pairs in panel_sims:
        seen: set[tuple[int, ...]] = set()
        rows: list[np.ndarray] = []
        for _, hist in pairs:
            key = tuple(int(x) for x in hist)
            if key not in seen:
                seen.add(key)
                rows.append(hist)
        unique_histories = np.stack(rows, axis=0)
        print(f"fig6 L={length}: {len(rows)} unique histories of length {length}")
        pmo_cache.update(_pmos_for_panel(unique_histories, w, rng))

    # 3. Assemble per-sim records by looking up the cached PMOs.
    records: list[dict] = []
    for length, pairs in panel_sims:
        for sim_idx, (true_model, hist) in enumerate(pairs):
            cached = pmo_cache[tuple(int(x) for x in hist)]
            pmo_true = cached["pmo_sse"] if true_model == "sse" else cached["pmo_ssi"]
            records.append(
                {
                    "history_length": length,
                    "sim_index": sim_idx,
                    "true_model": true_model,
                    "history": " ".join(str(int(x)) for x in hist),
                    "pmo_true": pmo_true,
                    **cached,
                }
            )

    df = pd.DataFrame.from_records(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig6_pmo_error.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
