"""Compute and save results for Figure 7: PMO error across a 5-model ensemble.

Same structure as Figure 6 (``results_6_pmo_error.py``) but the candidate
ensemble is wider:

* SSE with ``k = DEFAULT_K`` (0.18)
* SSE with ``k = FIG7_K_LARGE`` (1.0)
* SSI with ``k = DEFAULT_K``
* SSI with ``k = FIG7_K_LARGE``
* Poisson offspring (the ``k -> infty`` limit)

For each panel (history length ``L``) and each of ``FIG7_N_SIM``
simulations, draw the true model uniformly at random from the five and
forward-simulate ``L`` weeks from a single index case. Then compute the
analytic PMO under each candidate model and the model-averaged PMO via
``pmo_ensemble``. The "true" PMO for each row is the analytic PMO under
the model that generated it.

Only analytic estimators are computed (``pmo_ensemble`` currently
supports analytic and MCMC only; for fig7 we stay closed-form).
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
    FIG7_HISTORY_LENGTHS,
    FIG7_K_LARGE,
    FIG7_N_SIM,
    RESULTS_DIR,
    SIM_SEED,
    SIM_THRESHOLD,
)
from tqdm.auto import tqdm

from sse_ssi_pmo import (
    discretise_gamma,
    pmo_ensemble,
    simulate_poisson,
    simulate_sse,
    simulate_ssi,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
K_LARGE: float = FIG7_K_LARGE
SI_MEAN: float = DEFAULT_SI_MEAN
SI_SD: float = DEFAULT_SI_SD
SI_MAX: int = DEFAULT_SI_MAX
HISTORY_LENGTHS: list[int] = FIG7_HISTORY_LENGTHS
N_SIM: int = FIG7_N_SIM

# Identifiers for the five ensemble members; used as both column names in the
# output CSV and keys in the model-spec list.
MODEL_IDS: list[str] = ["sse_k", "sse_k_large", "ssi_k", "ssi_k_large", "poisson"]
MODEL_SPECS: list[dict] = [
    {"model": "sse", "R0": R0, "k": K},
    {"model": "sse", "R0": R0, "k": K_LARGE},
    {"model": "ssi", "R0": R0, "k": K},
    {"model": "ssi", "R0": R0, "k": K_LARGE},
    {"model": "poisson", "R0": R0},
]
PRIORS: list[float] = [1.0 / len(MODEL_IDS)] * len(MODEL_IDS)


def _simulate_history(
    model_id: str, length: int, rng: np.random.Generator, w: np.ndarray
) -> np.ndarray:
    """Return a length-``length`` history simulated under ``model_id`` from I_0 = 1.

    Right-pads with zeros if the simulator terminates early on a major
    outbreak or extinction. ``model_id`` is one of :data:`MODEL_IDS`.
    """
    if model_id == "sse_k":
        traj = simulate_sse(
            R0=R0,
            k=K,
            w=w,
            init_incidence=(1,),
            threshold=SIM_THRESHOLD,
            t_max=length,
            rng=rng,
        )
    elif model_id == "sse_k_large":
        traj = simulate_sse(
            R0=R0,
            k=K_LARGE,
            w=w,
            init_incidence=(1,),
            threshold=SIM_THRESHOLD,
            t_max=length,
            rng=rng,
        )
    elif model_id == "ssi_k":
        traj = simulate_ssi(
            R0=R0,
            k=K,
            w=w,
            init_incidence=1,
            threshold=SIM_THRESHOLD,
            t_max=length,
            rng=rng,
        )
    elif model_id == "ssi_k_large":
        traj = simulate_ssi(
            R0=R0,
            k=K_LARGE,
            w=w,
            init_incidence=1,
            threshold=SIM_THRESHOLD,
            t_max=length,
            rng=rng,
        )
    elif model_id == "poisson":
        traj = simulate_poisson(
            R0=R0,
            w=w,
            init_incidence=(1,),
            threshold=SIM_THRESHOLD,
            t_max=length,
            rng=rng,
        )
    else:
        raise ValueError(f"unknown model_id {model_id!r}")
    if traj.size < length:
        out = np.zeros(length, dtype=np.int64)
        out[: traj.size] = traj
        return out
    return traj


def _pmos_for_panel(
    unique_histories: np.ndarray, w: np.ndarray
) -> dict[tuple[int, ...], dict[str, float]]:
    """Compute per-model and ensemble PMOs for every unique history in a panel.

    Returns a dict keyed by the row's tuple-of-ints; each value contains
    ``pmo_<model_id>`` for each ensemble member and ``pmo_ensemble``.
    """
    res = pmo_ensemble(
        models=MODEL_SPECS,
        priors=PRIORS,
        w=w,
        history=unique_histories,
        method="analytic",
    )
    # 2-D ``history`` input ⇒ res.pmo: (M,); res.pmo_per_model: (M, N). np.asarray
    # narrows the public ``float | NDArray`` return type for the type-checker.
    pmo_arr = np.asarray(res.pmo)
    pmo_per_model_arr = np.asarray(res.pmo_per_model)
    cache: dict[tuple[int, ...], dict[str, float]] = {}
    for m in range(unique_histories.shape[0]):
        key = tuple(int(x) for x in unique_histories[m])
        row: dict[str, float] = {
            f"pmo_{mid}": float(pmo_per_model_arr[m, i]) for i, mid in enumerate(MODEL_IDS)
        }
        row["pmo_ensemble"] = float(pmo_arr[m])
        cache[key] = row
    return cache


def main() -> None:
    w = discretise_gamma(mean=SI_MEAN, sd=SI_SD, max_val=SI_MAX)
    rng = np.random.default_rng(SIM_SEED)

    # 1. Generate per-sim (true model, history) pairs for each panel.
    panel_sims: list[tuple[int, list[tuple[str, np.ndarray]]]] = []
    for length in HISTORY_LENGTHS:
        true_models = rng.choice(MODEL_IDS, size=N_SIM)  # uniform prior over models
        pairs: list[tuple[str, np.ndarray]] = []
        for tm in tqdm(true_models, desc=f"fig7 simulate L={length}", leave=False):
            pairs.append((str(tm), _simulate_history(str(tm), length, rng, w)))
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
        print(f"fig7 L={length}: {len(rows)} unique histories of length {length}")
        pmo_cache.update(_pmos_for_panel(unique_histories, w))

    # 3. Assemble per-sim records by looking up the cached PMOs.
    records: list[dict] = []
    for length, pairs in panel_sims:
        for sim_idx, (true_model, hist) in enumerate(pairs):
            cached = pmo_cache[tuple(int(x) for x in hist)]
            pmo_true = cached[f"pmo_{true_model}"]
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
    out_path = RESULTS_DIR / "fig7_pmo_ensemble_error.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
