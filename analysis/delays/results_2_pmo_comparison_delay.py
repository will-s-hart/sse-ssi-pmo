"""Compute and save results for Figure 2: onset-anchored PMO across histories.

Symptom-onset-anchored analogue of Figure 4. The observed histories are symptom
onsets; PMO is estimated by simulation only (rejection sampling) for the
onset-anchored SSE and SSI models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis._shared.defaults import (
    DEFAULT_K,
    DEFAULT_R0,
    DELAY_INC_MAX,
    DELAY_INC_MEAN,
    DELAY_INC_SD,
    SIM_SEED,
    SIM_T_MAX,
    SIM_THRESHOLD,
)
from analysis.delays.defaults import (
    DELAY_SIM_BATCH,
    DELAY_SIM_MAX_ATTEMPTS,
    DELAY_SIM_N,
    DELAY_TOST_MAX,
    DELAY_TOST_MEAN,
    DELAY_TOST_SD,
    FIG2_HISTORIES,
    RESULTS_DIR,
)
from sse_ssi_pmo import (
    discretise_gamma,
    pmo_sse_delay,
    pmo_ssi_delay,
)

R0: float = DEFAULT_R0
K: float = DEFAULT_K
HISTORIES: list[list[int]] = FIG2_HISTORIES


def main() -> None:
    tost = discretise_gamma(
        mean=DELAY_TOST_MEAN, sd=DELAY_TOST_SD, max_val=DELAY_TOST_MAX, allow_zero=True
    )
    inc = discretise_gamma(
        mean=DELAY_INC_MEAN, sd=DELAY_INC_SD, max_val=DELAY_INC_MAX, allow_zero=False
    )
    onset_si_mean = (tost * np.arange(tost.size)).sum() + (inc * np.arange(1, inc.size + 1)).sum()
    print(f"onset-anchored SI mean (TOST + incubation) = {onset_si_mean:.2f} weeks")
    rng = np.random.default_rng(SIM_SEED)

    n = len(HISTORIES)
    sse_sim = np.empty(n)
    ssi_sim = np.empty(n)

    for i, hist in enumerate(tqdm(HISTORIES, desc="fig12 histories")):
        history = np.array(hist, dtype=np.int64)
        sse_sim[i] = pmo_sse_delay(
            R0=R0,
            k=K,
            tost=tost,
            inc=inc,
            history=history,
            n_sims=DELAY_SIM_N,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
            batch_size=DELAY_SIM_BATCH,
            max_attempts=DELAY_SIM_MAX_ATTEMPTS,
        )
        ssi_sim[i] = pmo_ssi_delay(
            R0=R0,
            k=K,
            tost=tost,
            inc=inc,
            history=history,
            n_sims=DELAY_SIM_N,
            threshold=SIM_THRESHOLD,
            t_max=SIM_T_MAX,
            rng=rng,
            batch_size=DELAY_SIM_BATCH,
            max_attempts=DELAY_SIM_MAX_ATTEMPTS,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig2_pmo_comparison_delay.csv"
    df = pd.DataFrame(
        {
            "history": [str(h) for h in HISTORIES],
            "sse_delay_sim": sse_sim,
            "ssi_delay_sim": ssi_sim,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
