"""Default inputs shared by the base and delays analysis papers.

Paper-specific constants (output directories, per-figure settings) live in
``analysis.base.defaults`` and ``analysis.delays.defaults``; this module holds
only what both papers use.
"""

from __future__ import annotations

from pathlib import Path

# --- core epidemiological parameters ---
DEFAULT_R0: float = 2.0
DEFAULT_K: float = 0.18
DEFAULT_R_WEEKS: int = 1
DEFAULT_SI_MEAN: float = 2.19  # weeks
DEFAULT_SI_SD: float = 1.3  # weeks
DEFAULT_SI_MAX: int = 30  # discretisation upper bound (weeks)

# --- simulation-overlay defaults ---
SIM_THRESHOLD: int = 100  # single-step incidence treated as a major outbreak
SIM_T_MAX: int = 200  # max trajectory length (weeks)
SIM_N_SSE: int = 50_000  # SSE sims per parameter set
SIM_N_SSI: int = 5_000  # matching SSI sims per parameter set
SIM_BATCH_SSI: int = 10_000  # SSI batch size for rejection sampling
SIM_MAX_ATTEMPTS_SSI: int = 4_000_000  # safety cap on SSI sim attempts
SIM_SUBSET_POINTS: int = 20  # # of param values to overlay sims on for figs 2/3
SIM_SEED: int = 20260506

# --- incubation-period distribution (weeks; infection -> symptom onset, lag >= 1) ---
# Shared by the base 'bridge' figure (generation-time renewal + independent
# incubation) and the delays paper's onset-anchored models. Ebola-plausible
# (~8-9 days); chosen so the implied onset serial interval is comparable to
# DEFAULT_SI_MEAN.
DELAY_INC_MEAN: float = 1.2  # incubation-period mean (weeks)
DELAY_INC_SD: float = 1.0  # incubation-period SD (weeks)
DELAY_INC_MAX: int = 10  # incubation discretisation upper bound (weeks)

# committed input datasets (shared)
DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"

# --- shared short incidence/onset histories for the comparison figures ---
COMPARISON_HISTORIES: list[list[int]] = [
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [2, 2],
    [1, 0, 0],
    [1, 0, 1],
    [1, 0, 2],
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 2],
    [1, 2, 0],
    [1, 2, 1],
    [1, 2, 2],
]
