"""Shared default inputs for the analysis scripts."""

from __future__ import annotations

from pathlib import Path

DEFAULT_R0: float = 2.0
DEFAULT_K: float = 0.18
DEFAULT_R_WEEKS: int = 1
DEFAULT_SI_MEAN: float = 2.19  # weeks
DEFAULT_SI_SD: float = 1.3  # weeks
DEFAULT_SI_MAX: int = 30  # discretisation upper bound (weeks)

FIG1_R_MAX: int = 5
FIG2_R0_MIN: float = 0.5
FIG2_R0_MAX: float = 4.0
FIG3_K_MIN: float = 0.05
FIG3_K_MAX: float = 10.0
DEFAULT_N_POINTS: int = 200

# --- simulation-overlay defaults ---
SIM_THRESHOLD: int = 20  # single-step incidence treated as a major outbreak
SIM_T_MAX: int = 200  # max trajectory length (weeks)
SIM_N_SSE: int = 5000  # SSE sims per parameter set
SIM_N_SSI: int = 2000  # matching SSI sims per parameter set
SIM_BATCH_SSI: int = 5000  # SSI batch size for rejection sampling
SIM_MAX_ATTEMPTS_SSI: int = 4_000_000  # safety cap on SSI sim attempts
SIM_SUBSET_POINTS: int = 20  # # of param values to overlay sims on for figs 2/3
SIM_SEED: int = 20260504

OUT_DIR: Path = Path(__file__).resolve().parents[1] / "figures"
