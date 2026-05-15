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
SIM_THRESHOLD: int = 100  # single-step incidence treated as a major outbreak
SIM_T_MAX: int = 200  # max trajectory length (weeks)
SIM_N_SSE: int = 10_000  # SSE sims per parameter set
SIM_N_SSI: int = 5_000  # matching SSI sims per parameter set
SIM_BATCH_SSI: int = 10_000  # SSI batch size for rejection sampling
SIM_MAX_ATTEMPTS_SSI: int = 4_000_000  # safety cap on SSI sim attempts
SIM_SUBSET_POINTS: int = 20  # # of param values to overlay sims on for figs 2/3
SIM_SEED: int = 20260506

OUT_DIR: Path = Path(__file__).resolve().parents[1] / "figures"
RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results"

# --- figure 4 (PMO comparison across histories) ---
FIG4_HISTORIES: list[list[int]] = [
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
FIG4_MCMC_DRAWS: int = 1000
FIG4_MCMC_TUNE: int = 1000
FIG4_MCMC_CHAINS: int = 4

# --- figure 5 (model-uncertain PMO across histories) ---
FIG5_HISTORIES: list[list[int]] = FIG4_HISTORIES
FIG5_PRIOR_SSE: float = 0.5
FIG5_SIM_N: int = 5_000
FIG5_SIM_BATCH: int = 10_000
FIG5_SIM_MAX_ATTEMPTS: int = 8_000_000

# --- figure 6 (PMO error under model misspecification vs. averaging) ---
FIG6_PRIOR_SSE: float = 0.5
FIG6_HISTORY_LENGTHS: list[int] = [2, 3, 4]
FIG6_N_SIM: int = 5000

# --- figure 7 (PMO ensemble error across 5 models) ---
FIG7_HISTORY_LENGTHS: list[int] = [2, 3]
FIG7_N_SIM: int = 5000
FIG7_K_LARGE: float = 1.0  # secondary k value alongside DEFAULT_K

# --- figure 8 (PMO vs r with uncertain R0) ---
FIG8_R_MAX: int = FIG1_R_MAX
FIG8_R0_PRIOR_MEAN: float = DEFAULT_R0
FIG8_R0_PRIOR_SD: float = 0.5
FIG8_MCMC_DRAWS: int = FIG4_MCMC_DRAWS
FIG8_MCMC_TUNE: int = FIG4_MCMC_TUNE
FIG8_MCMC_CHAINS: int = FIG4_MCMC_CHAINS
FIG8_SIM_N: int = SIM_N_SSI
FIG8_SIM_BATCH: int = SIM_BATCH_SSI
FIG8_SIM_MAX_ATTEMPTS: int = SIM_MAX_ATTEMPTS_SSI
FIG8_CRI_LEVEL: float = 0.95

# --- figure 9 (model-uncertain PMO across histories with uncertain R0) ---
FIG9_HISTORIES: list[list[int]] = FIG5_HISTORIES
FIG9_R0_PRIOR_MEAN: float = FIG8_R0_PRIOR_MEAN
FIG9_R0_PRIOR_SD: float = FIG8_R0_PRIOR_SD
FIG9_PRIOR_SSE: float = FIG5_PRIOR_SSE
FIG9_MCMC_DRAWS: int = FIG4_MCMC_DRAWS
FIG9_MCMC_TUNE: int = FIG4_MCMC_TUNE
FIG9_MCMC_CHAINS: int = FIG4_MCMC_CHAINS
FIG9_SIM_N: int = FIG5_SIM_N
FIG9_SIM_BATCH: int = FIG5_SIM_BATCH
FIG9_SIM_MAX_ATTEMPTS: int = FIG5_SIM_MAX_ATTEMPTS

# --- figure 10 (PMO error under model misspecification with uncertain R0) ---
FIG10_PRIOR_SSE: float = FIG6_PRIOR_SSE
FIG10_HISTORY_LENGTHS: list[int] = FIG6_HISTORY_LENGTHS
FIG10_N_SIM: int = FIG6_N_SIM
FIG10_R0_PRIOR_MEAN: float = FIG8_R0_PRIOR_MEAN
FIG10_R0_PRIOR_SD: float = FIG8_R0_PRIOR_SD
FIG10_MCMC_DRAWS: int = FIG8_MCMC_DRAWS
FIG10_MCMC_TUNE: int = FIG8_MCMC_TUNE
FIG10_MCMC_CHAINS: int = FIG8_MCMC_CHAINS
FIG10_SIM_N: int = FIG5_SIM_N
FIG10_SIM_BATCH: int = FIG5_SIM_BATCH
FIG10_SIM_MAX_ATTEMPTS: int = FIG5_SIM_MAX_ATTEMPTS
