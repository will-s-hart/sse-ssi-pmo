"""Base-paper analysis defaults (infection-anchored PMO, figures 1-12)."""

from __future__ import annotations

from pathlib import Path

from analysis._shared.defaults import (
    COMPARISON_HISTORIES,
    DEFAULT_R0,
    SIM_BATCH_SSI,
    SIM_MAX_ATTEMPTS_SSI,
    SIM_N_SSI,
)

OUT_DIR: Path = Path(__file__).resolve().parents[2] / "figures" / "base"
RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "results" / "base"

FIG1_R_MAX: int = 5
FIG2_R0_MIN: float = 0.5
FIG2_R0_MAX: float = 4.0
FIG3_K_MIN: float = 0.05
FIG3_K_MAX: float = 10.0
DEFAULT_N_POINTS: int = 200

# --- figure 4 (PMO comparison across histories) ---
FIG4_HISTORIES: list[list[int]] = COMPARISON_HISTORIES
FIG4_MCMC_DRAWS: int = 1000
FIG4_MCMC_TUNE: int = 1000
FIG4_MCMC_CHAINS: int = 4

# --- figure 5 (model-uncertain PMO across histories) ---
FIG5_HISTORIES: list[list[int]] = COMPARISON_HISTORIES
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
FIG9_HISTORIES: list[list[int]] = COMPARISON_HISTORIES
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

# Figures 11 (naive) and 12 (bridge) -- real-time PMO over the 2018 Equateur EVD
# outbreak -- are added in a later phase alongside their results/figure scripts.
