"""Delays-paper analysis defaults (symptom-onset-anchored models, figures 1-6)."""

from __future__ import annotations

from pathlib import Path

from analysis._shared.defaults import COMPARISON_HISTORIES

OUT_DIR: Path = Path(__file__).resolve().parents[2] / "figures" / "delays"
RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "results" / "delays"

# --- symptom-onset-anchored (delay) models: TOST + incubation (weeks) ---
# Observed data are symptom onsets, not infections; transmission is anchored to
# a case's own onset via the TOST distribution (from lag 0) and infections are
# mapped forward to onsets via the (shared) incubation period (from lag 1). The
# TOST mean is chosen so the implied onset-to-onset SI (= TOST + incubation)
# roughly matches DEFAULT_SI_MEAN (~2.2 wk after Cori discretisation); tunable.
DELAY_TOST_MEAN: float = 0.8  # time from onset to transmission mean (weeks)
DELAY_TOST_SD: float = 0.7  # TOST SD (weeks)
DELAY_TOST_MAX: int = 10  # TOST discretisation upper bound (weeks)
DELAY_SIM_N: int = 5_000  # matching sims per parameter set / history
DELAY_SIM_BATCH: int = 10_000  # rejection-sampling batch size
DELAY_SIM_MAX_ATTEMPTS: int = 8_000_000  # safety cap on sim attempts

# --- figure 1 (onset-anchored PMO vs r) ---
FIG1_R_MAX: int = 5

# --- figure 2 (onset-anchored PMO comparison across histories) ---
FIG2_HISTORIES: list[list[int]] = COMPARISON_HISTORIES

# --- figure 3 (real-time onset-anchored PMO over a simulated SSI outbreak) ---
# A weekly onset history simulated from the *SSI* model (the true model) with the
# default parameters; the real-time model averaging should recover SSI as the
# weeks accumulate. FIG3_SIM_SEED selects a realisation that survives several
# weeks and stays within the particle filter's exact-match range.
FIG3_SIM_SEED: int = 46
FIG3_N_PARTICLES: int = 50_000  # bootstrap particle-filter population size
FIG3_PRIOR_SSE: float = 0.5  # prior probability of the SSE model (SSI = 1 - this)

# --- figure 4 (real-time onset-anchored PMO over a real EVD outbreak) ---
# Symptom-onset dates (dd/mm/yyyy) for the 2017 Likati (Bas-Uele, DRC) EVD
# outbreak; binned into calendar (Mon-Sun) weeks. PMO at each week conditions
# on the onset history observed up to that week.
FIG4_ONSET_DATES: list[str] = [
    "27/03/2017",
    "18/04/2017",
    "24/04/2017",
    "01/05/2017",
    "26/04/2017",
    "30/04/2017",
    "02/05/2017",
    "11/05/2017",
]
FIG4_N_PARTICLES: int = 50_000  # bootstrap particle-filter population size
FIG4_PRIOR_SSE: float = 0.5  # prior probability of the SSE model (SSI = 1 - this)

# --- figure 5 (real-time onset-anchored PMO over the 2018 Equateur outbreak) ---
# Daily disease-incidence series for the 2018 Equateur (DRC) EVD outbreak
# (committed CSV, one row per day from the first case on 5 Apr 2018), converted
# to calendar (Mon-Sun) weeks. The response team arrived on 8 May 2018 (week 5
# here), marked as the response/decision point.
FIG5_DATA_FILE: str = "equateur_2018_incidence.csv"
FIG5_RESPONSE_WEEK: int = 5  # calendar week containing the 8 May 2018 response arrival
FIG5_N_PARTICLES: int = 200_000  # weekly counts reach the teens
FIG5_PRIOR_SSE: float = 0.5  # prior probability of the SSE model (SSI = 1 - this)

# --- figure 6 (real-time onset-anchored PMO over the 2020 Equateur outbreak) ---
# Case line list for the 2020 Equateur (DRC) EVD outbreak (committed CSV of
# reported dates). NB these are *reporting* dates, used here as a proxy for the
# symptom-onset timeline the model assumes (a reporting delay is not modelled).
# Binned into calendar (Mon-Sun) weeks. The outbreak was officially declared on
# 1 Jun 2020 (week 4 here), marked as the response/decision point.
FIG6_DATA_FILE: str = "equateur_2020_reported.csv"
FIG6_RESPONSE_WEEK: int = 4  # calendar week containing the 1 Jun 2020 declaration
FIG6_N_PARTICLES: int = 100_000  # larger population (weekly counts reach the teens)
FIG6_PRIOR_SSE: float = 0.5  # prior probability of the SSE model (SSI = 1 - this)
