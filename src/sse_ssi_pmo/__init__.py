"""SSE/SSI/Poisson probability-of-major-outbreak library.

See ``notes/notes.tex`` for the mathematical derivation. This package exposes:

* :func:`discretise`, :func:`discretise_gamma` — discretise a continuous
  serial-interval distribution (Cori et al.).
* :func:`cumulative` — convert weights ``w_s`` to ``F_r``.
* :func:`pmo_sse`, :func:`pmo_ssi`, :func:`pmo_poisson` — probability of
  major outbreak after an observed incidence history, dispatching between
  an analytic closed-form and a Monte-Carlo / MCMC backend (Poisson is
  analytic-only).
* :func:`pmo_uncertain` — Bayesian model average across SSE and SSI given a
  prior on the SSE model; returns a :class:`PmoUncertainResult` with the
  averaged PMO, posterior model probability, and per-model PMOs.
* :func:`pmo_ensemble` — Bayesian model average across an arbitrary list
  of model specs (SSE / SSI / Poisson, each with its own scalar
  parameters) and prior probabilities; returns a
  :class:`PmoEnsembleResult`.
* :class:`Prior` — Gamma / LogNormal prior on :math:`R_0` or :math:`k`,
  accepted in place of a scalar by :func:`pmo_sse`, :func:`pmo_ssi`, and
  :func:`pmo_uncertain` to trigger parameter-uncertainty handling via
  MCMC or rejection sampling.
* :func:`simulate_sse`, :func:`simulate_ssi`, :func:`simulate_poisson` —
  single forward trajectories.
* :func:`pmo_sse_realtime` / :func:`pmo_ssi_realtime` — real-time (per-week)
  PMO over an observed infection history via a bootstrap particle filter;
  :func:`pmo_sse_incubation_realtime` / :func:`pmo_ssi_incubation_realtime` —
  the same generation-time renewal observed through symptom onsets (an
  independent incubation delay); :func:`pmo_sse_delay_realtime` /
  :func:`pmo_ssi_delay_realtime` — the onset-anchored (TOST) variant. All
  return a :class:`PmoRealtimeResult`.
"""

from sse_ssi_pmo.inference import fit_sse, fit_ssi
from sse_ssi_pmo.pmo import (
    PmoEnsembleResult,
    PmoRealtimeResult,
    PmoUncertainResult,
    pmo_ensemble,
    pmo_poisson,
    pmo_sse,
    pmo_sse_delay,
    pmo_sse_delay_realtime,
    pmo_sse_incubation_realtime,
    pmo_sse_realtime,
    pmo_ssi,
    pmo_ssi_delay,
    pmo_ssi_delay_realtime,
    pmo_ssi_incubation_realtime,
    pmo_ssi_realtime,
    pmo_uncertain,
)
from sse_ssi_pmo.priors import Prior
from sse_ssi_pmo.serial_interval import cumulative, discretise, discretise_gamma
from sse_ssi_pmo.simulation import simulate_poisson, simulate_sse, simulate_ssi
from sse_ssi_pmo.simulation_delay import simulate_sse_delay, simulate_ssi_delay

__all__ = [
    "PmoEnsembleResult",
    "PmoRealtimeResult",
    "PmoUncertainResult",
    "Prior",
    "cumulative",
    "discretise",
    "discretise_gamma",
    "fit_sse",
    "fit_ssi",
    "pmo_ensemble",
    "pmo_poisson",
    "pmo_sse",
    "pmo_sse_delay",
    "pmo_sse_delay_realtime",
    "pmo_sse_incubation_realtime",
    "pmo_sse_realtime",
    "pmo_ssi",
    "pmo_ssi_delay",
    "pmo_ssi_delay_realtime",
    "pmo_ssi_incubation_realtime",
    "pmo_ssi_realtime",
    "pmo_uncertain",
    "simulate_poisson",
    "simulate_sse",
    "simulate_sse_delay",
    "simulate_ssi",
    "simulate_ssi_delay",
]
