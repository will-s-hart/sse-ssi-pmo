"""SSE/SSI probability-of-major-outbreak library.

See ``notes/notes.tex`` for the mathematical derivation. This package exposes:

* :func:`discretise`, :func:`discretise_gamma` — discretise a continuous
  serial-interval distribution (Cori et al.).
* :func:`cumulative` — convert weights ``w_s`` to ``F_r``.
* :func:`pmo_sse`, :func:`pmo_ssi` — probability of major outbreak after an
  observed incidence history, dispatching between an analytic closed-form
  and a Monte-Carlo simulation backend.
* :func:`pmo_uncertain` — Bayesian model average across SSE and SSI given a
  prior on the SSE model; returns a :class:`PmoUncertainResult` with the
  averaged PMO, posterior model probability, and per-model PMOs.
* :class:`Prior` — Gamma / LogNormal prior on :math:`R_0` or :math:`k`,
  accepted in place of a scalar by the ``pmo_*`` functions to trigger
  parameter-uncertainty handling via MCMC or rejection sampling.
* :func:`simulate_sse`, :func:`simulate_ssi` — single forward trajectories.
"""

from sse_ssi_pmo.inference import fit_sse, fit_ssi
from sse_ssi_pmo.pmo import PmoUncertainResult, pmo_sse, pmo_ssi, pmo_uncertain
from sse_ssi_pmo.priors import Prior
from sse_ssi_pmo.serial_interval import cumulative, discretise, discretise_gamma
from sse_ssi_pmo.simulation import simulate_sse, simulate_ssi

__all__ = [
    "PmoUncertainResult",
    "Prior",
    "cumulative",
    "discretise",
    "discretise_gamma",
    "fit_sse",
    "fit_ssi",
    "pmo_sse",
    "pmo_ssi",
    "pmo_uncertain",
    "simulate_sse",
    "simulate_ssi",
]
