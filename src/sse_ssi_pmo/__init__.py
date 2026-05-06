"""SSE/SSI probability-of-major-outbreak library.

See ``notes/notes.tex`` for the mathematical derivation. This package exposes:

* :func:`discretise`, :func:`discretise_gamma` — discretise a continuous
  serial-interval distribution (Cori et al.).
* :func:`cumulative` — convert weights ``w_s`` to ``F_r``.
* :func:`pmo_sse`, :func:`pmo_ssi` — probability of major outbreak after an
  observed incidence history, dispatching between an analytic closed-form
  and a Monte-Carlo simulation backend.
* :func:`simulate_sse`, :func:`simulate_ssi` — single forward trajectories.
"""

from sse_ssi_pmo.inference import fit_sse, fit_ssi
from sse_ssi_pmo.pmo import pmo_sse, pmo_ssi
from sse_ssi_pmo.serial_interval import cumulative, discretise, discretise_gamma
from sse_ssi_pmo.simulation import simulate_sse, simulate_ssi

__all__ = [
    "cumulative",
    "discretise",
    "discretise_gamma",
    "fit_sse",
    "fit_ssi",
    "pmo_sse",
    "pmo_ssi",
    "simulate_sse",
    "simulate_ssi",
]
