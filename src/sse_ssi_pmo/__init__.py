"""SSE/SSI probability-of-major-outbreak library.

See ``notes/notes.tex`` for the mathematical derivation. This package exposes:

* :func:`discretise_gamma` — discretise a Gamma serial interval (Cori et al.).
* :func:`cumulative` — convert weights ``w_s`` to ``F_r``.
* :func:`q_unconditional`, :func:`q_sse`, :func:`q_ssi` — analytical
  extinction-probability solvers.
* :func:`pmo_unconditional`, :func:`pmo_sse`, :func:`pmo_ssi` — convenience
  ``1 - q`` wrappers.
* :func:`simulate_sse`, :func:`simulate_ssi` — single forward trajectories.
* :func:`pmo_sse_sim`, :func:`pmo_ssi_sim` — Monte-Carlo PMO estimates after
  an observed incidence history (SSI uses rejection sampling).
"""

from sse_ssi_pmo.extinction import (
    pmo_sse,
    pmo_ssi,
    pmo_unconditional,
    q_sse,
    q_ssi,
    q_unconditional,
)
from sse_ssi_pmo.serial_interval import cumulative, discretise, discretise_gamma
from sse_ssi_pmo.simulation import (
    pmo_sse_sim,
    pmo_ssi_sim,
    simulate_sse,
    simulate_ssi,
)

__all__ = [
    "cumulative",
    "discretise",
    "discretise_gamma",
    "pmo_sse",
    "pmo_sse_sim",
    "pmo_ssi",
    "pmo_ssi_sim",
    "pmo_unconditional",
    "q_sse",
    "q_ssi",
    "q_unconditional",
    "simulate_sse",
    "simulate_ssi",
]
