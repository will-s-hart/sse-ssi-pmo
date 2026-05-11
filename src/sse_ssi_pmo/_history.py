"""Tiny utilities for inspecting an observed incidence history.

Shared helpers used by both ``extinction.py`` (PMO) and ``likelihood.py``
(marginal likelihood); housed here to avoid a circular import between those
two modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def w_at(w: NDArray[np.float64], s: int) -> float:
    """Return ``w_s`` (the serial-interval weight at lag ``s``), or 0 outside support.

    ``w[0] = w_1`` (lag-1 weight), so the convention here is one-indexed: ``s``
    in ``{1, ..., len(w)}`` is the supported range.
    """
    if s < 1 or s > w.size:
        return 0.0
    return float(w[s - 1])


def classify_history(history: NDArray[np.int64]) -> dict:
    """Classify the shape of ``history`` for analytic SSI closed forms.

    Returns a dict with ``"kind"`` in ``{"day0_only", "one_later", "two_later",
    "general"}`` and the relevant counts/indices.
    """
    nonzero_after = np.flatnonzero(history[1:] != 0)
    n_later = nonzero_after.size
    out: dict = {"I_0": int(history[0])}
    if n_later == 0:
        out["kind"] = "day0_only"
    elif n_later == 1:
        i = int(nonzero_after[0]) + 1
        out["kind"] = "one_later"
        out["i"] = i
        out["I_i"] = int(history[i])
    elif n_later == 2:
        i = int(nonzero_after[0]) + 1
        j = int(nonzero_after[1]) + 1
        out["kind"] = "two_later"
        out["i"] = i
        out["I_i"] = int(history[i])
        out["j"] = j
        out["I_j"] = int(history[j])
    else:
        out["kind"] = "general"
    return out
