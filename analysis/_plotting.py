"""Shared plotting helpers for the analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

# Two-model colour scheme (colour-blind friendly: blue / vermillion from the
# Wong 2011 palette).
SSE_COLOUR = "#0072B2"
SSI_COLOUR = "#D55E00"
SSE_LABEL = "SSE model"
SSI_LABEL = "SSI model"
# SSE solid, SSI dashed: when the two coincide the dashed line shows through.
SSE_LINESTYLE = "-"
SSI_LINESTYLE = "--"

# Simulation overlay markers.
SSE_SIM_MARKER = "o"
SSI_SIM_MARKER = "s"
SIM_MARKERSIZE = 5
SIM_LABEL_SUFFIX = " (sim)"


def set_style() -> None:
    """Apply a consistent matplotlib style across figures."""
    mpl.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: Figure, name: str, out_dir: Path) -> None:
    """Save ``fig`` to ``out_dir`` as both ``<name>.pdf`` and ``<name>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
