"""Shared plotting helpers for the analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

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

# Model-average / ensemble line and epi-curve underlay (see the real-time figures).
ENSEMBLE_COLOUR = "#009E73"  # Wong palette green; distinct from SSE/SSI
ENSEMBLE_MARKER = "D"
BAR_COLOUR = "#DDDDDD"


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


def _cases_axis(ax_left: Axes, weeks: NDArray, cases: NDArray) -> Axes:
    """Twin a right axis onto ``ax_left`` and draw the weekly onset bars behind it."""
    ax_cases = ax_left.twinx()
    ax_cases.bar(
        weeks,
        cases,
        width=0.9,
        color=BAR_COLOUR,
        edgecolor="white",
        zorder=0,
        label="Weekly onsets",
    )
    ax_cases.set_ylabel("Symptom onsets per week")
    ax_cases.set_ylim(0, max(4, int(cases.max()) + 1))
    ax_cases.set_zorder(ax_left.get_zorder() - 1)  # bars behind the lines
    ax_left.patch.set_visible(False)
    return ax_cases


def realtime_two_panel(
    df: pd.DataFrame, *, title: str, xtick_labels: list[str], prior_sse: float
) -> Figure:
    """Two-panel real-time onset-anchored PMO figure (shared by the delay figures).

    ``df`` must have columns ``week``, ``cases``, ``sse_delay_pmo``,
    ``ssi_delay_pmo``, ``ensemble_pmo``, ``post_sse`` and ``post_ssi``. The top
    panel shows the real-time PMO of the SSE/SSI models and their Bayesian model
    average (left axis) with the weekly onset epi curve on a twin right axis; the
    bottom panel shows the posterior model probabilities with the same epi-curve
    underlay. Returns the Figure.
    """
    weeks = df["week"].to_numpy()
    cases = df["cases"].to_numpy()
    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, (ax_pmo, ax_post) = plt.subplots(
        2, 1, sharex=True, figsize=(1.5 * default_w, 1.7 * default_h)
    )

    # Top: real-time PMO (left) + epi curve (right).
    ax_pmo.plot(
        weeks,
        df["sse_delay_pmo"],
        color=SSE_COLOUR,
        linestyle=SSE_LINESTYLE,
        marker=SSE_SIM_MARKER,
        zorder=3,
        label=SSE_LABEL,
    )
    ax_pmo.plot(
        weeks,
        df["ssi_delay_pmo"],
        color=SSI_COLOUR,
        linestyle=SSI_LINESTYLE,
        marker=SSI_SIM_MARKER,
        zorder=3,
        label=SSI_LABEL,
    )
    ax_pmo.plot(
        weeks,
        df["ensemble_pmo"],
        color=ENSEMBLE_COLOUR,
        linestyle="-",
        marker=ENSEMBLE_MARKER,
        zorder=4,
        label=rf"Model average ($\pi_\mathrm{{SSE}}={prior_sse}$)",
    )
    ax_pmo.set_ylabel("Probability of major outbreak")
    ax_pmo.set_ylim(0, 1)
    ax_pmo.grid(True, axis="y", alpha=0.3)
    ax_pmo_cases = _cases_axis(ax_pmo, weeks, cases)
    h1, l1 = ax_pmo.get_legend_handles_labels()
    h2, l2 = ax_pmo_cases.get_legend_handles_labels()
    ax_pmo.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    ax_pmo.set_title(title)

    # Bottom: posterior model probabilities (left) + epi curve (right).
    ax_post.plot(
        weeks,
        df["post_sse"],
        color=SSE_COLOUR,
        linestyle=SSE_LINESTYLE,
        marker=SSE_SIM_MARKER,
        zorder=3,
        label=SSE_LABEL,
    )
    ax_post.plot(
        weeks,
        df["post_ssi"],
        color=SSI_COLOUR,
        linestyle=SSI_LINESTYLE,
        marker=SSI_SIM_MARKER,
        zorder=3,
        label=SSI_LABEL,
    )
    ax_post.axhline(prior_sse, color="0.5", linestyle=":", linewidth=1, zorder=1)
    ax_post.set_ylabel("Posterior model probability")
    ax_post.set_ylim(0, 1)
    ax_post.grid(True, axis="y", alpha=0.3)
    _cases_axis(ax_post, weeks, cases)
    ax_post.legend(loc="upper left", ncol=2)
    ax_post.set_xlabel("Week of outbreak")
    ax_post.set_xticks(weeks)
    ax_post.set_xticklabels(xtick_labels)
    return fig
