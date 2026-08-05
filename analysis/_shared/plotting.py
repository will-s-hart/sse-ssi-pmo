"""Shared plotting helpers for the analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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


def _cases_axis(
    ax_left: Axes,
    weeks: NDArray,
    cases: NDArray,
    *,
    ylabel: str = "Symptom onsets per week",
    bar_label: str = "Weekly onsets",
) -> Axes:
    """Twin a right axis onto ``ax_left`` and draw the weekly case bars behind it."""
    ax_cases = ax_left.twinx()
    ax_cases.bar(
        weeks,
        cases,
        width=0.9,
        color=BAR_COLOUR,
        edgecolor="white",
        zorder=0,
        label=bar_label,
    )
    ax_cases.set_ylabel(ylabel)
    ax_cases.set_ylim(0, max(4, int(cases.max()) + 1))
    ax_cases.set_zorder(ax_left.get_zorder() - 1)  # bars behind the lines
    ax_left.patch.set_visible(False)
    return ax_cases


def realtime_two_panel(
    df: pd.DataFrame,
    *,
    title: str,
    xtick_labels: list[str],
    prior_sse: float,
    sse_col: str = "sse_delay_pmo",
    ssi_col: str = "ssi_delay_pmo",
    ensemble_col: str = "ensemble_pmo",
    cases_ylabel: str = "Symptom onsets per week",
    cases_bar_label: str = "Weekly onsets",
    highlight_week: int | None = None,
    highlight_label: str | None = None,
    pmo_legend_loc: str = "upper left",
    post_legend_loc: str = "upper left",
) -> Figure:
    """Two-panel real-time PMO figure (shared by the real-time particle-filter figures).

    ``df`` must have columns ``week``, ``cases``, ``post_sse``, ``post_ssi`` and
    the per-model / ensemble PMO columns named by ``sse_col`` / ``ssi_col`` /
    ``ensemble_col`` (defaulting to the onset-anchored delay names). The top panel
    shows the real-time PMO of the SSE/SSI models and their Bayesian model average
    (left axis) with the weekly epi curve on a twin right axis; the bottom panel
    shows the posterior model probabilities with the same epi-curve underlay.
    ``highlight_week`` optionally marks a week (e.g. a response / decision point)
    with a vertical line on both panels, labelled ``highlight_label``. Returns the
    Figure.
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
        df[sse_col],
        color=SSE_COLOUR,
        linestyle=SSE_LINESTYLE,
        marker=SSE_SIM_MARKER,
        zorder=3,
        label=SSE_LABEL,
    )
    ax_pmo.plot(
        weeks,
        df[ssi_col],
        color=SSI_COLOUR,
        linestyle=SSI_LINESTYLE,
        marker=SSI_SIM_MARKER,
        zorder=3,
        label=SSI_LABEL,
    )
    ax_pmo.plot(
        weeks,
        df[ensemble_col],
        color=ENSEMBLE_COLOUR,
        linestyle="-",
        marker=ENSEMBLE_MARKER,
        zorder=4,
        label=rf"Model average ($\pi_\mathrm{{SSE}}={prior_sse}$)",
    )
    ax_pmo.set_ylabel("Probability of major outbreak")
    ax_pmo.set_ylim(0, 1)
    ax_pmo.grid(True, axis="y", alpha=0.3)
    ax_pmo_cases = _cases_axis(ax_pmo, weeks, cases, ylabel=cases_ylabel, bar_label=cases_bar_label)
    h1, l1 = ax_pmo.get_legend_handles_labels()
    h2, l2 = ax_pmo_cases.get_legend_handles_labels()
    ax_pmo.legend(h1 + h2, l1 + l2, loc=pmo_legend_loc, ncol=2)
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
    _cases_axis(ax_post, weeks, cases, ylabel=cases_ylabel, bar_label=cases_bar_label)
    ax_post.legend(loc=post_legend_loc, ncol=2)
    ax_post.set_xlabel("Week of outbreak")
    ax_post.set_xticks(weeks)
    ax_post.set_xticklabels(xtick_labels)

    if highlight_week is not None:
        for ax in (ax_pmo, ax_post):
            ax.axvline(highlight_week, color="0.3", linestyle="-.", linewidth=1.5, zorder=2)
        if highlight_label:
            ax_post.annotate(
                highlight_label,
                xy=(highlight_week, 0.5),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
                color="0.3",
                bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.7},
            )
    return fig


def _highlight(ax_pmo: Axes, ax_post: Axes, week: int | None, label: str | None) -> None:
    """Mark a decision-point week with a vertical line on both panels."""
    if week is None:
        return
    for ax in (ax_pmo, ax_post):
        ax.axvline(week, color="0.3", linestyle="-.", linewidth=1.5, zorder=2)
    if label:
        ax_post.annotate(
            label,
            xy=(week, 0.5),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color="0.3",
            bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.7},
        )


def realtime_two_panel_compare(
    df: pd.DataFrame,
    *,
    title: str,
    xtick_labels: list[str],
    prior_sse: float,
    cases_ylabel: str = "Cases per week",
    cases_bar_label: str = "Weekly cases",
    highlight_week: int | None = None,
    highlight_label: str | None = None,
) -> Figure:
    """Two-panel real-time PMO figure overlaying two estimation methods.

    Draws the particle-filter estimates as lines and the MCMC estimates as
    markers, so agreement shows as markers sitting on the lines. ``df`` must have
    columns ``week``, ``cases`` and, for ``m`` in ``{pf, mcmc}``: ``m_sse_pmo``,
    ``m_ssi_pmo``, ``m_ensemble_pmo``, ``m_post_sse``, ``m_post_ssi``. Used by the
    naive-infection EVD figure to validate the particle filter against the
    established analytic / MCMC machinery.
    """
    weeks = df["week"].to_numpy()
    cases = df["cases"].to_numpy()
    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, (ax_pmo, ax_post) = plt.subplots(
        2, 1, sharex=True, figsize=(1.5 * default_w, 1.7 * default_h)
    )
    mcmc_marker = "x"
    pmo_series = [
        ("sse", SSE_COLOUR, SSE_LABEL, SSE_LINESTYLE),
        ("ssi", SSI_COLOUR, SSI_LABEL, SSI_LINESTYLE),
        ("ensemble", ENSEMBLE_COLOUR, rf"Model average ($\pi_\mathrm{{SSE}}={prior_sse}$)", "-"),
    ]
    for key, colour, lab, ls in pmo_series:
        ax_pmo.plot(weeks, df[f"pf_{key}_pmo"], color=colour, linestyle=ls, zorder=3, label=lab)
        ax_pmo.plot(
            weeks,
            df[f"mcmc_{key}_pmo"],
            color=colour,
            linestyle="none",
            marker=mcmc_marker,
            markersize=7,
            zorder=4,
        )
    ax_pmo.set_ylabel("Probability of major outbreak")
    ax_pmo.set_ylim(0, 1)
    ax_pmo.grid(True, axis="y", alpha=0.3)
    ax_pmo_cases = _cases_axis(ax_pmo, weeks, cases, ylabel=cases_ylabel, bar_label=cases_bar_label)
    pf_proxy = Line2D([], [], color="0.3", linestyle="-", label="particle filter")
    mcmc_proxy = Line2D([], [], color="0.3", linestyle="none", marker=mcmc_marker, label="MCMC")
    h1, l1 = ax_pmo.get_legend_handles_labels()
    h2, l2 = ax_pmo_cases.get_legend_handles_labels()
    ax_pmo.legend(
        [*h1, pf_proxy, mcmc_proxy, *h2],
        [*l1, "particle filter", "MCMC", *l2],
        loc="lower right",
        ncol=2,
        fontsize=9,
    )
    ax_pmo.set_title(title)

    for key, colour, lab, ls in [
        ("sse", SSE_COLOUR, SSE_LABEL, SSE_LINESTYLE),
        ("ssi", SSI_COLOUR, SSI_LABEL, SSI_LINESTYLE),
    ]:
        ax_post.plot(weeks, df[f"pf_post_{key}"], color=colour, linestyle=ls, zorder=3, label=lab)
        ax_post.plot(
            weeks,
            df[f"mcmc_post_{key}"],
            color=colour,
            linestyle="none",
            marker=mcmc_marker,
            markersize=7,
            zorder=4,
        )
    ax_post.axhline(prior_sse, color="0.5", linestyle=":", linewidth=1, zorder=1)
    ax_post.set_ylabel("Posterior model probability")
    ax_post.set_ylim(0, 1)
    ax_post.grid(True, axis="y", alpha=0.3)
    _cases_axis(ax_post, weeks, cases, ylabel=cases_ylabel, bar_label=cases_bar_label)
    # Model identity is given by the top-panel legend (same colours/styles); a
    # second legend here would collide with the decision-point annotation.
    ax_post.set_xlabel("Week of outbreak")
    ax_post.set_xticks(weeks)
    ax_post.set_xticklabels(xtick_labels)

    _highlight(ax_pmo, ax_post, highlight_week, highlight_label)
    return fig
