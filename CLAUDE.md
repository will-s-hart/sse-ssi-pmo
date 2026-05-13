# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code accompanying the manuscript in `notes/notes.tex` (compiled to `notes/notes.pdf`). Computes the **probability of major outbreak (PMO)** after an observed incidence history under three branching-process offspring models — **SSE** (Superspreading Events, NB offspring), **SSI** (Superspreading Individuals, latent-Gamma Poisson offspring), and **Poisson** (the `k → ∞` limit of either). Supports closed-form, Monte-Carlo, and MCMC backends, plus Bayesian model averaging across them.

`notes/notes.tex` is the source of truth for the derivations the code implements — read it (or the compiled PDF) before changing any analytic formula.

## Tooling

Environment is managed by **pixi** (conda + editable PyPI install). All commands run via `pixi run`:

| Command | What it does |
| --- | --- |
| `pixi run lint` | `ruff check src analysis` |
| `pixi run fmt` | `ruff format src analysis` |
| `pixi run typecheck` | `ty check src analysis` (clears `VIRTUAL_ENV` first) |
| `pixi run res{1..7}` | Compute and write `results/figN_*.csv` for figure N |
| `pixi run results` | Run all `res1..res7` (depends-on aggregate) |
| `pixi run fig{1..7}` | Render `figures/figN_*.pdf` + `.png` from the corresponding CSV |
| `pixi run figs` | Render all figures |

There is **no test suite**. Validation is by reproducing the figures and cross-checking analytic vs. simulation/MCMC overlays.

To run a one-off script (e.g. during development): `pixi run python analysis/results_1_pmo_vs_r.py`. Plain `python` outside `pixi run` will not find dependencies.

The package is installed editable via `[tool.pixi.pypi-dependencies]` — edits under `src/sse_ssi_pmo/` take effect immediately.

## Architecture

### `src/sse_ssi_pmo/` — library

Single flat package; modules are layered to avoid circular imports.

- **`pmo.py`** — public dispatcher API. Five top-level functions: `pmo_sse`, `pmo_ssi`, `pmo_poisson`, `pmo_uncertain`, `pmo_ensemble`. Each takes `R0`, `k`, `w`, `history`, `method`, validates inputs, broadcasts `(R0, k)`, and routes to a private backend. **All user-facing input validation lives here** — private helpers in other modules trust their inputs.
- **`extinction.py`** — analytic + MCMC PMO backends (`_pmo_sse_analytic`, `_pmo_ssi_analytic`, `_pmo_ssi_mcmc`, `_pmo_poisson_analytic`, `_pmo_uncertain_*`, `_pmo_ensemble_*`). Builds on `likelihood.py` and `_history.py`.
- **`simulation.py`** — Monte-Carlo backends. Public single-trajectory `simulate_sse/ssi/poisson`; private `_pmo_*_sim*` PMO estimators. SSI uses **shared rejection sampling**: multi-history calls run one batched simulation pass matched against every history, requiring all rows to share the same `I_0`.
- **`inference.py`** — `fit_sse` / `fit_ssi`: PyMC HMC fits returning an `xr.DataTree`. SSI samples latent infectivities `Y_t` for every day with `I_t > 0` (variable `"infectivity"`). Used by the MCMC PMO/ensemble paths.
- **`likelihood.py`** — marginal log-likelihoods `log L_M` for Bayesian model averaging. SSE/Poisson closed-form; SSI closed-form for the three "special" history shapes, else MCMC via `"naive"` / `"importance_sampling"` / `"bridge"` (Meng-Wong; the default).
- **`serial_interval.py`** — `discretise`, `discretise_gamma`, `cumulative` (Cori et al. discretisation).
- **`_history.py`** — `classify_history` (returns `"day0_only" | "one_later" | "two_later" | "general"`) and `w_at`. Shared by `extinction.py` and `likelihood.py`; lives in its own module to break the circular import.

### Conventions and gotchas

- **Serial-interval weights `w` are 1-indexed**: `w[s-1] = w_s` for `s = 1, 2, ...`. `w_at(w, s)` returns 0 outside `[1, len(w)]`.
- **`history` is non-negative integer, must have `history[0] ≥ 1`.** May be 1-D `(L,)` or 2-D `(M, L)` (or list-of-lists). 2-D input adds a leading length-`M` axis to the return. Validation happens in `pmo.py:_validate_histories`.
- **`R0, k` broadcast** like NumPy. Scalar inputs → Python `float`; array inputs → `NDArray` of the broadcast shape; the dispatcher loops over `(R0, k)` combinations (`tqdm` wraps the outer loop when `show_progress=True`).
- **Analytic SSI / `pmo_uncertain` / `pmo_ensemble` only support three history kinds** (day-0-only, one-later, two-later). For a "general" history use `method='simulation'` or `method='mcmc'`. The dispatcher raises with a pointer to the offending row.
- **`pmo_poisson` is analytic-only.** For forward Monte-Carlo use `simulate_poisson` directly.
- **`pmo_ensemble` model specs** are dicts: `{"model": "sse"|"ssi", "R0": float, "k": float}` or `{"model": "poisson", "R0": float}`. `priors` is a length-`len(models)` non-negative simplex.
- **Ruff ignores `N802/N803/N806`** so mathematical capitalisation (`R0`, `F_r`, `Y`, `Lambda`) matches `notes.tex` — keep that convention rather than renaming.
- Private helpers prefixed `_` do **not** validate inputs; that is the dispatcher's job. Don't re-add validation there.

### `analysis/` — paper figures

Two scripts per figure, run in order:

1. `results_N_*.py` → writes `results/figN_*.csv`. Expensive (sims/MCMC).
2. `figure_N_*.py` → reads the CSV and writes `figures/figN_*.pdf` and `.png`.

Shared constants (defaults for `R0`, `k`, serial interval, simulation budgets, history sets for figs 4–7) live in `analysis/analysis_defaults.py`. Plotting helpers in `analysis/_plotting.py`. `figures/` is gitignored; CSVs in `results/` are committed.

The two-step split means: changing a plot style only re-runs `fig*`; changing a model means re-running the matching `res*` first.
