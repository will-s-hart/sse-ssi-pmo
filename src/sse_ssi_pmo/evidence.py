"""Model-evidence building blocks for the SSE and SSI models.

For an observed incidence history :math:`I_0, I_1, \\ldots, I_r` and a chosen
model :math:`M`, the (log) model evidence
:math:`\\log Z_M = \\log p_M(I_1, \\ldots, I_r \\mid I_0)` underwrites Bayesian
model averaging across :math:`M \\in \\{\\text{SSE}, \\text{SSI}\\}`. When
:math:`R_0` and / or :math:`k` are also given priors, the evidence becomes a
joint integral over the latents and the parameters; the estimators below
extend to that case (see :mod:`sse_ssi_pmo.priors`).

* SSE — closed-form for any history at fixed :math:`(R_0, k)` (product of
  per-day NB pmfs); see :func:`_log_evidence_sse_general`. Under a prior on
  :math:`(R_0, k)` the parameter dimension is marginalised by one of three
  estimators (see :func:`_log_evidence_sse_mcmc`).
* SSI — closed-form for the same three special cases as for the extinction
  probability (cases on day 0 only, on day 0 and one later day, on day 0 and
  two later days); see :func:`_log_evidence_ssi_analytic`. For a general
  history, three MCMC-based estimators are available, all routed through the
  dispatcher :func:`_log_evidence_ssi_mcmc`:

  * ``"naive"`` — Monte Carlo with the Gamma prior as proposal. No MCMC trace
    needed; a no-MCMC fast path. Variance grows when the prior and posterior
    diverge, but is acceptable on histories of moderate length.
  * ``"importance_sampling"`` — IS with an independent-Gamma proposal fitted
    by method-of-moments to the MCMC posterior marginals.
  * ``"bridge"`` — Meng-Wong bridge sampling using the same MoM-fitted
    proposal (Gronau et al. 2017 log-space iteration). The most rigorous; the
    default for :func:`pmo_uncertain` ``method='mcmc'``.

The functions in this module are private helpers used by ``pmo.py`` and
``extinction.py``; they do not validate user-facing inputs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.special
import scipy.stats
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from sse_ssi_pmo._history import classify_history, w_at
from sse_ssi_pmo.priors import Prior
from sse_ssi_pmo.serial_interval import cumulative

# ---------------------------------------------------------------------------
# helpers shared between SSI model-evidence estimators
# ---------------------------------------------------------------------------


def _active_prior_idx(history: NDArray[np.int64]) -> NDArray[np.int64]:
    """Indices ``t`` in ``{0, ..., r-1}`` with ``I_t > 0`` — the active prior latents.

    ``Y_r`` never enters :math:`p(I_{1:r} \\mid Y_{0:r-1})`, so we exclude it
    even when ``I_r > 0``. Days with ``I_t = 0`` collapse to a point mass at
    zero in the Gamma prior, so we exclude those too.
    """
    r = history.size - 1
    return np.flatnonzero(history[:r] > 0).astype(np.int64)


def _log_lik_given_Y(
    Y_active: NDArray[np.float64],
    R0: float | NDArray[np.float64],
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    prior_idx: NDArray[np.int64],
) -> NDArray[np.float64]:
    """``log p(I_{1:r} | Y, R0)`` for each row of ``Y_active``.

    ``Y_active`` has shape ``(n_samples, len(prior_idx))``; days outside
    ``prior_idx`` (and day ``r``) are zero-padded into a length-``r+1`` vector
    when computing the per-day FOI. ``R0`` may be a scalar or an
    ``(n_samples,)`` array (per-row, for the parameter-uncertain path).
    """
    r = history.size - 1
    n_samples = Y_active.shape[0]
    L_w = w.size

    Y_full = np.zeros((n_samples, r + 1), dtype=np.float64)
    Y_full[:, prior_idx] = Y_active

    log_lik = np.zeros(n_samples, dtype=np.float64)
    for t in range(1, r + 1):
        lam = np.zeros(n_samples, dtype=np.float64)
        for s in range(t):
            lag = t - s
            if 1 <= lag <= L_w:
                lam += w[lag - 1] * Y_full[:, s]
        I_t = int(history[t])
        if I_t == 0:
            log_lik = log_lik - R0 * lam
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_R0_lam = np.where(lam > 0, np.log(R0 * lam), -np.inf)
            log_lik = log_lik + I_t * log_R0_lam - R0 * lam - scipy.special.gammaln(I_t + 1)
    return log_lik


def _log_prior_Y(
    Y_active: NDArray[np.float64],
    k: float | NDArray[np.float64],
    history: NDArray[np.int64],
    prior_idx: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Sum over active days of ``log Gamma(k I_t, k)`` prior pdf at ``Y_t``.

    ``k`` may be a scalar (uniform over rows) or an ``(n_samples,)`` array
    (per-row, for the parameter-uncertain code path).
    """
    out = np.zeros(Y_active.shape[0], dtype=np.float64)
    for col, t in enumerate(prior_idx):
        out += scipy.stats.gamma.logpdf(Y_active[:, col], a=k * history[t], scale=1.0 / k)
    return out


def _log_proposal_Y(
    Y_active: NDArray[np.float64],
    alpha: NDArray[np.float64],
    beta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum over columns of ``log Gamma(alpha_t, beta_t)`` proposal pdf at ``Y_t``."""
    out = np.zeros(Y_active.shape[0], dtype=np.float64)
    for col in range(Y_active.shape[1]):
        out += scipy.stats.gamma.logpdf(Y_active[:, col], a=alpha[col], scale=1.0 / beta[col])
    return out


def _mcmc_active_samples(
    datatree: xr.DataTree,
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Posterior samples for the active prior latents from a fit_ssi trace.

    The ``infectivity`` array has shape ``(chain, draw, n_nonzero)`` where
    columns correspond to days where ``incidence_vec > 0`` in ascending
    order. If ``I_r > 0`` the last column is ``Y_r`` — drop it because it
    doesn't enter ``L_SSI``.
    """
    Y_mcmc = datatree["posterior"].ds["infectivity"].values
    n_active_full = Y_mcmc.shape[-1]
    Y_flat = Y_mcmc.reshape(-1, n_active_full)
    if history[-1] > 0:
        return Y_flat[:, :-1]
    return Y_flat


def _fit_gamma_mom(
    Y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit independent ``Gamma(alpha_t, beta_t)`` per column by method of moments.

    Returns ``(alpha, beta)`` arrays of shape ``(n_cols,)``. Method-of-moments:
    ``alpha = mean^2 / var``, ``beta = mean / var``.
    """
    mean = Y.mean(axis=0)
    var = Y.var(axis=0, ddof=1)
    var = np.maximum(var, 1e-12)
    alpha = mean**2 / var
    beta = mean / var
    return alpha, beta


# ---------------------------------------------------------------------------
# Case-(iii) mixture coefficient — used by both PMO (extinction.py) and
# the closed-form SSI likelihood below.
# ---------------------------------------------------------------------------


def _log_c_m_two_later_days(
    R0_b: NDArray[np.float64],
    k_b: NDArray[np.float64],
    I_0: int,
    I_i: int,
    I_j: int,
    w_i: float,
    w_j: float,
    w_jmi: float,
    F_r: float,
    F_rmi: float,
) -> NDArray[np.float64]:
    """Log of the mixture coefficients ``c_m`` (two-later-days case) for ``m = 0, ..., I_j``.

    Returns an array of shape ``(I_j + 1,) + R0_b.shape``. The unused ``w_i``
    parameter is kept for symmetry with the calling sites and ignored here
    (it appears in the prefactor of the likelihood, not in ``c_m``).
    """
    del w_i  # used elsewhere; kept as a parameter for symmetry
    m_arr = np.arange(I_j + 1)
    expand = (I_j + 1,) + (1,) * R0_b.ndim
    m_e = m_arr.reshape(expand)
    k_b_e = k_b[np.newaxis, ...]
    R0_b_e = R0_b[np.newaxis, ...]

    log_binom = (
        scipy.special.gammaln(I_j + 1)
        - scipy.special.gammaln(m_arr + 1)
        - scipy.special.gammaln(I_j - m_arr + 1)
    ).reshape(expand)
    exp_wj = (I_j - m_arr).astype(np.float64)
    exp_wjmi = m_arr.astype(np.float64)
    with np.errstate(divide="ignore"):
        log_wj = np.log(w_j) if w_j > 0 else -np.inf
        log_wjmi = np.log(w_jmi) if w_jmi > 0 else -np.inf
    log_w_pow_arr = np.where(exp_wj > 0, exp_wj * log_wj, 0.0) + np.where(
        exp_wjmi > 0, exp_wjmi * log_wjmi, 0.0
    )
    log_w_pow = log_w_pow_arr.reshape(expand)

    alpha_0 = k_b_e * I_0 + I_i + I_j - m_e
    alpha_i = k_b_e * I_i + m_e
    log_kpf = np.log(k_b_e + R0_b_e * F_r)
    log_kpfmi = np.log(k_b_e + R0_b_e * F_rmi)

    return (
        log_binom
        + log_w_pow
        + scipy.special.gammaln(alpha_0)
        - alpha_0 * log_kpf
        + scipy.special.gammaln(alpha_i)
        - alpha_i * log_kpfmi
    )


# ---------------------------------------------------------------------------
# SSE log model evidence (closed-form for any history at fixed (R0, k))
# ---------------------------------------------------------------------------


def _log_evidence_sse_general(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSE log model evidence ``log P(I_1, ..., I_r | I_0, R0, k)`` for an arbitrary history.

    Uses the per-day NB factorisation ``I_t | I_{<t} ~ NB(R0 lambda_t,
    k lambda_t)`` with ``lambda_t = sum_{s<t} w_{t-s} I_s`` (``w_s = 0`` for
    ``s > len(w)``). Broadcasts over ``(R0, k)``.
    """
    R0_b, k_b = np.broadcast_arrays(
        np.asarray(R0, dtype=np.float64), np.asarray(k, dtype=np.float64)
    )
    r = history.size - 1
    log_L = np.zeros(R0_b.shape, dtype=np.float64)
    L = w.size
    for t in range(1, r + 1):
        lam = 0.0
        for s in range(t):
            lag = t - s
            if 1 <= lag <= L:
                lam += float(w[lag - 1]) * float(history[s])
        I_t = int(history[t])
        if lam == 0.0:
            if I_t != 0:
                log_L = log_L + np.full(R0_b.shape, -np.inf)
            continue
        log_L = log_L + scipy.stats.nbinom.logpmf(I_t, k_b * lam, k_b / (k_b + R0_b))
    return log_L


def _log_evidence_poisson_general(
    R0: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Poisson log model evidence ``log P(I_1, ..., I_r | I_0, R0)`` for an arbitrary history.

    Per-day factorisation ``I_t | I_{<t} ~ Poisson(R0 lambda_t)`` with
    ``lambda_t = sum_{s<t} w_{t-s} I_s``; ``lam == 0`` and ``I_t != 0``
    contributes ``-inf`` (same edge as the SSE form). Broadcasts over
    ``R0``.
    """
    R0_b = np.asarray(R0, dtype=np.float64)
    r = history.size - 1
    log_L = np.zeros(R0_b.shape, dtype=np.float64)
    L = w.size
    for t in range(1, r + 1):
        lam = 0.0
        for s in range(t):
            lag = t - s
            if 1 <= lag <= L:
                lam += float(w[lag - 1]) * float(history[s])
        I_t = int(history[t])
        if lam == 0.0:
            if I_t != 0:
                log_L = log_L + np.full(R0_b.shape, -np.inf)
            continue
        log_L = log_L + scipy.stats.poisson.logpmf(I_t, R0_b * lam)
    return log_L


# ---------------------------------------------------------------------------
# SSI log model evidence: closed forms for cases (i)-(iii)
# ---------------------------------------------------------------------------


def _log_evidence_ssi_analytic(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSI log model evidence for histories with cases on day 0 and at most two later days.

    Closed forms for day-0-only, one-later-day, and two-later-day histories
    (see ``notes/notes.tex`` eqs ``L_ssi_case_i``--``L_ssi_case_iii``).
    Broadcasts over ``(R0, k)``.
    """
    R0_b, k_b = np.broadcast_arrays(
        np.asarray(R0, dtype=np.float64), np.asarray(k, dtype=np.float64)
    )
    cls = classify_history(history)
    if cls["kind"] == "general":
        raise ValueError(
            "_log_evidence_ssi_analytic supports histories with cases on day 0 and "
            "at most two later days."
        )
    F = cumulative(w)
    r = history.size - 1
    F_r = float(F[min(r, F.size - 1)])
    I_0 = cls["I_0"]
    if cls["kind"] == "day0_only":
        return -k_b * I_0 * np.log1p(R0_b * F_r / k_b)

    i = cls["i"]
    I_i = cls["I_i"]
    F_rmi = float(F[min(r - i, F.size - 1)])
    w_i = w_at(w, i)

    if cls["kind"] == "one_later":
        return (
            I_i * np.log(R0_b * w_i)
            - scipy.special.gammaln(I_i + 1)
            + scipy.special.gammaln(k_b * I_0 + I_i)
            - scipy.special.gammaln(k_b * I_0)
            + k_b * I_0 * np.log(k_b)
            - (k_b * I_0 + I_i) * np.log(k_b + R0_b * F_r)
            + k_b * I_i * (np.log(k_b) - np.log(k_b + R0_b * F_rmi))
        )

    j = cls["j"]
    I_j = cls["I_j"]
    F_rmj = float(F[min(r - j, F.size - 1)])
    w_j = w_at(w, j)
    w_jmi = w_at(w, j - i)

    log_c_m = _log_c_m_two_later_days(R0_b, k_b, I_0, I_i, I_j, w_i, w_j, w_jmi, F_r, F_rmi)
    log_sum_c = scipy.special.logsumexp(log_c_m, axis=0)
    return (
        I_i * np.log(R0_b * w_i)
        + I_j * np.log(R0_b)
        - scipy.special.gammaln(I_i + 1)
        - scipy.special.gammaln(I_j + 1)
        + (k_b * I_0 + k_b * I_i) * np.log(k_b)
        - scipy.special.gammaln(k_b * I_0)
        - scipy.special.gammaln(k_b * I_i)
        + k_b * I_j * (np.log(k_b) - np.log(k_b + R0_b * F_rmj))
        + log_sum_c
    )


# ---------------------------------------------------------------------------
# Generic log-evidence helpers (model- and proposal-agnostic)
#
# Each helper takes already-evaluated log-target / log-proposal arrays on
# pre-drawn samples and applies the corresponding estimator's arithmetic.
# Callers are responsible for drawing samples and evaluating the log targets;
# the SSE and SSI wrappers below build those closures.
# ---------------------------------------------------------------------------


EvidenceMethod = Literal["naive", "importance_sampling", "bridge"]


# Default sample sizes; can be overridden via ``n_samples`` / ``n_proposal_samples``.
_DEFAULT_NAIVE_SAMPLES = 100_000
_DEFAULT_IS_SAMPLES = 50_000
_DEFAULT_BRIDGE_MAX_ITER = 1000
_DEFAULT_BRIDGE_TOL = 1e-8


def _log_mean_exp(x: NDArray[np.float64]) -> float:
    """``log(mean(exp(x)))`` in a numerically stable way."""
    return float(scipy.special.logsumexp(x) - np.log(x.size))


def _naive_mc_from_log_target(log_target_at_prior_draws: NDArray[np.float64]) -> float:
    """Naive Monte-Carlo log-evidence: ``log E_prior[exp(log_target)]``.

    ``log_target_at_prior_draws[i]`` is the log of the unnormalised conditional
    target at the ``i``-th prior draw — typically ``log p(I | latents_i)``.
    """
    return _log_mean_exp(log_target_at_prior_draws)


def _importance_sampling_from_log_arrays(
    log_unnorm_target_at_proposal: NDArray[np.float64],
    log_proposal_at_proposal: NDArray[np.float64],
) -> float:
    """Importance-sampling log-evidence given precomputed log-arrays.

    The unnormalised target is ``log p(I, latents) = log p(I | latents) +
    log p(latents)``; the proposal density is whatever was used to draw the
    samples (typically MoM-fitted independent Gammas).
    """
    return _log_mean_exp(log_unnorm_target_at_proposal - log_proposal_at_proposal)


def _bridge_from_log_arrays(
    log_unnorm_target_at_p: NDArray[np.float64],
    log_unnorm_target_at_q: NDArray[np.float64],
    log_proposal_at_p: NDArray[np.float64],
    log_proposal_at_q: NDArray[np.float64],
    *,
    max_iter: int = _DEFAULT_BRIDGE_MAX_ITER,
    tol: float = _DEFAULT_BRIDGE_TOL,
) -> float:
    """Meng-Wong log-space bridge fixed-point iteration (Gronau et al. 2017).

    ``..._at_p`` arrays are evaluated on samples from the (unnormalised) target
    (typically the MCMC posterior trace), ``..._at_q`` on samples from the
    proposal. Returns the log model evidence.
    """
    N1 = log_unnorm_target_at_p.size
    N2 = log_unnorm_target_at_q.size
    l_p = log_unnorm_target_at_p - log_proposal_at_p
    l_q = log_unnorm_target_at_q - log_proposal_at_q

    log_s1 = np.log(N1 / (N1 + N2))
    log_s2 = np.log(N2 / (N1 + N2))

    log_r = float(np.median(l_q))
    for _ in range(max_iter):
        log_denom_q = np.logaddexp(log_s1, log_s2 + l_q - log_r)
        log_num = scipy.special.logsumexp(l_q - log_denom_q) - np.log(N2)

        log_denom_p = np.logaddexp(log_s1, log_s2 + l_p - log_r)
        log_den = scipy.special.logsumexp(-log_denom_p) - np.log(N1)

        log_r_new = log_num - log_den
        if abs(log_r_new - log_r) < tol:
            return float(log_r_new)
        log_r = log_r_new
    return float(log_r)


def _sample_indep_gamma(
    alpha: NDArray[np.float64],
    beta: NDArray[np.float64],
    n_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw ``n_samples`` rows of independent ``Gamma(alpha_j, beta_j)`` columns."""
    out = np.empty((n_samples, alpha.size), dtype=np.float64)
    for col in range(alpha.size):
        out[:, col] = rng.gamma(shape=alpha[col], scale=1.0 / beta[col], size=n_samples)
    return out


# ---------------------------------------------------------------------------
# Helpers for parameter-uncertain evidence (when R0 and / or k are Priors)
# ---------------------------------------------------------------------------


def _sample_param(
    x: float | Prior, n_samples: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Per-sample parameter array: draw from a :class:`Prior`, else broadcast a scalar."""
    if isinstance(x, Prior):
        return x.sample(n_samples, rng)
    return np.full(n_samples, float(x), dtype=np.float64)


def _prior_logpdf_or_zero(
    x: float | Prior, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``log π(values)`` when ``x`` is a :class:`Prior`; else a zero array.

    For fixed parameters this contributes nothing to the importance weight or
    bridge ratio — the prior factor is absent from the target.
    """
    if isinstance(x, Prior):
        return x.log_pdf(values)
    return np.zeros(values.shape, dtype=np.float64)


def _param_posterior_samples(
    datatree: xr.DataTree, name: str
) -> NDArray[np.float64] | None:
    """Flatten the ``(chain, draw)`` posterior of ``name``; ``None`` if absent.

    ``name`` is the PyMC RV name used by :func:`sse_ssi_pmo.inference.fit_sse`
    and :func:`fit_ssi`: ``"rep_no"`` for :math:`R_0` and ``"dispersion"`` for
    :math:`k`.
    """
    if name not in datatree["posterior"].ds:
        return None
    return datatree["posterior"].ds[name].values.reshape(-1).astype(np.float64)


def _augmented_posterior_samples(
    datatree: xr.DataTree,
    history: NDArray[np.int64],
    *,
    R0_uncertain: bool,
    k_uncertain: bool,
    include_Y: bool,
) -> tuple[NDArray[np.float64], int]:
    """Stack posterior samples for the (Y_active?, R0?, k?) augmented latent vector.

    Returns ``(samples, n_y_cols)`` where ``samples`` has shape
    ``(n_draws, n_y_cols + n_uncertain_params)`` — Y columns first, then R0,
    then k. ``n_y_cols`` is the SSI Y-active count (0 when ``include_Y`` is
    False, e.g. for the SSE-side estimators).
    """
    cols: list[NDArray[np.float64]] = []
    if include_Y:
        Y = _mcmc_active_samples(datatree, history)
        cols.append(Y)
        n_y_cols = Y.shape[1]
    else:
        n_y_cols = 0
    if R0_uncertain:
        rep = _param_posterior_samples(datatree, "rep_no")
        if rep is None:
            raise ValueError("fit trace has no 'rep_no' posterior — was R0 passed as Prior?")
        cols.append(rep.reshape(-1, 1))
    if k_uncertain:
        disp = _param_posterior_samples(datatree, "dispersion")
        if disp is None:
            raise ValueError("fit trace has no 'dispersion' posterior — was k passed as Prior?")
        cols.append(disp.reshape(-1, 1))
    if not cols:
        raise ValueError("_augmented_posterior_samples called with nothing to stack")
    if len(cols) == 1:
        return cols[0], n_y_cols
    n_draws = cols[0].shape[0]
    aligned = [c if c.shape[0] == n_draws else c[:n_draws] for c in cols]
    return np.concatenate(aligned, axis=1).astype(np.float64), n_y_cols


def _unpack_augmented(
    samples: NDArray[np.float64],
    n_y_cols: int,
    *,
    R0: float | Prior,
    k: float | Prior,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split ``samples`` into ``(Y_active, R0_per_sample, k_per_sample)``.

    For fixed parameters the per-sample array is a constant broadcast. The
    column order in ``samples`` is ``[Y_active, R0?, k?]``.
    """
    n_samples = samples.shape[0]
    Y_active = samples[:, :n_y_cols]
    col = n_y_cols
    if isinstance(R0, Prior):
        R0_arr = samples[:, col].astype(np.float64)
        col += 1
    else:
        R0_arr = np.full(n_samples, float(R0), dtype=np.float64)
    if isinstance(k, Prior):
        k_arr = samples[:, col].astype(np.float64)
        col += 1
    else:
        k_arr = np.full(n_samples, float(k), dtype=np.float64)
    return Y_active, R0_arr, k_arr


def _sample_Y_given_k_per_row(
    k_arr: NDArray[np.float64],
    history: NDArray[np.int64],
    prior_idx: NDArray[np.int64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample ``Y_active`` from its SSI prior with per-sample ``k_i`` shape parameter.

    ``Y[i, j] ~ Gamma(k_i * I_{prior_idx[j]}, k_i)``. Used by the SSI naive MC
    estimator under a prior on :math:`k`.
    """
    n_samples = k_arr.size
    Y = np.empty((n_samples, prior_idx.size), dtype=np.float64)
    for col, t in enumerate(prior_idx):
        Y[:, col] = rng.gamma(k_arr * float(history[t]), 1.0 / k_arr)
    return Y


# ---------------------------------------------------------------------------
# SSI log model evidence: MCMC-based estimators for general histories
# ---------------------------------------------------------------------------


def _log_evidence_ssi_naive_mc(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Naive Monte Carlo: average ``p(I|Y, R0)`` over (R0, k, Y) draws from the prior.

    With fixed ``(R0, k)`` reduces to drawing ``Y`` from the SSI Gamma prior
    and averaging ``p(I|Y)``. With ``R0`` and / or ``k`` a
    :class:`~sse_ssi_pmo.priors.Prior` we additionally draw each parameter
    from its prior; ``Y`` is then drawn from ``Gamma(k_i I_t, k_i)`` with the
    per-sample ``k_i`` (so the joint draw matches the SSI generative model).

    No MCMC required. Variance grows as the data become more informative
    (prior and posterior diverge); under a wide prior on ``(R0, k)`` the
    variance also grows with the additional parameter dimensions.
    """
    prior_idx = _active_prior_idx(history)
    R0_arr = _sample_param(R0, n_samples, rng)
    k_arr = _sample_param(k, n_samples, rng)
    Y = _sample_Y_given_k_per_row(k_arr, history, prior_idx, rng)
    log_lik = _log_lik_given_Y(Y, R0_arr, w, history, prior_idx)
    return _naive_mc_from_log_target(log_lik)


def _log_evidence_ssi_importance_sampling(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    datatree: xr.DataTree,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Importance sampling with an independent-Gamma proposal fitted to MCMC marginals.

    The proposal is the same family as the SSI prior on the latent
    infectivities (independent Gammas), with parameters set by
    method-of-moments to match the per-day MCMC marginal moments. When
    ``R0`` and / or ``k`` are :class:`~sse_ssi_pmo.priors.Prior` instances the
    proposal is augmented with the corresponding MoM-fitted marginal(s) from
    the trace, and the importance weight gains the matching prior log-pdf
    factor.
    """
    prior_idx = _active_prior_idx(history)
    R0_uncertain = isinstance(R0, Prior)
    k_uncertain = isinstance(k, Prior)
    samples_p, n_y = _augmented_posterior_samples(
        datatree, history, R0_uncertain=R0_uncertain, k_uncertain=k_uncertain, include_Y=True
    )
    alpha, beta = _fit_gamma_mom(samples_p)
    samples_q = _sample_indep_gamma(alpha, beta, n_samples, rng)
    Y_q, R0_q, k_q = _unpack_augmented(samples_q, n_y, R0=R0, k=k)

    log_unnorm_target = (
        _log_lik_given_Y(Y_q, R0_q, w, history, prior_idx)
        + _log_prior_Y(Y_q, k_q, history, prior_idx)
        + _prior_logpdf_or_zero(R0, R0_q)
        + _prior_logpdf_or_zero(k, k_q)
    )
    log_proposal = _log_proposal_Y(samples_q, alpha, beta)
    return _importance_sampling_from_log_arrays(log_unnorm_target, log_proposal)


def _log_evidence_ssi_bridge(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    datatree: xr.DataTree,
    n_proposal_samples: int,
    rng: np.random.Generator,
    max_iter: int = _DEFAULT_BRIDGE_MAX_ITER,
    tol: float = _DEFAULT_BRIDGE_TOL,
) -> float:
    """Meng-Wong bridge sampling, log-space iterative form (Gronau et al. 2017).

    Proposal: independent Gammas fitted to the augmented posterior marginals
    (same family as IS). The augmented latent space is
    ``(Y_active, R0?, k?)`` — the ``?`` columns are present when the
    corresponding parameter is a :class:`~sse_ssi_pmo.priors.Prior`. Iterates
    the bridge fixed-point in log space until ``|log r_{t+1} - log r_t| <
    tol`` or ``max_iter`` is reached.
    """
    prior_idx = _active_prior_idx(history)
    R0_uncertain = isinstance(R0, Prior)
    k_uncertain = isinstance(k, Prior)
    samples_p, n_y = _augmented_posterior_samples(
        datatree, history, R0_uncertain=R0_uncertain, k_uncertain=k_uncertain, include_Y=True
    )
    alpha, beta = _fit_gamma_mom(samples_p)
    samples_q = _sample_indep_gamma(alpha, beta, n_proposal_samples, rng)

    Y_p, R0_p, k_p = _unpack_augmented(samples_p, n_y, R0=R0, k=k)
    Y_q, R0_q, k_q = _unpack_augmented(samples_q, n_y, R0=R0, k=k)

    log_unnorm_p = (
        _log_lik_given_Y(Y_p, R0_p, w, history, prior_idx)
        + _log_prior_Y(Y_p, k_p, history, prior_idx)
        + _prior_logpdf_or_zero(R0, R0_p)
        + _prior_logpdf_or_zero(k, k_p)
    )
    log_unnorm_q = (
        _log_lik_given_Y(Y_q, R0_q, w, history, prior_idx)
        + _log_prior_Y(Y_q, k_q, history, prior_idx)
        + _prior_logpdf_or_zero(R0, R0_q)
        + _prior_logpdf_or_zero(k, k_q)
    )
    log_g_p = _log_proposal_Y(samples_p, alpha, beta)
    log_g_q = _log_proposal_Y(samples_q, alpha, beta)
    return _bridge_from_log_arrays(
        log_unnorm_p, log_unnorm_q, log_g_p, log_g_q, max_iter=max_iter, tol=tol
    )


def _log_evidence_ssi_mcmc(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    evidence_method: EvidenceMethod,
    datatree: xr.DataTree | None = None,
    rng: np.random.Generator | None = None,
    n_samples: int | None = None,
    show_progress: bool = False,
    **mcmc_kwargs,
) -> float:
    """Dispatch ``log Z_SSI`` estimation to the chosen MCMC-based method.

    Parameters
    ----------
    R0, k
        Fixed scalars or :class:`~sse_ssi_pmo.priors.Prior` instances. When
        priors are supplied the trace is fitted with ``R0=None`` / ``k=None``
        (the relevant parameter becomes an inferred quantity), and the IS /
        bridge proposal is augmented to include those marginals.
    evidence_method
        ``"naive"``, ``"importance_sampling"``, or ``"bridge"``.
    datatree
        Pre-existing ``fit_ssi`` trace. Required for ``"importance_sampling"``
        and ``"bridge"``; if ``None`` it is computed by running ``fit_ssi``.
        Ignored (no MCMC needed) for ``"naive"``.
    rng
        NumPy ``Generator``; ``None`` ⇒ a fresh default generator.
    n_samples
        Number of Monte-Carlo draws (naive / IS) or proposal samples (bridge).
        Defaults: 100_000 (naive), 50_000 (IS), total MCMC draws (bridge).
    show_progress
        Forwarded to ``fit_ssi`` if MCMC is run internally.
    **mcmc_kwargs
        Forwarded to ``pm.sample`` when MCMC is run internally; ignored when
        a ``datatree`` is supplied or the method is ``"naive"``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if evidence_method == "naive":
        n_naive = n_samples if n_samples is not None else _DEFAULT_NAIVE_SAMPLES
        return _log_evidence_ssi_naive_mc(R0, k, w, history, n_samples=n_naive, rng=rng)

    if datatree is None:
        from sse_ssi_pmo.inference import fit_ssi

        thin = mcmc_kwargs.pop("thin", 1)
        mcmc_kwargs.setdefault("progressbar", show_progress)
        priors_kw = _fit_priors_from_prior_args(R0, k)
        datatree = fit_ssi(
            history,
            w,
            R0=_fixed_or_none(R0),
            k=_fixed_or_none(k),
            priors=priors_kw,
            thin=thin,
            **mcmc_kwargs,
        )

    if evidence_method == "importance_sampling":
        n_is = n_samples if n_samples is not None else _DEFAULT_IS_SAMPLES
        return _log_evidence_ssi_importance_sampling(
            R0, k, w, history, datatree=datatree, n_samples=n_is, rng=rng
        )

    if evidence_method == "bridge":
        if n_samples is None:
            sizes = datatree["posterior"].ds.sizes
            n_proposal = int(sizes["chain"]) * int(sizes["draw"])
        else:
            n_proposal = n_samples
        return _log_evidence_ssi_bridge(
            R0, k, w, history, datatree=datatree, n_proposal_samples=n_proposal, rng=rng
        )

    raise ValueError(
        "evidence_method must be 'naive', 'importance_sampling', or 'bridge'; "
        f"got {evidence_method!r}"
    )


def _fit_priors_from_prior_args(
    R0: float | Prior, k: float | Prior
) -> dict | None:
    """Build the ``priors=`` dict for :func:`fit_sse` / :func:`fit_ssi`.

    Only includes entries for parameters that are :class:`Prior` instances —
    fixed parameters never reach the prior dispatch in the inference module.
    Returns ``None`` if neither argument is a Prior (no override).
    """
    import pymc as pm

    priors: dict = {}
    if isinstance(R0, Prior):
        cls = pm.Gamma if R0.family == "gamma" else pm.LogNormal
        priors["rep_no"] = (cls, dict(R0.pymc_params))
    if isinstance(k, Prior):
        cls = pm.Gamma if k.family == "gamma" else pm.LogNormal
        priors["dispersion"] = (cls, dict(k.pymc_params))
    return priors or None


def _fixed_or_none(x: float | Prior) -> float | None:
    """``None`` when ``x`` is a :class:`Prior` (signalling 'infer'); else ``float(x)``."""
    if isinstance(x, Prior):
        return None
    return float(x)


# ---------------------------------------------------------------------------
# SSE log model evidence: MCMC-based estimators under priors on (R0, k)
# ---------------------------------------------------------------------------


def _log_evidence_sse_naive_mc(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Naive Monte Carlo SSE evidence: average closed-form likelihood over θ-draws.

    Samples ``(R0, k)`` from the supplied priors (fixed parameters broadcast)
    and averages ``exp(_log_evidence_sse_general(R0_i, k_i, w, history))`` —
    the SSE conditional likelihood is closed-form for any history.
    """
    R0_arr = _sample_param(R0, n_samples, rng)
    k_arr = _sample_param(k, n_samples, rng)
    log_lik = _log_evidence_sse_general(R0_arr, k_arr, w, history)
    return _naive_mc_from_log_target(log_lik)


def _log_evidence_sse_importance_sampling(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    datatree: xr.DataTree,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """IS SSE evidence on the (R0?, k?) augmented prior axis using a fit_sse trace.

    The proposal is independent Gammas MoM-fitted to each posterior marginal
    of the uncertain parameter(s). At least one of ``R0`` / ``k`` must be a
    :class:`Prior` — otherwise the SSE evidence is closed-form, and the
    caller should use that path directly.
    """
    if not (isinstance(R0, Prior) or isinstance(k, Prior)):
        raise ValueError(
            "_log_evidence_sse_importance_sampling: at least one of R0, k must be a Prior; "
            "use _log_evidence_sse_general directly for the fixed-(R0, k) case."
        )
    R0_uncertain = isinstance(R0, Prior)
    k_uncertain = isinstance(k, Prior)
    samples_p, _ = _augmented_posterior_samples(
        datatree, history, R0_uncertain=R0_uncertain, k_uncertain=k_uncertain, include_Y=False
    )
    alpha, beta = _fit_gamma_mom(samples_p)
    samples_q = _sample_indep_gamma(alpha, beta, n_samples, rng)
    _Y, R0_q, k_q = _unpack_augmented(samples_q, 0, R0=R0, k=k)

    log_unnorm_target = (
        _log_evidence_sse_general(R0_q, k_q, w, history)
        + _prior_logpdf_or_zero(R0, R0_q)
        + _prior_logpdf_or_zero(k, k_q)
    )
    log_proposal = _log_proposal_Y(samples_q, alpha, beta)
    return _importance_sampling_from_log_arrays(log_unnorm_target, log_proposal)


def _log_evidence_sse_bridge(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    datatree: xr.DataTree,
    n_proposal_samples: int,
    rng: np.random.Generator,
    max_iter: int = _DEFAULT_BRIDGE_MAX_ITER,
    tol: float = _DEFAULT_BRIDGE_TOL,
) -> float:
    """Meng-Wong bridge sampling for the SSE evidence under priors on (R0, k).

    Mirrors the SSI bridge but on the (R0?, k?) parameter axis only — the
    SSE conditional likelihood has no latent variables to marginalise.
    """
    if not (isinstance(R0, Prior) or isinstance(k, Prior)):
        raise ValueError(
            "_log_evidence_sse_bridge: at least one of R0, k must be a Prior; "
            "use _log_evidence_sse_general directly for the fixed-(R0, k) case."
        )
    R0_uncertain = isinstance(R0, Prior)
    k_uncertain = isinstance(k, Prior)
    samples_p, _ = _augmented_posterior_samples(
        datatree, history, R0_uncertain=R0_uncertain, k_uncertain=k_uncertain, include_Y=False
    )
    alpha, beta = _fit_gamma_mom(samples_p)
    samples_q = _sample_indep_gamma(alpha, beta, n_proposal_samples, rng)

    _Yp, R0_p, k_p = _unpack_augmented(samples_p, 0, R0=R0, k=k)
    _Yq, R0_q, k_q = _unpack_augmented(samples_q, 0, R0=R0, k=k)

    log_unnorm_p = (
        _log_evidence_sse_general(R0_p, k_p, w, history)
        + _prior_logpdf_or_zero(R0, R0_p)
        + _prior_logpdf_or_zero(k, k_p)
    )
    log_unnorm_q = (
        _log_evidence_sse_general(R0_q, k_q, w, history)
        + _prior_logpdf_or_zero(R0, R0_q)
        + _prior_logpdf_or_zero(k, k_q)
    )
    log_g_p = _log_proposal_Y(samples_p, alpha, beta)
    log_g_q = _log_proposal_Y(samples_q, alpha, beta)
    return _bridge_from_log_arrays(
        log_unnorm_p, log_unnorm_q, log_g_p, log_g_q, max_iter=max_iter, tol=tol
    )


def _log_evidence_sse_mcmc(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    evidence_method: EvidenceMethod,
    datatree: xr.DataTree | None = None,
    rng: np.random.Generator | None = None,
    n_samples: int | None = None,
    show_progress: bool = False,
    **mcmc_kwargs,
) -> float:
    """Dispatch ``log Z_SSE`` estimation under a prior on (R0, k).

    With both ``R0`` and ``k`` scalar this is just the closed-form
    :func:`_log_evidence_sse_general` and the dispatcher should not be
    needed; for the parameter-uncertain case the dispatcher picks between
    the three estimators (analogue of :func:`_log_evidence_ssi_mcmc`).
    """
    if not isinstance(R0, Prior) and not isinstance(k, Prior):
        # Closed-form short-circuit so callers can hit a single dispatcher.
        return float(_log_evidence_sse_general(R0, k, w, history))
    if rng is None:
        rng = np.random.default_rng()

    if evidence_method == "naive":
        n_naive = n_samples if n_samples is not None else _DEFAULT_NAIVE_SAMPLES
        return _log_evidence_sse_naive_mc(R0, k, w, history, n_samples=n_naive, rng=rng)

    if datatree is None:
        from sse_ssi_pmo.inference import fit_sse

        thin = mcmc_kwargs.pop("thin", 1)
        mcmc_kwargs.setdefault("progressbar", show_progress)
        priors_kw = _fit_priors_from_prior_args(R0, k)
        datatree = fit_sse(
            history,
            w,
            R0=_fixed_or_none(R0),
            k=_fixed_or_none(k),
            priors=priors_kw,
            thin=thin,
            **mcmc_kwargs,
        )

    if evidence_method == "importance_sampling":
        n_is = n_samples if n_samples is not None else _DEFAULT_IS_SAMPLES
        return _log_evidence_sse_importance_sampling(
            R0, k, w, history, datatree=datatree, n_samples=n_is, rng=rng
        )

    if evidence_method == "bridge":
        if n_samples is None:
            sizes = datatree["posterior"].ds.sizes
            n_proposal = int(sizes["chain"]) * int(sizes["draw"])
        else:
            n_proposal = n_samples
        return _log_evidence_sse_bridge(
            R0, k, w, history, datatree=datatree, n_proposal_samples=n_proposal, rng=rng
        )

    raise ValueError(
        "evidence_method must be 'naive', 'importance_sampling', or 'bridge'; "
        f"got {evidence_method!r}"
    )
