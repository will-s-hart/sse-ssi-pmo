"""Marginal-likelihood building blocks for the SSE and SSI models.

For an observed incidence history :math:`I_0, I_1, \\ldots, I_r`, the marginal
log-likelihood :math:`\\log L_M = \\log p_M(I_1, \\ldots, I_r \\mid I_0)`
underwrites Bayesian model averaging across :math:`M \\in \\{\\text{SSE},
\\text{SSI}\\}`.

* SSE — closed-form for any history (product of per-day NB pmfs); see
  :func:`_log_likelihood_sse_general`.
* SSI — closed-form for the same three special cases as for the extinction
  probability (cases on day 0 only, on day 0 and one later day, on day 0 and
  two later days); see :func:`_log_likelihood_ssi_analytic`. For a general
  history, three MCMC-based estimators are available, all routed through the
  dispatcher :func:`_log_likelihood_ssi_mcmc`:

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
from sse_ssi_pmo.serial_interval import cumulative

# ---------------------------------------------------------------------------
# helpers shared between SSI marginal-likelihood estimators
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
    R0: float,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    prior_idx: NDArray[np.int64],
) -> NDArray[np.float64]:
    """``log p(I_{1:r} | Y)`` for each row of ``Y_active``.

    ``Y_active`` has shape ``(n_samples, len(prior_idx))``; days outside
    ``prior_idx`` (and day ``r``) are zero-padded into a length-``r+1`` vector
    when computing the per-day FOI.
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
    k: float,
    history: NDArray[np.int64],
    prior_idx: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Sum over active days of ``log Gamma(k I_t, k)`` prior pdf at ``Y_t``."""
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
# SSE log-likelihood (closed-form for any history)
# ---------------------------------------------------------------------------


def _log_likelihood_sse_general(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSE log-likelihood ``log P(I_1, ..., I_r | I_0)`` for an arbitrary history.

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


def _log_likelihood_poisson_general(
    R0: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Poisson log-likelihood ``log P(I_1, ..., I_r | I_0)`` for an arbitrary history.

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
# SSI log-likelihood: closed forms for cases (i)-(iii)
# ---------------------------------------------------------------------------


def _log_likelihood_ssi_analytic(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSI log-likelihood for histories with cases on day 0 and at most two later days.

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
            "_log_likelihood_ssi_analytic supports histories with cases on day 0 and "
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
# SSI log-likelihood: MCMC-based estimators for general histories
# ---------------------------------------------------------------------------


SsiEvidenceMethod = Literal["naive", "importance_sampling", "bridge"]


# Default sample sizes; can be overridden via ``n_samples`` / ``n_proposal_samples``.
_DEFAULT_NAIVE_SAMPLES = 100_000
_DEFAULT_IS_SAMPLES = 50_000
_DEFAULT_BRIDGE_MAX_ITER = 1000
_DEFAULT_BRIDGE_TOL = 1e-8


def _log_likelihood_ssi_naive_mc(
    R0: float,
    k: float,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Naive Monte Carlo: average ``p(I|Y)`` over draws from the Gamma prior.

    No MCMC required. Variance grows as the data become more informative
    (prior and posterior diverge). On the histories tested in Phase A
    (incidence up to 20, up to 4 later days) gives SD < 0.013 in log space at
    100k samples.
    """
    prior_idx = _active_prior_idx(history)
    Y = np.empty((n_samples, prior_idx.size), dtype=np.float64)
    for col, t in enumerate(prior_idx):
        Y[:, col] = rng.gamma(shape=k * float(history[t]), scale=1.0 / k, size=n_samples)
    log_lik = _log_lik_given_Y(Y, R0, w, history, prior_idx)
    return float(scipy.special.logsumexp(log_lik) - np.log(n_samples))


def _log_likelihood_ssi_importance_sampling(
    R0: float,
    k: float,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    datatree: xr.DataTree,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Importance sampling with an independent-Gamma proposal fitted to MCMC marginals.

    The proposal is the same family as the prior (independent Gammas), with
    parameters set by method-of-moments to match the per-day MCMC marginal
    moments. Importance weights are bounded by construction.
    """
    prior_idx = _active_prior_idx(history)
    Y_mcmc = _mcmc_active_samples(datatree, history)
    alpha, beta = _fit_gamma_mom(Y_mcmc)

    Y = np.empty((n_samples, prior_idx.size), dtype=np.float64)
    for col in range(prior_idx.size):
        Y[:, col] = rng.gamma(shape=alpha[col], scale=1.0 / beta[col], size=n_samples)

    log_w_imp = (
        _log_lik_given_Y(Y, R0, w, history, prior_idx)
        + _log_prior_Y(Y, k, history, prior_idx)
        - _log_proposal_Y(Y, alpha, beta)
    )
    return float(scipy.special.logsumexp(log_w_imp) - np.log(n_samples))


def _log_likelihood_ssi_bridge(
    R0: float,
    k: float,
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

    Proposal: independent Gammas fitted to MCMC marginals (same as IS).
    Iterates the bridge fixed-point in log space until ``|log r_{t+1} - log r_t|
    < tol`` or ``max_iter`` is reached. Phase A converged in ≤ 5 iterations on
    every history tested.
    """
    prior_idx = _active_prior_idx(history)
    Y_p = _mcmc_active_samples(datatree, history)
    N1 = Y_p.shape[0]

    alpha, beta = _fit_gamma_mom(Y_p)

    N2 = n_proposal_samples
    Y_q = np.empty((N2, prior_idx.size), dtype=np.float64)
    for col in range(prior_idx.size):
        Y_q[:, col] = rng.gamma(shape=alpha[col], scale=1.0 / beta[col], size=N2)

    log_p_tilde_p = _log_lik_given_Y(Y_p, R0, w, history, prior_idx) + _log_prior_Y(
        Y_p, k, history, prior_idx
    )
    log_p_tilde_q = _log_lik_given_Y(Y_q, R0, w, history, prior_idx) + _log_prior_Y(
        Y_q, k, history, prior_idx
    )
    log_g_p = _log_proposal_Y(Y_p, alpha, beta)
    log_g_q = _log_proposal_Y(Y_q, alpha, beta)

    l_p = log_p_tilde_p - log_g_p
    l_q = log_p_tilde_q - log_g_q

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
            log_r = log_r_new
            break
        log_r = log_r_new
    return float(log_r)


def _log_likelihood_ssi_mcmc(
    R0: float,
    k: float,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    ssi_evidence_method: SsiEvidenceMethod,
    datatree: xr.DataTree | None = None,
    rng: np.random.Generator | None = None,
    n_samples: int | None = None,
    show_progress: bool = False,
    **mcmc_kwargs,
) -> float:
    """Dispatch ``log L_SSI`` estimation to the chosen MCMC-based method.

    Parameters
    ----------
    ssi_evidence_method
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

    if ssi_evidence_method == "naive":
        n_naive = n_samples if n_samples is not None else _DEFAULT_NAIVE_SAMPLES
        return _log_likelihood_ssi_naive_mc(R0, k, w, history, n_samples=n_naive, rng=rng)

    if datatree is None:
        from sse_ssi_pmo.inference import fit_ssi

        thin = mcmc_kwargs.pop("thin", 1)
        mcmc_kwargs.setdefault("progressbar", show_progress)
        datatree = fit_ssi(history, w, R0=R0, k=k, thin=thin, **mcmc_kwargs)

    if ssi_evidence_method == "importance_sampling":
        n_is = n_samples if n_samples is not None else _DEFAULT_IS_SAMPLES
        return _log_likelihood_ssi_importance_sampling(
            R0, k, w, history, datatree=datatree, n_samples=n_is, rng=rng
        )

    if ssi_evidence_method == "bridge":
        if n_samples is None:
            sizes = datatree["posterior"].ds.sizes
            n_proposal = int(sizes["chain"]) * int(sizes["draw"])
        else:
            n_proposal = n_samples
        return _log_likelihood_ssi_bridge(
            R0, k, w, history, datatree=datatree, n_proposal_samples=n_proposal, rng=rng
        )

    raise ValueError(
        "ssi_evidence_method must be 'naive', 'importance_sampling', or 'bridge'; "
        f"got {ssi_evidence_method!r}"
    )
