"""Analytic extinction-probability and likelihood building blocks for the SSE and SSI models.

For an offspring distribution :math:`\\mathrm{NB}(\\text{mean}=\\mu,\\text{disp}=\\kappa)`,
branching-process theory says the extinction probability ``q`` is the smallest
non-negative fixed point of the pgf

.. math::

    G(s) = \\bigl(1 + (\\mu / \\kappa)(1 - s)\\bigr)^{-\\kappa}.

If ``mu <= 1`` the only fixed point in ``[0, 1]`` is ``q = 1``; otherwise the
unique fixed point in ``(0, 1)`` is found by ``scipy.optimize.brentq`` on
``f(q) = G(q) - q``.

After conditioning on an observed incidence history ``I_0, ..., I_r``, the
extinction probability is ``q_r = G_r(q)`` where ``G_r`` is the pgf of the
remaining offspring distribution of the observed cases. See ``notes/notes.tex``
for the SSE and SSI derivations. The closed forms exposed here are:

* SSE (general history): ``q_r = q ** Lambda`` with
  ``Lambda = sum_{s=0..r} I_s * (1 - F_{r-s})``; the joint likelihood is the
  product of per-day NB pmfs (``_log_likelihood_sse_general``).
* SSI: closed-form ``q_r`` and likelihood for histories with cases on day 0
  only, on day 0 and one later day, or on day 0 and two later days.
  See ``_pmo_ssi_analytic`` and ``_log_likelihood_ssi_analytic``.

The functions in this module are private helpers used by ``pmo.py``; they do
not validate user-facing inputs (the dispatcher is responsible for that).
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from sse_ssi_pmo.serial_interval import cumulative

# Upper bracket for brentq: just below 1 so we don't trivially hit q = 1.
_Q_UPPER = 1.0 - 1e-12


def _nb_extinction_prob(mean: float, disp: float) -> float:
    """Extinction probability for a single ``NB(mean, disp)`` offspring distribution.

    ``disp`` is the NB shape/dispersion parameter (the exponent ``-kappa`` in
    the pgf). When the offspring distribution has ``mean <= 1`` extinction is
    certain, so we return 1.
    """
    if mean <= 1.0 or disp <= 0.0:
        return 1.0

    def f(q: float) -> float:
        return (1.0 + (mean / disp) * (1.0 - q)) ** (-disp) - q

    # f(0) = G(0) > 0; f(_Q_UPPER) < 0 because mean > 1 implies G'(1) > 1, so
    # G(q) < q just below 1. Thus brentq finds the unique root in (0, 1).
    return float(scipy.optimize.brentq(f, 0.0, _Q_UPPER))


def _q_array(
    R0: ArrayLike, k: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve ``q`` on the broadcast of ``(R0, k)`` — one root-find per pair.

    Returns ``(q, R0_b, k_b)`` all sharing the broadcast shape, so callers can
    reuse the broadcast ``R0_b`` / ``k_b`` when evaluating closed forms.
    """
    R0_arr = np.asarray(R0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    R0_b, k_b = np.broadcast_arrays(R0_arr, k_arr)
    q = np.empty(R0_b.shape, dtype=np.float64)
    for idx in np.ndindex(R0_b.shape):
        q[idx] = _nb_extinction_prob(float(R0_b[idx]), float(k_b[idx]))
    return q, R0_b, k_b


def _lambda_from_history(w: NDArray[np.float64], history: NDArray[np.int64]) -> float:
    """Pooled effective weight ``Lambda = sum_{s=0..r} I_s (1 - F_{r-s})``.

    Indices ``r - s`` exceeding the support of ``w`` are clipped to
    ``F.size - 1``, where ``F = sum(w)``: the residual weight at lags beyond
    ``len(w)`` is taken to be zero (consistent with how ``cumulative``
    truncates the serial-interval distribution).
    """
    F = cumulative(w)
    r = history.size - 1
    idx = np.minimum(r - np.arange(r + 1), F.size - 1)
    return float((history.astype(np.float64) * (1.0 - F[idx])).sum())


def _pmo_sse_analytic(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSE PMO ``1 - q ** Lambda`` for the general observed history.

    Broadcasts over ``(R0, k)``; ``w`` and ``history`` are single 1-D arrays.
    """
    Lambda = _lambda_from_history(w, history)
    q, _, _ = _q_array(R0, k)
    return 1.0 - q**Lambda


def _pmo_ssi_mcmc(
    R0: float,
    k: float,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    show_progress: bool = False,
    **mcmc_kwargs,
) -> float:
    """SSI PMO for a general history, estimated via MCMC over latent infectivities.

    Samples $(Y_s)_{s=0}^r$ from their posterior given the observed history,
    then averages ``exp(R0 * Lambda * (q - 1))`` over draws, where
    ``Lambda = sum_s Y_s * (1 - F_{r-s})``.  See ``notes/notes.tex`` for the
    derivation.

    ``mcmc_kwargs`` are forwarded to ``pm.sample`` (e.g. ``draws``, ``tune``,
    ``chains``, ``thin``).
    """
    from sse_ssi_pmo.inference import fit_ssi

    thin = mcmc_kwargs.pop("thin", 1)
    mcmc_kwargs.setdefault("progressbar", show_progress)
    datatree = fit_ssi(
        history,
        w,
        R0=R0,
        k=k,
        thin=thin,
        **mcmc_kwargs,
    )

    # Posterior infectivity samples: shape (chain, draw, n_nonzero).
    Y_nonzero_samples = datatree["posterior"].ds["infectivity"].values
    n_nonzero = Y_nonzero_samples.shape[-1]
    Y_nonzero_flat = Y_nonzero_samples.reshape(-1, n_nonzero)  # (n_samples, n_nonzero)

    # Reconstruct full Y array including zeros for zero-incidence days.
    r = history.size - 1
    nonzero_idx = np.flatnonzero(history > 0)
    n_samples = Y_nonzero_flat.shape[0]
    Y_full = np.zeros((n_samples, r + 1), dtype=np.float64)
    Y_full[:, nonzero_idx] = Y_nonzero_flat

    # Weights: (1 - F_{r-s}) for s = 0, ..., r.
    F = cumulative(w)
    idx = np.minimum(r - np.arange(r + 1), F.size - 1)
    weights = 1.0 - F[idx]  # shape (r+1,)

    Lambda_samples = Y_full @ weights  # shape (n_samples,)

    q = _nb_extinction_prob(R0, k)
    q_r = float(np.mean(np.exp(R0 * Lambda_samples * (q - 1.0))))
    return 1.0 - q_r


def _w_at(w: NDArray[np.float64], s: int) -> float:
    """Return ``w_s`` (the serial-interval weight at lag ``s``), or ``0`` for
    ``s`` outside ``{1, ..., len(w)}``.
    """
    if s < 1 or s > w.size:
        return 0.0
    return float(w[s - 1])


def _classify_history(history: NDArray[np.int64]) -> dict:
    """Classify the shape of ``history`` for the analytic SSI closed forms.

    Returns a dict with key ``"kind"`` in ``{"day0_only", "one_later",
    "two_later", "general"}`` and the relevant indices/counts:

    * ``day0_only``: ``I_0``.
    * ``one_later``: ``I_0``, ``i``, ``I_i``.
    * ``two_later``: ``I_0``, ``i``, ``I_i``, ``j``, ``I_j``.
    * ``general``: just the kind.
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


def _beta(
    R0_b: NDArray[np.float64],
    k_b: NDArray[np.float64],
    q: NDArray[np.float64],
    F: float,
) -> NDArray[np.float64]:
    """Gamma-MGF factor ``1 + R_0(1-q)(1-F) / (k + R_0 F)``."""
    return 1.0 + R0_b * (1.0 - q) * (1.0 - F) / (k_b + R0_b * F)


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
    parameter is accepted for symmetry with the calling sites and ignored
    here (it appears in the prefactor of the likelihood, not in ``c_m``).
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
    # Handle log(0) safely: w_j^0 = 1, w_{j-i}^0 = 1 by convention even when
    # the base is 0. Compute exponent-zero terms separately to avoid 0*-inf.
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


def _pmo_ssi_analytic_day0(
    R0: ArrayLike,
    k: ArrayLike,
    I_0: int,
    F_r: float,
) -> NDArray[np.float64]:
    """SSI PMO for the day-0-only case ``q_r = (1 + R0(1-F_r)(1-q)/(k+R0 F_r))^{-k I_0}``."""
    q, R0_b, k_b = _q_array(R0, k)
    return 1.0 - _beta(R0_b, k_b, q, F_r) ** (-k_b * I_0)


def _pmo_ssi_analytic_one_later(
    R0: ArrayLike,
    k: ArrayLike,
    I_0: int,
    I_i: int,
    F_r: float,
    F_rmi: float,
) -> NDArray[np.float64]:
    """SSI PMO for histories with cases on day 0 and one later day ``i``.

    ``q_r = beta_0^{-(k I_0 + I_i)} * beta_i^{-k I_i}`` with
    ``beta_a = 1 + R0(1-q)(1-F_{r-a})/(k+R0 F_{r-a})``.
    """
    q, R0_b, k_b = _q_array(R0, k)
    beta_0 = _beta(R0_b, k_b, q, F_r)
    beta_i = _beta(R0_b, k_b, q, F_rmi)
    qr = beta_0 ** (-(k_b * I_0 + I_i)) * beta_i ** (-(k_b * I_i))
    return 1.0 - qr


def _pmo_ssi_analytic_two_later(
    R0: ArrayLike,
    k: ArrayLike,
    I_0: int,
    I_i: int,
    I_j: int,
    w_i: float,
    w_j: float,
    w_jmi: float,
    F_r: float,
    F_rmi: float,
    F_rmj: float,
) -> NDArray[np.float64]:
    """SSI PMO for histories with cases on day 0 and two later days ``i < j``.

    ``q_r = beta_j^{-k I_j} * sum_m rho_m beta_0^{-(k I_0 + I_i + I_j - m)}
    beta_i^{-(k I_i + m)}`` with ``rho_m = c_m / sum_{m'} c_{m'}``; see
    ``_log_c_m_two_later_days`` and ``notes/notes.tex``.
    """
    q, R0_b, k_b = _q_array(R0, k)
    beta_0 = _beta(R0_b, k_b, q, F_r)
    beta_i = _beta(R0_b, k_b, q, F_rmi)
    beta_j = _beta(R0_b, k_b, q, F_rmj)

    log_c_m = _log_c_m_two_later_days(R0_b, k_b, I_0, I_i, I_j, w_i, w_j, w_jmi, F_r, F_rmi)
    log_norm = scipy.special.logsumexp(log_c_m, axis=0, keepdims=True)
    log_rho = log_c_m - log_norm

    m_arr = np.arange(I_j + 1)
    expand = (I_j + 1,) + (1,) * R0_b.ndim
    m_e = m_arr.reshape(expand)
    k_b_e = k_b[np.newaxis, ...]
    alpha_0 = k_b_e * I_0 + I_i + I_j - m_e
    alpha_i = k_b_e * I_i + m_e

    log_beta_0 = np.log(beta_0)[np.newaxis, ...]
    log_beta_i = np.log(beta_i)[np.newaxis, ...]
    log_term = log_rho - alpha_0 * log_beta_0 - alpha_i * log_beta_i
    log_qr = -k_b * I_j * np.log(beta_j) + scipy.special.logsumexp(log_term, axis=0)
    return 1.0 - np.exp(log_qr)


def _pmo_ssi_analytic(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSI PMO for any history with cases on day 0 and at most two later days.

    Dispatches to the day-0-only / one-later-day / two-later-day closed form. Raises
    ``ValueError`` for histories with three or more later non-zero days.
    """
    cls = _classify_history(history)
    if cls["kind"] == "general":
        raise ValueError(
            "_pmo_ssi_analytic supports histories with cases on day 0 and at "
            "most two later days; use the MCMC backend for general histories."
        )
    F = cumulative(w)
    r = history.size - 1
    F_r = float(F[min(r, F.size - 1)])
    if cls["kind"] == "day0_only":
        return _pmo_ssi_analytic_day0(R0, k, cls["I_0"], F_r)
    if cls["kind"] == "one_later":
        i = cls["i"]
        F_rmi = float(F[min(r - i, F.size - 1)])
        return _pmo_ssi_analytic_one_later(R0, k, cls["I_0"], cls["I_i"], F_r, F_rmi)
    # two_later
    i, j = cls["i"], cls["j"]
    F_rmi = float(F[min(r - i, F.size - 1)])
    F_rmj = float(F[min(r - j, F.size - 1)])
    return _pmo_ssi_analytic_two_later(
        R0, k, cls["I_0"], cls["I_i"], cls["I_j"],
        _w_at(w, i), _w_at(w, j), _w_at(w, j - i),
        F_r, F_rmi, F_rmj,
    )


def _log_likelihood_sse_general(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
) -> NDArray[np.float64]:
    """SSE log-likelihood ``log P(I_1, ..., I_r | I_0)`` for an arbitrary history.

    Uses the per-day NB factorisation ``I_t | I_{<t} ~ NB(R0 lambda_t,
    k lambda_t)`` with ``lambda_t = sum_{s<t} w_{t-s} I_s`` (``w_s = 0``
    for ``s > len(w)``). Broadcasts over ``(R0, k)``.
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
    cls = _classify_history(history)
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
    w_i = _w_at(w, i)

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
    w_j = _w_at(w, j)
    w_jmi = _w_at(w, j - i)

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


def _pmo_uncertain_analytic(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    prior_sse: float,
) -> dict[str, NDArray[np.float64]]:
    """Model-averaged PMO for histories with cases on day 0 and at most two later days.

    Combines the per-model PMOs (``_pmo_sse_analytic`` / ``_pmo_ssi_analytic``)
    with the per-model log-likelihoods (``_log_likelihood_sse_general`` /
    ``_log_likelihood_ssi_analytic``) via Bayes' theorem. Returns a dict with keys
    ``pmo``, ``posterior_sse``, ``pmo_sse``, ``pmo_ssi``; each value
    broadcasts over ``(R0, k)``. Raises ``ValueError`` if ``history`` has
    three or more later non-zero days.
    """
    if not 0.0 <= prior_sse <= 1.0:
        raise ValueError("prior_sse must lie in [0, 1]")

    pmo_sse_arr = _pmo_sse_analytic(R0, k, w, history)
    pmo_ssi_arr = _pmo_ssi_analytic(R0, k, w, history)
    log_L_sse = _log_likelihood_sse_general(R0, k, w, history)
    log_L_ssi = _log_likelihood_ssi_analytic(R0, k, w, history)

    with np.errstate(divide="ignore"):
        log_prior_sse = np.log(prior_sse)
        log_prior_ssi = np.log1p(-prior_sse)
    a_sse = log_prior_sse + log_L_sse
    a_ssi = log_prior_ssi + log_L_ssi
    a_max = np.maximum(a_sse, a_ssi)
    log_norm = a_max + np.log(np.exp(a_sse - a_max) + np.exp(a_ssi - a_max))
    posterior_sse = np.exp(a_sse - log_norm)

    pmo = posterior_sse * pmo_sse_arr + (1.0 - posterior_sse) * pmo_ssi_arr
    return {
        "pmo": pmo,
        "posterior_sse": posterior_sse,
        "pmo_sse": pmo_sse_arr,
        "pmo_ssi": pmo_ssi_arr,
    }
