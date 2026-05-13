"""Analytic extinction-probability (PMO) building blocks for the SSE and SSI models.

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
  ``Lambda = sum_{s=0..r} I_s * (1 - F_{r-s})``; the SSE log model evidence
  used for Bayesian model averaging lives in :mod:`~sse_ssi_pmo.evidence`.
* SSI: closed-form ``q_r`` for histories with cases on day 0 only, on day 0
  and one later day, or on day 0 and two later days
  (:func:`_pmo_ssi_analytic`); an MCMC-based estimator for general histories
  (:func:`_pmo_ssi_mcmc`).
* Bayesian model average across SSE and SSI: analytic
  (:func:`_pmo_uncertain_analytic`, restricted to the three closed-form SSI
  cases) and MCMC-based (:func:`_pmo_uncertain_mcmc`, any history).

The functions in this module are private helpers used by ``pmo.py``; they do
not validate user-facing inputs (the dispatcher is responsible for that).
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import scipy.special
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from sse_ssi_pmo._history import classify_history, w_at
from sse_ssi_pmo.evidence import (
    EvidenceMethod,
    _fit_priors_from_prior_args,
    _fixed_or_none,
    _log_c_m_two_later_days,
    _log_evidence_sse_general,
    _log_evidence_sse_mcmc,
    _log_evidence_ssi_analytic,
    _log_evidence_ssi_mcmc,
    _param_posterior_samples,
)
from sse_ssi_pmo.priors import Prior
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


def _param_per_draw(
    x: float | Prior,
    trace_name: str,
    datatree: xr.DataTree,
    n_draws: int,
) -> NDArray[np.float64]:
    """Per-draw values of ``x`` from the trace (if Prior) or broadcast scalar."""
    if isinstance(x, Prior):
        arr = _param_posterior_samples(datatree, trace_name)
        if arr is None:
            raise ValueError(
                f"trace has no '{trace_name}' posterior; pass {trace_name} as a Prior "
                "to fit_ssi/fit_sse"
            )
        return arr[:n_draws]
    return np.full(n_draws, float(x), dtype=np.float64)


def _q_per_draw(
    R0_draws: NDArray[np.float64], k_draws: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-draw extinction probability ``q_i`` via ``_nb_extinction_prob``."""
    n = R0_draws.size
    q = np.empty(n, dtype=np.float64)
    for i in range(n):
        q[i] = _nb_extinction_prob(float(R0_draws[i]), float(k_draws[i]))
    return q


def _pmo_ssi_mcmc_from_trace(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    datatree: xr.DataTree,
) -> float:
    """Core of the SSI MCMC PMO estimator, given a pre-existing ``fit_ssi`` trace.

    Averages ``exp(R0_i * Lambda_i * (q_i - 1))`` over the posterior draws,
    where ``Lambda_i = sum_s Y_{s,i} * (1 - F_{r-s})``. With ``R0``/``k`` as
    :class:`~sse_ssi_pmo.priors.Prior` instances each draw uses its own
    ``R0_i``/``k_i`` read from the trace, and ``q_i`` is recomputed
    per-draw; otherwise scalars are broadcast.
    """
    Y_nonzero_samples = datatree["posterior"].ds["infectivity"].values
    n_nonzero = Y_nonzero_samples.shape[-1]
    Y_nonzero_flat = Y_nonzero_samples.reshape(-1, n_nonzero)

    r = history.size - 1
    nonzero_idx = np.flatnonzero(history > 0)
    n_samples = Y_nonzero_flat.shape[0]
    Y_full = np.zeros((n_samples, r + 1), dtype=np.float64)
    Y_full[:, nonzero_idx] = Y_nonzero_flat

    F = cumulative(w)
    idx = np.minimum(r - np.arange(r + 1), F.size - 1)
    weights = 1.0 - F[idx]
    Lambda_samples = Y_full @ weights

    R0_arr = _param_per_draw(R0, "rep_no", datatree, n_samples)
    k_arr = _param_per_draw(k, "dispersion", datatree, n_samples)
    q_arr = _q_per_draw(R0_arr, k_arr)
    q_r = float(np.mean(np.exp(R0_arr * Lambda_samples * (q_arr - 1.0))))
    return 1.0 - q_r


def _pmo_ssi_mcmc(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    datatree: xr.DataTree | None = None,
    show_progress: bool = False,
    **mcmc_kwargs,
) -> float:
    """SSI PMO for a general history, estimated via MCMC over latent infectivities.

    Runs ``fit_ssi`` internally and averages the conditional extinction
    probability over the posterior draws. With ``R0``/``k`` as
    :class:`~sse_ssi_pmo.priors.Prior` instances the trace also samples
    those parameters; ``q`` is then recomputed per-draw.

    To share a single MCMC fit between ``_pmo_ssi_mcmc`` and
    ``_log_evidence_ssi_mcmc`` (the case under
    ``pmo_uncertain(method='mcmc')``), pass a pre-existing trace via
    ``datatree``; in that case ``mcmc_kwargs`` is ignored.
    """
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
    return _pmo_ssi_mcmc_from_trace(R0, k, w, history, datatree)


def _pmo_sse_mcmc(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    *,
    datatree: xr.DataTree | None = None,
    show_progress: bool = False,
    **mcmc_kwargs,
) -> float:
    """SSE PMO under priors on ``R0`` / ``k``, averaging the closed-form over (R0, k) draws.

    At fixed ``(R0, k)`` the SSE PMO is closed-form (:func:`_pmo_sse_analytic`)
    and this MCMC backend isn't needed. With priors we fit ``R0`` / ``k`` via
    :func:`fit_sse` (or reuse a supplied ``datatree``) and average
    :func:`_pmo_sse_analytic` over the draws.

    Pass a pre-existing trace via ``datatree`` to share the fit (e.g. with a
    figure script that also wants the posterior of ``R0`` for plotting); in
    that case ``mcmc_kwargs`` is ignored.
    """
    if not isinstance(R0, Prior) and not isinstance(k, Prior):
        # No latents to average over — closed form is the answer.
        return float(_pmo_sse_analytic(R0, k, w, history))
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
    n_draws = int(datatree["posterior"].ds.sizes["chain"]) * int(
        datatree["posterior"].ds.sizes["draw"]
    )
    R0_arr = _param_per_draw(R0, "rep_no", datatree, n_draws)
    k_arr = _param_per_draw(k, "dispersion", datatree, n_draws)
    pmo_per_draw = _pmo_sse_analytic(R0_arr, k_arr, w, history)
    return float(np.mean(np.asarray(pmo_per_draw, dtype=np.float64)))


def _beta(
    R0_b: NDArray[np.float64],
    k_b: NDArray[np.float64],
    q: NDArray[np.float64],
    F: float,
) -> NDArray[np.float64]:
    """Gamma-MGF factor ``1 + R_0(1-q)(1-F) / (k + R_0 F)``."""
    return 1.0 + R0_b * (1.0 - q) * (1.0 - F) / (k_b + R0_b * F)


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
    :func:`sse_ssi_pmo.evidence._log_c_m_two_later_days` and ``notes/notes.tex``.
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
    cls = classify_history(history)
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
        R0,
        k,
        cls["I_0"],
        cls["I_i"],
        cls["I_j"],
        w_at(w, i),
        w_at(w, j),
        w_at(w, j - i),
        F_r,
        F_rmi,
        F_rmj,
    )


def _bayes_model_average(
    pmo_sse_arr: NDArray[np.float64],
    pmo_ssi_arr: NDArray[np.float64],
    log_L_sse: NDArray[np.float64],
    log_L_ssi: NDArray[np.float64],
    prior_sse: float,
) -> dict[str, NDArray[np.float64]]:
    """Combine per-model PMOs and log model evidences via Bayes' theorem.

    Shared between :func:`_pmo_uncertain_analytic` and
    :func:`_pmo_uncertain_mcmc` — the only difference between them is how
    ``pmo_ssi_arr`` and ``log_L_ssi`` are computed.
    """
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


def _pmo_uncertain_analytic(
    R0: ArrayLike,
    k: ArrayLike,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    prior_sse: float,
) -> dict[str, NDArray[np.float64]]:
    """Model-averaged PMO for histories with cases on day 0 and at most two later days.

    Combines the per-model PMOs (:func:`_pmo_sse_analytic` /
    :func:`_pmo_ssi_analytic`) with the per-model log model evidences
    (:func:`~sse_ssi_pmo.evidence._log_evidence_sse_general` /
    :func:`~sse_ssi_pmo.evidence._log_evidence_ssi_analytic`) via Bayes'
    theorem. Returns a dict with keys ``pmo``, ``posterior_sse``, ``pmo_sse``,
    ``pmo_ssi``; each value broadcasts over ``(R0, k)``. Raises ``ValueError``
    if ``history`` has three or more later non-zero days.
    """
    if not 0.0 <= prior_sse <= 1.0:
        raise ValueError("prior_sse must lie in [0, 1]")

    pmo_sse_arr = _pmo_sse_analytic(R0, k, w, history)
    pmo_ssi_arr = _pmo_ssi_analytic(R0, k, w, history)
    log_L_sse = _log_evidence_sse_general(R0, k, w, history)
    log_L_ssi = _log_evidence_ssi_analytic(R0, k, w, history)
    return _bayes_model_average(pmo_sse_arr, pmo_ssi_arr, log_L_sse, log_L_ssi, prior_sse)


def _pmo_uncertain_mcmc(
    R0: float | Prior,
    k: float | Prior,
    w: NDArray[np.float64],
    history: NDArray[np.int64],
    prior_sse: float,
    *,
    evidence_method: EvidenceMethod = "bridge",
    rng: np.random.Generator | None = None,
    n_evidence_samples: int | None = None,
    datatree_sse: xr.DataTree | None = None,
    datatree_ssi: xr.DataTree | None = None,
    show_progress: bool = False,
    **mcmc_kwargs,
) -> dict[str, float]:
    """Model-averaged PMO for any history, using MCMC for the SSI side and (under
    priors) the SSE side as well.

    With fixed ``(R0, k)``: runs ``fit_ssi`` once and reuses the trace for
    both the SSI PMO and the SSI log model evidence; the SSE side stays
    closed-form. With ``R0``/``k`` as :class:`~sse_ssi_pmo.priors.Prior`
    instances: additionally runs ``fit_sse`` and uses
    :func:`~sse_ssi_pmo.evidence._log_evidence_sse_mcmc` for the SSE
    evidence; both PMOs are averaged over their respective traces.

    Pass pre-fitted ``datatree_sse`` / ``datatree_ssi`` to skip the inner
    ``fit_*`` calls; either may be supplied independently.

    Returns the same dict shape as :func:`_pmo_uncertain_analytic`:
    ``{pmo, posterior_sse, pmo_sse, pmo_ssi}``, each a scalar.
    """
    if not 0.0 <= prior_sse <= 1.0:
        raise ValueError("prior_sse must lie in [0, 1]")

    param_uncertain = isinstance(R0, Prior) or isinstance(k, Prior)

    from sse_ssi_pmo.inference import fit_ssi

    thin = mcmc_kwargs.pop("thin", 1)
    mcmc_kwargs.setdefault("progressbar", show_progress)

    if datatree_ssi is None:
        priors_kw = _fit_priors_from_prior_args(R0, k)
        datatree_ssi = fit_ssi(
            history,
            w,
            R0=_fixed_or_none(R0),
            k=_fixed_or_none(k),
            priors=priors_kw,
            thin=thin,
            **mcmc_kwargs,
        )

    pmo_ssi = _pmo_ssi_mcmc_from_trace(R0, k, w, history, datatree_ssi)
    log_L_ssi = _log_evidence_ssi_mcmc(
        R0,
        k,
        w,
        history,
        evidence_method=evidence_method,
        datatree=datatree_ssi,
        rng=rng,
        n_samples=n_evidence_samples,
    )

    if param_uncertain:
        if datatree_sse is None:
            from sse_ssi_pmo.inference import fit_sse

            priors_kw = _fit_priors_from_prior_args(R0, k)
            datatree_sse = fit_sse(
                history,
                w,
                R0=_fixed_or_none(R0),
                k=_fixed_or_none(k),
                priors=priors_kw,
                thin=thin,
                **mcmc_kwargs,
            )
        pmo_sse = _pmo_sse_mcmc(R0, k, w, history, datatree=datatree_sse)
        log_L_sse = _log_evidence_sse_mcmc(
            R0,
            k,
            w,
            history,
            evidence_method=evidence_method,
            datatree=datatree_sse,
            rng=rng,
            n_samples=n_evidence_samples,
        )
    else:
        # Fixed-parameter fast paths.
        pmo_sse = float(_pmo_sse_analytic(R0, k, w, history))
        log_L_sse = float(_log_evidence_sse_general(R0, k, w, history))

    result = _bayes_model_average(
        np.asarray(pmo_sse),
        np.asarray(pmo_ssi),
        np.asarray(log_L_sse),
        np.asarray(log_L_ssi),
        prior_sse,
    )
    return {key: float(np.asarray(val).reshape(())) for key, val in result.items()}
