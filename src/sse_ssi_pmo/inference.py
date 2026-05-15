"""Bayesian inference for the SSE and SSI epidemic models.

Two public functions, one per model, fit the model to an observed incidence
vector ``(I_0, I_1, ..., I_r)`` and return a posterior trace as an
``xr.DataTree``:

* :func:`fit_ssi` — SSI model; latent infectivities ``Y_t`` are sampled via
  HMC.  The variable ``"infectivity"`` in the posterior DataTree holds the
  per-draw values of ``Y_t`` for every day ``t`` with ``I_t > 0``, in the
  order those days appear in the incidence vector.
* :func:`fit_sse` — SSE model; no latent variables, so the posterior contains
  only ``R0`` and/or ``k`` when they are inferred.

Both functions accept ``R0`` and ``k`` as either:

* a ``float`` — treated as a fixed known value (no prior placed on it), or
* ``None`` — a prior is placed on the parameter and it is inferred jointly.

The ``priors`` argument accepts a dict with keys ``"rep_no"`` and/or
``"dispersion"`` mapping to ``(pm_distribution_class, kwargs_dict)`` tuples,
overriding the module-level ``DEFAULT_PRIORS``.

All remaining keyword arguments are forwarded to ``pm.sample``.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import scipy.stats
import xarray as xr
from numpy.typing import ArrayLike, NDArray


def _lognormal_from_median_and_lower(median: float, lower_2_5: float) -> dict:
    """LogNormal parameters matching a given median and 2.5th percentile."""
    mu = np.log(median)
    sigma = (mu - np.log(lower_2_5)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}


DEFAULT_PRIORS: dict[str, tuple] = {
    "rep_no": (
        pm.LogNormal,
        _lognormal_from_median_and_lower(1.0, 0.2),
    ),
    "dispersion": (
        pm.LogNormal,
        _lognormal_from_median_and_lower(0.2, 0.1),
    ),
}


def _make_w_padded(w: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    """Return ``w`` zero-padded or truncated to ``length``."""
    out = np.zeros(length, dtype=np.float64)
    n = min(len(w), length)
    out[:n] = w[:n]
    return out


def fit_ssi(
    incidence_vec: ArrayLike,
    w: ArrayLike,
    R0: float | None = None,
    k: float | None = None,
    priors: dict | None = None,
    thin: int = 1,
    **sample_kwargs,
) -> xr.DataTree:
    """Fit the SSI model to observed incidence via MCMC.

    Parameters
    ----------
    incidence_vec
        Observed incidence ``(I_0, I_1, ..., I_r)`` as a 1-D non-negative
        integer array with ``I_0 >= 1``.
    w
        Discrete serial-interval weights ``w[s-1] = w_s`` for ``s = 1, 2, ...``.
    R0
        Reproduction number. Pass a ``float`` to fix it; ``None`` to infer
        from the ``"rep_no"`` prior.
    k
        Dispersion parameter. Pass a ``float`` to fix it; ``None`` to infer
        from the ``"dispersion"`` prior.
    priors
        Override entries in ``DEFAULT_PRIORS``; keys are ``"rep_no"`` and/or
        ``"dispersion"``, values are ``(pm_class, kwargs)`` tuples.
    thin
        Keep every ``thin``-th posterior draw.
    **sample_kwargs
        Forwarded to ``pm.sample`` (e.g. ``draws``, ``tune``, ``chains``,
        ``progressbar``).

    Returns
    -------
    xr.DataTree
        Posterior trace.  The SSI latent infectivities are stored under
        ``posterior["infectivity"]`` with shape ``(chain, draw, n_nonzero)``
        where ``n_nonzero = sum(incidence_vec > 0)``.
    """
    priors = priors or {}
    incidence_vec = np.asarray(incidence_vec, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    t_stop = len(incidence_vec)

    w_padded = _make_w_padded(w_arr, t_stop - 1)

    # mult_matrix[t, s] = w_{t-s} for s < t, else 0.
    # Multiplying this by the infectivity vector gives the FOI at each t.
    mult_matrix = np.zeros((t_stop, t_stop), dtype=np.float64)
    for t in range(1, t_stop):
        mult_matrix[t, :t] = w_padded[:t][::-1]

    # Observations: I_0 is the initial condition, not a likelihood entry.
    incidence_local = np.zeros(t_stop, dtype=np.float64)
    incidence_local[1:] = incidence_vec[1:]

    nonzero_incidence_idx = incidence_vec > 0
    n_nonzero = int(nonzero_incidence_idx.sum())

    # FOI at each time, using the mean incidence (used only for validation).
    foi_expected = mult_matrix @ incidence_vec
    nonzero_foi_idx = foi_expected > 0

    if np.any(incidence_local[~nonzero_foi_idx] > 0):
        raise ValueError(
            "Positive incidence observed at a time step where the force of "
            "infection is zero — inconsistent with the SSI model."
        )

    with pm.Model():
        # -- reproduction number -----------------------------------------------
        if R0 is None:
            prior_cls, prior_kw = priors.get("rep_no", DEFAULT_PRIORS["rep_no"])
            R0_rv = prior_cls("rep_no", **prior_kw)
        else:
            R0_rv = float(R0)

        # -- dispersion --------------------------------------------------------
        if k is None:
            prior_cls, prior_kw = priors.get("dispersion", DEFAULT_PRIORS["dispersion"])
            k_rv = prior_cls("dispersion", **prior_kw)
        else:
            k_rv = float(k)

        # -- latent infectivities ----------------------------------------------
        infectivity_rv = pm.Gamma(
            "infectivity",
            alpha=k_rv * incidence_vec[nonzero_incidence_idx],
            beta=k_rv,
            shape=n_nonzero,
        )

        # Full FOI vector: dot mult_matrix columns for non-zero-incidence days
        # with the corresponding latent infectivities.
        foi_vec = pm.math.dot(mult_matrix[:, nonzero_incidence_idx], infectivity_rv)

        # rep_no_vec is a length-t_stop vector; trivially constant for now but
        # structured for easy extension to time-varying R0.
        rep_no_vec = pm.math.full(t_stop, R0_rv)
        expected_local = rep_no_vec * foi_vec

        pm.Poisson(
            "likelihood",
            mu=expected_local[nonzero_foi_idx],
            observed=incidence_local[nonzero_foi_idx],
        )

        trace = pm.sample(
            idata_kwargs={"log_likelihood": True, "log_prior": True},
            **sample_kwargs,
        )

    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.assign_coords(draw=np.arange(len(trace.posterior.draw)))

    return az.convert_to_datatree(trace)


def fit_sse(
    incidence_vec: ArrayLike,
    w: ArrayLike,
    R0: float | None = None,
    k: float | None = None,
    priors: dict | None = None,
    thin: int = 1,
    **sample_kwargs,
) -> xr.DataTree:
    """Fit the SSE model to observed incidence via MCMC.

    Parameters
    ----------
    incidence_vec
        Observed incidence ``(I_0, I_1, ..., I_r)`` as a 1-D non-negative
        integer array with ``I_0 >= 1``.
    w
        Discrete serial-interval weights ``w[s-1] = w_s`` for ``s = 1, 2, ...``.
    R0
        Reproduction number. Pass a ``float`` to fix it; ``None`` to infer
        from the ``"rep_no"`` prior.
    k
        Dispersion parameter. Pass a ``float`` to fix it; ``None`` to infer
        from the ``"dispersion"`` prior.
    priors
        Override entries in ``DEFAULT_PRIORS``; keys are ``"rep_no"`` and/or
        ``"dispersion"``, values are ``(pm_class, kwargs)`` tuples.
    thin
        Keep every ``thin``-th posterior draw.
    **sample_kwargs
        Forwarded to ``pm.sample``.

    Returns
    -------
    xr.DataTree
        Posterior trace.
    """
    priors = priors or {}
    incidence_vec = np.asarray(incidence_vec, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    t_stop = len(incidence_vec)

    w_padded = _make_w_padded(w_arr, t_stop - 1)

    # Deterministic FOI at each time step.
    foi_vec = np.zeros(t_stop, dtype=np.float64)
    for t in range(1, t_stop):
        foi_vec[t] = np.dot(incidence_vec[:t][::-1], w_padded[:t])

    incidence_local = np.zeros(t_stop, dtype=np.float64)
    incidence_local[1:] = incidence_vec[1:]

    nonzero_foi_idx = foi_vec > 0

    if np.any(incidence_local[~nonzero_foi_idx] > 0):
        raise ValueError(
            "Positive incidence observed at a time step where the force of "
            "infection is zero — inconsistent with the SSE model."
        )

    with pm.Model():
        # -- reproduction number -----------------------------------------------
        if R0 is None:
            prior_cls, prior_kw = priors.get("rep_no", DEFAULT_PRIORS["rep_no"])
            R0_rv = prior_cls("rep_no", **prior_kw)
        else:
            R0_rv = float(R0)

        # -- dispersion --------------------------------------------------------
        if k is None:
            prior_cls, prior_kw = priors.get("dispersion", DEFAULT_PRIORS["dispersion"])
            k_rv = prior_cls("dispersion", **prior_kw)
        else:
            k_rv = float(k)

        rep_no_vec = pm.math.full(t_stop, R0_rv)
        expected_local = rep_no_vec * foi_vec

        pm.NegativeBinomial(
            "likelihood",
            mu=expected_local[nonzero_foi_idx],
            alpha=k_rv * foi_vec[nonzero_foi_idx],
            observed=incidence_local[nonzero_foi_idx],
        )

        trace = pm.sample(
            idata_kwargs={"log_likelihood": True, "log_prior": True},
            **sample_kwargs,
        )

    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.assign_coords(draw=np.arange(len(trace.posterior.draw)))

    return az.convert_to_datatree(trace)


__all__ = ["DEFAULT_PRIORS", "fit_sse", "fit_ssi"]
