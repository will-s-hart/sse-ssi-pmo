"""Prior distributions for the parameters :math:`R_0` and :math:`k`.

The :class:`Prior` dataclass wraps either a Gamma or LogNormal distribution and
exposes the three views the rest of the package needs:

* :attr:`Prior.scipy_dist` — a frozen ``scipy.stats`` distribution, used for
  sampling and log-pdf evaluation in the Monte-Carlo / rejection-sampling
  paths.
* :meth:`Prior.pymc_rv` — a PyMC random variable factory, used by
  :func:`sse_ssi_pmo.inference.fit_sse` / :func:`fit_ssi` when ``R0=None`` or
  ``k=None``.
* :meth:`Prior.sample` / :meth:`Prior.log_pdf` — thin numpy-friendly wrappers
  around the scipy distribution.

Construct via the classmethods :meth:`Prior.gamma` (``mean`` and ``sd``) or
:meth:`Prior.lognormal` (``median`` and ``sd_log``); both are positive-support
distributions covering common epidemiological priors for :math:`R_0` and
:math:`k`. The ``pmo_*`` dispatchers use ``isinstance(x, Prior)`` to decide
between the fixed-parameter and parameter-uncertain code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.stats
from numpy.typing import NDArray

PriorFamily = Literal["gamma", "lognormal"]


@dataclass(frozen=True)
class Prior:
    """A frozen prior distribution on :math:`R_0` or :math:`k`.

    Use the classmethods :meth:`gamma` or :meth:`lognormal` to construct;
    direct construction with arbitrary parameter dicts is supported but is not
    the intended user-facing API.

    ``scipy_params`` are the canonical positional/keyword arguments for the
    frozen ``scipy.stats`` distribution (e.g. ``{"a": 4, "scale": 0.5}`` for
    Gamma). ``pymc_params`` are the canonical kwargs for the matching PyMC
    distribution (e.g. ``{"alpha": 4, "beta": 2}``).
    """

    family: PriorFamily
    scipy_params: dict[str, float]
    pymc_params: dict[str, float]

    @classmethod
    def gamma(cls, *, mean: float, sd: float) -> Prior:
        """Gamma prior parameterised by mean and standard deviation.

        Equivalent to ``Gamma(shape=mean**2/sd**2, rate=mean/sd**2)``.
        """
        if mean <= 0 or sd <= 0:
            raise ValueError("Prior.gamma: mean and sd must be positive")
        shape = (mean / sd) ** 2
        rate = mean / sd**2
        return cls(
            family="gamma",
            scipy_params={"a": shape, "scale": 1.0 / rate},
            pymc_params={"alpha": shape, "beta": rate},
        )

    @classmethod
    def lognormal(cls, *, median: float, sd_log: float) -> Prior:
        """LogNormal prior parameterised by median and log-scale standard deviation.

        ``sd_log`` is the standard deviation of ``log X`` (so ``X`` is
        ``LogNormal(mu = log(median), sigma = sd_log)``).
        """
        if median <= 0 or sd_log <= 0:
            raise ValueError("Prior.lognormal: median and sd_log must be positive")
        mu = float(np.log(median))
        return cls(
            family="lognormal",
            scipy_params={"s": sd_log, "scale": median},
            pymc_params={"mu": mu, "sigma": sd_log},
        )

    @property
    def scipy_dist(self) -> Any:
        """Frozen ``scipy.stats`` distribution matching this prior."""
        if self.family == "gamma":
            return scipy.stats.gamma(**self.scipy_params)
        return scipy.stats.lognorm(**self.scipy_params)

    def pymc_rv(self, name: str) -> Any:
        """Build a PyMC random variable of this prior's family.

        Imported lazily so importing :mod:`sse_ssi_pmo.priors` does not pull in
        PyMC.
        """
        import pymc as pm

        # Cast to ``Any`` so the type-checker doesn't try to match the kwargs
        # against PyMC's positional-arg signature (ty mis-resolves it).
        kwargs: Any = dict(self.pymc_params)
        if self.family == "gamma":
            return pm.Gamma(name, **kwargs)
        return pm.LogNormal(name, **kwargs)

    def sample(self, size: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw ``size`` samples from the prior using the supplied generator."""
        return self.scipy_dist.rvs(size=size, random_state=rng).astype(np.float64)

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-sample log density at ``x``."""
        return self.scipy_dist.logpdf(x).astype(np.float64)


__all__ = ["Prior"]
