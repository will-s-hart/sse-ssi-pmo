"""Paper figures for the SSE/SSI PMO project.

Split into two subpackages that share the ``sse_ssi_pmo`` library and the
helpers in :mod:`analysis._shared`:

- :mod:`analysis.base` -- infection-anchored PMO under model uncertainty.
- :mod:`analysis.delays` -- symptom-onset-anchored (delayed-transmission) models.

Scripts are run in module form, e.g. ``python -m analysis.base.results_1_pmo_vs_r``.
"""
