"""Schwartz–Smith two-factor model with deterministic seasonality.

Specification (CONTEXT.md § Pieza 2: modelo de la curva forward):

    log F(t, T) = X_t · e^(−κ (T − t)) + L_t + s(T)

where X_t is the short-term mean-reverting factor, L_t is the long-term
random walk, and s(T) is a deterministic seasonal function of delivery.

Calibration: Kalman filter + MLE on OMIP forward prices, conditioned to
reproduce the current OMIE spot and the current OMIP forward curve.
Risk-neutral drift adjustments (market price of risk for X_t and L_t)
are part of the parameter vector — see calibration/kalman.py.
"""

from __future__ import annotations


class SchwartzSmith2F:
    """Placeholder for the Schwartz–Smith two-factor forward model."""

    def __init__(self) -> None:
        raise NotImplementedError("SchwartzSmith2F not implemented yet")
