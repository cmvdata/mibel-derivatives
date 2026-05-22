"""Mean-reverting jump-diffusion model for the hourly MIBEL spot log-price.

Specification (CONTEXT.md § Pieza 1: modelo del spot):

    d log S_t = κ (θ_t − log S_t) dt + σ dW_t + J · dN_t

with θ_t carrying deterministic annual + weekly + daily seasonality, jumps
J of size drawn from a parametric family (Normal or double-exponential —
choice deferred to calibration), and N_t a Poisson process of intensity λ.

Calibration path: detect jumps via bipower variation (or fixed threshold on
standardised increments), then maximum likelihood on the non-jump residuals.
"""

from __future__ import annotations


class SpotMRJD:
    """Placeholder for the mean-reverting jump-diffusion spot model.

    Concrete implementation pending — see CONTEXT.md for the SDE and the
    open implementation questions raised at scaffolding time.
    """

    def __init__(self) -> None:
        raise NotImplementedError("SpotMRJD not implemented yet")
