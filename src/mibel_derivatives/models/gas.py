"""Gas spot model and electricity-gas correlation.

Specification (CONTEXT.md § Tolling agreement / Modelo de gas):

- MIBGAS as primary market for the short/medium forward curve.
- TTF as the long-end anchor (> 3 years) due to liquidity. The MIBGAS-TTF
  basis is treated as a deterministic spread in v1.
- Spot gas: mean-reverting jump-diffusion, calibrated analogously to the
  electricity spot model in models/spot.py.
- Electricity-gas correlation calibrated on residuals (non-jump component).
"""

from __future__ import annotations


class GasMRJD:
    """Placeholder for the gas spot mean-reverting jump-diffusion model."""

    def __init__(self) -> None:
        raise NotImplementedError("GasMRJD not implemented yet")
