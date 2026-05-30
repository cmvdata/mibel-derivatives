"""Structured energy products built on the MIBEL price models.

Each module here prices a contract whose cashflows are a deterministic
function of one or more simulated risk factors (spot price, solar
generation). The risk-factor models live in :mod:`mibel_derivatives.models`;
the product modules compose them and add the contract payoff.

Currently implemented:

- :mod:`mibel_derivatives.products.ppa` — solar PPA (Pieza 5).
"""

from __future__ import annotations
