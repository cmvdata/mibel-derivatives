"""Swing option pricer (Longstaff-Schwartz Monte Carlo).

Specification (CONTEXT.md § Swing option):

Annual MIBEL electricity contract with a fixed strike, a daily exercise
cap, and an annual maximum volume. Pricing via Longstaff-Schwartz with
basis functions of (S_t, remaining_volume, time_to_maturity).

Open spec items at scaffolding time (deferred to design phase):
- Exercise frequency (hourly vs daily) and whether the daily cap applies
  to hourly or daily totals.
- Whether the contract carries a minimum annual take-or-pay and, if so,
  the under-take penalty structure.
"""

from __future__ import annotations


def price_swing(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for the swing-option LSMC pricer."""
    raise NotImplementedError("price_swing not implemented yet")
