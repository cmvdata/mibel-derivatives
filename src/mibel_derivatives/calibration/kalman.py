"""Kalman filter / smoother for the Schwartz–Smith two-factor model.

Observation equation maps the latent state (X_t, L_t) to the log of
observed OMIP forward prices for each delivery contract. The seasonal
component s(T) is subtracted from observations before filtering.

Identifiability note: Schwartz–Smith is weakly identified without
sufficient cross-sectional maturities per date. The set of OMIP
contracts used per filtering step (month-ahead, quarter-ahead, year-ahead)
must be chosen explicitly when implementing — see CONTEXT.md and the
end-of-scaffolding open questions.
"""

from __future__ import annotations


class SchwartzSmithKalman:
    """Placeholder for the Kalman filter targeting the Schwartz–Smith state."""

    def __init__(self) -> None:
        raise NotImplementedError("SchwartzSmithKalman not implemented yet")
