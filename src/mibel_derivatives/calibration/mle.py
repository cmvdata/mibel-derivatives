"""Maximum-likelihood estimators for the spot and gas MRJD models.

Two-stage procedure for each spot process:

1. Jump-detection pass on standardised increments (bipower variation or
   threshold rule). The flagged observations are removed from the
   continuous-diffusion likelihood.
2. MLE on the remaining residuals for (kappa, sigma, seasonal theta_t parameters),
   then a separate ML fit for the jump frequency lambda and jump-size
   parameters on the flagged subsample.

Joint refinement under a full likelihood is left as an extension.
"""

from __future__ import annotations


def fit_mrjd(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for the MRJD MLE driver."""
    raise NotImplementedError("fit_mrjd not implemented yet")
