"""Smoke tests: the package and every public submodule import cleanly."""

from __future__ import annotations

import importlib


def test_package_imports() -> None:
    pkg = importlib.import_module("mibel_derivatives")
    assert pkg.__version__ == "0.1.0"


def test_submodules_import() -> None:
    for name in [
        "mibel_derivatives.data.loaders",
        "mibel_derivatives.models.spot",
        "mibel_derivatives.models.forward",
        "mibel_derivatives.models.gas",
        "mibel_derivatives.pricing.swing",
        "mibel_derivatives.pricing.tolling",
        "mibel_derivatives.pricing.ppa_solar",
        "mibel_derivatives.calibration.kalman",
        "mibel_derivatives.calibration.mle",
        "mibel_derivatives.evaluation.sensitivities",
    ]:
        importlib.import_module(name)
