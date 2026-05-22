"""Repo-root and lakehouse-zone path helpers.

Every scraper resolves its filesystem locations through these helpers so
the tree layout lives in one place. The repo root is detected by walking
up from this file (no env var, no CWD dependence).
"""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
STAGING_DIR: Path = DATA_DIR / "staging"
CURATED_DIR: Path = DATA_DIR / "curated"
MANIFEST_PATH: Path = DATA_DIR / "_manifest.jsonl"


def raw_path(source: str, *parts: str, filename: str) -> Path:
    """Return a raw-zone path for an artifact and create parent dirs."""
    p = RAW_DIR.joinpath(source, *parts, filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def staging_path(source: str, filename: str) -> Path:
    """Return a staging-zone path and create parent dirs."""
    p = STAGING_DIR.joinpath(source, filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def curated_path(filename: str) -> Path:
    """Return a curated-zone path and create the curated dir if missing."""
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    return CURATED_DIR / filename
