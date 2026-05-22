"""Hash and manifest helpers for the data lakehouse.

Every byte written under `data/raw/` should be accompanied by an entry
in `data/_manifest.jsonl` so re-runs can be audited and bad files
quarantined. The manifest is append-only; nothing is mutated in place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._paths import MANIFEST_PATH, ROOT


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ProvenanceRecord:
    """One line in `data/_manifest.jsonl`."""

    source: str
    url: str
    raw_path: str  # POSIX path relative to repo root
    http_status: int
    bytes: int
    sha256: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


def append_manifest(record: ProvenanceRecord) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")


def record_from_download(
    source: str,
    url: str,
    raw_path: Path,
    http_status: int,
    payload: bytes,
    params: dict[str, Any] | None = None,
) -> ProvenanceRecord:
    """Build a record from an in-memory payload that was just written to disk."""
    resolved = raw_path.resolve()
    try:
        rel = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        # Outside the repo (e.g. pytest tmp_path) — record absolute path.
        rel = resolved.as_posix()
    return ProvenanceRecord(
        source=source,
        url=url,
        raw_path=rel,
        http_status=http_status,
        bytes=len(payload),
        sha256=sha256_bytes(payload),
        params=params or {},
    )
