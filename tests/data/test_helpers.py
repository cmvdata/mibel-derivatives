"""Unit tests for _paths, _provenance and _http (offline only)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mibel_derivatives.data import _http, _paths, _provenance


def test_paths_resolve_under_repo_root(tmp_path: Path) -> None:
    assert _paths.ROOT.name == "Mibel_derivatives"
    assert _paths.RAW_DIR == _paths.DATA_DIR / "raw"


def test_raw_path_creates_parents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_paths, "RAW_DIR", tmp_path)
    p = _paths.raw_path("demo", "partition=A", filename="x.bin")
    assert p == tmp_path / "demo" / "partition=A" / "x.bin"
    assert p.parent.is_dir()


def test_sha256_bytes_known_vector() -> None:
    assert _provenance.sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_append_manifest_roundtrips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_manifest = tmp_path / "_manifest.jsonl"
    monkeypatch.setattr(_provenance, "MANIFEST_PATH", fake_manifest)
    rec = _provenance.ProvenanceRecord(
        source="demo",
        url="https://example.com/x",
        raw_path="data/raw/demo/x.bin",
        http_status=200,
        bytes=3,
        sha256="abc",
        params={"k": "v"},
    )
    _provenance.append_manifest(rec)
    _provenance.append_manifest(rec)
    lines = fake_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["source"] == "demo"
    assert parsed["sha256"] == "abc"
    assert parsed["timestamp_utc"].endswith("Z")


def test_throttled_session_enforces_min_interval() -> None:
    sess = _http.ThrottledSession(min_interval_seconds=0.15, max_retries=1)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

    calls: list[float] = []

    def fake_get(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(time.monotonic())
        return FakeResponse()

    with patch.object(sess._session, "get", side_effect=fake_get):
        sess.get("https://example.com")
        sess.get("https://example.com")
    assert calls[1] - calls[0] >= 0.14


def test_throttled_session_retries_on_503() -> None:
    sess = _http.ThrottledSession(
        min_interval_seconds=0.0,
        max_retries=3,
        initial_backoff=0.01,
        max_sleep=0.05,
    )

    responses = [
        type("R", (), {"status_code": 503, "raise_for_status": lambda self: None})(),
        type("R", (), {"status_code": 503, "raise_for_status": lambda self: None})(),
        type("R", (), {"status_code": 200, "raise_for_status": lambda self: None})(),
    ]

    def fake_get(*args, **kwargs):  # type: ignore[no-untyped-def]
        return responses.pop(0)

    with patch.object(sess._session, "get", side_effect=fake_get):
        resp = sess.get("https://example.com")
    assert resp.status_code == 200
