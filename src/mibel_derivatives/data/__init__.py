"""Data layer for mibel-derivatives.

Three-zone lakehouse:
- raw/      bytes as downloaded, audited by data/_manifest.jsonl
- staging/  parsed and typed parquet, one per source
- curated/  analysis-ready parquet, one per logical dataset

Shared utils:
- _paths.py        repo-root + zone path helpers
- _http.py         throttled, retrying HTTP session
- _provenance.py   SHA-256 hashing and manifest append

One module per source: omip, mibgas, omie, esios, pvgis, bce, eua, ttf.
"""

from __future__ import annotations
