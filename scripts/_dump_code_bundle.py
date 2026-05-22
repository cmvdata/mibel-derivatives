"""Dump every executable file in the repo into a single audit-friendly bundle.

Discovery is by glob, not a fixed list (deviation from the
mibel-forecasting bundler): every file matching a section's patterns is
included, in deterministic alphabetical order within the section. The
sections themselves are topological — config → package source → tests →
scripts → notebooks → reports — so the bundle still reads like a slow
execution trace from leaves to roots.

Output: code_bundle.txt at the repo root. The file is gitignored.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code_bundle.txt"

# Each tuple: (section title, [glob patterns relative to ROOT]).
# Order within a section is the deterministic sort of resolved paths.
SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1. CONFIG / TOOLING",
        [
            "pyproject.toml",
            ".env.example",
            ".gitignore",
            "Makefile",
            ".github/workflows/*.yml",
            "README.md",
            "CONTEXT.md",
            "data/README.md",
        ],
    ),
    (
        "2. PACKAGE SOURCE",
        [
            "src/mibel_derivatives/**/*.py",
        ],
    ),
    (
        "3. TESTS",
        [
            "tests/**/*.py",
        ],
    ),
    (
        "4. STANDALONE SCRIPTS",
        [
            "scripts/**/*.py",
        ],
    ),
    (
        "5. NOTEBOOKS",
        [
            "notebooks/**/*.ipynb",
        ],
    ),
    (
        "6. DIAGNOSTIC REPORTS",
        [
            "reports/**/*.md",
        ],
    ),
]

# Files matching any of these patterns are excluded even if globbed.
EXCLUDE: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/.ipynb_checkpoints/**",
    "**/*.executed.ipynb",
)


def _discover(patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pat in patterns:
        for p in ROOT.glob(pat):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            if any(p.match(ex) for ex in EXCLUDE):
                continue
            # exclude __pycache__ etc by path prefix too
            if "__pycache__" in rel or ".ipynb_checkpoints" in rel:
                continue
            found.add(p)
    return sorted(found, key=lambda x: x.relative_to(ROOT).as_posix())


def _render_notebook(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            out.append(f"# === CELL {i} (markdown) ===")
            for line in src.splitlines():
                out.append(f"# {line}")
        else:
            out.append(f"# === CELL {i} (code) ===")
            out.append(src.rstrip())
        out.append("")
    return "\n".join(out)


def _numbered(text: str) -> str:
    """Prefix every line with a line number so citation by line works."""
    lines = text.splitlines()
    width = max(4, len(str(len(lines))))
    return "\n".join(f"{i:>{width}d}  {line}" for i, line in enumerate(lines, start=1))


def _file_header(rel: str, full: Path) -> list[str]:
    """Header block for a file: path, mtime, optional last-touching commit."""
    header = ["=" * 80, f"FILE: {rel}"]
    if not full.exists():
        header.append("(file not found)")
        header.append("=" * 80)
        return header
    mtime = dt.datetime.fromtimestamp(full.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    header.append(f"mtime: {mtime}")
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--pretty=%h %ad %s", "--date=short", "--", rel],
            capture_output=True, text=True, cwd=ROOT, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            header.append(f"last commit: {out.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    header.append("=" * 80)
    return header


def main() -> None:
    chunks: list[str] = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chunks.append("MIBEL-DERIVATIVES - CODE AUDIT BUNDLE")
    chunks.append(f"Generated: {now}")
    chunks.append("Source: scripts/_dump_code_bundle.py")
    chunks.append("Regenerate manually: `python scripts/_dump_code_bundle.py`")
    chunks.append("Discovery is by glob, not by fixed list.")
    chunks.append("Per-file lines are numbered (LLLL  source) for citation.")
    chunks.append("Each file carries its mtime and last git commit (if any).")
    chunks.append("")
    chunks.append("INDEX")
    chunks.append("-" * 80)

    resolved: list[tuple[str, list[Path]]] = [
        (title, _discover(pats)) for title, pats in SECTIONS
    ]

    file_count = 0
    for title, paths in resolved:
        chunks.append(f"  {title}")
        if not paths:
            chunks.append("      (no files)")
            continue
        for p in paths:
            chunks.append(f"      + {p.relative_to(ROOT).as_posix()}")
            file_count += 1
    chunks.append("")
    chunks.append(f"  Total entries: {file_count}")
    chunks.append("")

    for title, paths in resolved:
        chunks.append("")
        chunks.append("#" * 80)
        chunks.append(f"## {title}")
        chunks.append("#" * 80)
        for full in paths:
            rel = full.relative_to(ROOT).as_posix()
            chunks.append("")
            chunks.extend(_file_header(rel, full))
            if rel.endswith(".ipynb"):
                rendered = _render_notebook(full)
            else:
                rendered = full.read_text(encoding="utf-8").rstrip()
            chunks.append(_numbered(rendered))

    body = "\n".join(chunks) + "\n"
    OUT.write_text(body, encoding="utf-8")
    n_lines = body.count("\n")
    print(f"Wrote {OUT}  ({n_lines} lines, {len(body):,} bytes)")


if __name__ == "__main__":
    main()
