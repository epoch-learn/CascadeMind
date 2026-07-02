#!/usr/bin/env python3
"""Public-release and camera-ready sanity checks for CascadeMind."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_TEX = ROOT / "paper" / "latest-paperfeb12026" / "semeval2026_final.tex"
CANONICAL_BIB = ROOT / "paper" / "latest-paperfeb12026" / "references.bib"

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".env.example",
    "data/README.md",
    "paper/latest-paperfeb12026/semeval2026_final.tex",
    "paper/latest-paperfeb12026/references.bib",
    "scripts/check_release.py",
    "scripts/run_camera_ready_experiments.py",
]

EXPECTED_DATA_ROWS = {
    "data/dev_track_a.jsonl": 200,
    "data/test_track_a.jsonl": 400,
    "data/dev_track_b.jsonl": 479,
    "data/test_track_b.jsonl": 849,
    "data/synthetic_data_for_classification.jsonl": 1900,
}

SECRET_PATTERNS = [
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("planning Gemini key", re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}\b")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b")),
]

FORBIDDEN_TRACKED_PATTERNS = [
    re.compile(r"^artifacts/"),
    re.compile(r"^data/.*\.(jsonl|csv|zip)$"),
    re.compile(r"^paper/.*/review_package/"),
    re.compile(r"\.zip$"),
    re.compile(r"\.pdf$"),
]


def git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def tracked_files() -> list[str]:
    proc = git(["ls-files"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    return sorted(line for line in proc.stdout.splitlines() if line)


def is_tracked(rel: str) -> bool:
    return git(["ls-files", "--error-unmatch", rel]).returncode == 0


def is_ignored(rel: str) -> bool:
    return git(["check-ignore", "-q", rel]).returncode == 0


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except FileNotFoundError:
        return False


def is_probably_text(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\x00" not in chunk


def scan_secrets(paths: list[str], errors: list[str]) -> None:
    for rel in paths:
        path = ROOT / rel
        if not path.exists() or not path.is_file() or not is_probably_text(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"{rel}: possible {label} literal")


def check_required_files(errors: list[str]) -> None:
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required file: {rel}")
            continue
        if not is_tracked(rel):
            errors.append(f"{rel}: required file is not tracked")
        if is_ignored(rel):
            errors.append(f"{rel}: required file is ignored by .gitignore")


def check_tracked_file_policy(paths: list[str], errors: list[str]) -> None:
    for rel in paths:
        path = ROOT / rel
        if path.exists() and is_lfs_pointer(path):
            errors.append(f"{rel}: Git LFS pointer stub is tracked")
        name = Path(rel).name
        if name.startswith(".env") and name != ".env.example":
            errors.append(f"{rel}: environment file must not be tracked")
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if pattern.search(rel):
                errors.append(f"{rel}: generated/data artifact should not be tracked")
                break


def citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    cite_re = re.compile(r"\\cite\w*(?:\[[^\]]*\]){0,2}\{([^}]+)\}")
    for match in cite_re.finditer(tex):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def bib_keys(bib: str) -> set[str]:
    return {match.group(1).strip() for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)", bib)}


def check_paper(errors: list[str]) -> None:
    if not CANONICAL_TEX.exists() or not CANONICAL_BIB.exists():
        return
    tex = CANONICAL_TEX.read_text(encoding="utf-8")
    bib = CANONICAL_BIB.read_text(encoding="utf-8")

    required_snippets = [
        (r"72.75\%", "official 72.75% result"),
        ("10th", "official 10th-place table wording"),
        ("https://github.com/epoch-learn/CascadeMind", "public code URL"),
        ("Epoch Learn", "author affiliation"),
    ]
    for snippet, label in required_snippets:
        if snippet not in tex:
            errors.append(f"paper missing {label}")

    forbidden = ["11th", "TeamX", "Geffen Academy", "confirms that"]
    lower_tex = tex.lower()
    for snippet in forbidden:
        if snippet.lower() in lower_tex:
            errors.append(f"paper still contains stale/overstrong wording: {snippet}")

    missing = sorted(citation_keys(tex) - bib_keys(bib))
    if missing:
        errors.append("missing bibliography keys: " + ", ".join(missing))


def check_markdown_links(paths: list[str], errors: list[str]) -> None:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for rel in paths:
        if not rel.endswith(".md"):
            continue
        path = ROOT / rel
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in link_re.finditer(text):
            target = match.group(1).split()[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local_target = unquote(target.split("#", 1)[0])
            if not local_target:
                continue
            candidate = (path.parent / local_target).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                errors.append(f"{rel}: markdown link escapes repository: {target}")
                continue
            if not candidate.exists():
                errors.append(f"{rel}: broken markdown link: {target}")


def count_jsonl(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.relative_to(ROOT)}:{line_no}: invalid JSON: {exc}") from exc
            rows += 1
    return rows


def check_data(strict: bool, errors: list[str], warnings: list[str]) -> None:
    for rel, expected in EXPECTED_DATA_ROWS.items():
        path = ROOT / rel
        if not path.exists():
            if strict:
                errors.append(f"{rel}: missing local data file")
            continue
        if is_lfs_pointer(path):
            errors.append(f"{rel}: Git LFS pointer stub, not restored data")
            continue
        try:
            rows = count_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if rows != expected:
            errors.append(f"{rel}: expected {expected} rows, found {rows}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-data", action="store_true", help="fail when expected local datasets are missing")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    try:
        paths = tracked_files()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    check_required_files(errors)
    check_tracked_file_policy(paths, errors)
    scan_secrets(paths, errors)
    check_paper(errors)
    check_markdown_links(paths, errors)
    check_data(args.strict_data, errors, warnings)

    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"FAIL: {error}")

    if errors:
        print(f"\nRelease check failed with {len(errors)} error(s).")
        return 1

    print("PASS: release checks completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
