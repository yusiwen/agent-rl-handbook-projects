#!/usr/bin/env python3
"""Repository checks that do not need a build: characters, payloads, secrets.

    python3 scripts/check_repo.py            # all three
    python3 scripts/check_repo.py --chars    # one at a time

Three things this repository promises and can therefore be caught breaking:

1. **English, with a short allowlist.**  The only non-ASCII characters allowed are the
   tier marks in ``README.md`` and the legacy of the book's own gate: check-mark,
   warning sign, its variation selector, and cross-mark.  Everything else fails, named
   by file, line and codepoint - so a stray full-width punctuation mark is reported
   instead of shipped.
2. **No dataset payloads.**  Nothing under ``datasets/`` may be a data file, and no file
   anywhere may exceed the size cap.  Some of this project's datasets are share-alike
   (DS-1000 is CC-BY-SA-4.0); committing them would be a licence problem before it was
   a size problem.
3. **No secrets in a file.**  Keys are read from the environment.  The patterns here are
   deliberately narrow: they match credential-shaped strings, not the word "token",
   which this project uses constantly for something else.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Directories never scanned: generated, cached, or not ours.
SKIP_DIRS = {".git", "runs", "__pycache__", ".direnv", ".venv", "node_modules", "book", "result"}

#: Non-ASCII characters that are allowed, by codepoint.  Kept as an explicit list, not a
#: range: a range would quietly permit the next full-width accident.
ALLOWED_CODEPOINTS = {
    0x2705,  # check-mark
    0x26A0,  # warning sign
    0xFE0F,  # variation selector 16, part of the emoji presentation of the warning sign
    0x274C,  # cross-mark
}

#: Extensions that mean "a dataset payload", if they appear under ``datasets/``.
PAYLOAD_SUFFIXES = {
    ".parquet", ".csv", ".tsv", ".jsonl", ".json", ".arrow", ".feather", ".gz", ".zip",
    ".tar", ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".db", ".sqlite", ".h5",
}

#: Directories under ``datasets/`` whose contents are deliberately committed.  A digest is
#: provenance - a handful of hashes saying which revision was used - and it is the one
#: thing a run record must cite.  It is not a payload, despite being JSON.
PAYLOAD_EXEMPT = ("datasets/digests",)

#: Files larger than this are almost certainly a payload that got committed by accident.
MAX_BYTES = 256 * 1024

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][A-Za-z0-9/+_\-]{16,}['\"]"),
)


def iter_files() -> list[Path]:
    """Files the repository is responsible for: tracked, plus new files not yet ignored.

    Uses the index rather than the filesystem walk, because the promise is about what is
    *committed*: a dataset pulled into ``datasets/payloads/`` is ignored and therefore
    none of this check's business.  Run ``git add`` before the gate so new files are seen.
    Falls back to a full walk outside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(path for path in REPO.rglob("*") if path.is_file()
                      and not any(part in SKIP_DIRS for part in path.relative_to(REPO).parts))

    files: list[Path] = []
    for name in result.stdout.split("\0"):
        if not name:
            continue
        path = REPO / name
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(REPO).parts):
            files.append(path)
    return sorted(files)


def read_text(path: Path) -> str | None:
    """Text or nothing: binary files are checked by size, not by character."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def check_chars() -> list[str]:
    problems: list[str] = []
    for path in iter_files():
        text = read_text(path)
        if text is None:
            continue
        relative = path.relative_to(REPO)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for column, char in enumerate(line, start=1):
                codepoint = ord(char)
                if codepoint > 127 and codepoint not in ALLOWED_CODEPOINTS:
                    problems.append(
                        f"{relative}:{lineno}:{column}: U+{codepoint:04X} {char!r} is not allowed"
                    )
    return problems


def check_payloads() -> list[str]:
    problems: list[str] = []
    datasets = REPO / "datasets"
    for path in iter_files():
        relative = path.relative_to(REPO)
        size = path.stat().st_size
        exempt = any(str(relative).startswith(prefix + "/") for prefix in PAYLOAD_EXEMPT)
        if datasets in path.parents and path.suffix in PAYLOAD_SUFFIXES and not exempt:
            problems.append(
                f"{relative}: payload files do not belong in the repository "
                f"(pull them per machine; commit the digest instead)"
            )
        elif size > MAX_BYTES:
            problems.append(f"{relative}: {size} bytes exceeds the {MAX_BYTES} byte cap")
    return problems


def check_secrets() -> list[str]:
    problems: list[str] = []
    for path in iter_files():
        text = read_text(path)
        if text is None:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                lineno = text[: match.start()].count("\n") + 1
                problems.append(
                    f"{path.relative_to(REPO)}:{lineno}: looks like a credential "
                    f"({match.group(0)[:8]}...) - read it from the environment instead"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository-wide checks with no build step.")
    parser.add_argument("--chars", action="store_true", help="only the character check")
    parser.add_argument("--payloads", action="store_true", help="only the payload check")
    parser.add_argument("--secrets", action="store_true", help="only the secret check")
    args = parser.parse_args(argv)

    selected = args.chars or args.payloads or args.secrets
    checks: list[tuple[str, list[str]]] = []
    if not selected or args.chars:
        checks.append(("characters", check_chars()))
    if not selected or args.payloads:
        checks.append(("payloads", check_payloads()))
    if not selected or args.secrets:
        checks.append(("secrets", check_secrets()))

    failed = False
    for name, problems in checks:
        if problems:
            failed = True
            print(f"[FAIL] {name}: {len(problems)} problem(s)")
            for problem in problems[:20]:
                print(f"       {problem}")
            if len(problems) > 20:
                print(f"       ... and {len(problems) - 20} more")
        else:
            print(f"[PASS] {name}")
    print("check_repo: all clean" if not failed else "check_repo: problems found")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
