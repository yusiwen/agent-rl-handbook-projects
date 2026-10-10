#!/usr/bin/env python3
"""Repository checks that do not need a build: characters, payloads, secrets.

    python3 scripts/check_repo.py             # all three
    python3 scripts/check_repo.py --chars     # one at a time
    python3 scripts/check_repo.py --selftest  # prove the checks can still fail

Three things this repository promises and can therefore be caught breaking:

1. **English prose, with a stated allowlist.**  Allowed: ASCII, the Latin-1 letters minus
   no-break space and soft hyphen, the punctuation this project's prose uses (dashes,
   ellipsis, arrows), the tier marks in the README, and the Greek letters used as
   notation.  The list is deliberately the same as the handbook's gate - one author, one
   voice, and two gates that drift apart would judge the same sentence two ways.
   Everything else fails by file, line and codepoint, so a stray full-width mark is
   reported rather than shipped.
2. **No dataset payloads.**  Nothing under ``datasets/`` may be a data file, and no file
   anywhere may exceed the size cap.  Two of this project's datasets are share-alike
   (DS-1000 and HotpotQA are CC-BY-SA-4.0), so committing them would be a licence problem
   before it was a size problem.  ``datasets/digests/`` is exempt: a digest is provenance.
3. **No secrets in a file.**  Keys are read from the environment.  The patterns are
   deliberately narrow - they match credential-shaped strings, not the word "token", which
   this project uses constantly for something else.

``--selftest`` runs the checks against inputs whose verdict is known, because a check that
has only ever seen clean input has been hoped at rather than tested.
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

#: Non-ASCII characters that are allowed, by codepoint.  An explicit list, never a range:
#: a range would quietly permit the next full-width accident.
ALLOWED_CODEPOINTS = {
    0x2013,  # en dash
    0x2014,  # em dash
    0x2026,  # ellipsis
    0x2192,  # right arrow
    0x2194,  # left-right arrow
    0x2248,  # almost equal to
    0x25BA,  # black right-pointing pointer
    0x25B2,  # black up-pointing triangle
    0x2500,  # box-drawing horizontal
    0x2502,  # box-drawing vertical
    0x250C,  # box-drawing down and right
    0x2510,  # box-drawing down and left
    0x2514,  # box-drawing up and right
    0x2518,  # box-drawing up and left
    0x2705,  # check mark - a compute tier
    0x26A0,  # warning sign - a compute tier
    0xFE0F,  # variation selector 16, part of the warning sign's emoji presentation
    0x274C,  # cross mark - a compute tier
    0x03B2,  # beta
    0x03B3,  # gamma
    0x03B4,  # delta
    0x03B8,  # theta
    0x03BB,  # lambda
    0x03C0,  # pi
    0x03C4,  # tau
}

#: Latin-1 letters are allowed, minus the two characters that exist only to be invisible:
#: a no-break space is a space you cannot see, and a soft hyphen is a hyphen you cannot see.
#: Both have cost somebody an afternoon of "why does this string not match".
LATIN1_DENIED = {0x00A0, 0x00AD}

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


def char_problem(char: str) -> str | None:
    """Why this character is not allowed, or ``None`` if it is.

    One predicate, used by the repository scan and by the selftest alike, so the selftest
    tests the rule that actually runs rather than a copy of it that can drift.
    """
    codepoint = ord(char)
    if codepoint < 128:
        return None
    if codepoint in ALLOWED_CODEPOINTS:
        return None
    if 0xA0 <= codepoint <= 0xFF and codepoint not in LATIN1_DENIED:
        return None
    return f"U+{codepoint:04X} {char!r} is not allowed"


def iter_files() -> list[Path]:
    """Files the repository is responsible for: tracked, plus new files not yet ignored.

    Uses the index rather than the filesystem walk, because the promise is about what is
    *committed*: a dataset pulled into ``datasets/payloads/`` is ignored and therefore none
    of this check's business.  Run ``git add`` before the gate so new files are seen.  Falls
    back to a full walk where there is no git repository at all - a sandboxed build, for
    instance, whose source tree is already only tracked files.
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
                reason = char_problem(char)
                if reason:
                    problems.append(f"{relative}:{lineno}:{column}: {reason}")
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


def selftest() -> list[str]:
    """Fail the gate if the checks have stopped being able to fail.

    Every case below is a verdict something else depends on: a full-width mark and an
    ideograph must be rejected (those are the mistakes that have actually happened here),
    the tier marks and dashes must be accepted (the README and every doc use them), and the
    secret patterns must fire on a credential while ignoring the word "token".

    The rejected characters are written as escapes on purpose: this file is scanned by its
    own character rule, so a checker that spelled them literally would fail the gate it
    implements.
    """
    failures: list[str] = []

    for char, should_be_allowed in (
        ("A", True), ("7", True), ("-", True), ("_", True),
        ("\u2014", True), ("\u2026", True), ("\u26a0", True), ("\ufe0f", True),
        ("\u03bb", True), ("\u00e9", True),
        ("\u3000", False), ("\uff5c", False), ("\u6c49", False),
        ("\u00a0", False), ("\u00ad", False),
    ):
        reason = char_problem(char)
        if should_be_allowed and reason is not None:
            failures.append(f"selftest: {char!r} should be allowed and was not ({reason})")
        elif not should_be_allowed and reason is None:
            failures.append(f"selftest: U+{ord(char):04X} should be rejected and was not")

    for sample, should_match in (
        ("sk-" + "a" * 24, True),
        ("ghp_" + "b" * 30, True),
        # Assembled rather than written out: this file is scanned by its own secret rule,
        # and a credential-shaped literal in it - even as a test fixture - is a finding.
        ("api_key = '" + "abcdefghijklmnop" + "1234'", True),
        ("the loss mask says token 120 belongs to a tool segment", False),
    ):
        matched = any(pattern.search(sample) for pattern in SECRET_PATTERNS)
        if matched != should_match:
            failures.append(
                f"selftest: the secret patterns {'missed' if should_match else 'false-positived on'} "
                f"{sample[:24]!r}"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository-wide checks with no build step.")
    parser.add_argument("--chars", action="store_true", help="only the character check")
    parser.add_argument("--payloads", action="store_true", help="only the payload check")
    parser.add_argument("--secrets", action="store_true", help="only the secret check")
    parser.add_argument("--selftest", action="store_true",
                        help="check that the checks can still fail")
    args = parser.parse_args(argv)

    if args.selftest:
        problems = selftest()
        if problems:
            print(f"[FAIL] selftest: {len(problems)} problem(s)")
            for problem in problems:
                print(f"       {problem}")
            return 1
        print("[PASS] selftest: the checks still reject what they must reject")
        return 0

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
