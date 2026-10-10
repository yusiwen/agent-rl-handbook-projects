#!/usr/bin/env python3
"""Repository checks that do not need a build: characters, payloads, secrets, imports.

    python3 scripts/check_repo.py             # all four
    python3 scripts/check_repo.py --chars     # one at a time
    python3 scripts/check_repo.py --selftest  # prove the checks can still fail

Four things this repository promises and can therefore be caught breaking:

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
4. **No third-party imports in the shared layers.**  ``common/``, ``p1-verifiers/src/``,
   ``p1-verifiers/checks/``, ``datasets/`` and ``scripts/`` may import the standard library and
   each other, and nothing else.  The projects under ``p2..p6`` are deliberately out of scope:
   they are meant to import frameworks.  That difference is the rule, so it is checked rather
   than asserted in a document.

``--selftest`` runs the checks against inputs whose verdict is known, because a check that
has only ever seen clean input has been hoped at rather than tested.
"""

from __future__ import annotations

import argparse
import ast
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

#: Layers that must stay importable with nothing but an interpreter. The checks live here too:
#: they are tools rather than libraries, and a tool that needs a framework to print a report is
#: a tool that cannot run on the machine where the problem is.
IMPORT_FREE_ROOTS = ("common", "p1-verifiers/checks", "p1-verifiers/src", "datasets", "scripts")

#: First-party names those layers may import.  `pull` is the dataset tool, which is not a
#: package because the project's directories have hyphens; its tests import it by name.
ALLOWED_FIRST_PARTY = {"common", "p1_verifiers", "pull"}

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


def imported_names(source: str, relative: Path) -> tuple[list[tuple[int, str]], str | None]:
    """Every module this source imports, with the line it is imported on.

    Parsed rather than searched: ``import`` inside a docstring or a comment is not an
    import, and the dataset tool deliberately imports its two libraries inside a function -
    so a grep would report the opposite of the truth in both directions.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], f"{relative}: unparseable ({exc})"

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import is first-party by definition
                continue
            if node.module:
                found.append((node.lineno, node.module))
    return found, None


def check_imports() -> list[str]:
    """No third-party imports in the layers that must stay dependency-free.

    ``sys.stdlib_module_names`` is the definition of "standard library" here, so a module
    can leave it in one release and be caught by the next gate run rather than by a reader
    noticing.
    """
    if not hasattr(sys, "stdlib_module_names"):
        return [f"this interpreter ({sys.version.split()[0]}) exposes no stdlib_module_names, "
                f"so the import rule cannot be checked here"]
    stdlib = set(sys.stdlib_module_names)

    problems: list[str] = []
    for path in iter_files():
        relative = path.relative_to(REPO)
        if path.suffix != ".py" or not str(relative).startswith(IMPORT_FREE_ROOTS):
            continue
        source = read_text(path)
        if source is None:
            continue
        found, error = imported_names(source, relative)
        if error:
            problems.append(error)
            continue
        for lineno, module in found:
            top = module.split(".")[0]
            if top in stdlib or top in ALLOWED_FIRST_PARTY:
                continue
            problems.append(
                f"{relative}:{lineno}: imports {top!r}, which is neither the standard library "
                f"nor first-party; the shared layers must stay runnable with nothing but an "
                f"interpreter (see AGENTS.md)"
            )
    return problems


def selftest() -> list[str]:
    """Fail the gate if the checks have stopped being able to fail.

    Every case below is a verdict something else depends on: a full-width mark and an
    ideograph must be rejected (those are the mistakes that have actually happened here),
    the tier marks and dashes must be accepted (the README and every doc use them), the
    secret patterns must fire on a credential while ignoring the word "token", and the
    import rule must reject a framework while allowing a lazy import of a third-party
    library.

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

    stdlib_guard = "torch" not in getattr(sys, "stdlib_module_names", ())
    for source, expected_count, why in (
        ("import os\nimport json\n", 0, "the standard library"),
        ("import torch\n", 1 if stdlib_guard else 0, "a framework at module level"),
        ("def f():\n    import torch\n", 1 if stdlib_guard else 0, "a framework inside a function"),
        ('"""import torch"""\nimport os\n', 0, "the word import in a docstring"),
        ("from . import sibling\n", 0, "a relative import"),
    ):
        found, error = imported_names(source, Path("selftest.py"))
        if error:
            failures.append(f"selftest: parsing failed for {why} ({error})")
            continue
        offending = [name for _, name in found
                     if name.split(".")[0] not in set(sys.stdlib_module_names)
                     and name.split(".")[0] not in ALLOWED_FIRST_PARTY]
        if len(offending) != expected_count:
            failures.append(
                f"selftest: the import rule found {len(offending)} problem(s) in {why}, "
                f"expected {expected_count}"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository-wide checks with no build step.")
    parser.add_argument("--chars", action="store_true", help="only the character check")
    parser.add_argument("--payloads", action="store_true", help="only the payload check")
    parser.add_argument("--secrets", action="store_true", help="only the secret check")
    parser.add_argument("--imports", action="store_true", help="only the import check")
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

    selected = args.chars or args.payloads or args.secrets or args.imports
    checks: list[tuple[str, list[str]]] = []
    if not selected or args.chars:
        checks.append(("characters", check_chars()))
    if not selected or args.payloads:
        checks.append(("payloads", check_payloads()))
    if not selected or args.secrets:
        checks.append(("secrets", check_secrets()))
    if not selected or args.imports:
        checks.append(("imports", check_imports()))

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
