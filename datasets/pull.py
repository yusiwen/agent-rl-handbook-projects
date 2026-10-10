#!/usr/bin/env python3
"""Pull a dataset into datasets/payloads/ and record exactly what was pulled.

    python3 datasets/pull.py --list
    python3 datasets/pull.py gsm8k
    python3 datasets/pull.py gsm8k mbpp

Why this is a Python program rather than a shell script wrapping the `hf` command:

* The revision is resolved *before* the transfer and written into the digest, so a run
  record cites a commit instead of "main, that day".
* The code that downloads is the code that hashes, so a digest cannot describe a
  half-finished download.
* Everything except the transfer is standard library, and the transfer is injected, so the
  logic is tested offline - see ``datasets/tests/test_pull.py``.

``huggingface_hub`` and ``yaml`` are imported inside the functions that use them and are
provided by the dev shell, which pins them. A bare ``python3`` fails with an explanation
rather than a traceback, and the gate never needs either: it has no network, and the pull
tool has no role in it.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "datasets"
MANIFEST = DATASETS / "manifest.yaml"
PAYLOADS = DATASETS / "payloads"
DIGESTS = DATASETS / "digests"

#: Shown whenever a dependency is missing. The point is that the reader does not have to
#: guess where the library was supposed to come from - and it names both routes, because the
#: reference machine (a DGX Spark) has no nix on it.
PULL_HINT = (
    "This tool needs huggingface_hub and PyYAML. With nix, the dev shell pins both:\n"
    "    nix develop --command python3 datasets/pull.py <name>\n"
    "Without nix, a virtual environment works:\n"
    "    python3 -m venv .venv && .venv/bin/pip install huggingface_hub pyyaml\n"
    "    .venv/bin/python datasets/pull.py <name>"
)


class PullError(RuntimeError):
    """Anything that stops a pull, with a message a reader can act on."""


class MissingDependency(PullError):
    """A library this tool needs is not importable here."""


class UnknownDataset(PullError):
    """The name given is not in the manifest."""


def require(module: str, hint: str = PULL_HINT):
    """Import a dependency, or explain how to get it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependency(f"{module} is not importable here ({exc}).\n{hint}") from exc


#: A downloader takes keyword arguments repo_id, revision, target, allow_patterns and
#: returns the commit that was actually fetched.
Downloader = Callable[..., str]


def load_manifest(path: Path | str = MANIFEST, loader: Callable[[str], Any] | None = None) -> dict:
    """Read ``datasets/manifest.yaml``.

    ``loader`` is injected so the tests do not need a YAML parser to test anything else.
    """
    text = Path(path).read_text(encoding="utf-8")
    if loader is None:
        parser = require("yaml").safe_load
    else:
        parser = loader
    manifest = parser(text)
    if not isinstance(manifest, dict) or "datasets" not in manifest:
        raise PullError(f"{path}: no 'datasets' key - this is not the manifest")
    return manifest


def dataset_entries(manifest: dict) -> dict[str, dict]:
    """The manifest's datasets, keyed by repository id."""
    entries: dict[str, dict] = {}
    for entry in manifest["datasets"]:
        repo_id = entry.get("id")
        if not repo_id:
            raise PullError("a manifest entry has no 'id'")
        entries[repo_id] = entry
    return entries


def find_entry(manifest: dict, name: str) -> tuple[str, dict]:
    """Look a dataset up by repository id or by short name (``gsm8k``, not ``openai/gsm8k``)."""
    entries = dataset_entries(manifest)
    if name in entries:
        return name, entries[name]
    for repo_id, entry in entries.items():
        if entry.get("name") == name or repo_id.split("/")[-1] == name:
            return repo_id, entry
    known = ", ".join(sorted(entry.get("name") or repo_id.split("/")[-1]
                             for repo_id, entry in entries.items()))
    raise UnknownDataset(f"no dataset named {name!r}. Known names: {known}")


def default_downloader(*, repo_id: str, revision: str, target: Path,
                       allow_patterns: list[str] | None = None) -> str:
    """Fetch a snapshot with ``huggingface_hub`` and return the commit that was fetched.

    Two calls on purpose: the first resolves the revision to a commit *before* any bytes
    move, so the pin is known even if the transfer fails half way. ``snapshot_download``
    is then told to fetch exactly that commit rather than the branch name.
    """
    hub = require("huggingface_hub")
    resolved = hub.HfApi().dataset_info(repo_id, revision=revision).sha
    hub.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved,
        local_dir=str(target),
        allow_patterns=allow_patterns or None,
    )
    return resolved


def stray_files(files: Mapping[str, Any], patterns: list[str]) -> list[str]:
    """Pulled files that no ``allow_pattern`` matches.

    ``snapshot_download`` adds to a directory rather than replacing it, so a second pull
    with narrower patterns leaves the first pull's files in place. The digest hashes what is
    on disk, so without this check it would describe more than the patterns claim - and the
    patterns field exists precisely to say which subset was fetched.

    Matching uses ``fnmatch``, the semantics huggingface_hub applies, so what is attested is
    what was requested rather than a second opinion about it.

    No patterns means no claim, so nothing is stray: the caller asked for everything.
    """
    if not patterns:
        return []
    return [name for name in files if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]


def clean_payload(target: Path | str) -> int:
    """Empty a payload directory before pulling into it. Returns the number of files removed.

    Never called on its own: ``--clean`` is explicit, and the default behaviour on a
    mismatched directory is to refuse and explain. Deleting silently would trade one kind of
    quiet wrongness (a digest that lies) for another (data that vanished).
    """
    root = Path(target)
    removed = 0
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
            removed += 1
        elif path.is_dir():
            path.rmdir()
    return removed


def digest_tree(root: Path | str, *, dataset: str, revision: str, licence: str,
                patterns: list[str] | None = None, pulled_at: str | None = None) -> dict:
    """Hash what was pulled.

    Two decisions worth naming:

    * Any path with a dot-prefixed component is skipped. The client's own ``.cache/``
      bookkeeping is not the dataset, and a digest that changes when a client updates is
      not a pin. This cost an afternoon once; now it is a test.
    * ``allow_patterns`` is recorded alongside the hashes, because the digest describes the
      subset that was fetched, not the dataset. Without the patterns, a later reader cannot
      tell a filtered pull from a complete one.
    """
    root = Path(root)
    files: dict[str, dict] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(block)
        size = path.stat().st_size
        total += size
        files[relative.as_posix()] = {"sha256": hasher.hexdigest(), "bytes": size}

    if not files:
        raise PullError(f"{root}: nothing was pulled, so there is nothing to record")

    # The digest must not describe more than the patterns claim. A file left behind by an
    # earlier, wider pull is the ordinary way that happens.
    if patterns:
        stray = stray_files(files, patterns)
        if stray:
            shown = ", ".join(stray[:3]) + ("" if len(stray) <= 3 else f", and {len(stray) - 3} more")
            raise PullError(
                f"{root} holds {len(stray)} file(s) outside allow_patterns {patterns}: {shown}\n"
                f"The digest would claim a filtered pull while describing an unfiltered one.\n"
                f"Pull into an empty directory, or pass --clean to remove what is already there."
            )

    return {
        "dataset": dataset,
        "revision": revision,
        "licence": licence,
        "allow_patterns": list(patterns or []),
        "files": files,
        "total_bytes": total,
        "pulled_at": pulled_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_digest(digest: dict, name: str, digests: Path | str = DIGESTS) -> Path:
    """Write ``datasets/digests/<name>.digest.json`` - committed, unlike the payload."""
    directory = Path(digests)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.digest.json"
    path.write_text(json.dumps(digest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def pull(name: str, *, manifest: dict | None = None, downloader: Downloader | None = None,
         revision: str = "main", payloads: Path | str = PAYLOADS,
         digests: Path | str = DIGESTS, clean: bool = False) -> tuple[Path, Path, dict]:
    """Pull one dataset. Returns (payload directory, digest path, digest).

    ``clean`` empties the payload directory first. It is not the default because
    ``snapshot_download`` adds to a directory rather than replacing it, and the honest
    response to a leftover file is to say so - not to delete something the operator did not
    ask about. When it is passed, what it removed is printed, so a deletion is never silent.
    """
    manifest = manifest if manifest is not None else load_manifest()
    repo_id, entry = find_entry(manifest, name)
    short = entry.get("name") or repo_id.split("/")[-1]
    target = Path(payloads) / short
    target.mkdir(parents=True, exist_ok=True)
    patterns = entry.get("allow_patterns") or []

    if clean:
        removed = clean_payload(target)
        if removed:
            print(f"  --clean removed {removed} file(s) from {target}")

    resolved = (downloader or default_downloader)(
        repo_id=repo_id, revision=revision, target=target, allow_patterns=patterns or None)

    digest = digest_tree(target, dataset=repo_id, revision=resolved,
                         licence=entry.get("licence", "unstated"), patterns=patterns)
    digest_path = write_digest(digest, short, digests)
    return target, digest_path, digest


def human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull datasets named in datasets/manifest.yaml into datasets/payloads/.")
    parser.add_argument("names", nargs="*", help="short names, or repository ids")
    parser.add_argument("--list", action="store_true", help="show the manifest and exit")
    parser.add_argument("--revision", default="main",
                        help="branch, tag or commit to resolve before pulling (default: main)")
    parser.add_argument("--clean", action="store_true",
                        help="empty the payload directory first (a download adds rather than replaces)")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest()
    except MissingDependency as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    except (PullError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entries = dataset_entries(manifest)

    if args.list or not args.names:
        print(f"datasets in {MANIFEST.relative_to(REPO)} (checked {manifest.get('checked', '?')}):")
        for repo_id in sorted(entries):
            entry = entries[repo_id]
            short = entry.get("name") or repo_id.split("/")[-1]
            patterns = ", ".join(entry.get("allow_patterns") or []) or "(all files)"
            print(f"  {short:<20} {repo_id:<42} {entry.get('licence', 'unstated'):<14} {patterns}")
        if not args.names:
            print("\npull one with: python3 datasets/pull.py <name>")
        return 0

    failures = 0
    for name in args.names:
        try:
            target, digest_path, digest = pull(name, manifest=manifest, revision=args.revision,
                                               clean=args.clean)
        except (PullError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"{digest['dataset']} at revision {digest['revision']}")
        print(f"  licence : {digest['licence']}")
        print(f"  files   : {len(digest['files'])}, {human_bytes(digest['total_bytes'])}")
        print(f"  payload : {target.relative_to(REPO)} (ignored by git)")
        print(f"  digest  : {digest_path.relative_to(REPO)} (commit this)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
