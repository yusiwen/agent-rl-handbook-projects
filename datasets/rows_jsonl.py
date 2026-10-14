"""Materialise one split of a dataset as JSONL, over HTTP, with no third-party library.

Why this exists.  `datasets/pull.py` downloads the files a dataset publishes, and for
`openai/gsm8k` those files are parquet - readable only with a columnar library.  The
decontamination check in `common/contamination.py` compares *text* against the evaluation
set, and `common/` may import only the standard library, so the text has to be reachable
from a stdlib-only program.  The Hugging Face datasets server serves the same rows as JSON,
so this script asks it for them a page at a time and writes a JSONL view next to the
payload that `pull.py` already downloaded::

    python3 datasets/rows_jsonl.py openai/gsm8k --config main --split test
    -> datasets/payloads/gsm8k/rows/test.jsonl

Three deliberate decisions:

* **The digest is printed, and the check reports it.**  A contamination verdict names the
  exact bytes it compared against, so "we checked" is a statement about a file, not a mood.
* **A short page is an error, not the end of the data.**  The server pages 100 rows at a
  time; treating an early end as completion would silently produce a smaller evaluation
  set than the manifest claims, which is the failure mode that makes a check worthless.
* **No token.**  The rows endpoint is public, so nothing here needs an API key and nothing
  here can leak one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "datasets"
PAYLOADS = DATASETS / "payloads"

#: The rows endpoint of the Hugging Face datasets server.
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

#: Its documented maximum page size for this endpoint.
PAGE = 100

#: Identify the tool, because an anonymous request is the first one to be rate-limited and a
#: rate limit reads exactly like a missing dataset.  (Learned the hard way against the arXiv
#: API, where the default curl agent is refused.)
USER_AGENT = "agent-rl-handbook-projects/1.0 (+https://github.com/yusiwen/agent-rl-handbook-projects)"


class RowsError(RuntimeError):
    """Anything that stops the export, with a message a reader can act on."""


class EmptyPage(RowsError):
    """The server stopped returning rows before it said it would run out."""


Fetcher = Callable[[str], dict]


def rows_url(
    dataset: str,
    *,
    config: str = "main",
    split: str = "test",
    offset: int = 0,
    length: int = PAGE,
) -> str:
    """The URL for one page of rows."""
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    return f"{ROWS_ENDPOINT}?{query}"


def default_fetcher(url: str) -> dict:
    """Fetch one page and parse it, turning HTTP problems into a readable error."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https endpoint
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RowsError(
            f"the datasets server answered {exc.code} for {url}\n"
            "A 404 usually means that config/split does not exist for this dataset; "
            "a 429 means the request came too fast."
        ) from exc
    except urllib.error.URLError as exc:
        raise RowsError(f"could not reach the datasets server for {url} ({exc.reason})") from exc
    except json.JSONDecodeError as exc:
        raise RowsError(f"the datasets server returned something that is not JSON for {url}") from exc


def fetch_page(
    dataset: str,
    *,
    config: str = "main",
    split: str = "test",
    offset: int = 0,
    length: int = PAGE,
    fetch: Fetcher = default_fetcher,
) -> tuple[list[dict], int]:
    """One page of rows, and the total the server says exists."""
    payload = fetch(rows_url(dataset, config=config, split=split, offset=offset, length=length))
    total = payload.get("num_rows_total")
    if not isinstance(total, int):
        raise RowsError(
            "the datasets server did not report num_rows_total, so how many rows exist "
            "cannot be known; refusing to guess (this is how a truncated evaluation set "
            "gets mistaken for a complete one)"
        )
    rows = payload.get("rows") or []
    return [entry.get("row", entry) for entry in rows], total


def iter_rows(
    dataset: str,
    *,
    config: str = "main",
    split: str = "test",
    limit: int | None = None,
    fetch: Fetcher = default_fetcher,
    page: int = PAGE,
) -> Iterator[dict]:
    """Every row of the split, a page at a time."""
    offset = 0
    total: int | None = None
    while True:
        want = page if limit is None else min(page, limit - offset)
        if want <= 0:
            return
        rows, total = fetch_page(
            dataset, config=config, split=split, offset=offset, length=want, fetch=fetch
        )
        if not rows:
            raise EmptyPage(
                f"the server returned no rows at offset {offset} but reported {total} in total; "
                "stopping rather than writing a short file"
            )
        for row in rows:
            yield row
        offset += len(rows)
        if offset >= total:
            return


def write_jsonl(rows: Iterator[dict], path: Path) -> int:
    """Write rows as JSONL.  Escaped, so the file stays ASCII whatever the payload holds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
            written += 1
    return written


def digest_of(path: Path) -> str:
    """The same short digest shape the run records use, so a check can name its input."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()[:16]


def materialise(
    dataset: str,
    *,
    config: str = "main",
    split: str = "test",
    name: str | None = None,
    limit: int | None = None,
    root: Path | str = PAYLOADS,
    fetch: Fetcher = default_fetcher,
    quiet: bool = False,
) -> Path:
    """Write the split's JSONL view and return its path."""
    stem = name or dataset.rstrip("/").split("/")[-1]
    path = Path(root) / stem / "rows" / f"{split}.jsonl"
    written = write_jsonl(
        iter_rows(dataset, config=config, split=split, limit=limit, fetch=fetch), path
    )
    if not quiet:
        print(f"{dataset} [{config}/{split}]: {written} row(s) -> {path}")
        print(f"digest {digest_of(path)} (pass this file to the contamination check with --eval)")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 datasets/rows_jsonl.py",
        description="Write one split of a dataset as JSONL, using the Hugging Face datasets server.",
    )
    parser.add_argument("dataset", help="dataset id, for example openai/gsm8k")
    parser.add_argument("--config", default="main", help="config name (default: main)")
    parser.add_argument("--split", default="test", help="split name (default: test)")
    parser.add_argument("--name", help="directory name under datasets/payloads (default: the dataset's last segment)")
    parser.add_argument("--limit", type=int, help="write only the first N rows (for a quick look)")
    parser.add_argument("--root", default=str(PAYLOADS), help="where payloads live")
    parser.add_argument("--quiet", action="store_true", help="print nothing but errors")
    args = parser.parse_args(argv)

    try:
        materialise(
            args.dataset,
            config=args.config,
            split=args.split,
            name=args.name,
            limit=args.limit,
            root=args.root,
            quiet=args.quiet,
        )
    except RowsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
