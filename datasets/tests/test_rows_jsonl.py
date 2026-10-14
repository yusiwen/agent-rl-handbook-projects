"""Tests for datasets/rows_jsonl.py.

    python3 -m unittest discover -s datasets/tests -t datasets/tests

Every test injects the fetcher, so nothing here touches the network, the real
`datasets/payloads` tree, or a token.  What is being tested is not HTTP: it is the rule
that a page which ends early is an error rather than the end of the data, because a
silently truncated evaluation set is exactly the kind of input that makes a
decontamination check say "clean" about half a benchmark.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("rows_jsonl", REPO / "datasets" / "rows_jsonl.py")
assert _SPEC and _SPEC.loader
rows_jsonl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rows_jsonl)


class PagedFetcher:
    """Serves `total` rows in pages, optionally going empty partway or hiding the total."""

    def __init__(self, total: int, *, empty_from: int | None = None, hide_total: bool = False) -> None:
        self.total = total
        self.empty_from = empty_from
        self.hide_total = hide_total
        self.calls: list[dict] = []

    def __call__(self, url: str) -> dict:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        self.calls.append({"offset": offset, "length": length})
        payload: dict = {}
        if not self.hide_total:
            payload["num_rows_total"] = self.total
        empty = self.empty_from is not None and offset >= self.empty_from
        payload["rows"] = (
            []
            if empty
            else [
                {"row_idx": index, "row": {"question": f"q{index}"}}
                for index in range(offset, min(offset + length, self.total))
            ]
        )
        return payload


class UrlTests(unittest.TestCase):
    def test_the_dataset_id_is_encoded(self) -> None:
        url = rows_jsonl.rows_url("openai/gsm8k", config="main", split="test")
        self.assertIn("dataset=openai%2Fgsm8k", url)
        self.assertIn("config=main", url)
        self.assertIn("split=test", url)
        self.assertIn("offset=0", url)


class PagingTests(unittest.TestCase):
    def test_pages_until_the_reported_total_is_reached(self) -> None:
        fetch = PagedFetcher(250)
        rows = list(rows_jsonl.iter_rows("x/y", fetch=fetch))
        self.assertEqual(len(rows), 250)
        self.assertEqual([call["offset"] for call in fetch.calls], [0, 100, 200])

    def test_a_short_page_is_an_error_not_the_end(self) -> None:
        fetch = PagedFetcher(250, empty_from=100)
        with self.assertRaises(rows_jsonl.EmptyPage) as caught:
            list(rows_jsonl.iter_rows("x/y", fetch=fetch))
        self.assertIn("reported 250 in total", str(caught.exception))

    def test_a_missing_total_is_refused(self) -> None:
        fetch = PagedFetcher(10, hide_total=True)
        with self.assertRaises(rows_jsonl.RowsError) as caught:
            list(rows_jsonl.iter_rows("x/y", fetch=fetch))
        self.assertIn("num_rows_total", str(caught.exception))

    def test_a_limit_stops_after_that_many_rows(self) -> None:
        fetch = PagedFetcher(250)
        rows = list(rows_jsonl.iter_rows("x/y", limit=5, fetch=fetch))
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(fetch.calls), 1)


class WriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_jsonl_is_one_row_per_line_and_stays_ascii(self) -> None:
        path = self.root / "rows" / "test.jsonl"
        written = rows_jsonl.write_jsonl(
            iter([{"question": "Janet\u2019s ducks \u2014 16 eggs per day"}, {"question": "plain"}]),
            path,
        )
        self.assertEqual(written, 2)
        raw = path.read_bytes()
        self.assertTrue(raw.decode("ascii"), "a payload view must survive an ASCII-only gate")
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[0]["question"], "Janet\u2019s ducks \u2014 16 eggs per day")

    def test_materialise_writes_under_the_named_directory(self) -> None:
        path = rows_jsonl.materialise(
            "openai/gsm8k", split="test", root=self.root, fetch=PagedFetcher(3), quiet=True
        )
        self.assertEqual(path, self.root / "gsm8k" / "rows" / "test.jsonl")
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)
        self.assertTrue(rows_jsonl.digest_of(path).startswith("sha256:"))

    def test_an_explicit_name_wins_over_the_dataset_id(self) -> None:
        path = rows_jsonl.materialise(
            "openai/gsm8k", name="eval-set", root=self.root, fetch=PagedFetcher(1), quiet=True
        )
        self.assertEqual(path.parent.parent.name, "eval-set")

    def test_the_digest_follows_the_content(self) -> None:
        first = self.root / "a.jsonl"
        second = self.root / "b.jsonl"
        first.write_text('{"question": "one"}\n', encoding="utf-8")
        second.write_text('{"question": "two"}\n', encoding="utf-8")
        self.assertNotEqual(rows_jsonl.digest_of(first), rows_jsonl.digest_of(second))


class LayoutTests(unittest.TestCase):
    def test_the_default_root_is_the_ignored_payload_directory(self) -> None:
        self.assertEqual(rows_jsonl.PAYLOADS, REPO / "datasets" / "payloads")
        default_root = inspect.signature(rows_jsonl.materialise).parameters["root"].default
        self.assertEqual(Path(default_root), REPO / "datasets" / "payloads")

    def test_that_directory_is_git_ignored(self) -> None:
        """A view written by this tool must not be committable by accident."""
        ignored = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("datasets/payloads/", [line.strip() for line in ignored])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
