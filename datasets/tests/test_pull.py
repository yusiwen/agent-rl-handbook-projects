"""Tests for datasets/pull.py.

    python3 -m unittest discover -s datasets/tests -t datasets/tests

The download is injected, so everything here runs offline and without a token. That the
whole file imports and passes with a plain interpreter is itself a test of the design: the
YAML and huggingface_hub imports live inside the functions that need them, so nothing else
in the repository can come to depend on them by accident.

The tests never touch the real datasets/payloads or datasets/digests: both are redirected
into a temporary directory, so a test run cannot leave a payload behind or overwrite a
committed digest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("pull", REPO / "datasets" / "pull.py")
assert _SPEC and _SPEC.loader
pull = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pull)

try:
    # find_spec rather than `import yaml`: the import check parses this file, and a
    # third-party import here would be reported as a violation of the rule it sits next to.
    HAVE_YAML = importlib.util.find_spec("yaml") is not None
except (ImportError, ValueError):  # a broken yaml install still means "cannot test it"
    HAVE_YAML = False


class DigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_hashes_and_totals_are_recorded(self) -> None:
        self.write("a.txt", "alpha")
        self.write("nested/b.txt", "beta")
        digest = pull.digest_tree(self.root, dataset="x/y", revision="c0ffee", licence="MIT")
        self.assertEqual(digest["dataset"], "x/y")
        self.assertEqual(digest["revision"], "c0ffee")
        self.assertEqual(set(digest["files"]), {"a.txt", "nested/b.txt"})
        self.assertEqual(digest["files"]["a.txt"]["sha256"],
                         hashlib.sha256(b"alpha").hexdigest())
        self.assertEqual(digest["total_bytes"], len("alpha") + len("beta"))

    def test_the_clients_cache_is_not_part_of_the_dataset(self) -> None:
        # The regression this exists for: hashing everything under the output directory
        # made the digest change whenever the client's bookkeeping changed, which means it
        # was never a pin.
        self.write("real.parquet", "x")
        self.write(".cache/huggingface/download/meta", "changing")
        self.write("nested/.hidden", "also changing")
        digest = pull.digest_tree(self.root, dataset="x/y", revision="c0ffee", licence="MIT")
        self.assertEqual(set(digest["files"]), {"real.parquet"})

    def test_patterns_are_recorded_so_a_filtered_pull_is_distinguishable(self) -> None:
        self.write("a.txt", "alpha")
        digest = pull.digest_tree(self.root, dataset="x/y", revision="c0ffee", licence="MIT",
                                  patterns=["a.*"])
        self.assertEqual(digest["allow_patterns"], ["a.*"])

    def test_pulled_at_is_recorded_utc(self) -> None:
        self.write("a.txt", "alpha")
        digest = pull.digest_tree(self.root, dataset="x/y", revision="c0ffee", licence="MIT")
        self.assertRegex(digest["pulled_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_an_empty_payload_is_an_error_not_an_empty_digest(self) -> None:
        with self.assertRaises(pull.PullError):
            pull.digest_tree(self.root, dataset="x/y", revision="c0ffee", licence="MIT")


class FakeDownloader:
    """Writes files the way a real snapshot would, and records how it was called."""

    def __init__(self, files: dict[str, str], resolved: str = "1" * 40) -> None:
        self.files = files
        self.resolved = resolved
        self.calls: list[dict] = []

    def __call__(self, *, repo_id, revision, target, allow_patterns=None) -> str:
        self.calls.append({"repo_id": repo_id, "revision": revision,
                           "allow_patterns": allow_patterns})
        for name, content in self.files.items():
            path = Path(target) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        cache = Path(target) / ".cache" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "meta").write_text("bookkeeping", encoding="utf-8")
        return self.resolved


MANIFEST = {
    "checked": "2026-10-09",
    "datasets": [
        {"id": "openai/gsm8k", "name": "gsm8k", "licence": "MIT",
         "allow_patterns": ["README.md", "main/*"]},
        {"id": "xlangai/DS-1000", "name": "ds1000", "licence": "CC-BY-SA-4.0"},
    ],
}

#: A loader that returns the manifest above, so these tests need no YAML parser.
fake_loader = lambda text: MANIFEST  # noqa: E731


class FindEntryTests(unittest.TestCase):
    def test_a_short_name_resolves(self) -> None:
        repo_id, entry = pull.find_entry(MANIFEST, "gsm8k")
        self.assertEqual(repo_id, "openai/gsm8k")
        self.assertEqual(entry["licence"], "MIT")

    def test_a_repository_id_resolves(self) -> None:
        repo_id, _ = pull.find_entry(MANIFEST, "xlangai/DS-1000")
        self.assertEqual(repo_id, "xlangai/DS-1000")

    def test_the_last_path_segment_resolves_when_no_name_is_declared(self) -> None:
        repo_id, _ = pull.find_entry(MANIFEST, "DS-1000")
        self.assertEqual(repo_id, "xlangai/DS-1000")

    def test_an_unknown_name_says_what_the_known_ones_are(self) -> None:
        with self.assertRaises(pull.UnknownDataset) as ctx:
            pull.find_entry(MANIFEST, "mnist")
        self.assertIn("gsm8k", str(ctx.exception))
        self.assertIn("ds1000", str(ctx.exception))


class ManifestTests(unittest.TestCase):
    def test_an_injected_loader_avoids_the_dependency(self) -> None:
        path = Path(tempfile.mkdtemp()) / "manifest.yaml"
        path.write_text("datasets: []\n", encoding="utf-8")
        self.assertEqual(pull.load_manifest(path, loader=fake_loader), MANIFEST)

    def test_a_file_that_is_not_the_manifest_is_rejected(self) -> None:
        path = Path(tempfile.mkdtemp()) / "manifest.yaml"
        path.write_text("datasets: []\n", encoding="utf-8")
        with self.assertRaises(pull.PullError):
            pull.load_manifest(path, loader=lambda text: {"something": "else"})

    @unittest.skipUnless(HAVE_YAML, "no YAML parser here; run the tests from the dev shell")
    def test_the_real_manifest_parses_and_declares_what_a_digest_needs(self) -> None:
        manifest = pull.load_manifest()
        entries = pull.dataset_entries(manifest)
        self.assertGreaterEqual(len(entries), 8)
        for repo_id, entry in entries.items():
            self.assertIn("licence", entry, f"{repo_id} has no licence note")
            self.assertIn("used_by", entry, f"{repo_id} says nothing about which project uses it")
            if "allow_patterns" in entry:
                self.assertIsInstance(entry["allow_patterns"], list)

    @unittest.skipUnless(HAVE_YAML, "no YAML parser here; run the tests from the dev shell")
    def test_every_name_in_the_manifest_is_unique_and_resolvable(self) -> None:
        manifest = pull.load_manifest()
        names = []
        for repo_id in pull.dataset_entries(manifest):
            resolved, entry = pull.find_entry(manifest, repo_id)
            self.assertEqual(resolved, repo_id)
            names.append(entry.get("name") or repo_id.split("/")[-1])
        self.assertEqual(len(names), len(set(names)), f"duplicate short names: {names}")


class PullTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.payloads = Path(self.tmp.name) / "payloads"
        self.digests = Path(self.tmp.name) / "digests"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_pull(self, downloader: Callable[..., str]):
        return pull.pull("gsm8k", manifest=MANIFEST, downloader=downloader,
                         payloads=self.payloads, digests=self.digests)

    def test_the_digest_records_the_resolved_commit_not_the_branch(self) -> None:
        downloader = FakeDownloader({"README.md": "x", "main/train.parquet": "y"})
        _, _, digest = self.run_pull(downloader)
        self.assertEqual(digest["revision"], downloader.resolved)
        self.assertNotEqual(digest["revision"], "main")

    def test_the_patterns_from_the_manifest_reach_the_downloader(self) -> None:
        downloader = FakeDownloader({"README.md": "x"})
        self.run_pull(downloader)
        self.assertEqual(downloader.calls[0]["allow_patterns"], ["README.md", "main/*"])
        self.assertEqual(downloader.calls[0]["repo_id"], "openai/gsm8k")

    def test_the_licence_comes_from_the_manifest(self) -> None:
        downloader = FakeDownloader({"README.md": "x"})
        _, _, digest = self.run_pull(downloader)
        self.assertEqual(digest["licence"], "MIT")

    def test_a_dataset_with_no_patterns_asks_for_everything(self) -> None:
        downloader = FakeDownloader({"test/file.parquet": "x"})
        pull.pull("ds1000", manifest=MANIFEST, downloader=downloader,
                  payloads=self.payloads, digests=self.digests)
        self.assertIsNone(downloader.calls[0]["allow_patterns"])

    def test_payload_and_digest_land_where_they_belong(self) -> None:
        downloader = FakeDownloader({"README.md": "x", "main/train.parquet": "y"})
        target, digest_path, digest = self.run_pull(downloader)
        self.assertEqual(target, self.payloads / "gsm8k")
        self.assertTrue((target / "main" / "train.parquet").exists())
        self.assertEqual(digest_path, self.digests / "gsm8k.digest.json")
        on_disk = json.loads(digest_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, digest)

    def test_a_failing_download_leaves_no_digest(self) -> None:
        def broken(**kwargs):
            raise pull.PullError("network")

        with self.assertRaises(pull.PullError):
            self.run_pull(broken)
        self.assertFalse(self.digests.exists() and any(self.digests.iterdir()))


class PatternAttestationTests(unittest.TestCase):
    """The digest must not claim a filtered pull while describing an unfiltered one."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.payloads = Path(self.tmp.name) / "payloads"
        self.digests = Path(self.tmp.name) / "digests"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_stray_file_is_refused_rather_than_absorbed(self) -> None:
        # Exactly what happened live: an earlier, wider pull left `socratic/*` behind, and
        # the digest went on to claim `main/*`.
        downloader = FakeDownloader({"README.md": "x", "main/train.parquet": "y",
                                     "socratic/train.parquet": "z"})
        with self.assertRaises(pull.PullError) as ctx:
            pull.pull("gsm8k", manifest=MANIFEST, downloader=downloader,
                      payloads=self.payloads, digests=self.digests)
        message = str(ctx.exception)
        self.assertIn("socratic/train.parquet", message)
        self.assertIn("--clean", message)

    def test_files_inside_the_patterns_are_accepted(self) -> None:
        downloader = FakeDownloader({"README.md": "x", "main/train.parquet": "y"})
        _, _, digest = pull.pull("gsm8k", manifest=MANIFEST, downloader=downloader,
                                 payloads=self.payloads, digests=self.digests)
        self.assertEqual(set(digest["files"]), {"README.md", "main/train.parquet"})

    def test_no_patterns_means_no_attestation(self) -> None:
        self.assertEqual(pull.stray_files({"anything": {}}, []), [])

    def test_matching_follows_the_clients_fnmatch_semantics(self) -> None:
        patterns = ["main/*", "*.md"]
        self.assertEqual(pull.stray_files({"main/a.parquet": {}, "README.md": {}}, patterns), [])
        self.assertEqual(pull.stray_files({"socratic/a.parquet": {}}, patterns),
                         ["socratic/a.parquet"])

    def test_clean_empties_the_directory_and_reports_how_much(self) -> None:
        downloader = FakeDownloader({"README.md": "x", "socratic/train.parquet": "z"})
        with self.assertRaises(pull.PullError):
            pull.pull("gsm8k", manifest=MANIFEST, downloader=downloader,
                      payloads=self.payloads, digests=self.digests)
        target = self.payloads / "gsm8k"
        self.assertTrue((target / "socratic" / "train.parquet").exists())
        removed = pull.clean_payload(target)
        # Three, not one: the fake downloader also writes the client's cache metadata, which
        # a real pull leaves behind too - which is why the live run reported 24.
        self.assertEqual(removed, 3)
        self.assertTrue(target.exists())
        self.assertEqual(list(target.rglob("*")), [])

    def test_clean_does_not_excuse_a_downloader_that_writes_outside_the_patterns(self) -> None:
        downloader = FakeDownloader({"README.md": "x", "socratic/train.parquet": "z"})
        # The downloader stays stubborn - it writes the stray file again - so this asserts
        # the check fires even after a clean, not that cleaning fixes a bad downloader.
        with self.assertRaises(pull.PullError):
            pull.pull("gsm8k", manifest=MANIFEST, downloader=downloader,
                      payloads=self.payloads, digests=self.digests, clean=True)


class DependencyTests(unittest.TestCase):
    def test_a_missing_library_explains_how_to_get_it(self) -> None:
        with self.assertRaises(pull.MissingDependency) as ctx:
            pull.require("definitely_not_a_module_xyz")
        self.assertIn("definitely_not_a_module_xyz", str(ctx.exception))
        self.assertIn("nix develop", str(ctx.exception))

    def test_the_module_imports_without_the_optional_libraries(self) -> None:
        # This test is the assertion: the module was imported at the top of this file, and
        # datasets/tests runs under an interpreter that need not have either library.
        self.assertTrue(callable(pull.digest_tree))
        self.assertTrue(callable(pull.default_downloader))


class HumanBytesTests(unittest.TestCase):
    def test_units(self) -> None:
        self.assertEqual(pull.human_bytes(0), "0 B")
        self.assertEqual(pull.human_bytes(999), "999 B")
        self.assertEqual(pull.human_bytes(2048), "2.0 KB")
        self.assertEqual(pull.human_bytes(5 * 1024 * 1024), "5.0 MB")


if __name__ == "__main__":
    unittest.main()
