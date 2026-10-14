"""Run records: round summaries, spreads, stable ids, on-disk runs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common.runs import (
    RunHeader,
    RunWriter,
    digest_of,
    round_record,
    sample_id,
    sample_ids,
    spread_fraction,
    spread_per_group,
)
from common.schema import read_rollouts

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def rows(name: str) -> list:
    return read_rollouts(str(FIXTURES / name))


class SpreadTests(unittest.TestCase):
    def test_spread_per_group_is_best_minus_worst(self) -> None:
        spreads = spread_per_group(rows("round-000.jsonl"))
        self.assertAlmostEqual(spreads["p1"], 1.6)
        self.assertAlmostEqual(spreads["p2"], 1.1)

    def test_every_group_is_usable_in_the_fixture(self) -> None:
        self.assertAlmostEqual(spread_fraction(rows("round-000.jsonl")), 1.0)

    def test_no_rows_means_no_spread_not_a_crash(self) -> None:
        self.assertEqual(spread_fraction([]), 0.0)


class SampleIdTests(unittest.TestCase):
    def test_sample_id_is_prompt_plus_index(self) -> None:
        first = rows("round-000.jsonl")[0]
        self.assertEqual(sample_id(first), "p1#0")

    def test_ids_are_sorted_and_stable_across_rounds(self) -> None:
        self.assertEqual(sample_ids(rows("round-000.jsonl")), sample_ids(rows("round-001.jsonl")))
        self.assertEqual(sample_ids(rows("round-000.jsonl"))[:2], ["p1#0", "p1#1"])


class RoundRecordTests(unittest.TestCase):
    def test_summary_of_the_first_fixture_round(self) -> None:
        record = round_record(0, rows("round-000.jsonl"))
        self.assertEqual(record.round_index, 0)
        self.assertEqual((record.groups, record.rollouts), (2, 6))
        self.assertAlmostEqual(record.reward_mean, 0.45)
        self.assertAlmostEqual(record.term_means["rule"], 0.5)
        self.assertAlmostEqual(record.term_means["judge"], -0.05)
        self.assertAlmostEqual(record.spread_fraction, 1.0)
        self.assertEqual(record.tokens, 1200)
        self.assertEqual(record.trained_tokens, 660)
        self.assertEqual(len(record.sample_ids), 6)

    def test_duplicate_episode_ids_are_rejected(self) -> None:
        first = rows("round-000.jsonl")[0]
        with self.assertRaises(ValueError):
            round_record(0, [first, first])

    def test_an_empty_round_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            round_record(0, [])

    def test_render_names_the_terms_and_the_spread(self) -> None:
        text = round_record(3, rows("round-000.jsonl")).render()
        self.assertIn("round 3", text)
        self.assertIn("rule=", text)
        self.assertIn("spread=100%", text)
        self.assertIn("660/1200 tokens trained", text)


class RunWriterTests(unittest.TestCase):
    def test_header_and_rounds_are_written_and_read_back(self) -> None:
        header = RunHeader(
            run_id="test-run",
            env_version="fixtures/v1",
            config_hash="sha256:cfg",
            code_sha="git:deadbeef",
            project="p1-verifiers",
            dataset="fixtures",
        )
        with tempfile.TemporaryDirectory() as tmp:
            writer = RunWriter(header, root=tmp)
            record = round_record(0, rows("round-000.jsonl"))
            writer.append_round(record)
            writer.append_rollouts(0, rows("round-000.jsonl"))

            written_header = json.loads((Path(tmp) / "test-run" / "header.json").read_text())
            self.assertEqual(written_header["run_id"], "test-run")
            self.assertEqual(written_header["code_sha"], "git:deadbeef")
            self.assertIn("started_at", written_header)

            back = writer.read_rounds()
            self.assertEqual(len(back), 1)
            self.assertEqual(back[0]["round_index"], 0)

            raw = (Path(tmp) / "test-run" / "round-000.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(raw), 6)
            self.assertEqual(json.loads(raw[0])["episode_id"], "p1-s0")

    def test_reading_a_run_with_no_rounds_is_empty(self) -> None:
        header = RunHeader(run_id="empty", env_version="v", config_hash="h",
                           code_sha="c", project="p1-verifiers")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(RunWriter(header, root=tmp).read_rounds(), [])


class RunWriterCollisionTests(unittest.TestCase):
    """A run record is evidence: two runs must never share a directory.

    Measured cause: run ids are second-resolution timestamps, and a script that finished one
    configuration and started the next inside the same second produced the same id. The second
    run's header and round went over the first's, and a whole configuration's rows were lost.
    """

    HEADER = dict(env_version="test", config_hash="sha256:x", code_sha="git:0", project="p",
                  dataset="d", dataset_digest="sha256:d", notes="n")

    def test_a_second_writer_with_the_same_id_gets_its_own_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = RunWriter(RunHeader(run_id="run-1", **self.HEADER), root=tmp)
            second = RunWriter(RunHeader(run_id="run-1", **self.HEADER), root=tmp)
            self.assertEqual(first.dir.name, "run-1")
            self.assertEqual(second.dir.name, "run-1-2")
            self.assertEqual(second.header.run_id, "run-1-2")
            self.assertTrue((first.dir / "header.json").is_file())
            self.assertTrue((second.dir / "header.json").is_file())

    def test_the_first_record_is_not_overwritten_by_the_second(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = RunWriter(RunHeader(run_id="run-1", **self.HEADER), root=tmp)
            RunWriter(RunHeader(run_id="run-1", **self.HEADER), root=tmp)
            header = json.loads((first.dir / "header.json").read_text(encoding="utf-8"))
            self.assertEqual(header["run_id"], "run-1")


class DigestTests(unittest.TestCase):
    def test_digest_is_stable_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            first.write_text("{}\n", encoding="utf-8")
            digest = digest_of(first)
            self.assertTrue(digest.startswith("sha256:"))
            self.assertEqual(digest, digest_of(first))
            first.write_text("{}\n{}\n", encoding="utf-8")
            self.assertNotEqual(digest, digest_of(first))


if __name__ == "__main__":
    unittest.main()
