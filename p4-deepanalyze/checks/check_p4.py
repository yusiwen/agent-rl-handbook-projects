"""The gate step for P4: prove the loop works before any model is involved.

Four things are checked, and each one is a link that can silently break:

1. **The fixtures are right.** Every task's expected value is recomputed by an independent
   program from its own CSV, run in the sandbox, and compared with the stored answer. If this
   fails, no result from a model means anything.
2. **The protocol and the loop work end to end.** A scripted client emits the oracle program
   and then the number it printed; the episode must come out solved, with the observation in
   the transcript.
3. **A prose reply is refused, not guessed at.** The most common failure of a small model is
   answering without the markers; the count of protocol errors is what makes it visible.
4. **A wrong answer is unsolved, and a timeout is not a wrong answer.** The three verdicts the
   chapter distinguishes are separated in the record.

It runs with the local executor, so it works on a machine with no container runtime. The
Docker executor is checked by the run itself (`summary.json` carries the health probe).
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "src"))

from p4_deepanalyze.agent import run_episode  # noqa: E402
from p4_deepanalyze.client import ScriptedClient  # noqa: E402
from p4_deepanalyze.sandbox import LocalSandbox, Observation  # noqa: E402
from p4_deepanalyze.tasks import load  # noqa: E402
from p4_deepanalyze.verify import is_correct  # noqa: E402

FIXTURES = HERE.parent / "fixtures" / "tasks.jsonl"

from p4_deepanalyze.oracles import ORACLE  # noqa: E402

FIXTURES_SET = (FIXTURES, HERE.parent / "fixtures" / "tasks-heldout.jsonl")


def summaries_of(rewards: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rewards:
        counts[row["task_id"]] = counts.get(row["task_id"], 0) + (1 if row["solved"] else 0)
    return counts


class FixtureTests(unittest.TestCase):
    def test_every_task_family_has_an_oracle(self):
        for path in FIXTURES_SET:
            with self.subTest(fixtures=path.name):
                self.assertEqual({task.family for task in load(path)}, set(ORACLE))

    def test_the_rl_group_report_agrees_with_its_rewards(self):
        """The chapter quotes "2 of 6 groups taught anything": check it against the rows."""
        root = HERE.parent / "fixtures" / "rl-r1"
        rewards = [json.loads(line) for line in (root / "rewards.jsonl").read_text(encoding="utf-8").splitlines()]
        groups = json.loads((root / "groups.json").read_text(encoding="utf-8"))
        summary = groups.pop("_summary")
        self.assertEqual(summary["groups"], len({row["task_id"] for row in rewards}))
        self.assertEqual(summaries_of(rewards), {task: group["solved"] for task, group in groups.items()})
        self.assertEqual(summary["informative_groups"], sum(1 for group in groups.values() if group["teaches"]))
        full = json.loads((root / "update-p4-rl-full.json").read_text(encoding="utf-8"))
        positive = json.loads((root / "update-p4-rl-positive.json").read_text(encoding="utf-8"))
        informative = {task for task, group in groups.items() if group["teaches"]}
        self.assertEqual(full["used"], sum(1 for row in rewards if row["task_id"] in informative))
        self.assertEqual(positive["used"], sum(1 for row in rewards if row["reward"] > 0 and row["task_id"] in informative))
        self.assertEqual(full["samples"], len(rewards))
        self.assertEqual(full["steps"], full["used"])
        self.assertFalse(full["positive_only"])
        self.assertTrue(positive["positive_only"])

    def test_both_fixture_files_are_current(self):
        build = HERE.parent / "fixtures" / "build_tasks.py"
        done = subprocess.run([sys.executable, str(build), "--check"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_every_expected_answer_is_reproduced_by_its_oracle(self):
        sandbox = LocalSandbox(timeout_s=10)
        for path in FIXTURES_SET:
            for task in load(path):
                with self.subTest(task=task.id):
                    observation = sandbox.run(ORACLE[task.family], task.files())
                    self.assertEqual(observation.exit_code, 0, observation.stderr)
                    self.assertTrue(
                        is_correct(observation.stdout.strip(), task.expected, task.tolerance),
                        f"{task.id}: oracle printed {observation.stdout.strip()!r}, fixture says {task.expected!r}",
                    )


class LoopTests(unittest.TestCase):
    def test_an_oracle_episode_solves(self):
        sandbox = LocalSandbox(timeout_s=10)
        for task in load(FIXTURES):
            with self.subTest(task=task.id):
                answer = sandbox.run(ORACLE[task.family], task.files()).stdout.strip()
                client = ScriptedClient([f"<code>{ORACLE[task.family]}</code>", f"<report>{answer}</report>"])
                episode = run_episode(task, client, sandbox, max_turns=3)
                self.assertTrue(episode.solved, episode.reason)
                self.assertEqual(episode.protocol_errors, 0)
                self.assertEqual(episode.sandbox_errors, 0)
                self.assertEqual(len(episode.turns), 2)
                self.assertEqual(episode.lost_turns, 0, "an oracle prints exactly one value")
                self.assertEqual(episode.turns[0].warnings, [])

    def test_prose_counts_as_a_protocol_error(self):
        task = load(FIXTURES)[0]
        episode = run_episode(task, ScriptedClient(["The answer is 3."]), LocalSandbox(), max_turns=2)
        self.assertEqual(episode.protocol_errors, 1)
        self.assertFalse(episode.solved)

    def test_a_timeout_is_not_a_wrong_answer(self):
        task = load(FIXTURES)[0]
        timed = Observation(exit_code=None, stdout="", stderr="", timed_out=True, duration_ms=20000)
        sandbox = _OneShot(timed)
        episode = run_episode(
            task, ScriptedClient(["<code>while True: pass</code>", "<report>3</report>"]), sandbox, max_turns=3
        )
        self.assertEqual(episode.timeouts, 1)
        self.assertEqual(episode.sandbox_errors, 0)

    def test_a_wrong_answer_is_recorded_with_its_reason(self):
        task = load(FIXTURES)[0]
        episode = run_episode(task, ScriptedClient(["<report>-1</report>"]), LocalSandbox(), max_turns=2)
        self.assertFalse(episode.solved)
        self.assertIn("reported", episode.reason)


class _OneShot:
    def __init__(self, observation: Observation) -> None:
        self.observation = observation

    def run(self, code: str, files=None) -> Observation:
        return self.observation


if __name__ == "__main__":
    unittest.main(verbosity=2)
