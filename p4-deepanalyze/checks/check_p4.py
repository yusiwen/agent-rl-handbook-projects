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

#: One program per task, written against the CSV rather than against the stored answer.
ORACLE: dict[str, str] = {
    "p4-sales-north": """\
import csv
with open('table.csv') as handle:
    rows = list(csv.DictReader(handle))
print(sum(float(r['units']) * float(r['unit_price']) for r in rows if r['region'] == 'North'))
""",
    "p4-sales-top-product": """\
import csv
from collections import Counter
counts = Counter()
with open('table.csv') as handle:
    for row in csv.DictReader(handle):
        counts[row['product']] += float(row['units'])
print(max(counts.items(), key=lambda kv: kv[1])[0])
""",
    "p4-temps-mean": """\
import csv
with open('table.csv') as handle:
    temps = [float(r['temp_c']) for r in csv.DictReader(handle)]
print(round(sum(temps) / len(temps), 1))
""",
    "p4-students-pass-rate": """\
import csv
with open('table.csv') as handle:
    scores = [float(r['score']) for r in csv.DictReader(handle)]
print(round(sum(1 for s in scores if s >= 60) / len(scores), 4))
""",
    "p4-orders-biggest-month": """\
import csv
with open('table.csv') as handle:
    rows = list(csv.DictReader(handle))
print(max(rows, key=lambda r: float(r['amount']))['month'])
""",
    "p4-inventory-shortfall": """\
import csv
with open('table.csv') as handle:
    rows = list(csv.DictReader(handle))
print(int(max(float(r['reorder_level']) - float(r['stock']) for r in rows)))
""",
}


class FixtureTests(unittest.TestCase):
    def test_every_task_has_an_oracle(self):
        self.assertEqual({task.id for task in load(FIXTURES)}, set(ORACLE))

    def test_every_expected_answer_is_reproduced_by_its_oracle(self):
        sandbox = LocalSandbox(timeout_s=10)
        for task in load(FIXTURES):
            with self.subTest(task=task.id):
                observation = sandbox.run(ORACLE[task.id], task.files())
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
                answer = sandbox.run(ORACLE[task.id], task.files()).stdout.strip()
                client = ScriptedClient([f"<code>{ORACLE[task.id]}</code>", f"<report>{answer}</report>"])
                episode = run_episode(task, client, sandbox, max_turns=3)
                self.assertTrue(episode.solved, episode.reason)
                self.assertEqual(episode.protocol_errors, 0)
                self.assertEqual(episode.sandbox_errors, 0)
                self.assertEqual(len(episode.turns), 2)

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
