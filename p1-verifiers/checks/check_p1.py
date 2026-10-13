#!/usr/bin/env python3
"""Grade P1: run the harness on fixtures and check it against the book's promises.

    python3 p1-verifiers/checks/check_p1.py

Runs with no network, no weights and no GPU - which is the point.  If this fails, the
problem is in this repository; if it passes and a real run fails, the problem is in the
environment or in the model.

What it checks:

1. **coverage** - the task set is reported as covered, declined or unregistered, and the
   numbers are counted rather than estimated.
2. **the four shared checks** - mask, boundaries, terms, advantage spread, all on rows
   produced by this project's harness rather than by the fixtures in ``common/``.
3. **the fixture contract** - the fixture is built to have a known pass rate and a known
   number of mixed groups, so an accidental edit to it is caught rather than absorbed.
4. **the stub contract** - an unimplemented verifier declines tasks and raises when called,
   so it can never quietly score zero.

Exit status is 0 only when every part passes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))              # for `common`
sys.path.insert(0, str(HERE.parent / "src"))  # for `p1_verifiers`

from common.checks import (  # noqa: E402
    check_advantage_spread,
    check_boundaries,
    check_mask,
    check_terms,
)
from common.runs import RunHeader, RunWriter, digest_of, round_record  # noqa: E402
from common.verifier import MissingVerifier  # noqa: E402

from p1_verifiers import (  # noqa: E402
    UNIMPLEMENTED,
    FixtureResponder,
    LocalSandbox,
    Provenance,
    default_registry,
    load_tasks,
    partition_by_coverage,
    pass_rate,
    report,
    run_taskset,
)
from p1_verifiers.harness import PSEUDO_TOKENISER  # noqa: E402

FIXTURES = HERE.parent / "fixtures"
TASKS = FIXTURES / "tasks.jsonl"
RESPONSES = FIXTURES / "responses.jsonl"

#: What the fixture is built to produce.  Editing the fixture without editing these
#: numbers fails the check on purpose: a fixture whose contract is not stated drifts.
EXPECTED_TASKS = 18
EXPECTED_COVERED = 18
EXPECTED_SAMPLES_PER_TASK = 2
EXPECTED_PASSED = 18
EXPECTED_MIXED_GROUPS = 16


def code_sha() -> str:
    """The current commit, or ``git:none`` before the first commit exists."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return f"git:{result.stdout.strip()}" if result.returncode == 0 else "git:none"


def main() -> int:
    failures: list[str] = []
    print("== P1 verifier check ==")

    tasks = load_tasks(TASKS)
    responder = FixtureResponder.from_jsonl(RESPONSES)
    # The unit-test verifier executes code, so its sandbox is a choice rather than a default.
    # The registry hands out the container, because what it runs in a real run came from a
    # model; this script passes a local one because the fixture's attempts are written in this
    # repository and the offline gate must not need a container runtime.  Every row it produces
    # says which sandbox produced it, and why.
    registry = default_registry(
        sandbox=LocalSandbox(why="the fixture's attempts are written in this repository")
    )

    print(f"tasks: {len(tasks)} loaded from {TASKS.name}")
    if len(tasks) != EXPECTED_TASKS:
        failures.append(f"expected {EXPECTED_TASKS} tasks, found {len(tasks)}")

    coverage = registry.coverage(task.as_mapping() for task in tasks)
    print(coverage.render())
    if coverage.covered != EXPECTED_COVERED:
        failures.append(f"expected {EXPECTED_COVERED} covered tasks, counted {coverage.covered}")

    covered, uncovered = partition_by_coverage(tasks, registry)
    for task, reason in uncovered:
        print(f"  unfinished: {task.id} - {reason}")

    rows = run_taskset(
        covered,
        responder,
        registry,
        samples_per_task=EXPECTED_SAMPLES_PER_TASK,
        provenance=Provenance(
            env_version=PSEUDO_TOKENISER,
            config_hash=digest_of(TASKS),
            code_sha=code_sha(),
        ),
    )
    print(f"rollouts: {len(rows)} "
          f"({len(covered)} tasks x {EXPECTED_SAMPLES_PER_TASK} samples, covered subset)")
    print(report(rows))

    expected_rows = EXPECTED_COVERED * EXPECTED_SAMPLES_PER_TASK
    if len(rows) != expected_rows:
        failures.append(f"expected {expected_rows} rollouts, got {len(rows)}")

    for check in (
        check_mask(rows),
        check_boundaries(rows),
        check_terms(rows),
        check_advantage_spread(rows),
    ):
        print(check.render())
        if not check.ok:
            failures.append(f"{check.name} check failed")

    passed, total = pass_rate(rows)
    if (passed, total) != (EXPECTED_PASSED, expected_rows):
        failures.append(
            f"the fixture contract says {EXPECTED_PASSED}/{expected_rows} pass, counted {passed}/{total}"
        )

    spreads = [row for row in {row.group_id: row for row in rows}.values()]
    mixed = sum(
        1
        for group_id in {row.group_id for row in rows}
        if len({row.reward() for row in rows if row.group_id == group_id}) > 1
    )
    print(f"fixture contract: {passed}/{total} passed, {mixed}/{len(spreads)} groups mixed")
    if mixed != EXPECTED_MIXED_GROUPS:
        failures.append(f"expected {EXPECTED_MIXED_GROUPS} mixed groups, counted {mixed}")

    # The stubs must be honestly unimplemented: never a silent zero.  An empty UNIMPLEMENTED
    # means the suite is complete, and this says so rather than iterating nothing and printing
    # an empty list of names.
    if UNIMPLEMENTED:
        for name in UNIMPLEMENTED:
            verifier = registry.get(name)
            probe = {"id": "probe", "prompt": "irrelevant", "verifier": name, "answer": "1",
                     "tolerance": 0.1, "tests": "true", "solution": {}}
            if verifier.can_verify(probe):
                failures.append(f"{name}: can_verify() claims coverage but verify() is a stub")
            try:
                verifier.verify(probe, "anything")
            except NotImplementedError as exc:
                if "not implemented yet" not in str(exc):
                    failures.append(f"{name}: NotImplementedError does not say what to do")
            else:
                failures.append(
                    f"{name}: verify() returned instead of raising - it may score zero silently"
                )
        print(f"stub contract: {', '.join(UNIMPLEMENTED)} all decline and raise as expected")
    else:
        print("stub contract: no stubs left - every declared verifier is implemented")

    run_id = f"p1-fixtures-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    # Where the run record lands. Overridable because a sandboxed run (nix build, CI) has a
    # read-only source tree: the record is an artefact, and artefacts go where the caller says.
    runs_root = Path(os.environ.get("ARL_RUNS_DIR") or REPO / "runs")
    writer = RunWriter(
        RunHeader(
            run_id=run_id,
            env_version=PSEUDO_TOKENISER,
            config_hash=digest_of(TASKS),
            code_sha=code_sha(),
            project="p1-verifiers",
            dataset="fixtures",
            dataset_digest=digest_of(TASKS),
            notes=f"responses fixture digest {digest_of(RESPONSES)}",
        ),
        root=runs_root,
    )
    writer.append_round(round_record(0, rows))
    writer.append_rollouts(0, rows)
    shown = runs_root.relative_to(REPO) if runs_root.is_relative_to(REPO) else runs_root
    print(f"run record: {shown}/ (header.json, rounds.jsonl, round-000.jsonl)")

    if UNIMPLEMENTED:
        print(
            f"-- your turn: implement {', '.join(UNIMPLEMENTED)} in "
            "p1-verifiers/src/p1_verifiers/verifiers.py --"
        )
    else:
        print("-- all four verifiers are implemented: what is left is the task set --")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall P1 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
