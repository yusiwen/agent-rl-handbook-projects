"""P1: a verifier suite and its coverage report.

The package lives inside ``p1-verifiers/src`` because the directory name has a hyphen
and therefore cannot be a Python identifier.  The check script puts this directory on
``sys.path``; an installed environment would use ``pip install -e p1-verifiers``.
"""

from .harness import (
    FixtureResponder,
    Provenance,
    Task,
    TaskSetError,
    load_tasks,
    partition_by_coverage,
    pass_rate,
    report,
    run_episode,
    run_taskset,
)
from .verifiers import (
    UNIMPLEMENTED,
    ExactMatchVerifier,
    LogicPuzzleVerifier,
    NumericVerifier,
    UnitTestVerifier,
    default_registry,
    normalise,
)

__all__ = [
    "UNIMPLEMENTED",
    "ExactMatchVerifier",
    "FixtureResponder",
    "LogicPuzzleVerifier",
    "NumericVerifier",
    "Provenance",
    "Task",
    "TaskSetError",
    "UnitTestVerifier",
    "default_registry",
    "load_tasks",
    "normalise",
    "partition_by_coverage",
    "pass_rate",
    "report",
    "run_episode",
    "run_taskset",
]
