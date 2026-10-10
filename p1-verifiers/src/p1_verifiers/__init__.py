"""P1: a verifier suite and its coverage report.

The package lives inside ``p1-verifiers/src`` because the directory name has a hyphen
and therefore cannot be a Python identifier.  The check script puts this directory on
``sys.path``; an installed environment would use ``pip install -e p1-verifiers``.
"""

from .harness import (
    PSEUDO,
    FixtureResponder,
    Provenance,
    PseudoTokeniser,
    Task,
    TaskSetError,
    Tokeniser,
    load_tasks,
    partition_by_coverage,
    pass_rate,
    report,
    run_episode,
    run_taskset,
)
from .remote import (
    OpenAIChatResponder,
    RemoteError,
    ServerTokeniser,
    list_models,
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
    "PSEUDO",
    "UNIMPLEMENTED",
    "ExactMatchVerifier",
    "FixtureResponder",
    "LogicPuzzleVerifier",
    "NumericVerifier",
    "OpenAIChatResponder",
    "Provenance",
    "PseudoTokeniser",
    "RemoteError",
    "ServerTokeniser",
    "Task",
    "TaskSetError",
    "Tokeniser",
    "UnitTestVerifier",
    "default_registry",
    "list_models",
    "load_tasks",
    "normalise",
    "partition_by_coverage",
    "pass_rate",
    "report",
    "run_episode",
    "run_taskset",
]
