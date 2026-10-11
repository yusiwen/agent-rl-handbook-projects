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
    DEFAULT_ATTEMPT_FILE,
    DEFAULT_TEST_TIMEOUT_S,
    UNIMPLEMENTED,
    DockerSandbox,
    ExactMatchVerifier,
    LocalSandbox,
    LogicPuzzleVerifier,
    NumericVerifier,
    Sandbox,
    SandboxResult,
    SandboxUnavailable,
    UnitTestVerifier,
    attempt_source,
    default_registry,
    normalise,
)

__all__ = [
    "DEFAULT_ATTEMPT_FILE",
    "DEFAULT_TEST_TIMEOUT_S",
    "DockerSandbox",
    "ExactMatchVerifier",
    "FixtureResponder",
    "LocalSandbox",
    "LogicPuzzleVerifier",
    "NumericVerifier",
    "OpenAIChatResponder",
    "PSEUDO",
    "Provenance",
    "PseudoTokeniser",
    "RemoteError",
    "Sandbox",
    "SandboxResult",
    "SandboxUnavailable",
    "ServerTokeniser",
    "Task",
    "TaskSetError",
    "Tokeniser",
    "UNIMPLEMENTED",
    "UnitTestVerifier",
    "attempt_source",
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
