"""P4 - a data-analysis agent: a marker protocol, a sandbox, and an episode loop.

The package is deliberately small and dependency-free (standard library only).  The sandbox is
the only component that reaches outside the process, and it is the component the tests replace.
"""

from .agent import Episode, Turn, run_episode
from .protocol import Action, ProtocolError, parse
from .sandbox import DockerSandbox, LocalSandbox, Observation
from .tasks import Task, load
from .verify import is_correct, last_number

__all__ = [
    "Action", "DockerSandbox", "Episode", "LocalSandbox", "Observation",
    "ProtocolError", "Task", "Turn", "is_correct", "last_number", "load", "parse", "run_episode",
]
