"""``__init__`` for the p5 package: the world, the protocol, the loop, and the numbers."""

from p5_openclaw.world import (
    DEMO_TASKS,
    DOCS,
    EXTRA_TASKS,
    TASKS,
    Observation,
    Task,
    World,
    oracle_episode,
    solved,
)

__all__ = ["DEMO_TASKS", "DOCS", "EXTRA_TASKS", "TASKS", "Observation", "Task", "World",
           "oracle_episode", "solved"]
