"""Two deterministic task families, and the signals the environment gives away for free.

Chapter 6, Lesson 2's cheapest dense signal: in a tool-calling environment the *observation
itself* is feedback.  A command printed an error; a write changed nothing; the goal state was
reached.  That signal needs no model and no labels, and the only reason people train a critic
to predict it is that they forgot it was there.

What this module provides, and why each choice:

* **A family is a small simulated world with a fixed task list**, resettable between attempts.
  Not resettable means the same task scores differently across attempts and your reward
  differences are leaked state, not behaviour.
* **The world offers a finite action menu per turn.**  A tool-calling API is finite too - the
  model chooses an action from a declared set plus arguments - and a finite menu is what makes
  this project's policy learnable without a neural network, so the *credit-assignment* machinery
  can be studied on its own.  Swapping in ALFWorld or tau-bench replaces this module and nothing
  else; the seam is described in the README.
* **Every step returns an observation, a signal and a done flag.**  The signal is the free
  per-turn feedback (``ok``, ``error``, ``no-op``, ``done``), and the outcome check lives in
  ``solved()`` so that the trajectory-level truth is separate from the per-turn shaping.
* **No randomness anywhere.**  Two attempts of one task start from identical state, which is what
  makes a group's reward differences attributable to the policy's samples.

The pseudo-tokeniser.  `common.schema` counts tokens rather than storing them, so this module
counts words: one word is one token, and every run row names the convention in ``env_version``.
A real pipeline swaps in a real tokeniser and records its name; the mask arithmetic is identical
either way, which is the only thing this project needs from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

#: The free per-turn signals.  ``error`` and ``no-op`` are what a rule critic can read without a
#: model; ``done`` marks the turn that ended the episode.
SIGNALS = ("ok", "error", "no-op", "done")

#: One word is one token here; see the module docstring.
ENV_VERSION = "pseudo-words/v1 (p2 families, no tokeniser)"


def token_count(text: str) -> int:
    """Word count, floored at 1 so that an empty observation still costs a token."""
    return max(1, len(text.split()))


@dataclass(frozen=True)
class Task:
    """One task in a family."""

    task_id: str
    family: str
    prompt: str
    goal: str
    expected: str

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "prompt": self.prompt,
            "goal": self.goal,
            "expected": self.expected,
        }


@dataclass(frozen=True)
class StepResult:
    """What one action did."""

    observation: str
    signal: str
    done: bool
    answer: str | None = None

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError(f"unknown signal {self.signal!r}; expected one of {SIGNALS}")


class Attempt:
    """One live attempt at one task: the state a reset produces, and the actions on it.

    Subclasses own the world.  Everything a caller may touch is here, so the agent loop and the
    critics never reach into a family's internals.
    """

    family: str = "unknown"

    def __init__(self, task: Task) -> None:
        self.task = task
        self.history: list[tuple[str, StepResult]] = []
        self.finished = False
        self.answer: str | None = None
        self._reset_world()
        #: Distance-to-goal after every step, starting with the reset state.  Built *after* the
        #: reset, because it reads the world.  This is what makes a turn scoreable by *progress*
        #: rather than by activity: "the state changed" is not the same claim as "the state changed
        #: towards the goal", and a critic that confuses them teaches a policy to be busy
        #: (Chapter 6, Lesson 4's Echo Trap, which the first draft of this project reproduced).
        self.progress_trace: list[float] = [self.progress()]

    # ------------------------------------------------------------------ the world
    def _reset_world(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ the interface
    @property
    def observation(self) -> str:
        if not self.history:
            return self._opening_observation()
        return self.history[-1][1].observation

    def _opening_observation(self) -> str:
        raise NotImplementedError

    @property
    def signature(self) -> str:
        """A small state label the policy is conditioned on.

        Coarse on purpose: a table keyed on the exact state is not learnable from a few dozen
        attempts, and this project's subject is the credit signal, not sample efficiency.
        """
        return f"{self.task.task_id}:{self.stage}"

    @property
    def stage(self) -> str:
        raise NotImplementedError

    def menu(self) -> list[str]:
        """The finite set of actions legal right now, correct ones and distractors alike."""
        raise NotImplementedError

    def progress(self) -> float:
        """How close the world is to the task's goal, in [0, 1].  Each family says how."""
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        if self.finished:
            raise RuntimeError(f"task {self.task.task_id} is already finished")
        if action not in self.menu():
            # An action outside the menu is a format error, not an environment bug: the model
            # was offered a set and chose something else.  It is reported, never silently fixed.
            result = StepResult(
                observation=f"error: {action!r} is not one of the offered actions", signal="error", done=False
            )
            self.history.append((action, result))
            self.progress_trace.append(self.progress())
            return result
        result = self._apply(action)
        self.history.append((action, result))
        self.progress_trace.append(self.progress())
        if result.done:
            self.finished = True
            self.answer = result.answer
        return result

    def _apply(self, action: str) -> StepResult:
        raise NotImplementedError

    def goal_met(self) -> bool:
        """Whether the world is in the state the task asked for.  Each family says how."""
        raise NotImplementedError

    def solved(self) -> bool:
        """The trajectory-level truth: final state *and* the reported answer.

        Both halves matter.  A run that reports the right number after the wrong action has not
        solved the task, and a run that reaches the right state but reports the wrong number has
        not communicated it.  Checking only the answer would score ``rm draft.txt`` followed by
        ``done 1`` as a success on a rename task, which is exactly the kind of quiet false
        positive this repository exists to avoid.
        """
        reported = (self.answer or "").strip().lower() == self.task.expected.strip().lower()
        return bool(self.finished and reported and self.goal_met())


# ------------------------------------------------------------------ family 1: files


class FilesAttempt(Attempt):
    """A tiny directory.  The task is a path operation plus a count to report."""

    family = "files"

    def _reset_world(self) -> None:
        self.files: dict[str, str] = {
            "draft.txt": "first draft",
            "tmp.log": "noise",
            "keep.md": "notes",
        }
        self._stage = "start"

    def _opening_observation(self) -> str:
        return "directory: " + ", ".join(sorted(self.files))

    @property
    def stage(self) -> str:
        return self._stage

    def menu(self) -> list[str]:
        """Legal actions, generated per task.

        Generated rather than shared, and that is a bug fix with a lesson: the first draft offered
        one menu for the whole family, so the delete task's own action (``rm tmp.log``) was never
        on it and the task was unsolvable.  A menu that is missing the correct action is not a hard
        task - it is a broken environment, and no amount of training can show it.
        """
        if self._stage == "start":
            if self.task.task_id == "files-rename-count":
                return sorted(
                    {
                        "mv draft.txt final.txt",
                        "mv final.txt draft.txt",
                        "rm draft.txt",
                        "cat draft.txt",
                        "ls",
                        "done 1",
                        "done 2",
                        "done 3",
                    }
                )
            return sorted(
                {
                    "rm tmp.log",
                    "rm draft.txt",
                    "cat draft.txt",
                    "ls",
                    "done 0",
                    "done 1",
                    "done 2",
                }
            )
        return sorted({"ls", "done 0", "done 1", "done 2", "done 3"})

    def progress(self) -> float:
        if self.task.task_id == "files-rename-count":
            if self.goal_met():
                return 1.0
            if "final.txt" in self.files:
                return 0.6
            if "draft.txt" not in self.files:
                return 0.0
            return 0.25
        if self.task.task_id == "files-delete-count":
            if self.goal_met():
                return 1.0
            if "draft.txt" not in self.files:
                return 0.0  # removed the wrong file: worse than doing nothing
            return 0.25
        raise NotImplementedError(f"no progress measure for {self.task.task_id!r}")

    def goal_met(self) -> bool:
        if self.task.task_id == "files-rename-count":
            return "final.txt" in self.files and "draft.txt" not in self.files
        if self.task.task_id == "files-delete-count":
            return "tmp.log" not in self.files
        raise NotImplementedError(f"no goal check for {self.task.task_id!r}")

    def _apply(self, action: str) -> StepResult:
        parts = action.split()
        if parts[0] == "ls":
            return StepResult("directory: " + ", ".join(sorted(self.files)), "no-op", False)
        if parts[0] == "cat":
            name = parts[1]
            if name not in self.files:
                return StepResult(f"error: no such file {name}", "error", False)
            return StepResult(f"{name}: {self.files[name]}", "no-op", False)
        if parts[0] == "rm":
            name = parts[1]
            if name not in self.files:
                return StepResult(f"error: no such file {name}", "error", False)
            del self.files[name]
            self._stage = "after-change"
            return StepResult("directory: " + ", ".join(sorted(self.files)), "ok", False)
        if parts[0] == "mv":
            source, target = parts[1], parts[2]
            if source not in self.files:
                return StepResult(f"error: no such file {source}", "error", False)
            self.files[target] = self.files.pop(source)
            self._stage = "after-change"
            return StepResult("directory: " + ", ".join(sorted(self.files)), "ok", False)
        if parts[0] == "done":
            return StepResult(f"reported {parts[1]}", "done", True, answer=parts[1])
        return StepResult(f"error: unknown action {action!r}", "error", False)


# ----------------------------------------------------------------- family 2: ledger


class LedgerAttempt(Attempt):
    """A running balance.  The task is a sequence of movements plus the balance to report."""

    family = "ledger"

    def _reset_world(self) -> None:
        self.balance = 0
        self._stage = "start"
        self._moves: list[int] = []

    def _opening_observation(self) -> str:
        return f"balance: {self.balance}"

    @property
    def stage(self) -> str:
        return self._stage

    def menu(self) -> list[str]:
        if not self._moves:
            return sorted({"add 12", "add 5", "sub 5", "sub 12", "done 7", "done 12", "done 0"})
        return sorted({"add 12", "sub 5", "done 7", "done 12", "done 5"})

    def progress(self) -> float:
        """Progress along the *required* sequence, not distance to the final number.

        Distance alone was the first draft, and it is wrong in a way worth keeping: adding 5 moves
        the balance to 5, which is nearer to 7 than 12 is, so the distractor outscored the correct
        first move and the policy learned it.  A progress measure has to know the plan.
        """
        plan = [12, -5]
        if len(self._moves) > len(plan):
            return 0.0
        if self._moves != plan[: len(self._moves)]:
            return 0.0
        return len(self._moves) / len(plan)

    def goal_met(self) -> bool:
        return self.balance == 7

    def _apply(self, action: str) -> StepResult:
        parts = action.split()
        if parts[0] in {"add", "sub"}:
            amount = int(parts[1])
            self.balance += amount if parts[0] == "add" else -amount
            self._moves.append(amount if parts[0] == "add" else -amount)
            self._stage = "after-change"
            return StepResult(f"balance: {self.balance}", "ok", False)
        if parts[0] == "done":
            return StepResult(f"reported {parts[1]}", "done", True, answer=parts[1])
        return StepResult(f"error: unknown action {action!r}", "error", False)


# --------------------------------------------------------------------- families


@dataclass(frozen=True)
class Family:
    """A task family: its name, its version, its tasks, and how to start an attempt."""

    name: str
    version: str
    tasks: tuple[Task, ...]
    starter: Callable[[Task], Attempt]

    def task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"no task {task_id!r} in family {self.name!r}")

    def start(self, task_id: str) -> Attempt:
        return self.starter(self.task(task_id))


FILES_TASKS = (
    Task(
        task_id="files-rename-count",
        family="files",
        prompt=(
            "In the shared directory, rename draft.txt to final.txt, then report how many files "
            "start with the letter f.  Finish with: done <number>"
        ),
        goal="draft.txt is renamed to final.txt",
        expected="1",
    ),
    Task(
        task_id="files-delete-count",
        family="files",
        prompt=(
            "In the shared directory, delete tmp.log, then report how many files remain.  "
            "Finish with: done <number>"
        ),
        goal="tmp.log is gone",
        expected="2",
    ),
)

LEDGER_TASKS = (
    Task(
        task_id="ledger-net-seven",
        family="ledger",
        prompt="Add 12 to the balance, then subtract 5, then report the balance.  Finish with: done <number>",
        goal="the balance is 7",
        expected="7",
    ),
)


def families() -> dict[str, Family]:
    """The registered families, keyed by name."""
    return {
        "files": Family(
            name="files",
            version="files/v1",
            tasks=FILES_TASKS,
            starter=FilesAttempt,
        ),
        "ledger": Family(
            name="ledger",
            version="ledger/v1",
            tasks=LEDGER_TASKS,
            starter=LedgerAttempt,
        ),
    }


def all_tasks() -> list[Task]:
    return [task for family in families().values() for task in family.tasks]


def start(task_id: str) -> Attempt:
    """Start an attempt at ``task_id`` in whichever family owns it."""
    for family in families().values():
        if any(task.task_id == task_id for task in family.tasks):
            return family.start(task_id)
    raise KeyError(f"no family owns task {task_id!r}")
