"""P1's verifier suite: three implemented, one that is yours.

The book's Chapter 5 deliverable is a verifier suite plus a coverage report.  This
module gives you the shape and three worked examples (``ExactMatchVerifier``,
``NumericVerifier`` and ``UnitTestVerifier``); the last one is the exercise.  Each stub is
honest about its state:

* ``can_verify`` returns ``False`` while the verifier is unimplemented, so the coverage
  report tells the truth instead of promising a check that does not exist.
* ``verify`` raises ``NotImplementedError`` with what to do, so a stub can never
  quietly return a zero - an unscored task and a failed task would then look identical
  in a reward curve, and only one of them is the model's fault.

Implement one, flip its ``can_verify`` to a real test, and re-run
``p1-verifiers/checks/check_p1.py``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from common.verifier import Outcome, VerifierRegistry

#: Answers are compared after this normalisation.  Deliberately conservative: it folds
#: case, spacing and a trailing full stop, and nothing else.  Anything cleverer belongs
#: in a different verifier, where its assumptions are visible.
_WHITESPACE = re.compile(r"\s+")

#: A number as it appears in prose: an optional sign, either a thousands-separated group
#: (``1,000``) or a plain run of digits, an optional decimal part, and an optional exponent
#: (``1e3``).  Spelled out rather than cutting a slice of text and hoping ``float`` takes it,
#: because the job is "find the number" and the units around it are the task's business.
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: A fenced code block: the whole response can be one, or one can sit among prose.
_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def final_answer(response: str) -> str:
    """Extract the candidate answer from a response: the last non-empty line.

    Every verifier owns a parser, and this is the first one.  The policy is deliberately
    blunt - "the final answer is on its own last line", which the system prompt asks for -
    because a clever extractor is a new failure mode you cannot see in the reward curve.
    Two cases here show both directions of that:

    * ``"... reasoning ...\\\\n\\\\n150 minutes"`` is extracted as ``150 minutes`` and fails an
      exact-match check against ``150``: correct arithmetic, wrong claim about what the
      answer *is*.
    * ``"x + 7 = 19, so x = 12.\\\\n\\\\nx = 12"`` is extracted as ``x = 12`` and also fails:
      this extractor does not understand equations, and pretending otherwise would need a
      symbolic parser this project has not written.

    Returns an empty string when the response has no non-empty line, so the caller fails
    the task rather than crashing on it.
    """
    for line in reversed(response.splitlines()):
        candidate = line.strip()
        if candidate:
            return candidate[:-1] if candidate.endswith(".") else candidate
    return ""


def normalise(text: str) -> str:
    """Case-insensitive, whitespace-insensitive, trailing-punctuation-insensitive."""
    folded = _WHITESPACE.sub(" ", text.strip().lower())
    return folded.rstrip(".").strip()


def last_number(response: str) -> float | None:
    """The last number in a response, or ``None`` when it contains none.

    The *last* one, because the system prompt asks for the final answer on the last line
    and a response that shows its work ends with the number that matters::

        "Pi is 3.14159..., so to two decimal places it is 3.14."   ->   3.14

    Units are deliberately ignored.  That is what lets ``150 minutes`` count as the answer
    ``150`` - and it is also this helper's blind spot: ``150 kilometres`` scores the same
    as ``150 minutes``.  A task where the unit carries meaning wants ``ExactMatchVerifier``,
    which compares text and refuses both.

    A date or a version string reads as its last numeric fragment (``2026-10-10`` -> ``-10``),
    which is another reason those tasks belong to a different verifier.

    Returns ``None`` rather than raising: a response with no number in it is a failed
    attempt, not a defect in the harness, and the two must not look alike in a report.
    """
    matches = _NUMBER.findall(response)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def _as_number(value: Any, *, what: str) -> float:
    """A float from a task field, or an error that names the field.

    Loud rather than zero: a task whose ``answer`` field is not a number is a defect in the
    task set, and a verifier that scores it ``0.0`` turns that defect into a training
    signal.  ``common.verifier``'s rule for a missing verifier applies here too - an
    unscored task and a failed task must not look identical.
    """
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not a number: {value!r}") from exc


class ExactMatchVerifier:
    """Scores 1.0 for the expected answer as a string, 0.0 otherwise.

    Use it for answers with one agreed surface form - a number, a short name, a date.
    Do **not** reach for it when the surface form varies: ``1024.0`` and ``1024`` are
    the same answer to a mathematician, and this verifier will disagree.  That case is
    what ``NumericVerifier`` below is for.

    This one is implemented, because the harness needs at least one verifier that works
    before you can tell an environment problem from an implementation problem.
    """

    name = "exact_match"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        return "answer" in task and isinstance(task.get("answer"), str)

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        expected = normalise(str(task["answer"]))
        extracted = final_answer(response)
        actual = normalise(extracted)
        matched = expected == actual
        if matched:
            detail = "exact match"
        else:
            detail = f"expected {expected!r}, got {actual!r}"
            if extracted != response.strip():
                detail += f" (extracted from the last line: {extracted!r})"
        return Outcome(passed=matched, score=1.0 if matched else 0.0, detail=detail)


class NumericVerifier:
    """Scores a number within a tolerance.  Tasks carry ``answer`` plus ``tolerance``.

    The tolerance is what makes some tasks scoreable at all.  ``1024`` and ``1024.0`` are
    the same answer to a mathematician and different strings to ``ExactMatchVerifier``; a
    model that writes ``3.14159`` for "pi to two decimal places" is right, and a text
    comparison would call it wrong.  So the question here is "is the last number in the
    response close enough", and the answer is a pass or a fail rather than a graded score:
    partial credit for wrong arithmetic is a reward you cannot explain later.

    The comparison is ``abs(actual - expected) <= tolerance``.  One caveat belongs in the
    open: both numbers are binary floats, so a value that *looks* exactly one tolerance away
    can land on either side of it.  Measured, not assumed: ``abs(3.15 - 3.14)`` is
    ``0.009999999999999787``, which is inside a tolerance of ``0.01``.  A tolerance written
    to one significant figure is a statement about the answer, not about the last binary
    digit - do not put a task's boundary on a knife edge and expect the tie to resolve your
    way.

    What it will not do: understand units, algebra or equations, or guess which of several
    numbers in a response was meant.  It takes the last one.  When that is not good enough
    for a task, the task needs a different verifier, not a cleverer regex.
    """

    name = "numeric"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        """True when the task carries the fields this verifier needs.

        Presence only.  A task whose ``answer`` is not a number is a task-set defect, and
        ``verify`` reports it loudly rather than this predicate quietly shrinking coverage.
        """
        return "answer" in task and "tolerance" in task

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        task_id = task.get("id", "<no id>")
        expected = _as_number(task["answer"], what=f"task {task_id!r} answer")
        tolerance = _as_number(task["tolerance"], what=f"task {task_id!r} tolerance")
        if tolerance < 0:
            raise ValueError(
                f"task {task_id!r} has a negative tolerance ({tolerance!r}); a tolerance is a "
                "distance, so write the value you mean"
            )
        actual = last_number(response)
        if actual is None:
            return Outcome(
                passed=False,
                score=0.0,
                detail=(
                    f"expected {expected:g} +/- {tolerance:g}, but the response contains no number"
                ),
            )
        delta = abs(actual - expected)
        matched = delta <= tolerance
        return Outcome(
            passed=matched,
            score=1.0 if matched else 0.0,
            detail=f"expected {expected:g} +/- {tolerance:g}, got {actual:g} (off by {delta:g})",
        )


#: The file an attempt is written to when its task does not name one.  Tests are expected to
#: reach it as a module, which is why the default is a name a test can import.
DEFAULT_ATTEMPT_FILE = "main.py"

#: Wall-clock seconds a task's tests get before the attempt is called a failure.  Generous for
#: a unit test, short enough that a hung sandbox costs a round rather than a night.
DEFAULT_TEST_TIMEOUT_S = 20.0

#: Docker's own exit status for "the client could not run the container at all", as opposed to
#: a container that ran and whose command failed.  Only this one is an environment problem.
DOCKER_CLIENT_ERROR = 125


def attempt_source(response: str) -> str:
    """The code to run in the sandbox: the last fenced block, or the response itself.

    The one piece of presentation this verifier removes, and it removes exactly one: a model
    that answers with a fenced block has followed the instruction and wrapped it in markdown,
    and a verifier that wrote the fence lines into the file would be scoring the wrapper
    rather than the code.  The *last* block, for the same reason ``final_answer`` takes the
    last line - the prompt asks for one answer, and the last one is the answer.

    Everything else is left alone: prose around the code, a missing function, a syntax error.
    Those belong to the attempt, and the attempt's score should say so.
    """
    blocks = _FENCE.findall(response)
    return (blocks[-1] if blocks else response).strip()


@dataclass(frozen=True)
class SandboxResult:
    """What a sandbox reports back: status, whether it timed out, output, and who ran it."""

    exit_code: int | None
    timed_out: bool
    output: str
    sandbox: str

    def tail(self, limit: int = 200) -> str:
        """The last non-empty line of output, truncated: what a detail line can carry.

        Not the whole output: a row's detail is a pointer for a human, not a log, and a
        traceback pasted into every row is how a run record stops being readable.
        """
        for line in reversed(self.output.strip().splitlines()):
            if line.strip():
                return line.strip()[:limit]
        return ""


class SandboxUnavailable(RuntimeError):
    """Raised when the configured sandbox cannot run at all.

    Deliberately an error and not a zero score, for the same reason as ``MissingVerifier``:
    "this machine cannot run the tests" and "the attempt failed the tests" are different
    facts, and only one of them is the model's fault.  A verifier that scored an unavailable
    sandbox as 0.0 would be training on an environment bug.
    """


class Sandbox(Protocol):
    """Where a task's tests run.  ``name`` travels into every detail, so a row can say where."""

    name: str

    def run(self, command: str, workdir: Path, timeout: float) -> SandboxResult:  # pragma: no cover
        ...


@dataclass(frozen=True)
class LocalSandbox:
    """Runs the tests with ``subprocess`` on this machine.  No isolation whatsoever.

    It exists for one narrow case: attempts that *this repository* wrote, so that the offline
    gate needs no container runtime.  For anything a model produced it is the wrong tool - it
    would run generated code directly on the host - so it refuses to be anonymous.  It demands
    a ``why`` sentence, and every detail it produces carries that sentence: a row scored
    without a container says so, and says why.
    """

    why: str
    name = "local subprocess (no isolation)"

    def run(self, command: str, workdir: Path, timeout: float) -> SandboxResult:
        if not self.why.strip():
            raise ValueError("LocalSandbox needs a 'why': say whose code this is")
        try:
            completed = subprocess.run(
                ["sh", "-lc", command],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = _as_text(exc.stdout) + _as_text(exc.stderr)
            return SandboxResult(None, True, output, self.name)
        return SandboxResult(
            completed.returncode, False, completed.stdout + completed.stderr, self.name
        )


@dataclass(frozen=True)
class DockerSandbox:
    """Runs the tests in a container: no network, a memory cap, and the timeout from outside.

    The book's two non-negotiables - a container and a wall-clock timeout - are both here.  The
    third thing that matters is what happens when it is *missing*: no ``docker`` on PATH raises
    ``SandboxUnavailable`` rather than quietly running a model's code on the host.
    """

    image: str = "python:3.13-alpine"
    memory: str = "512m"
    name = "docker"

    def argv(self, command: str, workdir: Path, container: str) -> list[str]:
        """The docker command line, kept separate from running it so the flags are testable.

        ``--network=none`` because a unit test has no business on the network, and the
        alternative is generated code with a route out.  ``--memory`` so a runaway allocation
        does not take the host with it.  ``--rm`` plus a name, so a container cannot outlive
        the run that started it.
        """
        return [
            "docker", "run", "--rm",
            "--name", container,
            "--network=none",
            f"--memory={self.memory}",
            "-v", f"{workdir}:/work",
            "-w", "/work",
            self.image,
            "sh", "-lc", command,
        ]

    def run(self, command: str, workdir: Path, timeout: float) -> SandboxResult:
        if shutil.which("docker") is None:
            raise SandboxUnavailable(
                "docker is not on PATH, so the unit-test sandbox cannot run. Install a "
                "container runtime, or pass a LocalSandbox with a 'why' when the attempts "
                "are your own code rather than a model's."
            )
        container = f"p1-unit-{uuid.uuid4().hex[:12]}"
        try:
            completed = subprocess.run(
                self.argv(command, workdir, container),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # Killing the client does not stop the container, so kill it by name.  A leftover
            # container is a hang that costs the *next* run - the failure mode this verifier
            # exists to avoid, and one it would otherwise cause itself.
            subprocess.run(
                ["docker", "kill", container], capture_output=True, text=True, check=False
            )
            output = _as_text(exc.stdout) + _as_text(exc.stderr)
            return SandboxResult(None, True, output, self.name)
        if completed.returncode == DOCKER_CLIENT_ERROR:
            raise SandboxUnavailable(
                "docker could not run the container at all (its own exit status "
                f"{DOCKER_CLIENT_ERROR}): {completed.stderr.strip()[:200]}"
            )
        return SandboxResult(
            completed.returncode, False, completed.stdout + completed.stderr, self.name
        )


def _as_text(value: object) -> str:
    """Whatever a timeout left behind, as text: bytes, a string, or nothing at all."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class UnitTestVerifier:
    """Runs a task's tests in a sandbox and scores the exit status.  Tasks carry ``tests``.

    The two rules from the book are structure here rather than advice.  The tests run in a
    sandbox, and ``DockerSandbox`` is what the registry hands this verifier by default, because
    the code being executed came from a model.  The run is bounded by a wall-clock timeout, and
    the timeout expiring is a *failure* whose detail says "timed out" - an environment that
    hangs must not look like a model that is slow.  A sandbox that cannot start at all is
    neither: that raises ``SandboxUnavailable``.

    Optional task fields: ``attempt_file`` (default ``main.py``) is where the attempt is
    written in the sandbox's working directory, and ``test_timeout`` overrides the default of
    20 seconds.  Every detail names the sandbox that produced it, so a row scored without a
    container is distinguishable from one scored inside one.
    """

    name = "unit_test"

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        """True when the task states the tests to run.  Presence only, like ``numeric``."""
        tests = task.get("tests")
        return isinstance(tests, str) and bool(tests.strip())

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        task_id = task.get("id", "<no id>")
        tests = str(task["tests"])
        raw_attempt = task.get("attempt_file")
        attempt_file = DEFAULT_ATTEMPT_FILE if raw_attempt is None else str(raw_attempt)
        if attempt_file in {"", ".", ".."} or "/" in attempt_file or "\\" in attempt_file:
            raise ValueError(
                f"task {task_id!r} names an attempt_file that is not a plain file name: "
                f"{attempt_file!r}"
            )
        # Explicitly about None: `task.get("test_timeout") or DEFAULT` treats a zero as
        # "absent", and a task that says 0 means 0 - which is a defect, not a default.
        raw_timeout = task.get("test_timeout")
        timeout = DEFAULT_TEST_TIMEOUT_S if raw_timeout is None else float(raw_timeout)
        if timeout <= 0:
            raise ValueError(f"task {task_id!r} has a non-positive test_timeout: {timeout!r}")
        source = attempt_source(response)
        if not source:
            return Outcome(
                passed=False,
                score=0.0,
                detail="the response is empty, so there is nothing to run the tests against",
            )
        with tempfile.TemporaryDirectory(prefix="p1-unit-") as tmp:
            workdir = Path(tmp)
            (workdir / attempt_file).write_text(source + "\n", encoding="utf-8")
            result = self.sandbox.run(tests, workdir, timeout)

        where = f"{result.sandbox}; attempt written to {attempt_file}"
        if result.timed_out:
            return Outcome(
                passed=False,
                score=0.0,
                detail=(
                    f"timed out after {timeout:g}s ({where}) - a hung sandbox is a failed "
                    "attempt, not an error"
                ),
            )
        if result.exit_code == 0:
            return Outcome(passed=True, score=1.0, detail=f"tests passed ({where})")
        detail = f"tests failed with exit status {result.exit_code} ({where})"
        if result.tail():
            detail += f": {result.tail()}"
        return Outcome(passed=False, score=0.0, detail=detail)


class LogicPuzzleVerifier:
    """YOUR TURN.  Check a structured solution (a grid, an assignment) rather than a string.

    Tasks that want this verifier carry a ``solution`` entry: the constraint set the
    attempt must satisfy.
    """

    name = "logic_puzzle"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        return False  # until implemented

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        raise NotImplementedError(
            "logic-puzzle verification is not implemented yet.  Parse the assignment out "
            "of the response and check it against task['solution'] constraint by "
            "constraint, so that a near miss is reported as a near miss instead of a "
            "bare zero."
        )


#: Every verifier this project declares.  ``common.checks``-style honesty applies: a
#: name in here that declines a task shows up in the coverage report, not in a log.
def default_registry(sandbox: Sandbox | None = None) -> VerifierRegistry:
    """The suite, with a choice to make about the unit-test verifier's sandbox.

    The default is ``DockerSandbox``: this verifier executes code that a model produced, and
    running it directly on the host is not a trade-off worth making for speed.  Pass a
    ``LocalSandbox`` only when the attempts are this repository's own - which is what
    ``check_p1.py`` does, so that the offline gate needs no container runtime.
    """
    return VerifierRegistry(
        [
            ExactMatchVerifier(),
            NumericVerifier(),
            UnitTestVerifier(sandbox or DockerSandbox()),
            LogicPuzzleVerifier(),
        ]
    )


#: Names whose ``verify`` is expected to raise ``NotImplementedError``.  The check
#: script asserts this, so a stub cannot silently become a zero-scoring verifier.
UNIMPLEMENTED = ("logic_puzzle",)
