"""P1's verifier suite: all four verifiers implemented.

The book's Chapter 5 deliverable is a verifier suite plus a coverage report, and this module is
the suite: four verifiers, each with a docstring that says what it will *not* do - which is the
part that decides whether you can trust the numbers it produces.

The pattern to keep if you add another one: a verifier that cannot judge a task declines it
(``can_verify`` returns ``False``) instead of scoring it, and a verifier that is not written yet
raises ``NotImplementedError`` instead of returning a zero.  An unscored task and a failed task
look identical in a reward curve, and only one of them is the model's fault; the check script
asserts both halves of that rule for every name in ``UNIMPLEMENTED``.
"""

from __future__ import annotations

import json
import os
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


def numbers_in(text: str) -> list[float]:
    """Every number in a piece of text, in the order it appears."""
    return [float(match.replace(",", "")) for match in _NUMBER.findall(text)]


#: What every ``NumericVerifier`` detail starts with, so a run record can be read for *why* a
#: group was flat without re-running it.  ``form=ok`` means the response declared its answer in
#: the form the task asked for and the value was scored; ``form=no`` means it did not and the
#: value never entered the reward.
FORM_PREFIX = "form="


def declared_number(response: str) -> tuple[float | None, str]:
    """The number a response declares as its answer, plus the reason when it declares none.

    The rule is the one the tasks state and the system prompt asks for: **the final non-empty
    line is the declaration, and it carries exactly one number.**  So ``150``, ``150 minutes``,
    ``**36 litres**`` and ``3.14.`` declare an answer, while ``"So the tank contains 36 litres
    after 12 minutes."`` does not - that line asserts two quantities, and a verifier that picks
    one of them is choosing, not measuring.

    The rule this replaces was "the last number anywhere in the response", and it was wrong in
    both directions, measured rather than imagined:

    * a **false negative**: ``"There are 150 minutes in 2.5 hours."`` is the right answer in one
      line, and the last number in it is ``2.5`` - so a correct attempt scored zero, on every
      sample, deterministically;
    * **false variance**: four samples of a tank problem all computed 36 correctly, and exactly
      the two that ended ``"contains 36 litres after 12 minutes"`` scored zero because ``12``
      came last.  A reward that flips on where a sentence stops is a reward nobody can defend,
      and training on it teaches trailing digits rather than arithmetic.

    Units are still deliberately ignored **when the answer is declared**: a final line of ``150
    minutes`` is the answer ``150``, and ``150 kilometres`` is too.  A task where the unit carries
    meaning wants ``ExactMatchVerifier``, which compares text and refuses both.

    Returns ``(None, reason)`` rather than raising: a response that declares nothing is a failed
    attempt, not a defect in the harness, and the two must not look alike in a report.
    """
    line = final_answer(response)
    if not line:
        return None, "the response is empty"
    found = numbers_in(line)
    if not found:
        if not numbers_in(response):
            return None, "the response contains no number"
        return None, f"the final line states no number ({line!r})"
    if len(found) > 1:
        seen = ", ".join(f"{value:g}" for value in found)
        return None, f"the final line states {len(found)} numbers ({seen}): {line!r}"
    return found[0], ""


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
    """Scores a *declared* number within a tolerance.  Tasks carry ``answer`` and ``tolerance``.

    Two questions, in this order, and the order is the whole design:

    1. **Did the response declare an answer in the form the task asked for?**  The final
       non-empty line must carry exactly one number (``declared_number``).  If it does not,
       the attempt scores ``0.0`` - not because the arithmetic was wrong, but because there is
       no answer to judge, and the detail says exactly that.
    2. **Is that number within the tolerance?**  ``abs(declared - expected) <= tolerance``.

    The first question is the format gate, and it exists because of a measurement rather than a
    preference: with the looser rule ("the last number anywhere"), a correct one-line answer of
    ``"There are 150 minutes in 2.5 hours."`` scored zero on every sample, and four identical
    correct tank answers split two passes and two failures depending on whether the sentence
    ended at ``"...36 litres"`` or ``"...36 litres after 12 minutes"``.  A reward that turns on
    where a sentence stops cannot be defended, and a group whose samples disagree for that
    reason is not a training signal - it is noise with a confidence interval.

    Because the gate is a *condition* and not a bonus, the reward stays in ``[0, 1]``: an attempt
    that formats without computing earns nothing, and the value never enters the reward when the
    form is wrong.  Both facts are readable in the detail, which always begins ``form=ok`` or
    ``form=no`` (``FORM_PREFIX``), so a flat group can be attributed: ``form=no`` throughout means
    the model will not answer in the asked-for form, ``form=ok`` with a wrong value means it
    cannot do the arithmetic.  Those are different problems with different fixes, and before this
    rule they produced the same reward curve.

    The tolerance is what makes some tasks scoreable at all: ``1024`` and ``1024.0`` are the same
    answer to a mathematician and different strings to ``ExactMatchVerifier``, and a model that
    writes ``3.14159`` for "pi to two decimal places" is right.  One caveat belongs in the open:
    both numbers are binary floats, so a value that *looks* exactly one tolerance away can land on
    either side of it.  Measured, not assumed: ``abs(3.15 - 3.14)`` is ``0.009999999999999787``,
    which is inside a tolerance of ``0.01``.  A tolerance written to one significant figure is a
    statement about the answer, not about the last binary digit - do not put a task's boundary on
    a knife edge and expect the tie to resolve your way.

    What it will not do: understand units, understand algebra or equations, or read an answer out
    of a sentence that asserts several quantities.  When a task needs that, it needs a different
    verifier, not a cleverer regex.
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

        declared, reason = declared_number(response)
        if declared is None:
            detail = (
                f"{FORM_PREFIX}no; expected {expected:g} +/- {tolerance:g}, but {reason} - "
                "the answer must be declared on its own last line, as the task asks"
            )
            # Name the near miss when the arithmetic was actually right: a reader deciding
            # whether to fix the prompt or the task set needs to know which half failed.
            seen = numbers_in(final_answer(response))
            if any(abs(value - expected) <= tolerance for value in seen):
                detail += "; the expected value is stated there, so the arithmetic looks right"
            return Outcome(passed=False, score=0.0, detail=detail)

        delta = abs(declared - expected)
        matched = delta <= tolerance
        return Outcome(
            passed=matched,
            score=1.0 if matched else 0.0,
            detail=(
                f"{FORM_PREFIX}ok; expected {expected:g} +/- {tolerance:g}, "
                f"got {declared:g} (off by {delta:g})"
            ),
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


def _caller_identity() -> str:
    """The calling user as ``uid:gid``, for ``docker run --user``.

    Windows has no ``getuid``; this sandbox targets Linux and macOS, and a machine without one
    falls back to the conventional first user.  All that matters is that container output stays
    deletable by whoever owns the working directory.
    """
    if hasattr(os, "getuid"):
        return f"{os.getuid()}:{os.getgid()}"
    return "1000:1000"


@dataclass(frozen=True)
class DockerSandbox:
    """Runs the tests in a container: no network, a memory cap, and the timeout from outside.

    The book's two non-negotiables - a container and a wall-clock timeout - are both here.  The
    third thing that matters is what happens when it is *missing*: no ``docker`` on PATH raises
    ``SandboxUnavailable`` rather than quietly running a model's code on the host.
    """

    image: str = "python:3.13-alpine"
    memory: str = "512m"
    #: ``uid:gid`` the container runs as.  Empty means "whoever is running the verifier", which
    #: is the only value that leaves the working directory's owner able to clean it up.
    user: str = ""
    name = "docker"

    def argv(self, command: str, workdir: Path, container: str) -> list[str]:
        """The docker command line, kept separate from running it so the flags are testable.

        ``--network=none`` because a unit test has no business on the network, and the
        alternative is generated code with a route out.  ``--memory`` so a runaway allocation
        does not take the host with it.  ``--rm`` plus a name, so a container cannot outlive
        the run that started it.

        ``--user`` is the flag that came from a real run rather than from reasoning, because of
        what the default does: a container runs as root, ``import main`` writes ``__pycache__``
        *as root* into the bind mount, and the host user then cannot delete its own working
        directory.  Measured on the first live run of this path, where that turned a scored
        attempt into a traceback.  A sandbox must not leave files its caller cannot remove.
        """
        return [
            "docker", "run", "--rm",
            "--name", container,
            "--network=none",
            f"--memory={self.memory}",
            "--user", self.user or _caller_identity(),
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
        # ignore_cleanup_errors: cleanup is housekeeping, and housekeeping must never turn a
        # scored attempt into an exception.  Measured, not assumed - the first live run of this
        # path raised PermissionError from this __exit__, because the container had left a
        # root-owned __pycache__ in the work directory, and the whole run died instead of
        # reporting a score.  (``--user`` is what stops it happening; this is what stops it
        # mattering.)
        with tempfile.TemporaryDirectory(prefix="p1-unit-", ignore_cleanup_errors=True) as tmp:
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


#: The constraint kinds a ``solution`` may use.  A short list on purpose: each kind is a
#: statement a reader can check by eye, and an unknown kind is an error rather than a
#: constraint that is quietly ignored.
SOLUTION_KINDS = ("distinct", "equals", "not_equals")

#: A ``name = value`` or ``name: value`` line.  Only lines naming a variable the task declares
#: are read, which is what keeps a sentence like "note: this took a while" out of the answer.
_ASSIGNMENT_LINE = re.compile(r"^\s*([A-Za-z][\w.-]*)\s*(?:=|:)\s*(\S.*?)\s*$")

#: A brace-delimited object, as a *candidate* for JSON rather than as a JSON parser: tried from
#: the end of the response backwards and rejected when it does not parse.  Nested objects are
#: out of scope - an assignment is flat, and a flat object is what the prompt asks for.
_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_assignment(response: str, variables: Mapping[str, Any] | list[str]) -> dict[str, str]:
    """The assignment a response states, as ``{variable: value}``, both sides normalised.

    Two shapes, in this order, because they are the two a model actually produces:

    1. the last brace-delimited object in the response that parses as a flat object;
    2. ``name = value`` / ``name: value`` lines, a later line overriding an earlier one.

    Only declared variables survive: an assignment is judged against the task's variables, and
    a model that mentions something else along the way has not changed its answer.  That filter
    is also what keeps prose out - a line must name a declared variable to be read at all.

    Returns an empty mapping when neither shape is present, so the caller reports "no
    assignment" instead of scoring an empty answer against the constraints and calling the
    missing entries failures.
    """
    declared = {normalise(str(variable)) for variable in variables}
    for candidate in reversed(_JSON_OBJECT.findall(response)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            found = {
                normalise(str(key)): normalise(str(value))
                for key, value in data.items()
                if normalise(str(key)) in declared
            }
            if found:
                return found
    assignment: dict[str, str] = {}
    for line in response.splitlines():
        match = _ASSIGNMENT_LINE.match(line)
        if match and normalise(match.group(1)) in declared:
            assignment[normalise(match.group(1))] = normalise(match.group(2))
    return assignment


def _solution_parts(solution: Mapping[str, Any], task_id: Any) -> tuple[list[str], list[Any]]:
    """``(variables, constraints)`` from a solution, or an error naming what is missing.

    A puzzle nothing can check is not a scoreable task, and scoring it 0.0 would report a
    defect in the task set as a failure of the model's.
    """
    variables = solution.get("variables")
    if not isinstance(variables, (list, tuple)) or not variables:
        raise ValueError(
            f"task {task_id!r} has a solution without 'variables': a logic puzzle has to say "
            "which names the assignment is about"
        )
    constraints = solution.get("constraints")
    if not isinstance(constraints, (list, tuple)) or not constraints:
        raise ValueError(f"task {task_id!r} has a solution without 'constraints'")
    for index, constraint in enumerate(constraints, start=1):
        if not isinstance(constraint, Mapping):
            raise ValueError(f"task {task_id!r} constraint #{index} is not an object")
        kind = constraint.get("kind")
        if kind not in SOLUTION_KINDS:
            raise ValueError(
                f"task {task_id!r} constraint #{index} has kind {kind!r}, which is not one of "
                f"{', '.join(SOLUTION_KINDS)}"
            )
    return [str(variable) for variable in variables], list(constraints)


def constraint_failure(
    constraint: Mapping[str, Any],
    assignment: Mapping[str, str],
    variables: list[str],
) -> str | None:
    """``None`` when the assignment satisfies the constraint, else a sentence saying why not.

    That sentence is the point of this verifier: a near miss has to read as a near miss.  A
    variable the answer never mentions fails whatever constraint names it - an incomplete
    assignment is not vacuously right about the parts it left out.
    """
    kind = constraint.get("kind")
    if kind == "distinct":
        missing = [variable for variable in variables if not assignment.get(normalise(variable))]
        if missing:
            return f"distinct: {' and '.join(missing)} missing from the answer"
        seen: dict[str, str] = {}
        for variable in variables:
            value = assignment[normalise(variable)]
            if value in seen:
                return f"distinct: {seen[value]} and {variable} are both {value!r}"
            seen[value] = variable
        return None
    name = normalise(str(constraint.get("var", "")))
    if not name:
        raise ValueError(f"constraint {kind!r} needs a 'var' naming the variable it is about")
    if "value" not in constraint:
        raise ValueError(f"constraint {kind!r} needs a 'value'")
    expected = normalise(str(constraint["value"]))
    actual = assignment.get(name)
    if actual is None:
        return f"{name} missing from the answer"
    if kind == "equals":
        return None if actual == expected else f"{name} must be {constraint['value']!r}, got {actual!r}"
    return None if actual != expected else f"{name} must not be {constraint['value']!r}, and is"


class LogicPuzzleVerifier:
    """Checks a structured assignment constraint by constraint.

    Tasks carry a ``solution`` with ``variables`` (the names the answer is about) and
    ``constraints`` (the facts it must satisfy), each constraint naming a ``kind`` from
    ``SOLUTION_KINDS``.  The response is parsed into an assignment, every constraint is
    checked, and the score is 1.0 when all of them hold and 0.0 otherwise.

    Why a hard zero rather than the fraction that held: the same reason ``NumericVerifier``
    refuses partial credit for wrong arithmetic - a reward you cannot explain later is worse
    than one you can defend.  What the fraction buys is in the *detail* instead: it names the
    constraints that failed, so a near miss reads as a near miss in the report even though it
    scores zero.
    """

    name = "logic_puzzle"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        """True when the task carries a solution object.  Presence only, like its siblings.

        Whether that solution is *usable* - variables, known constraint kinds - is a different
        question, and ``verify`` answers it loudly rather than this predicate quietly shrinking
        coverage.
        """
        return isinstance(task.get("solution"), Mapping)

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        task_id = task.get("id", "<no id>")
        solution = task["solution"]
        if not isinstance(solution, Mapping):
            raise ValueError(
                f"task {task_id!r} has a solution that is not an object: {solution!r}"
            )
        variables, constraints = _solution_parts(solution, task_id)
        assignment = parse_assignment(response, variables)
        if not assignment:
            return Outcome(
                passed=False,
                score=0.0,
                detail=(
                    "the response contains no assignment: expected a JSON object or 'name = "
                    f"value' lines naming {', '.join(variables)}"
                ),
            )
        failures: list[str] = []
        for index, constraint in enumerate(constraints, start=1):
            reason = constraint_failure(constraint, assignment, variables)
            if reason is not None:
                failures.append(f"{constraint.get('id') or f'#{index}'} ({reason})")
        if not failures:
            return Outcome(
                passed=True,
                score=1.0,
                detail=f"all {len(constraints)} constraints hold",
            )
        shown = "; ".join(failures[:3]) + (" ..." if len(failures) > 3 else "")
        return Outcome(
            passed=False,
            score=0.0,
            detail=(
                f"satisfied {len(constraints) - len(failures)} of {len(constraints)} constraints; "
                f"failed: {shown}"
            ),
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


#: Names whose ``verify`` is expected to raise ``NotImplementedError``.  The check script
#: asserts this, so a stub cannot silently become a zero-scoring verifier.  Empty means the
#: suite is complete: every declared verifier scores what it claims to score.
UNIMPLEMENTED: tuple[str, ...] = ()
