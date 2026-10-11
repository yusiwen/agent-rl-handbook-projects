"""P1's verifier suite: two implemented, two that are yours.

The book's Chapter 5 deliverable is a verifier suite plus a coverage report.  This
module gives you the shape and two worked examples (``ExactMatchVerifier`` and
``NumericVerifier``); the other two are the exercise.  Each stub is honest about its state:

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
from typing import Any, Mapping

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


class UnitTestVerifier:
    """YOUR TURN.  Run a task's tests in a sandbox and score by the exit status.

    Tasks that want this verifier carry a ``tests`` entry: a command (or a file path)
    that returns 0 when the attempt is correct.

    Two rules from the book apply and are not optional here: run it in a container with
    a timeout, and treat a timeout as a failure rather than as an error - an environment
    that hangs must not look like a model that is slow.
    """

    name = "unit_test"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        return False  # until implemented

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        raise NotImplementedError(
            "unit-test verification is not implemented yet.  Write the candidate code to "
            "a temporary directory, run task['tests'] in a container with a wall-clock "
            "timeout, and score 1.0 on exit status 0, 0.0 otherwise.  Record the timeout "
            "case in the detail so a hung environment is not read as a wrong answer."
        )


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
def default_registry() -> VerifierRegistry:
    return VerifierRegistry(
        [ExactMatchVerifier(), NumericVerifier(), UnitTestVerifier(), LogicPuzzleVerifier()]
    )


#: Names whose ``verify`` is expected to raise ``NotImplementedError``.  The check
#: script asserts this, so a stub cannot silently become a zero-scoring verifier.
UNIMPLEMENTED = ("unit_test", "logic_puzzle")
