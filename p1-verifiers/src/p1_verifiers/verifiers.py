"""P1's verifier suite: one implemented, three that are yours.

The book's Chapter 5 deliverable is a verifier suite plus a coverage report.  This
module gives you the shape and one worked example (``ExactMatchVerifier``); the other
three are the exercise.  Each stub is honest about its state:

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


def final_answer(response: str) -> str:
    """Extract the candidate answer from a response: the last non-empty line.

    Every verifier owns a parser, and this is the first one.  The policy is deliberately
    blunt - "the final answer is on its own last line", which the system prompt asks for -
    because a clever extractor is a new failure mode you cannot see in the reward curve.
    Two cases here show both directions of that:

    * ``"... reasoning ...\\n\\n150 minutes"`` is extracted as ``150 minutes`` and fails an
      exact-match check against ``150``: correct arithmetic, wrong claim about what the
      answer *is*.
    * ``"x + 7 = 19, so x = 12.\\n\\nx = 12"`` is extracted as ``x = 12`` and also fails:
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
    """YOUR TURN.  Score numbers within a tolerance instead of by text.

    Tasks that want this verifier carry ``answer`` plus ``tolerance``.
    """

    name = "numeric"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        # Honest while unimplemented: see the module docstring.  Flip this to
        # ``return "tolerance" in task`` when verify() below works.
        return False

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        raise NotImplementedError(
            "numeric verification is not implemented yet.  Parse the last number that "
            "appears in the response, compare it with task['answer'] within "
            "task['tolerance'], and return a score of 1.0 or 0.0.  When it works, make "
            "can_verify() return True for tasks that carry a tolerance."
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
UNIMPLEMENTED = ("numeric", "unit_test", "logic_puzzle")
