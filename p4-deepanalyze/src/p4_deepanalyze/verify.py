"""Scoring a report: one number, compared with a tolerance, plus the reason when it fails.

The rule the book keeps repeating applies here too: a metric must be defined in code, once.
``is_correct`` is that definition, and every table this project publishes is derived from it.

Two decisions worth naming:

* **The last number in the report is the answer.** A model that explains itself and then
  writes ``The answer is 42`` is doing the right thing; a model whose last number is an
  intermediate value is wrong, and that is the correct verdict for a task that asked for a
  final figure.
* **A non-numeric expectation is compared as text**, case- and whitespace-insensitive, so a
  task can ask for a column name instead of a count.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> float | None:
    matches = _NUMBER.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def is_correct(report: str | None, expected: str, tolerance: float = 1e-6) -> bool:
    if report is None:
        return False
    want = last_number(expected)
    if want is None:
        return " ".join((report or "").split()).strip().lower() == " ".join(expected.split()).strip().lower()
    got = last_number(report)
    if got is None:
        return False
    return abs(got - want) <= tolerance * max(1.0, abs(want))


def why(report: str | None, expected: str, tolerance: float = 1e-6) -> str:
    """A one-line explanation, for the row a reader will look at when the table is surprising."""
    if report is None:
        return "no report"
    want = last_number(expected)
    got = last_number(report)
    if want is None:
        return "match" if is_correct(report, expected, tolerance) else f"text differs from {expected!r}"
    if got is None:
        return "no number in the report"
    return f"reported {got:g}, expected {want:g}" if abs(got - want) > tolerance * max(1.0, abs(want)) else "match"
