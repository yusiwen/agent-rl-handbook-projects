"""Reading one action out of what the model said, and saying why it failed.

A parser that is lenient hides the thing this project measures. If it strips markdown, guesses the
closest menu entry and accepts a sentence that happens to contain `rm tmp.log`, the format
compliance number stops meaning anything, and a policy that cannot follow a one-line instruction
looks trained. So the parser normalises only whitespace and decoration, requires the action to be
on the menu character for character, and returns a reason for every rejection.

Reasons, and what each one means about the policy:

``ok``             the line was there and the action was on the menu
``no_action_line`` the model answered in prose; the instruction was not followed at all
``ambiguous``      two different actions were offered in one reply; the environment needs one
``not_in_menu``    an action was named, but not one that exists; an invented tool
``empty``          the reply was empty

The reward consequences are the environment's business, not the parser's: an unparsed reply is
passed through as an action the environment will reject, so the trace records what was said and
the policy pays for it the same way it pays for any other wrong action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .prompt import ACTION_PREFIX

_DECORATION = "*`\"'\u201c\u201d\u2018\u2019"


@dataclass(frozen=True)
class ParsedAction:
    """What the reply contained, and the action if one could be read from it."""

    action: str | None
    reason: str
    raw: str

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


def _strip_decoration(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed:
        changed = False
        for mark in _DECORATION:
            if stripped.startswith(mark):
                stripped = stripped[1:].lstrip()
                changed = True
            if stripped.endswith(mark):
                stripped = stripped[:-1].rstrip()
                changed = True
    return stripped


def _payload(line: str) -> str | None:
    """The text after ``ACTION:`` on this line, or ``None`` if this is not an action line."""
    candidate = _strip_decoration(line)
    lowered = candidate.lower()
    prefix = ACTION_PREFIX.lower()
    if not lowered.startswith(prefix):
        return None
    return _strip_decoration(candidate[len(ACTION_PREFIX) :]).rstrip(".").strip()


def _normalise(action: str) -> str:
    return " ".join(action.split())


def parse_action(text: str, menu: Sequence[str]) -> ParsedAction:
    """Read one action from a reply. Never raises: a bad reply is a result, not an exception."""
    payloads = []
    for line in (text or "").splitlines():
        value = _payload(line)
        if value is not None:
            payloads.append(value)
    if not payloads:
        return ParsedAction(None, "no_action_line", _normalise(text or ""))
    distinct = list(dict.fromkeys(_normalise(payload) for payload in payloads))
    if len(distinct) > 1:
        return ParsedAction(None, "ambiguous", " | ".join(distinct))
    action = distinct[0]
    if not action:
        return ParsedAction(None, "empty", action)
    for legal in menu:
        if action == _normalise(legal):
            return ParsedAction(legal, "ok", action)
    return ParsedAction(None, "not_in_menu", action)
