"""One tool call per reply, parsed strictly, refused by name.

The format is a single JSON object:

    {"tool": "search", "args": {"query": "q3 north revenue"}}

Nothing else counts.  A fenced block is accepted (models like fences), a sentence around the object
is not, and two objects in one reply are refused rather than merged - the same rule as Project 4's
markers, for the same reason: a parser that guesses turns a format failure into a wrong answer, and
those two need different fixes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

TOOLS = ("search", "open", "calc", "submit")

SYSTEM_PROMPT = """\
You answer questions about a small company by calling tools. Reply with exactly one JSON object and
nothing else, on one line:

  {"tool": "search", "args": {"query": "revenue q3 north"}}
  {"tool": "open", "args": {"doc": "q3-north"}}
  {"tool": "calc", "args": {"expression": "1200 * 4.50"}}
  {"tool": "submit", "args": {"answer": "5400.00"}}

You see the result of each call before your next reply. Use `search` to find documents, `open` to read
one, `calc` for arithmetic, and `submit` when you know the answer. Submit the value the question asks
for and nothing else: a number with two decimals if it is money, a region name if it asks for one.
"""

_OBJECT = re.compile(r"\{.*\}", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class ProtocolError(ValueError):
    def __init__(self, reason: str, reply: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.reply = reply


@dataclass(frozen=True)
class Call:
    tool: str
    args: dict


def parse(reply: str) -> Call:
    text = (reply or "").strip()
    if not text:
        raise ProtocolError("empty reply", text)
    fence = _FENCE.search(text)
    body = fence.group(1).strip() if fence else text

    if fence is None:
        if not body.startswith("{"):
            # Two objects may follow a sentence; only the sentence is the interesting part if the
            # object is not at the start, so both are named differently.
            raise ProtocolError("text around the object" if "{" in body else "no object", text)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            if "Extra data" in exc.msg:
                raise ProtocolError("not one object", text) from exc
            raise ProtocolError(f"bad json: {exc.msg}", text) from exc
    else:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"bad json: {exc.msg}", text) from exc
    if not isinstance(payload, dict):
        raise ProtocolError("not an object", text)
    tool = payload.get("tool")
    if tool not in TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}", text)
    args = payload.get("args")
    if not isinstance(args, dict):
        raise ProtocolError("args is not an object", text)
    return Call(tool=str(tool), args=args)


def first_message(question: str) -> str:
    return f"Question: {question}\n\nAvailable tools: {', '.join(TOOLS)}."


def format_observation(text: str, error: str = "", done: bool = False) -> str:
    if error:
        return f"<observation error>\n{error}\n</observation>"
    tail = "\n(episode over)" if done else ""
    return f"<observation>\n{text}{tail}\n</observation>"
