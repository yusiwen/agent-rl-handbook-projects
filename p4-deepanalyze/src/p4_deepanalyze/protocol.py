"""The marker protocol: how a model asks for code to be run, and how it answers.

Chapter 10's third lesson reads DeepAnalyze's turn structure; this module is the smallest
version of it that is still a contract rather than a convention:

* ``<code> ... </code>`` asks the harness to write the program to ``prog.py`` and run it;
* ``<report> ... </report>`` is the final answer, and it ends the episode;
* ``<analyze> ... </analyze>`` may precede either one and is ignored by the harness.

Exactly one of the first two per reply. Zero is a stalled turn; two is an ambiguous one.
Both are protocol errors and both are recorded as such instead of being guessed at - the
difference between "the model cannot follow the format" and "the sandbox broke" is the
difference between a data problem and an infrastructure problem, and a parser that
improvises destroys it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CODE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_REPORT = re.compile(r"<report>(.*?)</report>", re.DOTALL)
_ANALYZE = re.compile(r"<analyze>(.*?)</analyze>", re.DOTALL)
_OPEN_CODE = re.compile(r"<code>", re.IGNORECASE)
_OPEN_REPORT = re.compile(r"<report>", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are a data-analysis agent. A table is available as the file `table.csv` in the working
directory. You answer the user's question by writing Python, running it, and then reporting.

Reply with exactly one of these, and nothing else on that turn:

  <analyze>your reasoning</analyze>
  <code>
  import pandas as pd
  ...
  print(result)
  </code>

or, when you know the answer:

  <report>the final answer</report>

Rules:
- The code you write is run in a fresh sandbox with the table in the current directory.
  You see its stdout and stderr.
- Report the answer itself, not the code and not a description of how you got it.
- If a run fails, read the error and write a different program.
"""


class ProtocolError(ValueError):
    """A reply that cannot be turned into exactly one action."""

    def __init__(self, reason: str, reply: str) -> None:
        super().__init__(f"{reason}: {reply[:160]!r}")
        self.reason = reason
        self.reply = reply


@dataclass(frozen=True)
class Action:
    """One parsed action: either code to run, or a report to score."""

    kind: str  # "code" | "report"
    text: str
    analyzed: str = ""


def parse(reply: str) -> Action:
    """Turn one reply into one action, or refuse it.

    Refusals are deliberately specific: a caller records ``reason`` and a reader can tell a
    model that wrote prose from a model that wrote two actions from a model that wrote an
    opening marker and never closed it.
    """
    text = (reply or "").strip()
    if not text:
        raise ProtocolError("empty reply", reply)

    codes = _CODE.findall(text)
    reports = _REPORT.findall(text)
    if len(codes) > 1 or len(reports) > 1:
        raise ProtocolError("more than one action", reply)
    if codes and reports:
        raise ProtocolError("both code and report", reply)
    if not codes and not reports:
        if _OPEN_CODE.search(text) or _OPEN_REPORT.search(text):
            raise ProtocolError("unclosed marker", reply)
        raise ProtocolError("no marker", reply)

    analyzed = " ".join(part.strip() for part in _ANALYZE.findall(text))
    if codes:
        body = codes[0].strip()
        if not body:
            raise ProtocolError("empty code block", reply)
        return Action(kind="code", text=body, analyzed=analyzed)
    body = reports[0].strip()
    if not body:
        raise ProtocolError("empty report", reply)
    return Action(kind="report", text=body, analyzed=analyzed)


def format_observation(exit_code: int | None, stdout: str, stderr: str, timed_out: bool) -> str:
    """What the model is shown after a run.  A timeout is a different sentence from an error."""
    if timed_out:
        return (
            "<observation>\n"
            "The program was stopped: it ran longer than the sandbox timeout.\n"
            "</observation>"
        )
    parts = [f"<observation exit={exit_code}>"]
    if stdout.strip():
        parts.append("stdout:\n" + stdout.strip())
    if stderr.strip():
        parts.append("stderr:\n" + stderr.strip())
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    parts.append("</observation>")
    return "\n".join(parts)
