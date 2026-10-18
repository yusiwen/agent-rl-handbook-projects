"""The prompt a policy sees, and the one line it is asked to answer with.

The shape matters more than the wording, for a reason this project measured in Chapter 7: a
format instruction that names the exact mechanical slot ("exactly one line, in this form:
ACTION: <action>") moved compliance from 26% to 88% on the same samples, while prohibitions
("no units, no LaTeX") made the model delete its own work. So the action line is a slot to fill,
not a style to adopt, and the menu is printed verbatim next to it.

What the prompt deliberately does not contain: the answer, the number of turns left, or any hint
about which action is correct. The observation is the observation, the menu is the menu.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .env import Task

#: The one prefix the parser looks for. Exported so the parser and the prompt cannot disagree.
ACTION_PREFIX = "ACTION:"

SYSTEM_PROMPT = (
    "You control a small sandbox by choosing exactly one action per turn. "
    "Copy one action from the list you are given, character for character. "
    "Do not explain, do not add a second action, and do not invent actions that are not listed."
)


def render(
    task: Task,
    observation: str,
    menu: Sequence[str],
    history: Iterable[tuple[str, str, str]] = (),
) -> list[dict[str, str]]:
    """The chat messages for one turn.

    ``history`` is ``(action, signal, observation)`` per completed turn, which is what the
    environment already records; passing it in rather than reading the attempt keeps this function
    pure and testable.
    """
    lines = [f"Task: {task.prompt}", f"Observation: {observation}"]
    turns = list(history)
    if turns:
        lines.append("Turns so far:")
        for action, signal, observed in turns:
            lines.append(f"- {action} -> {signal}: {observed}")
    lines.append("Available actions (copy one exactly):")
    lines.extend(f"- {action}" for action in menu)
    lines.append(f"Reply with exactly one line, in this form: {ACTION_PREFIX} <action>")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
