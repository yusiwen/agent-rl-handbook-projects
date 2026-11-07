"""The episode: a question, up to N turns of code-and-observe, and a report.

The shape is the one Chapter 10 describes and Chapter 6 measured:

    user question
      -> model: <code>        (a model segment)
      -> sandbox: stdout      (a tool segment, produced by an executor, not by the model)
      -> model: <code>        (another model segment)
      -> sandbox: ...
      -> model: <report>      (the answer, scored)

What the episode records is what a reader needs to re-derive the verdict without trusting it:
every reply, every observation, every refusal with its reason, the number of runs, and
whether any run timed out. ``solved`` is only ever ``True`` when a report was produced *and*
matched; an episode that never reported is unsolved, not "partial".
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from . import protocol, verify
from .protocol import ProtocolError
from .sandbox import Observation
from .tasks import Task


@dataclass
class Turn:
    index: int
    reply: str
    kind: str  # "code" | "report" | "protocol-error"
    code: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    observation: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Episode:
    task_id: str
    turns: list[Turn] = field(default_factory=list)
    report: str | None = None
    solved: bool = False
    reason: str = ""
    protocol_errors: int = 0
    sandbox_errors: int = 0
    timeouts: int = 0
    lost_turns: int = 0  # runs whose output was empty or not a scalar
    model: str = ""
    code_sha: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["turns"] = [turn.to_dict() for turn in self.turns]
        payload["seconds"] = round(self.seconds, 2)
        return payload


def first_message(task: Task) -> str:
    return (
        f"Question: {task.question}\n\n"
        f"The table is `table.csv`. Columns: {', '.join(task.columns())}.\n"
        f"First rows:\n{task.preview()}"
    )


def run_episode(
    task: Task,
    client,
    sandbox,
    max_turns: int = 4,
    model: str = "",
    code_sha: str = "",
) -> Episode:
    """Drive one task to a report, a refusal, or the turn limit."""
    episode = Episode(task_id=task.id, model=model, code_sha=code_sha)
    messages = [
        {"role": "system", "content": protocol.SYSTEM_PROMPT},
        {"role": "user", "content": first_message(task)},
    ]
    started = time.monotonic()

    for index in range(max_turns):
        reply = client.ask(messages)
        try:
            action = protocol.parse(reply)
        except ProtocolError as exc:
            episode.turns.append(Turn(index=index, reply=reply, kind="protocol-error", error=exc.reason))
            episode.protocol_errors += 1
            episode.reason = f"protocol error: {exc.reason}"
            break

        if action.kind == "report":
            episode.turns.append(Turn(index=index, reply=reply, kind="report"))
            episode.report = action.text
            episode.solved = verify.is_correct(action.text, task.expected, task.tolerance)
            episode.reason = verify.why(action.text, task.expected, task.tolerance)
            break

        observation: Observation = sandbox.run(action.text, task.files())
        warnings = protocol.observation_warnings(
            observation.exit_code, observation.stdout, observation.stderr, observation.timed_out
        )
        episode.turns.append(
            Turn(
                index=index, reply=reply, kind="code", code=action.text,
                warnings=warnings, observation=observation.to_dict(),
            )
        )
        if observation.timed_out:
            episode.timeouts += 1
        elif observation.exit_code != 0:
            episode.sandbox_errors += 1
        if "no_output" in warnings or "multiline_output" in warnings:
            episode.lost_turns += 1
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": protocol.format_observation(
                    observation.exit_code, observation.stdout, observation.stderr,
                    observation.timed_out, warnings,
                ),
            }
        )
    else:
        episode.reason = episode.reason or f"no report within {max_turns} turns"

    episode.seconds = time.monotonic() - started
    return episode
