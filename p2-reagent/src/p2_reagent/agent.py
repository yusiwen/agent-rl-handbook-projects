"""The agent loop: sample an action per turn, record the turns, and return a rollout row.

The one thing this module is careful about is that **turn boundaries are recorded, not
reconstructed**.  `common.mask.turn_boundaries` can rebuild them from the segments, and
`common.checks` verifies the recorded ``turn_count`` against the rebuilt one - which means the
record is a claim a check can falsify.  That is the difference between plumbing and a comment.

The trace this builds, per attempt::

    user   the task prompt                      (context, never trained on)
    tool   the opening observation              (context, never trained on)
    model  turn 1's action                      (trained)
    tool   the environment's answer to it       (context)
    model  turn 2's action                      (trained)
    ...

The mask follows from the roles, and `common.mask.build_mask` is the only place that decides it.
Truncation is recorded, not scored as a quiet zero: an attempt that runs out of turns is a
different failure from one that reports a wrong answer, and Chapter 6's symptom table needs the
difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.mask import build_mask, turn_boundaries
from common.schema import Rollout, Segment

from . import env
from .policy import TabularPolicy

#: The hard stop on turns.  Two more than either family needs, so the cap is a guard rather than
#: a design constraint, and a policy that loops is caught instead of running forever.
DEFAULT_MAX_TURNS = 6


@dataclass
class Episode:
    """One attempt: the row that goes into the run record, plus what the loop needs to learn."""

    rollout: Rollout
    task: env.Task
    solved: bool
    truncated: bool
    turns: list[dict] = field(default_factory=list)
    #: Per turn, the key the policy was conditioned on, the menu it chose from, and its choice.
    signatures: list[str] = field(default_factory=list)
    menus: list[tuple[str, ...]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    #: Distance-to-goal after every turn, produced by the environment (not by a model).
    attempt_progress: list[float] = field(default_factory=list)

    @property
    def turn_scores_available(self) -> bool:
        return len(self.signals) == len(self.actions)


def run_episode(
    policy: TabularPolicy,
    task_id: str,
    *,
    sample_index: int = 0,
    max_turns: int = DEFAULT_MAX_TURNS,
    env_version: str = env.ENV_VERSION,
    code_sha: str = "git:none",
    config_hash: str = "unknown",
    policy_version: str | None = None,
    prompt_version: str = "prompt-v1",
) -> Episode:
    """Run one attempt at ``task_id`` with ``policy``, and return the row plus the turn detail."""
    attempt = env.start(task_id)
    segments: list[Segment] = [
        Segment(role="user", text=attempt.task.prompt, tokens=env.token_count(attempt.task.prompt)),
        Segment(role="tool", text=attempt.observation, tokens=env.token_count(attempt.observation)),
    ]
    episode = Episode(
        rollout=Rollout(  # placeholder terms, attached by the pipeline once the critics have run
            episode_id=f"{task_id}-s{sample_index}",
            prompt_id=task_id,
            sample_index=sample_index,
            group_id=task_id,
            segments=segments,
            reward_terms={},
            env_version=env_version,
            config_hash=config_hash,
            code_sha=code_sha,
        ),
        task=attempt.task,
        solved=False,
        truncated=False,
    )

    for _ in range(max_turns):
        if attempt.finished:
            break
        signature = attempt.signature
        menu = tuple(attempt.menu())
        # A tabular policy chooses from the state label; a served one needs the prompt, so it is
        # offered the attempt and may read the observation and the history from it.  The hook is
        # `choose`, and its absence is the whole difference between the two.
        action = policy.choose(attempt) if hasattr(policy, "choose") else policy.sample(signature, menu)
        result = attempt.step(action)

        episode.signatures.append(signature)
        episode.menus.append(menu)
        episode.actions.append(action)
        episode.signals.append(result.signal)
        episode.turns.append(
            {
                "index": len(episode.actions) - 1,
                "signature": signature,
                "menu": list(menu),
                "action": action,
                "signal": result.signal,
                "observation": result.observation,
                "done": result.done,
            }
        )
        segments.append(Segment(role="model", text=action, tokens=env.token_count(action)))
        segments.append(
            Segment(role="tool", text=result.observation, tokens=env.token_count(result.observation))
        )

    episode.solved = attempt.solved()
    episode.truncated = not attempt.finished
    episode.attempt_progress = list(attempt.progress_trace)

    rollout = episode.rollout
    rollout.segments = segments
    rollout.mask = build_mask(segments)
    rollout.turn_count = len(turn_boundaries(segments))
    rollout.reward_terms = {"rule_outcome": 1.0 if episode.solved else 0.0}
    rollout.extras = {
        "task": attempt.task.task_id,
        "family": attempt.task.family,
        "goal_met": attempt.goal_met() if attempt.finished else False,
        "reported": attempt.answer,
        "expected": attempt.task.expected,
        "truncated": episode.truncated,
        "turns": len(episode.actions),
        "signals": list(episode.signals),
        "policy_version": policy_version or policy.version,
        "prompt_version": prompt_version,
    }
    return episode


def policy_updates(episode: Episode, weights: list[float]) -> list:
    """Pair each turn's action with its weight, ready for ``policy.apply``.

    A turn with no weight has nothing to learn from, and dropping it is the honest move: an
    attempt whose group was flat contributes zeros, and an attempt with more turns than weights is
    a bug worth raising rather than padding.
    """
    from .policy import Update

    if len(weights) != len(episode.actions):
        raise ValueError(
            f"{episode.rollout.episode_id}: {len(weights)} weight(s) for {len(episode.actions)} turn(s)"
        )
    return [
        Update(
            signature=signature,
            menu=menu,
            action=action,
            weight=weight,
        )
        for signature, menu, action, weight in zip(
            episode.signatures, episode.menus, episode.actions, weights
        )
        if weight != 0.0
    ]
