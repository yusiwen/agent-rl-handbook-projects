"""P2 - process feedback and turn-level credit.

The package layout mirrors the chapter's four blocks:

* ``env`` - two deterministic task families, and the per-turn signals they give away for free;
* ``agent`` - the loop that samples actions and records turn boundaries;
* ``critics`` - the rule critic, the scripted stand-in, the served model, and the budget;
* ``fusion_policy`` - rule and critic in, one scalar out, with the weights written down;
* ``train`` - group-relative advantages and the two ways a turn gets a weight;
* ``loop`` - the four stages, and the checks that decide whether the loop really closed.
"""

from .agent import DEFAULT_MAX_TURNS, Episode, policy_updates, run_episode
from .critics import (
    BudgetedCritic,
    CriticParseError,
    FixtureCritic,
    ModelCritic,
    RuleCritic,
    TurnNote,
    Verdict,
    disagreement,
)
from .env import ENV_VERSION, Family, Task, all_tasks, families, start
from .fusion_policy import CONFLICT_POLICY, FULL, RULE_ONLY, WEIGHTS, fuse_terms, policy_for, terms_for
from .loop import Config, RoundResult, check_entropy, check_lineage, run_loop, run_round
from .policy import TabularPolicy, Update
from .train import CREDIT_MODES, TURN_SHAPING, group_advantages, turn_weights

__all__ = [
    "BudgetedCritic",
    "CONFLICT_POLICY",
    "CREDIT_MODES",
    "Config",
    "CriticParseError",
    "DEFAULT_MAX_TURNS",
    "ENV_VERSION",
    "Episode",
    "FULL",
    "Family",
    "FixtureCritic",
    "ModelCritic",
    "RULE_ONLY",
    "RoundResult",
    "RuleCritic",
    "TabularPolicy",
    "Task",
    "TURN_SHAPING",
    "TurnNote",
    "Update",
    "Verdict",
    "WEIGHTS",
    "all_tasks",
    "check_entropy",
    "check_lineage",
    "disagreement",
    "families",
    "fuse_terms",
    "group_advantages",
    "policy_for",
    "policy_updates",
    "run_episode",
    "run_loop",
    "run_round",
    "start",
    "terms_for",
    "turn_weights",
]
