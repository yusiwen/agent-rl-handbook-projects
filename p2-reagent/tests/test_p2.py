"""Tests for P2: the environment, the policy, the critics, the fusion, and the loop.

    python3 -m unittest discover -s p2-reagent/tests -t p2-reagent/tests

Everything here is offline and deterministic: no model, no container, no network.  That is the
point of the project's shape - the credit-assignment machinery is arithmetic, and arithmetic can be
tested exactly.

Four tests exist because their absence cost a bug during development, and they are marked so:

* ``test_each_task_can_be_solved_from_its_menu`` - the delete task once had no ``rm tmp.log`` in
  its menu, so it was unsolvable and the loop "failed to learn" a broken environment;
* ``test_progress_follows_the_plan_not_the_distance`` - a distance-to-goal progress let the ledger's
  distractor outscore the correct first move;
* ``test_padding_a_trajectory_does_not_raise_the_turn_term`` - a mean over turns made neutral
  filler beat acting, which is the Echo Trap arriving early;
* ``test_an_outcome_only_rule_has_no_gradient_at_a_cold_start`` - with outcome alone, nothing solves,
  every group is flat and the update is a no-op, so the loop cannot start.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE.parent / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.checks import check_advantage_spread, check_mask, check_terms  # noqa: E402
from common.mask import turn_boundaries  # noqa: E402

from p2_reagent import env, fusion_policy  # noqa: E402
from p2_reagent.agent import DEFAULT_MAX_TURNS, Episode, policy_updates, run_episode  # noqa: E402
from p2_reagent.critics import (  # noqa: E402
    BudgetedCritic,
    CriticParseError,
    FixtureCritic,
    ModelCritic,
    RuleCritic,
    TurnNote,
    Verdict,
    disagreement,
)
from p2_reagent.loop import (  # noqa: E402
    Config,
    RoundResult,
    check_budget,
    check_entropy,
    check_lineage,
    run_loop,
    run_round,
)
from p2_reagent.policy import TabularPolicy, Update  # noqa: E402
from p2_reagent.train import (  # noqa: E402
    TURN_SHAPING,
    group_advantages,
    standardise,
    turn_weights,
)

MENU = ("a", "b", "c", "d")
FIXTURES = HERE.parent / "fixtures"


def walk(task_id: str) -> env.Attempt:
    """An attempt driven to completion by an oracle that maximises progress."""
    attempt = env.start(task_id)
    for _ in range(DEFAULT_MAX_TURNS):
        if attempt.finished:
            break
        best: tuple[float, str] | None = None
        for action in attempt.menu():
            probe = env.start(task_id)
            for taken in [step[0] for step in attempt.history]:
                probe.step(taken)
            result = probe.step(action)
            score = 2.0 if (result.done and probe.solved()) else probe.progress()
            if best is None or score > best[0]:
                best = (score, action)
        assert best is not None
        attempt.step(best[1])
    return attempt


def episode_for(task_id: str, policy: TabularPolicy | None = None) -> Episode:
    policy = policy or TabularPolicy(seed=3)
    return run_episode(policy, task_id)


class EnvTests(unittest.TestCase):
    def test_families_are_registered_with_their_tasks(self) -> None:
        families = env.families()
        self.assertEqual(sorted(families), ["files", "ledger"])
        self.assertEqual(len(env.all_tasks()), 3)

    def test_each_task_can_be_solved_from_its_menu(self) -> None:
        """The regression that mattered: a menu missing the right action makes the task impossible."""
        for task in env.all_tasks():
            attempt = walk(task.task_id)
            self.assertTrue(attempt.solved(), f"{task.task_id} could not be solved: {attempt.history}")

    def test_two_attempts_start_identical(self) -> None:
        first = env.start("files-rename-count")
        second = env.start("files-rename-count")
        self.assertEqual(first.observation, second.observation)
        self.assertEqual(first.menu(), second.menu())
        self.assertEqual(first.progress_trace, second.progress_trace)

    def test_solved_needs_both_the_state_and_the_answer(self) -> None:
        wrong_action = env.start("files-rename-count")
        wrong_action.step("rm draft.txt")
        wrong_action.step("done 1")
        self.assertFalse(wrong_action.goal_met())
        self.assertFalse(wrong_action.solved())

        wrong_number = env.start("files-rename-count")
        wrong_number.step("mv draft.txt final.txt")
        wrong_number.step("done 2")
        self.assertTrue(wrong_number.goal_met())
        self.assertFalse(wrong_number.solved())

    def test_an_action_outside_the_menu_is_an_error_signal(self) -> None:
        attempt = env.start("files-rename-count")
        result = attempt.step("sudo rm -rf /")
        self.assertEqual(result.signal, "error")
        self.assertFalse(result.done)

    def test_progress_follows_the_plan_not_the_distance(self) -> None:
        """The ledger's distractor (add 5) used to look better than the correct first move (add 12)."""
        attempt = env.start("ledger-net-seven")
        attempt.step("add 5")
        self.assertEqual(attempt.progress(), 0.0)
        correct = env.start("ledger-net-seven")
        correct.step("add 12")
        self.assertGreater(correct.progress(), 0.0)

    def test_a_finished_attempt_refuses_more_actions(self) -> None:
        attempt = env.start("ledger-net-seven")
        attempt.step("add 12")
        attempt.step("sub 5")
        attempt.step("done 7")
        with self.assertRaises(RuntimeError):
            attempt.step("ls")

    def test_the_menu_shrinks_after_a_change(self) -> None:
        attempt = env.start("files-rename-count")
        before = attempt.menu()
        attempt.step("mv draft.txt final.txt")
        self.assertNotEqual(before, attempt.menu())
        self.assertNotIn("mv draft.txt final.txt", attempt.menu())


class PolicyTests(unittest.TestCase):
    def test_a_new_row_is_uniform(self) -> None:
        policy = TabularPolicy(seed=0)
        probs = policy.probs("sig", MENU)
        self.assertAlmostEqual(sum(probs), 1.0)
        for probability in probs:
            self.assertAlmostEqual(probability, 0.25)

    def test_a_positive_weight_moves_probability_towards_the_action(self) -> None:
        policy = TabularPolicy(learning_rate=1.0)
        before = policy.probs("sig", MENU)[MENU.index("b")]
        policy.apply([Update(signature="sig", menu=MENU, action="b", weight=1.0)])
        after = policy.probs("sig", MENU)[MENU.index("b")]
        self.assertGreater(after, before)

    def test_a_negative_weight_moves_probability_away(self) -> None:
        policy = TabularPolicy(learning_rate=1.0)
        before = policy.probs("sig", MENU)[MENU.index("b")]
        policy.apply([Update(signature="sig", menu=MENU, action="b", weight=-1.0)])
        after = policy.probs("sig", MENU)[MENU.index("b")]
        self.assertLess(after, before)

    def test_sampling_is_reproducible_from_the_seed(self) -> None:
        first = TabularPolicy(seed=11)
        second = TabularPolicy(seed=11)
        draws_first = [first.sample("sig", MENU) for _ in range(20)]
        draws_second = [second.sample("sig", MENU) for _ in range(20)]
        self.assertEqual(draws_first, draws_second)

    def test_entropy_starts_at_log_of_the_menu_size(self) -> None:
        policy = TabularPolicy(seed=0)
        import math

        self.assertAlmostEqual(policy.entropy("sig", MENU), math.log(len(MENU)))

    def test_entropy_falls_after_a_confident_update(self) -> None:
        policy = TabularPolicy(learning_rate=2.0)
        before = policy.entropy("sig", MENU)
        policy.apply([Update(signature="sig", menu=MENU, action="a", weight=1.0)] * 5)
        self.assertLess(policy.entropy("sig", MENU), before)

    def test_bumping_keeps_the_parameters_and_renames(self) -> None:
        policy = TabularPolicy(seed=0)
        policy.apply([Update(signature="sig", menu=MENU, action="a", weight=1.0)])
        clone = policy.bumped("policy-v9")
        self.assertEqual(clone.version, "policy-v9")
        self.assertEqual(clone.table(), policy.table())

    def test_an_update_outside_its_menu_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Update(signature="sig", menu=MENU, action="z", weight=1.0)

    def test_a_non_finite_weight_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Update(signature="sig", menu=MENU, action="a", weight=float("nan"))


class EpisodeTests(unittest.TestCase):
    def test_the_trace_alternates_and_the_mask_follows_it(self) -> None:
        episode = episode_for("files-rename-count")
        roles = [segment.role for segment in episode.rollout.segments]
        self.assertEqual(roles[0], "user")
        self.assertEqual(roles[1], "tool")
        self.assertEqual(roles[2], "model")
        self.assertEqual(roles[3], "tool")
        self.assertEqual(check_mask([episode.rollout]).messages, [])

    def test_the_recorded_turn_count_matches_the_rebuilt_one(self) -> None:
        episode = episode_for("ledger-net-seven")
        self.assertEqual(episode.rollout.turn_count, len(turn_boundaries(episode.rollout.segments)))
        self.assertEqual(episode.rollout.turn_count, len(episode.actions))

    def test_every_row_carries_its_provenance(self) -> None:
        episode = episode_for("files-delete-count")
        extras = episode.rollout.extras
        for key in ("task", "family", "policy_version", "prompt_version", "turns", "signals"):
            self.assertIn(key, extras)

    def test_truncation_is_recorded_rather_than_scored_quietly(self) -> None:
        """A policy that never closes the episode must be reported as truncated, not scored as 0."""

        class NeverDone(TabularPolicy):
            def sample(self, signature, menu):
                return next((action for action in menu if not action.startswith("done")), menu[0])

        episode = run_episode(NeverDone(seed=0), "files-rename-count", max_turns=2)
        self.assertTrue(episode.truncated)
        self.assertEqual(len(episode.actions), 2)
        self.assertTrue(episode.rollout.extras["truncated"])
        self.assertFalse(episode.solved)

    def test_the_outcome_term_is_the_only_term_the_agent_sets(self) -> None:
        episode = episode_for("files-delete-count")
        self.assertEqual(set(episode.rollout.reward_terms), {"rule_outcome"})

    def test_policy_updates_pair_turns_with_weights(self) -> None:
        episode = episode_for("ledger-net-seven")
        updates = policy_updates(episode, [1.0] * len(episode.actions))
        self.assertEqual(len(updates), len(episode.actions))

    def test_a_weight_count_that_does_not_match_raises(self) -> None:
        episode = episode_for("ledger-net-seven")
        with self.assertRaises(ValueError):
            policy_updates(episode, [1.0])


class CriticTests(unittest.TestCase):
    def test_the_rule_critic_scores_every_turn_once(self) -> None:
        episode = run_episode(TabularPolicy(seed=0), "files-rename-count")
        verdict = RuleCritic().critique(episode)
        self.assertEqual(len(verdict.per_turn), len(episode.actions))
        self.assertEqual([note.index for note in verdict.per_turn], list(range(len(episode.actions))))
        self.assertTrue(0.0 <= verdict.score <= 1.0)

    def test_an_error_turn_scores_zero(self) -> None:
        policy = TabularPolicy(seed=0)
        episode = run_episode(policy, "files-rename-count")
        # Force an error signal by editing the recorded signals: the critic reads them, so this is
        # also a check that it reads the record rather than the world.
        episode.signals[0] = "error"
        verdict = RuleCritic().critique(episode)
        self.assertEqual(verdict.per_turn[0].score, 0.0)

    def test_a_wrong_report_scores_zero_on_its_turn(self) -> None:
        attempt = env.start("files-rename-count")
        attempt.step("mv draft.txt final.txt")
        attempt.step("done 2")
        episode = run_episode(TabularPolicy(seed=0), "files-rename-count")
        episode.signals = [step[1].signal for step in attempt.history]
        episode.attempt_progress = attempt.progress_trace
        episode.solved = attempt.solved()
        verdict = RuleCritic().critique(episode)
        self.assertEqual(verdict.per_turn[-1].score, 0.0)

    def test_the_fixture_critic_uses_its_script(self) -> None:
        critic = FixtureCritic.from_jsonl(str(FIXTURES / "critic.jsonl"))
        episode = episode_for("files-rename-count")
        verdict = critic.critique(episode)
        self.assertEqual(verdict.per_turn[0].score, 0.95)
        self.assertEqual(verdict.source, "fixture")

    def test_the_fixture_critic_falls_back_to_its_rubric(self) -> None:
        critic = FixtureCritic({})
        episode = episode_for("ledger-net-seven")
        verdict = critic.critique(episode)
        self.assertTrue(all(0.0 <= note.score <= 1.0 for note in verdict.per_turn))

    def test_the_model_parser_reads_a_well_formed_reply(self) -> None:
        reply = "TURN 0 SCORE: 0.4\nWHY: partial\nTURN 1 SCORE: 0.9\nWHY: good"
        notes = ModelCritic.parse(reply, 2)
        self.assertEqual([note.score for note in notes], [0.4, 0.9])
        self.assertEqual(notes[1].critique, "good")

    def test_the_model_parser_refuses_a_partial_judgement(self) -> None:
        with self.assertRaises(CriticParseError):
            ModelCritic.parse("TURN 0 SCORE: 0.4\nWHY: partial", 2)

    def test_the_model_parser_refuses_prose_without_scores(self) -> None:
        with self.assertRaises(CriticParseError):
            ModelCritic.parse("This attempt was quite good overall.", 1)

    def test_a_verdict_outside_the_unit_interval_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            TurnNote(index=0, score=1.5, critique="too generous")

    def test_disagreements_are_reported_beyond_the_margin(self) -> None:
        left = Verdict(score=0.1, critique="", trace="", per_turn=[TurnNote(0, 0.1, "no")])
        right = Verdict(score=0.9, critique="", trace="", per_turn=[TurnNote(0, 0.9, "yes")])
        self.assertEqual(len(disagreement(left, right, margin=0.25)), 1)
        self.assertEqual(disagreement(left, right, margin=0.95), [])

    def test_the_budget_skips_the_expensive_critic_when_the_rule_is_confident(self) -> None:
        attempt = walk("files-rename-count")
        episode = episode_for("files-rename-count")
        episode.signals = [step[1].signal for step in attempt.history]
        episode.attempt_progress = attempt.progress_trace
        episode.solved = attempt.solved()
        critic = BudgetedCritic(RuleCritic(), FixtureCritic.from_jsonl(str(FIXTURES / "critic.jsonl")))
        verdict = critic.critique(episode)
        self.assertIn("skipped", verdict.gate)
        self.assertEqual(verdict.source, "rule")

    def test_the_budget_runs_the_expensive_critic_inside_the_band(self) -> None:
        policy = TabularPolicy(seed=0)
        episode = run_episode(policy, "files-rename-count")
        critic = BudgetedCritic(RuleCritic(), FixtureCritic.from_jsonl(str(FIXTURES / "critic.jsonl")), low=0.0, high=1.0)
        verdict = critic.critique(episode)
        self.assertIn("ran:", verdict.gate)
        self.assertEqual(verdict.source, "fixture")

    def test_the_budget_records_a_decision_for_every_attempt(self) -> None:
        policy = TabularPolicy(seed=0)
        critic = BudgetedCritic(RuleCritic(), FixtureCritic({}), low=0.0, high=1.0)
        for task in env.all_tasks():
            verdict = critic.critique(run_episode(policy, task.task_id))
            self.assertTrue(verdict.gate)


class FusionTests(unittest.TestCase):
    def test_the_rule_only_stage_has_no_critic_term(self) -> None:
        with self.assertRaises(ValueError):
            fusion_policy.terms_for(fusion_policy.RULE_ONLY, rule_outcome=1.0, rule_turn=0.5, critic=0.9)

    def test_the_full_stage_needs_a_critic_term(self) -> None:
        with self.assertRaises(ValueError):
            fusion_policy.terms_for(fusion_policy.FULL, rule_outcome=1.0, rule_turn=0.5, critic=None)

    def test_the_total_is_the_sum_of_the_contributions(self) -> None:
        terms = fusion_policy.terms_for(fusion_policy.FULL, rule_outcome=1.0, rule_turn=0.8, critic=0.6)
        fusion = fusion_policy.fuse_terms(terms, fusion_policy.FULL)
        self.assertAlmostEqual(fusion.total, sum(fusion.contributions.values()))

    def test_a_failed_outcome_with_a_good_process_is_conflicted_and_damped(self) -> None:
        terms = fusion_policy.terms_for(fusion_policy.FULL, rule_outcome=0.0, rule_turn=0.5, critic=0.9)
        fusion = fusion_policy.fuse_terms(terms, fusion_policy.FULL)
        self.assertTrue(any("halved" in conflict for conflict in fusion.conflicts))
        self.assertAlmostEqual(
            fusion.contributions["critic"], fusion_policy.WEIGHTS[fusion_policy.FULL]["critic"] * 0.9 * 0.5
        )

    def test_a_good_outcome_keeps_the_critic_contribution_whole(self) -> None:
        terms = fusion_policy.terms_for(fusion_policy.FULL, rule_outcome=1.0, rule_turn=0.9, critic=0.9)
        fusion = fusion_policy.fuse_terms(terms, fusion_policy.FULL)
        self.assertFalse(any("halved" in conflict for conflict in fusion.conflicts))

    def test_the_rule_only_weights_keep_the_turn_term(self) -> None:
        """The measured reason: outcome alone has no gradient at a cold start."""
        self.assertGreater(fusion_policy.WEIGHTS[fusion_policy.RULE_ONLY]["rule_turn"], 0.0)


class TrainTests(unittest.TestCase):
    def test_advantages_are_centred_and_scaled(self) -> None:
        """Mean-centred and divided by the population spread, so the pair (1, 3) lands at ±1.2247.

        The expected values are written out rather than described, because the whole point of this
        function is that its arithmetic is exact: 2 is one *spread* from the mean, the spread of
        (1, 2, 3) is sqrt(2/3), and 1 / sqrt(2/3) is 1.2247.
        """
        advantages = group_advantages([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(advantages), 0.0, places=9)
        self.assertAlmostEqual(advantages[0], -(1 / (2 / 3) ** 0.5), places=6)
        self.assertAlmostEqual(advantages[1], 0.0, places=9)
        self.assertAlmostEqual(advantages[2], 1 / (2 / 3) ** 0.5, places=6)

    def test_a_flat_group_has_no_advantage(self) -> None:
        self.assertEqual(group_advantages([0.4, 0.4, 0.4]), [0.0, 0.0, 0.0])

    def test_a_group_of_one_has_no_relative_information(self) -> None:
        self.assertEqual(group_advantages([0.7]), [0.0])

    def test_standardise_is_zero_for_a_constant_trajectory(self) -> None:
        self.assertEqual(standardise([0.5, 0.5]), [0.0, 0.0])

    def test_trajectory_mode_gives_every_turn_the_same_weight(self) -> None:
        episode = episode_for("ledger-net-seven")
        weights = turn_weights(episode, 0.7, mode="trajectory")
        self.assertEqual(weights, [0.7] * len(episode.actions))

    def test_turn_mode_redistributes_within_the_trajectory(self) -> None:
        episode = episode_for("ledger-net-seven")
        scores = [0.1, 0.9] + [0.5] * (len(episode.actions) - 2)
        weights = turn_weights(episode, 0.0, mode="turn", turn_scores=scores[: len(episode.actions)])
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(sum(weights), 0.0, places=9)

    def test_turn_mode_needs_scores(self) -> None:
        episode = episode_for("ledger-net-seven")
        with self.assertRaises(ValueError):
            turn_weights(episode, 0.0, mode="turn")

    def test_the_shaping_coefficient_is_written_down(self) -> None:
        self.assertTrue(0.0 < TURN_SHAPING <= 1.0)


class LoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_round_produces_one_row_per_attempt_with_the_shared_checks(self) -> None:
        config = Config(rounds=1, samples=4)
        policy = TabularPolicy(seed=5)
        result = run_round(policy, RuleCritic(), config, 0)
        self.assertEqual(len(result.episodes), config.samples * len(config.tasks))
        self.assertEqual(check_mask(result.rollouts).messages, [])
        self.assertTrue(any("terms seen" in message for message in check_terms(result.rollouts).messages))
        self.assertEqual(result.policy_version, policy.version)

    def test_the_row_terms_are_contributions_so_reward_is_the_fused_total(self) -> None:
        config = Config(rounds=1, samples=4)
        result = run_round(TabularPolicy(seed=5), RuleCritic(), config, 0)
        for episode in result.episodes:
            self.assertAlmostEqual(episode.rollout.reward(), result.totals[episode.rollout.episode_id], places=6)

    def test_the_loop_writes_a_run_record(self) -> None:
        config = Config(rounds=2, samples=3)
        run_loop(config, runs_dir=self.root)
        dirs = list(Path(self.root).iterdir())
        self.assertEqual(len(dirs), 1)
        self.assertTrue((dirs[0] / "header.json").exists())
        self.assertTrue((dirs[0] / "rounds.jsonl").exists())
        self.assertTrue((dirs[0] / "round-000.jsonl").exists())

    def test_the_same_sample_ids_appear_in_every_round(self) -> None:
        results = run_loop(Config(rounds=2, samples=3), runs_dir=self.root)
        first = {episode.rollout.episode_id for episode in results[0].episodes}
        second = {episode.rollout.episode_id for episode in results[1].episodes}
        self.assertEqual(first, second)

    def test_lineage_passes_when_each_round_carries_its_own_version(self) -> None:
        results = run_loop(Config(rounds=2, samples=3), runs_dir=self.root)
        self.assertTrue(check_lineage(results).ok)

    def test_lineage_fails_when_a_row_carries_a_stale_version(self) -> None:
        policy = TabularPolicy(seed=0)
        episode = run_episode(policy, "files-rename-count", policy_version="policy-v0")
        stale = RoundResult(
            index=1, policy_version="policy-v1", table_digest="sha256:changed", episodes=[episode]
        )
        first = RoundResult(index=0, policy_version="policy-v0", table_digest="sha256:first", episodes=[episode])
        result = check_lineage([first, stale])
        self.assertFalse(result.ok)
        self.assertTrue(any("landed in round 1" in message for message in result.messages))

    def test_lineage_fails_when_the_parameters_did_not_change(self) -> None:
        first = RoundResult(index=0, policy_version="policy-v0", table_digest="sha256:same")
        second = RoundResult(index=1, policy_version="policy-v1", table_digest="sha256:same")
        self.assertFalse(check_lineage([first, second]).ok)

    def test_entropy_passes_without_a_cliff(self) -> None:
        first = RoundResult(index=0, policy_version="v0", table_digest="a", entropy_before=1.9, entropy_after=1.6)
        second = RoundResult(index=1, policy_version="v1", table_digest="b", entropy_before=1.5, entropy_after=1.4)
        self.assertTrue(check_entropy([first, second]).ok)

    def test_entropy_fails_on_a_cliff(self) -> None:
        first = RoundResult(index=0, policy_version="v0", table_digest="a", entropy_before=1.9, entropy_after=1.6)
        second = RoundResult(index=1, policy_version="v1", table_digest="b", entropy_before=0.4, entropy_after=0.3)
        result = check_entropy([first, second])
        self.assertFalse(result.ok)
        self.assertTrue(any("25%" in message for message in result.messages))

    def test_the_budget_check_counts_what_the_logs_said(self) -> None:
        first = RoundResult(index=0, policy_version="v0", table_digest="a")
        first.gates = {"x-s0": "expensive critic ran: rule score 0.60 is inside [0.5, 0.9]"}
        second = RoundResult(index=1, policy_version="v1", table_digest="b")
        second.gates = {"x-s0": "expensive critic skipped: rule score 0.10 is outside [0.5, 0.9]"}
        result = check_budget([first, second], fusion_policy.FULL)
        self.assertTrue(result.ok)
        self.assertTrue(any("ran 1 time(s), skipped 1 time(s)" in message for message in result.messages))

    def test_a_cold_policy_is_learnable_only_because_of_the_turn_term(self) -> None:
        """Chapter 6's cold start, measured: outcome alone would leave nothing to learn from.

        With outcome alone, a policy that solves nothing earns zero on every attempt, so every
        group is flat, every advantage is zero and the update is a no-op.  The assertion is not
        "the policy is bad" - it is that the *turn* term is what supplies a gradient at all, which
        is why the rule-only stage weights it.
        """
        config = Config(rounds=1, samples=8, seed=5)
        result = run_round(TabularPolicy(seed=5), RuleCritic(), config, 0)
        outcomes = [episode.rollout.extras["terms_raw"]["rule_outcome"] for episode in result.episodes]
        turns = [episode.rollout.extras["terms_raw"]["rule_turn"] for episode in result.episodes]
        if all(value == 0.0 for value in outcomes):
            self.assertTrue(any(value > 0.0 for value in turns), "no signal at all: the loop cannot start")
            self.assertGreater(result.updates_applied, 0, "a cold start still needs a gradient")

    def test_padding_a_trajectory_does_not_raise_the_turn_term(self) -> None:
        """The Echo Trap's entry ticket: a mean over turns let neutral filler beat acting."""
        from p2_reagent.loop import progress_reached

        padded = episode_for("files-rename-count")
        padded.attempt_progress = [0.25, 0.25, 0.25, 0.25]
        reached = episode_for("files-rename-count")
        reached.attempt_progress = [0.25, 0.6]
        self.assertLess(progress_reached(padded), progress_reached(reached))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
