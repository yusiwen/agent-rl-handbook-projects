# P2 design: where the signal comes from, and what it costs

This is the design document [Chapter 6](../../agent-rl) asks for in its deliverables: the task as a
Markov decision process, the signal level chosen for each kind of decision, and the reason the level
below was not chosen. Claims are labelled, in the book's convention: **[measured]** points at a run
record in `runs/`, **[textbook]** is standard fact, **[judgement]** is my reasoning and could be
wrong.

Chapter 6 calls this "upgraded MDP" work; the point is not the formalism but that every choice below
has a price written next to it.

## 1. The task as an MDP

- **State.** The family's state (a temporary directory for `files`, a running total for `ledger`),
  plus the turn index and the action menu the task allows at that turn. The observation is the menu
  and a short instruction; the same prompt shape reaches a served model and the local one.
- **Action.** Exactly one line, `ACTION: <action>`, chosen from the menu printed in the prompt. The
  menu is finite (2 to 9 entries) and generated per task, which is what makes "the model chose an
  action that was not offered" a detectable event rather than a silent one (`actions.py` returns
  `not_in_menu` as one of five parse outcomes).
- **Transition.** Deterministic. The same action on the same state gives the same next state, which
  is deliberate: with a noisy environment, a reward difference between two attempts is not
  attributable to the policy alone.
- **Terminal.** The task's goal is reached, or the guard cap is hit (`DEFAULT_MAX_TURNS = 6` - two
  more than any family needs; `files` takes one or two actions plus the report, `ledger` two or three
  plus the report). Hitting the cap is a failure even if the state happens to look right, because
  "eventually right after enough retries" is a different skill from "right in the budget".
- **Episode, and group.** One episode is one attempt. A *group* is N samples of the *same task* at
  temperature 1.0; advantages are computed inside a group and never across tasks. This is the GRPO
  shape, and in this project the boundary is enforced in code (`run_episode` per task, per-task
  `summarise_groups`) after an early version averaged across the whole round.

## 2. Decision types, and the signal level chosen for each

The book's map (Chapter 6, Lesson 2) has four levels: token, step, turn, trajectory. "Why not the
level below" is answered per decision type, because the three types below are not the same problem.

### 2a. Which action to take on this turn

**Chosen: turn.** The turn score is the progress delta the action produced (see the three bugs in
`README.md` - an activity-based score taught the policy to be busy, and a mean over turns let it
pad [measured]). The signal is derived from the environment's response, which is the one thing a
turn has that a step does not.

**Why not step.** In these families a step *is* a turn: one action per turn, no sub-steps. A
step-level scheme would therefore buy nothing in resolution and would cost either per-step labels or
resampling from every step, and the label source's bias would then become the reward's bias
[textbook: Chapter 6, Lesson 2]. The turn already carries strictly more information - the action
*and* what the environment did about it.

**Why not token.** A token-level value needs a trained critic, and the criticism here would be of a
half-written sentence: the model emits a fixed prefix, then one action token, then stops. There is no
per-token preference that the task can grade. The measured basis for not wanting a critic here: at
the first supervised token of a bare arithmetic sample, the model's mass sat on prose continuation
(`Let` 30.25 nats, `To` 28.25) rather than on the answer (`4` 18.375) [measured: `env/README.md`,
P0b `--inspect`]. A critic asked to value that state would be scoring the model's formatting habits,
not the task.

### 2b. When to stop and report

**Chosen: turn (the terminating turn), with the trajectory outcome as the truth.** Reporting is the
decision that ends the episode, so it gets the credit that belongs to it: a correct report on a
reached goal is the episode's success, and an early report on an unreached goal is the failure mode
the rule critic exists to catch.

**Why not step.** "Done" is a boundary event, not a step. A per-step score over the tokens of one
`done` line has nothing to grade - there is no partial correctness in the word.

**Why not trajectory alone** - this is the measured one. With a trajectory-level advantage, every
turn of an attempt carries the same sign, and at a 5-sample budget the group is frequently all-right
or all-wrong, so the advantage is exactly zero. Eight rounds of that produced 19, 17, 20, 18, 16, 19,
19, 20 solved out of 25: an eight-point spread with no trend, and two rounds with *zero* weighted
turns because every group was flat [measured: `env/README.md`, step B]. The failure is not the
algorithm; it is the signal's resolution colliding with the sample budget.

### 2c. Which value to report (the ledger family)

**Chosen: trajectory (the checker).** The reported number is only correct relative to the final
state, so the checker is the only thing that can grade it.

**Why not turn.** The first attempt at this was a per-turn distance-to-goal score, and it rewarded the
distractor: in the ledger task `add 5` moved the running total *nearer* to 7 than the correct
`add 12` did, so the shaping term paid for the wrong action [measured: `README.md`, bug 3]. The fix
was to make progress follow the required sequence rather than the metric distance - which is a way of
saying the turn score must be derived from the final checker, not from a proxy that happens to be
available every turn.

## 3. Which of Chapter 6's four difficulties this task actually has

Naming the difficulty is half of choosing the level, so here is the honest inventory rather than a
list of all four.

- **Long trajectories: no.** One to three actions plus a report, against a guard cap of six. The
  density problem the book describes for 20,000-token traces does not apply here.
- **Sparse feedback: yes, and it is the dominant one.** Success is binary and the useful case - a
  failing attempt where the *right* turn is wrong - is rare. Two of the eight rounds in step B had
  no gradient at all for this reason [measured]. The response was calibration, not a smarter
  advantage: the same loop run on a task set chosen for mid-difficulty variance moved from 14-19% to
  36-45% to 100% of groups with any spread, changing nothing but the tasks [measured:
  `fixtures/calibration*.json`, `checks/calibrate_tasks.py`].
- **Boundaries that are not the model's: partly.** Action boundaries are ours and explicit, so
  boundary *detection* is cheap. But the observation text is not the policy's, which is the mask
  question in section 5, and it is the same difficulty wearing a different hat.
- **Noisy negative samples: yes, severely, and from a source that looked authoritative.** An 8B
  judge held the format on 35 of 35 traces and disagreed with the rule 106 times across the 35
  judged episodes - about three per episode, roughly three quarters of judged turns at least 0.25
  apart, and in one direction: it scored "inspect the state before writing" as a failure [measured:
  `checks/run_p2_fusion.py`, record `runs/p2-loop-full-turn-2`, `code_sha git:96e1688`]. That is
  structural label noise, not variance: no amount of averaging removes a judge that is wrong the
  same way every time.

## 4. Why the fusion weights are what they are, and why the judge was removed

Rule-only runs use `{outcome 0.6, turn 0.4}`; the full stage adds `{critic 0.2}`. The weights matter
less than the measurement that decided their fate: at 0.2, the model term still moved the reward on
roughly three quarters of the judged turns, always in the same direction [measured:
`runs/p2-loop-full-turn-2`]. A judge that disagrees more
than half the time is not a weight to tune down - it is a signal to remove, and it was removed from
the reward and kept as a diagnostic. The rule's own bias is documented in the same place (it scores a
neutral turn as 0.5, which is why the rule-vs-judge comparison needed a neutral band rather than a
strict inequality).

## 5. The loss mask

- **Masked out:** the prompt - system message, user message, and the chat template scaffolding
  around them - and any padding.
- **Supervised:** the reply tokens only, including the terminating `<|im_end|>`, because stopping is
  a behaviour the model has to learn like any other.
- **Where:** `env/sft.py:95`
  (`labels = [-100] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids):]`), and the mask is
  *printed* rather than asserted: `env/README.md` shows the prompt and the supervised span for the
  first sample, because a mask bug is invisible in a loss curve.

**One failure a wrong mask produces.** Supervise the prompt and the objective quietly becomes
"reproduce the prompt": the loss falls fast and the answers get worse, and nothing in the training
log looks wrong. The model's default is already continuation - measured at the first supervised token
of an arithmetic sample, where the mass sat on `Let`/`To` and the correct digit ranked third
[measured]. A mask bug therefore does not create a new failure mode; it amplifies the one the model
arrives with. A second, cheaper failure: mask the terminator and the model never learns to stop, so
turns run into the token cap and the parser sees truncated actions [judgement].

## 6. What this design does not cover

- **An update on a served policy.** The served loop reads weights over HTTP and never writes them;
  the updating loop (`env/rl_step.py`) trains where it generates, so weight sync never happens. The
  design for that join is Project 3's subject (Chapter 9), and until it exists the two halves are
  joined only in this document.
- **Credit below the turn.** No per-turn labels were built, so nothing here says how to score the
  individual tokens of an action.
- **Multi-seed stability.** Every number cited is one seed, stated in the run header as such.
