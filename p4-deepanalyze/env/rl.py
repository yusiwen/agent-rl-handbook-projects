"""The P4 reinforcement step, in three phases, because three different machines are needed.

    # 1. sample - needs the GPU and the model
    python3 p4-deepanalyze/env/rl.py --phase sample --run /work/rl/r1 --samples 4 --temperature 0.8

    # 2. reward - needs Docker, so it runs on the host, not in the training container
    python3 p4-deepanalyze/env/rl.py --phase reward --run /work/rl/r1 --image arlh-p4-sandbox:1

    # 3. update - needs the GPU again
    python3 p4-deepanalyze/env/rl.py --phase update --run /work/rl/r1 --lr 5e-5

Why three processes rather than one loop: the training container has no Docker socket, and a reward
computed by a different executor than the one that conditions the rollouts is a reward you cannot
compare with the eval.  So the sampling phase conditions on a local subprocess (and says so), the
reward phase re-executes every program in the real container and records the verdict it gets, and
the update consumes only the container's verdict.  Disagreements between the two are counted and
printed, because a silent disagreement is a reward you are training against by accident.

What this is not: an efficient trainer.  Every rollout is a separate conversation, the update is a
single step per sample with no critic and no KL anchor to a reference model, and the group is four
samples.  It is the smallest loop that has the property the chapter is about - the reward comes
from executing code - and the record says exactly that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))

from p4_deepanalyze import protocol, rlmath  # noqa: E402
from p4_deepanalyze.agent import first_message  # noqa: E402
from p4_deepanalyze.sandbox import DockerSandbox, LocalSandbox  # noqa: E402
from p4_deepanalyze.tasks import load  # noqa: E402
from p4_deepanalyze.verify import is_correct  # noqa: E402

FIXTURES = HERE.parent / "fixtures"


# --------------------------------------------------------------------------------------------
# phase 1: sample
# --------------------------------------------------------------------------------------------


def sample(args) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = load(args.tasks)

    tokeniser = AutoTokenizer.from_pretrained(args.model)
    tokeniser.padding_side = "left"
    if tokeniser.pad_token_id is None:
        tokeniser.pad_token = tokeniser.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.eval()

    sandbox = LocalSandbox(timeout_s=args.timeout)
    out = (run_dir / "rollouts.jsonl").open("w", encoding="utf-8")
    written = 0
    for task in tasks:
        prompts = [[
            {"role": "system", "content": protocol.SYSTEM_PROMPT},
            {"role": "user", "content": first_message(task)},
        ] for _ in range(args.samples)]
        finished = [False] * args.samples

        for turn in range(args.turns):
            texts = [
                tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in prompts
            ]
            batch = tokeniser(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    pad_token_id=tokeniser.pad_token_id,
                )
            replies = tokeniser.batch_decode(
                generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for index, (messages, reply) in enumerate(zip(prompts, replies)):
                if finished[index]:
                    continue
                reply = reply.strip()
                try:
                    action = protocol.parse(reply)
                except protocol.ProtocolError:
                    action = None
                history = list(messages) + [{"role": "assistant", "content": reply}]
                entry = {"task_id": task.id, "sample": index, "turn": turn, "messages": messages, "reply": reply}
                if action is not None and action.kind == "code":
                    observation = sandbox.run(action.text, task.files())
                    entry["observation"] = observation.to_dict()
                    entry["warnings"] = protocol.observation_warnings(
                        observation.exit_code, observation.stdout, observation.stderr, observation.timed_out
                    )
                    history.append({
                        "role": "user",
                        "content": protocol.format_observation(
                            observation.exit_code, observation.stdout, observation.stderr,
                            observation.timed_out, entry["warnings"],
                        ),
                    })
                elif action is not None and action.kind == "report":
                    entry["report"] = action.text
                    finished[index] = True
                else:
                    entry["protocol_error"] = True
                    finished[index] = True
                out.write(json.dumps(entry, sort_keys=True) + "\n")
                written += 1
                prompts[index] = history
            if args.turns == 1:
                break
        print(f"  {task.id}: {args.samples} samples, {args.turns} turn(s)")

    out.close()
    print(f"rollouts: {written} turn records -> {run_dir / 'rollouts.jsonl'}")
    return 0


# --------------------------------------------------------------------------------------------
# phase 2: reward, with the container executor
# --------------------------------------------------------------------------------------------


def replay(rollout: dict, task, sandbox) -> tuple[bool, bool, list[str]]:
    """Re-run every program in the rollout, in order, and score the final report."""
    ran_ok = False
    printed = ""
    for turn in rollout["turns"]:
        try:
            action = protocol.parse(turn["reply"])
        except protocol.ProtocolError:
            continue
        if action.kind == "code":
            observation = sandbox.run(action.text, task.files())
            if observation.exit_code == 0 and observation.stdout.strip():
                ran_ok = True
                printed = observation.stdout.strip()
    report = rollout.get("report")
    solved = bool(report) and is_correct(report, task.expected, task.tolerance)
    return solved, ran_ok, [printed]


def reward(args) -> int:
    run_dir = Path(args.run)
    tasks = {task.id: task for task in load(args.tasks)}
    sandbox = DockerSandbox(image=args.image, timeout_s=args.timeout)

    records: dict[tuple[str, int], dict] = {}
    for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        key = (entry["task_id"], entry["sample"])
        record = records.setdefault(
            key, {"task_id": entry["task_id"], "sample": entry["sample"], "turns": [], "report": None}
        )
        record["turns"].append(entry)
        if entry.get("report") is not None:
            record["report"] = entry["report"]

    groups: dict[str, list[float]] = {}
    out_lines: list[str] = []
    answered_without_running = 0
    for key, record in sorted(records.items()):
        task = tasks[record["task_id"]]
        solved, ran_ok, printed = replay(record, task, sandbox)
        if solved and not ran_ok:
            # The report was right and no program produced a value: a guess that happened to land.
            # Worth counting separately - it is the same failure the harness refuses in eval.
            answered_without_running += 1
        value = rlmath.reward_for(solved=solved, ran_ok=ran_ok)
        groups.setdefault(record["task_id"], []).append(value)
        out_lines.append(json.dumps({
            "task_id": record["task_id"],
            "sample": record["sample"],
            "solved": solved,
            "ran_ok": ran_ok,
            "reward": value,
            "printed": printed,
        }, sort_keys=True))

    (run_dir / "rewards.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    report = rlmath.group_report(groups)
    (run_dir / "groups.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report["_summary"]
    print(f"groups: {summary['groups']}, informative: {summary['informative_groups']}, "
          f"uninformative: {summary['uninformative_groups']}")
    print(f"solved without a runnable program (a lucky guess): {answered_without_running}")
    for task_id, row in report.items():
        if task_id == "_summary":
            continue
        print(f"  {task_id:26} solved {row['solved']}/{row['samples']}  spread {row['spread']}")
    return 0


# --------------------------------------------------------------------------------------------
# phase 3: update
# --------------------------------------------------------------------------------------------


def update(args) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.run)
    rewards = [json.loads(line) for line in (run_dir / "rewards.jsonl").read_text(encoding="utf-8").splitlines()]
    by_group: dict[str, list[dict]] = {}
    for row in rewards:
        by_group.setdefault(row["task_id"], []).append(row)
    for rows in by_group.values():
        values = [row["reward"] for row in rows]
        for row, advantage in zip(rows, rlmath.advantages(values)):
            row["advantage"] = advantage

    if args.positive_only:
        # Whole-reply REINFORCE punishes every token of a nearly correct program when the answer
        # is wrong, which teaches the model to avoid the API it used. Keeping only the winners is
        # a blunt experiment, and it is the cheapest way to find out whether that is the mechanism.
        for row in rewards:
            row["advantage"] = max(row["advantage"], 0.0)
    used = [row for row in rewards if row["advantage"] != 0.0]
    print(f"samples: {len(rewards)}, with a non-zero advantage: {len(used)}")
    if not used:
        print("nothing to learn: every group scored the same. This is a measurement, not a failure.")
        tag = Path(args.out).name or "update"
        (run_dir / f"update-{tag}.json").write_text(
            json.dumps({"samples": len(rewards), "used": 0, "advantage": "all zero", "steps": 0,
                        "lr": args.lr, "positive_only": bool(args.positive_only)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    rollouts: dict[tuple[str, int], list[dict]] = {}
    for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        rollouts.setdefault((entry["task_id"], entry["sample"]), []).append(entry)

    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.train()
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    steps = 0
    total_loss = 0.0
    for row in used:
        turns = rollouts[(row["task_id"], row["sample"])]
        loss = None
        for turn in turns:
            text = tokeniser.apply_chat_template(turn["messages"], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokeniser(text, add_special_tokens=False)["input_ids"]
            reply_ids = tokeniser(turn["reply"] + tokeniser.eos_token, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt_ids + reply_ids], device="cuda")
            labels = torch.tensor([[-100] * len(prompt_ids) + reply_ids], device="cuda")
            token_loss = model(input_ids=ids, labels=labels).loss
            loss = (loss or 0.0) + token_loss / len(turns)
        # REINFORCE with the group mean as the baseline; the sign does the work.
        weighted = -row["advantage"] * loss
        optimiser.zero_grad(set_to_none=True)
        weighted.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimiser.step()
        steps += 1
        total_loss += loss.item()
        print(f"  {row['task_id']:26} sample {row['sample']}  reward {row['reward']:.1f}  "
              f"adv {row['advantage']:+.2f}  loss {loss.item():.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokeniser.save_pretrained(out / "adapter")
    summary = {"samples": len(rewards), "used": len(used), "steps": steps,
               "mean_loss": round(total_loss / max(steps, 1), 4), "lr": args.lr,
               "positive_only": bool(args.positive_only)}
    tag = out.name or "update"
    (run_dir / f"update-{tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    print(f"adapter: {out / 'adapter'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4 reinforcement step (sample / reward / update)")
    parser.add_argument("--phase", choices=("sample", "reward", "update"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tasks", default=str(FIXTURES / "tasks.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--image", default="arlh-p4-sandbox:1")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--out", default="/work/p4-rl")
    args = parser.parse_args(argv)

    handlers = {"sample": sample, "reward": reward, "update": update}
    return handlers[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
