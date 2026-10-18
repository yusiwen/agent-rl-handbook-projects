"""P2-prime: supervised fine-tuning on oracle demonstrations, then a merged checkpoint.

    python3 /env/sft.py                        # build demonstrations, train one epoch, merge
    python3 /env/sft.py --epochs 2 --lr 1e-4   # anything else

What this is for, stated plainly: the loop's cold start is flat. At a uniform policy over the
43-task set, 18 of 43 groups show any reward variation at all, and the rest contribute a zero
gradient. Format compliance is already at 100% before training (see the README's baseline), so the
thing missing is not the ability to follow the instruction, it is knowing which action to take.

So the demonstrations are the oracle's own trajectories - the shortest path to the goal, computed by
search over the menu rather than written by hand - and the loss covers only the model's own reply.
Prompt tokens are masked. That mask is the same discipline Chapter 6 argues for, applied here in the
one place where it is easy to get wrong: a fine-tune that trains on the prompt learns to reproduce
the prompt.

The demonstration file is written to the workspace volume with its digest, so a later measurement
can say which demonstrations the checkpoint saw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2_reagent import env, prompt


def oracle(task_id: str, max_turns: int = 8) -> list[str]:
    """The shortest action sequence that solves the task, found by trying every action.

    Search rather than hand-writing, because a hand-written demonstration is a second description
    of the task and will eventually disagree with the environment - which is the failure this
    project keeps running into whenever two places describe the same thing.
    """
    attempt = env.start(task_id)
    taken: list[str] = []
    for _ in range(max_turns):
        if attempt.finished:
            break
        best = None
        for action in attempt.menu():
            probe = env.start(task_id)
            for previous in taken:
                probe.step(previous)
            result = probe.step(action)
            score = (2.0 if result.done and probe.solved() else probe.progress())
            if best is None or score > best[0]:
                best = (score, action)
        if best is None:
            break
        taken.append(best[1])
        attempt.step(best[1])
    if not attempt.solved():
        raise RuntimeError(f"oracle could not solve {task_id}: {taken}")
    return taken


def demonstrations() -> list[dict]:
    """One record per task: the prompts the policy saw and the reply it should have given."""
    records = []
    for task in env.all_tasks():
        actions = oracle(task.task_id)
        attempt = env.start(task.task_id)
        history: list[tuple[str, str, str]] = []
        turns = []
        for action in actions:
            turns.append(
                {
                    "messages": prompt.render(attempt.task, attempt.observation, attempt.menu(), history),
                    "target": f"{prompt.ACTION_PREFIX} {action}",
                }
            )
            result = attempt.step(action)
            history.append((action, result.signal, result.observation))
        if not attempt.solved():
            raise RuntimeError(f"{task.task_id}: replaying the oracle trajectory did not solve it")
        records.append({"task_id": task.task_id, "turns": turns, "actions": actions})
    return records


def encode(tokeniser, messages: list[dict[str, str]], target: str, seq_len: int, pad_id: int):
    """Prompt masked, reply supervised, one row."""
    prompt_text = tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokeniser(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tokeniser(target + tokeniser.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + reply_ids)[:seq_len]
    labels = [-100] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids) :]
    keep = len(ids)
    return (
        ids + [pad_id] * (seq_len - keep),
        labels + [-100] * (seq_len - keep),
        [1] * keep + [0] * (seq_len - keep),
        len(prompt_ids),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2-prime supervised fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=640)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out", default="/work/sft")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ---------------------------------------------------------------- the demonstrations
    records = demonstrations()
    demo_text = "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    (out / "demonstrations.jsonl").write_text(demo_text, encoding="utf-8")
    demo_digest = "sha256:" + hashlib.sha256(demo_text.encode("utf-8")).hexdigest()[:16]
    turns = sum(len(record["turns"]) for record in records)
    print(f"demonstrations: {len(records)} tasks, {turns} turns, {demo_digest}")

    # ---------------------------------------------------------------- the model
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pad_id = tokeniser.pad_token_id if tokeniser.pad_token_id is not None else tokeniser.eos_token_id

    rows = [turn for record in records for turn in record["turns"]]
    encoded = [encode(tokeniser, row["messages"], row["target"], args.seq_len, pad_id) for row in rows]
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )

    torch.cuda.reset_peak_memory_stats()
    model.train()
    losses = []
    train_started = time.time()
    for epoch in range(args.epochs):
        for start in range(0, len(encoded), args.batch):
            batch = encoded[start : start + args.batch]
            ids = torch.tensor([row[0] for row in batch], device="cuda")
            labels = torch.tensor([row[1] for row in batch], device="cuda")
            mask = torch.tensor([row[2] for row in batch], device="cuda")
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if len(losses) % 10 == 0 or len(losses) == 1:
                print(f"step {len(losses):3d}  loss {losses[-1]:.4f}")
    torch.cuda.synchronize()
    train_seconds = time.time() - train_started

    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tokeniser.save_pretrained(out / "merged")
    merged_digest = "sha256:" + hashlib.sha256(
        b"".join(sorted(p.name.encode() for p in (out / "merged").iterdir()))
    ).hexdigest()[:16]

    print()
    print(
        f"SFT_OK model={args.model} tasks={len(records)} turns={turns} epochs={args.epochs} "
        f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.3f}%) "
        f"loss={losses[0]:.4f}->{losses[-1]:.4f} steps={len(losses)} "
        f"train={train_seconds:.1f}s peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
        f"demos={demo_digest} merged_files={merged_digest} out={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
