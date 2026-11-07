"""The cold start: LoRA supervised fine-tuning on the oracle demonstrations.

    python3 /work/repo/p4-deepanalyze/env/sft.py --epochs 6 --lr 1e-4
    python3 .../sft.py --out /work/p4-sft --check-only     # encode and report, do not train

Why a cold start at all: the base policy already follows the marker format (6 of 6 on the measured
run), so what is missing is not the grammar.  It is two habits - print the value you computed, and
report what the observation says rather than what you remember.  Neither is a fact the model can be
told; both are visible in twelve demonstrations, and a LoRA on twelve rows is exactly the size of
intervention this needs.

The loss covers the model's own reply and nothing else.  Prompt tokens are masked with -100, which
is Chapter 6's rule applied in the one place where it is easiest to get wrong: a fine-tune that
trains on the prompt learns to reproduce the prompt.
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

HERE = Path(__file__).resolve().parent
DEFAULT_DEMOS = HERE.parent / "fixtures" / "sft.jsonl"


def encode(tokeniser, messages: list[dict[str, str]], target: str, seq_len: int, pad_id: int):
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


def load_demos(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"no demonstrations in {path}")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4 cold start (LoRA SFT on oracle demonstrations)")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--demos", default=str(DEFAULT_DEMOS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="/work/p4-sft")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    demos_path = Path(args.demos)
    demo_text = demos_path.read_text(encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(demo_text.encode("utf-8")).hexdigest()[:16]

    records = load_demos(demos_path)
    rows = [turn for record in records for turn in record["turns"]]
    print(f"demonstrations: {len(records)} tasks, {len(rows)} turns, {digest}")

    tokeniser = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokeniser.pad_token_id if tokeniser.pad_token_id is not None else tokeniser.eos_token_id
    encoded = [encode(tokeniser, row["messages"], row["target"], args.seq_len, pad_id) for row in rows]
    lengths = [row[3] for row in encoded]
    print(f"prompt tokens: min {min(lengths)} max {max(lengths)}; sequence budget {args.seq_len}")
    if args.check_only:
        for record, row in zip(records, rows):
            print(f"  {record['task_id']:28} {len(row['target'])} chars target")
        return 0

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
    print(f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    losses: list[float] = []
    started = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(encoded)).tolist()
        for start in range(0, len(encoded), args.batch):
            batch = [encoded[i] for i in order[start : start + args.batch]]
            ids = torch.tensor([row[0] for row in batch], device="cuda")
            labels = torch.tensor([row[1] for row in batch], device="cuda")
            mask = torch.tensor([row[2] for row in batch], device="cuda")
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            losses.append(loss.item())
        print(f"epoch {epoch + 1}/{args.epochs}  loss {losses[-1]:.4f}")
    torch.cuda.synchronize()
    seconds = time.time() - started

    model.save_pretrained(out / "adapter")
    tokeniser.save_pretrained(out / "adapter")
    manifest = {
        "base_model": args.model,
        "demonstrations": str(demos_path),
        "demonstration_digest": digest,
        "tasks": [record["task_id"] for record in records],
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "rank": args.rank,
        "trainable_params": trainable,
        "total_params": total,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "steps": len(losses),
        "seconds": round(seconds, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("loss_first", "loss_last", "steps", "seconds", "peak_memory_gb")}))
    print(f"adapter: {out / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
