"""Stage 1: LoRA supervised fine-tuning on the traced oracle episodes.

    python3 p5-openclaw-rl/env/sft.py --out /models/p5-sft --epochs 6

Same shape as the other projects' cold starts: prompt tokens masked, reply supervised, LoRA on the
four attention projections, a manifest with the demonstration digest.  What differs is the reply: one
JSON object per turn, which is a format a 1.7B base model can produce for a turn or two and then
loses - which is the point of supervising it here rather than discovering it in the loop.
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
DEMOS = HERE.parent / "fixtures" / "demonstrations.jsonl"


def encode(tokeniser, messages: list[dict[str, str]], reply: str, seq_len: int, pad_id: int):
    prompt_text = tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokeniser(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tokeniser(reply + tokeniser.eos_token, add_special_tokens=False)["input_ids"]
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
    parser = argparse.ArgumentParser(description="P5 stage 1: cold start")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--demos", default=str(DEMOS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="/models/p5-sft")
    args = parser.parse_args(argv)

    demos = Path(args.demos)
    text = demos.read_text(encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    print(f"demonstrations: {len(rows)} turns, {digest}")

    tokeniser = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokeniser.pad_token_id if tokeniser.pad_token_id is not None else tokeniser.eos_token_id
    encoded = [encode(tokeniser, row["messages"], row["reply"], args.seq_len, pad_id) for row in rows]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, bias="none",
                   task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
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

    out = Path(args.out)
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokeniser.save_pretrained(out / "adapter")
    manifest = {
        "base_model": args.model, "demonstration_digest": digest, "turns": len(rows),
        "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
        "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
        "steps": len(losses), "seconds": round(time.time() - started, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("loss_first", "loss_last", "steps", "seconds", "peak_memory_gb")}))
    print(f"adapter: {out / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
