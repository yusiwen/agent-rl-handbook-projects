"""P0b: ten LoRA steps on the real checkpoint, inside the training container.

    python3 /env/smoke_lora.py                       # Qwen/Qwen3-1.7B, 20 samples, 10 steps
    python3 /env/smoke_lora.py --steps 4 --batch 1   # smaller, for a machine with less memory

This is the smallest run that can answer the question P0 asks: does a backward pass through the
real checkpoint fit, converge, and leave an adapter that loads again? It is not training data and
it is not a result - twenty arithmetic pairs the model already knows, used as a load test.

The loss mask is not incidental. Prompt tokens are labelled -100, so the model is trained on the
answer only; that is the discipline Chapter 6 argues for, applied here so the smoke run exercises
the same code path the real fine-tune will.

Prints one `SMOKE_OK` line with the numbers, and exits non-zero if the loss does not fall or the
adapter does not reload.
"""

from __future__ import annotations

import argparse
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PAIRS = [
    ("What is 17 + 26?", "43"),
    ("What is 64 / 8?", "8"),
    ("What is 9 * 7?", "63"),
    ("What is 100 - 37?", "63"),
    ("What is 12 + 12?", "24"),
    ("What is 81 / 9?", "9"),
    ("What is 15 * 4?", "60"),
    ("What is 50 - 18?", "32"),
    ("What is 7 + 7 + 7?", "21"),
    ("What is 144 / 12?", "12"),
    ("What is 33 + 44?", "77"),
    ("What is 8 * 8?", "64"),
    ("What is 90 - 45?", "45"),
    ("What is 6 * 11?", "66"),
    ("What is 72 / 6?", "12"),
    ("What is 19 + 23?", "42"),
    ("What is 5 * 13?", "65"),
    ("What is 200 - 149?", "51"),
    ("What is 3 + 4 + 5?", "12"),
    ("What is 121 / 11?", "11"),
]


def build_batch(tokeniser, pairs, seq_len, pad_id):
    """Tokenise prompt+answer pairs with the prompt masked out of the loss."""
    rows_ids, rows_labels, rows_mask = [], [], []
    for question, answer in pairs:
        messages = [{"role": "user", "content": question}]
        try:
            # Qwen3's template opens a thinking block by default; the smoke run has nothing to
            # think about and the mask below would then cover reasoning tokens.
            prompt = tokeniser.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokeniser.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt_ids = tokeniser(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokeniser(prompt + answer + tokeniser.eos_token, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:seq_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        keep = len(full_ids)
        rows_ids.append(full_ids + [pad_id] * (seq_len - keep))
        rows_labels.append(labels + [-100] * (seq_len - keep))
        rows_mask.append([1] * keep + [0] * (seq_len - keep))
    return (
        torch.tensor(rows_ids, device="cuda"),
        torch.tensor(rows_labels, device="cuda"),
        torch.tensor(rows_mask, device="cuda"),
    )


def inspect(tokeniser, model, seq_len, pad_id) -> int:
    """Per-token cross entropy on the first sample, and what the model would have said instead."""
    ids, labels, mask = build_batch(tokeniser, PAIRS[:1], seq_len, pad_id)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
    positions = (labels[0] != -100).nonzero().flatten().tolist()
    first = positions[0]
    print(f"first supervised token: {tokeniser.decode([int(labels[0, first])])!r} at position {first}")
    top = logits[0, first - 1].topk(5)
    for value, index in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  wants {tokeniser.decode([index])!r:16s} logit {value:9.3f}")
    print()
    losses = []
    for position in positions:
        logit = logits[0, position - 1 : position]
        target = labels[0, position : position + 1]
        ce = torch.nn.functional.cross_entropy(logit, target)
        losses.append(ce.item())
        print(f"  {tokeniser.decode([int(labels[0, position])])!r:16s} ce {ce.item():8.3f}")
    print(f"\nmean over {len(losses)} tokens: {sum(losses) / len(losses):.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0b LoRA smoke run")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out", default="/work/p0b-adapter")
    parser.add_argument("--inspect", action="store_true", help="print per-token loss and exit")
    args = parser.parse_args(argv)

    started = time.time()
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    loaded = time.time()

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
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )

    pad_id = tokeniser.pad_token_id if tokeniser.pad_token_id is not None else tokeniser.eos_token_id
    batches = [
        PAIRS[i : i + args.batch]
        for i in range(0, len(PAIRS), args.batch)
    ][: args.steps]

    if args.inspect:
        return inspect(tokeniser, model, args.seq_len, pad_id)

    # The mask is the part that is easy to get wrong, so the record shows it rather than
    # asserting it: decode the first sample's supervised span and its prompt span.
    _ids, _labels, _mask = build_batch(tokeniser, PAIRS[:1], args.seq_len, pad_id)
    supervised = tokeniser.decode([t for t in _labels[0].tolist() if t != -100])
    first_prompt = tokeniser.apply_chat_template(
        [{"role": "user", "content": PAIRS[0][0]}], tokenize=False, add_generation_prompt=True
    )
    print(f"prompt    {first_prompt!r}")
    print(f"labels    {supervised!r}")
    print()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_started = time.time()
    losses = []
    model.train()
    for step, pairs in enumerate(batches, start=1):
        ids, labels, mask = build_batch(tokeniser, pairs, args.seq_len, pad_id)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
        print(f"step {step:2d}/{len(batches)}  loss {losses[-1]:.4f}")
    torch.cuda.synchronize()
    train_seconds = time.time() - train_started

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    model.save_pretrained(args.out)
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0}),
        args.out,
    )
    reloaded.eval()

    if losses[-1] >= losses[0]:
        print(f"FAIL: loss did not fall ({losses[0]:.4f} -> {losses[-1]:.4f})")
        return 2
    tokens = len(batches) * args.batch * args.seq_len
    print()
    print(
        f"SMOKE_OK model={args.model} steps={len(batches)} batch={args.batch} seq={args.seq_len} "
        f"rank={args.rank} trainable={trainable:,}/{total:,} ({100 * trainable / total:.3f}%) "
        f"loss={losses[0]:.4f}->{losses[-1]:.4f} "
        f"load={loaded - started:.1f}s train={train_seconds:.1f}s "
        f"s_per_step={train_seconds / len(batches):.2f} tok_per_s={tokens / train_seconds:.0f} "
        f"peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB "
        f"adapter={args.out} reloaded={type(reloaded).__name__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
