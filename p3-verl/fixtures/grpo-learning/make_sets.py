"""Assemble the training set from the calibrated band, and a fresh held-out evaluation set.

Training items are the ones the served base model neither always nor never solves (0 < rate < 1) in
either calibration pass.  Evaluation items are a fresh draw from the same four tiers with a different
seed: they are never trained on, and their difficulty is whatever it is - which is why the report has
to show the base model's rate on them too, or a post-training number means nothing.
"""
import json
import random
from pathlib import Path

OUT = Path("/tmp/p3c")
TAIL = " Think briefly, then end your reply with a final line of the form Answer: <number>."

band = []
for name in ("calib2.json", "calib3.json"):
    rows = json.loads((OUT / name).read_text())
    band += [r for r in rows if 0 < r["rate"] < 1]

seen, train = set(), []
for r in band:
    if r["id"] in seen:
        continue
    seen.add(r["id"])
    train.append({"id": r["id"], "family": r["family"], "prompt": r["prompt"], "answer": r["answer"], "rate": r["rate"]})

rng = random.Random(20261111)
eval_rows = []


def add(family, prompt, answer):
    eval_rows.append({"id": f"e-{family}-{len(eval_rows):02d}", "family": family,
                      "prompt": prompt + TAIL, "answer": str(answer)})


for _ in range(6):
    a, b = rng.randint(71, 99), rng.randint(71, 99)
    add("m2", f"What is {a} times {b}?", a * b)
for _ in range(6):
    a, b = rng.randint(310, 989), rng.randint(31, 98)
    add("m32", f"What is {a} times {b}?", a * b)
for _ in range(6):
    b = rng.randint(11, 19)
    q = rng.randint(200, 900)
    add("dv", f"What is {b * q} divided by {b}?", q)
for _ in range(6):
    whole = rng.randint(180, 960)
    pct = rng.choice([13, 17, 19, 23, 29, 37])
    add("pc", f"What is {pct} percent of {whole}? Round down if the result is not a whole number.", whole * pct // 100)

(OUT / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
(OUT / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in eval_rows))
print(f"train {len(train)} items (rate {min(r['rate'] for r in train):.3f}..{max(r['rate'] for r in train):.3f}), eval {len(eval_rows)} items")
from collections import Counter
print("train families:", dict(Counter(r["family"] for r in train)))
print("eval  families:", dict(Counter(r["family"] for r in eval_rows)))
