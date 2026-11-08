"""Third task set: the four families that showed variance, shifted one notch harder and multiplied.

The calibration - not the designer - decides the band; this set exists because set 2 produced only 13
items with any variance at all, which is too few to split into train and held-out.
"""
import json
import random
from pathlib import Path

rng = random.Random(20261110)
rows = []
TAIL = " Think briefly, then end your reply with a final line of the form Answer: <number>."


def add(family, prompt, answer):
    rows.append({"id": f"{family}-{len(rows):02d}", "family": family, "prompt": prompt + TAIL, "answer": str(answer)})


for _ in range(14):                     # 2x2 at the top of the two-digit range
    a, b = rng.randint(71, 99), rng.randint(71, 99)
    add("m2", f"What is {a} times {b}?", a * b)

for _ in range(14):                     # 3x2 with a three-digit first factor
    a, b = rng.randint(310, 989), rng.randint(31, 98)
    add("m32", f"What is {a} times {b}?", a * b)

for _ in range(14):                     # exact division, divisor 11..19, quotient 200..900
    b = rng.randint(11, 19)
    q = rng.randint(200, 900)
    add("dv", f"What is {b * q} divided by {b}?", q)

for _ in range(14):                     # percentage with a non-round whole
    whole = rng.randint(180, 960)
    pct = rng.choice([13, 17, 19, 23, 29, 37])
    add("pc", f"What is {pct} percent of {whole}? Round down if the result is not a whole number.", whole * pct // 100)

out = Path("/tmp/p3c/tasks3.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"{len(rows)} tasks -> {out}")
