"""Second task set: tiers around the difficulty band the first calibration found.

The first set was almost entirely solved (369/384), with only 2-digit x 2-digit anywhere near the
band (0.77).  So the tiers here sit one step up from that, and the calibration - not the designer -
decides which ones train.
"""
import json
import random
from pathlib import Path

rng = random.Random(20261109)
rows = []
TAIL = " Think briefly, then end your reply with a final line of the form Answer: <number>."


def add(family, prompt, answer):
    rows.append({"id": f"{family}-{len(rows):02d}", "family": family, "prompt": prompt + TAIL, "answer": str(answer)})


for _ in range(8):                                   # t1: 2x2, large digits
    a, b = rng.randint(61, 99), rng.randint(61, 99)
    add("mul2x2h", f"What is {a} times {b}?", a * b)

for _ in range(8):                                   # t2: 3x2
    a, b = rng.randint(112, 989), rng.randint(12, 98)
    add("mul3x2", f"What is {a} times {b}?", a * b)

for _ in range(8):                                   # t3: 4x1
    a, b = rng.randint(1120, 9899), rng.randint(3, 9)
    add("mul4x1", f"What is {a} times {b}?", a * b)

for _ in range(8):                                   # t4: percentage of a non-round number
    whole = rng.randint(140, 960)
    pct = rng.choice([12, 18, 24, 36])
    add("pct_hard", f"What is {pct} percent of {whole}? Round down if the result is not a whole number.", whole * pct // 100)

for _ in range(8):                                   # t5: exact division, 4-5 digits
    b = rng.randint(6, 19)
    q = rng.randint(120, 899)
    add("div_hard", f"What is {b * q} divided by {b}?", q)

for _ in range(8):                                   # t6: three-term with carries
    a, b, c = rng.randint(1200, 9800), rng.randint(1200, 9800), rng.randint(120, 980)
    add("sub3_big", f"What is {a} plus {b} minus {c}?", a + b - c)

for _ in range(8):                                   # t7: (a+b) x c with 3-digit terms
    a, b, c = rng.randint(120, 890), rng.randint(120, 890), rng.randint(12, 48)
    add("twostep_big", f"What is ({a} plus {b}) times {c}?", (a + b) * c)

out = Path("/tmp/p3c/tasks2.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"{len(rows)} tasks -> {out}")
