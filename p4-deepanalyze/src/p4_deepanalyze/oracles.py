"""One oracle program per task family: the shortest program that answers the question.

Two uses, and they are the same use twice.  In the gate, the oracle proves that a fixture's
expected answer is reachable from its own CSV through the real executor - if the oracle cannot
reproduce it, the fixture is wrong, not the model.  In the cold start, the oracle is the
demonstration: the trajectory the fine-tune imitates is the program that provably works, not a
program someone wrote by hand and hoped.

Programs are keyed by *family*, not by task id, because the held-out set is the same six schemas
over different rows.  A program that only works on the training rows would prove nothing about
generalisation.
"""

from __future__ import annotations

ORACLE: dict[str, str] = {
    "sales-north": """\
import csv
with open('table.csv') as handle:
    rows = list(csv.DictReader(handle))
print(sum(float(r['units']) * float(r['unit_price']) for r in rows if r['region'] == 'North'))
""",
    "sales-top-product": """\
import csv
from collections import Counter
counts = Counter()
with open('table.csv') as handle:
    for row in csv.DictReader(handle):
        counts[row['product']] += float(row['units'])
print(max(counts.items(), key=lambda kv: kv[1])[0])
""",
    "temps-mean": """\
import csv
with open('table.csv') as handle:
    temps = [float(r['temp_c']) for r in csv.DictReader(handle)]
print(round(sum(temps) / len(temps), 1))
""",
    "students-pass-rate": """\
import csv
with open('table.csv') as handle:
    scores = [float(r['score']) for r in csv.DictReader(handle)]
print(round(sum(1 for s in scores if s >= 60) / len(scores), 4))
""",
    "orders-biggest-month": """\
import csv
with open('table.csv') as handle:
    rows = list(csv.DictReader(handle))
print(max(rows, key=lambda r: float(r['amount']))['month'])
""",
    "inventory-shortfall": """\
import csv
with open('table.csv') as handle:
    rows = list(csv.DictReader(handle))
print(int(max(float(r['reorder_level']) - float(r['stock']) for r in rows)))
""",
}
