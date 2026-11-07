"""Build the P4 fixtures: six training tables, six held-out ones, and the checks that keep them true.

The point of this script is not convenience.  If an expected answer is typed by hand, the fixture
can be wrong in a way no test will catch: the model reports the right number, the verifier says it
is wrong, and the chapter's conclusion is backwards.  So every answer here is computed from the CSV
in this file, and the files are regenerated rather than edited.

Two sets, because the cold start needs them:

* **train** (`tasks.jsonl`) - the six tasks whose oracle programs become the SFT demonstrations;
* **held-out** (`tasks-heldout.jsonl`) - the same six schemas over different rows, never trained on,
  which is the only way a fine-tune's score says anything about the model rather than about recall.

    python3 p4-deepanalyze/fixtures/build_tasks.py            # writes both files
    python3 p4-deepanalyze/fixtures/build_tasks.py --check    # fails if either is stale
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_OUT = HERE / "tasks.jsonl"
HELDOUT_OUT = HERE / "tasks-heldout.jsonl"

TABLES: dict[str, str] = {
    "sales": """region,product,units,unit_price
North,widget,120,4.50
South,widget,80,4.50
North,gadget,45,12.00
East,widget,200,4.50
East,gadget,30,12.00
South,gizmo,15,30.00
North,gizmo,22,30.00""",
    "temps": """day,temp_c
Mon,18.5
Tue,21.0
Wed,19.5
Thu,23.0
Fri,20.0
Sat,17.5""",
    "students": """name,score
Ana,72
Bo,58
Chen,91
Dara,60
Eli,45
Fen,83""",
    "orders": """month,amount
January,1200.00
February,980.50
March,2400.75
April,1750.25
May,2999.99""",
    "inventory": """item,stock,reorder_level
bolts,120,200
nuts,340,150
washers,75,100
screws,45,80
anchors,12,40""",
}

HELDOUT_TABLES: dict[str, str] = {
    "sales": """region,product,units,unit_price
South,widget,60,4.50
North,widget,30,4.50
South,gadget,90,12.00
West,widget,150,4.50
North,gizmo,8,30.00
West,gadget,70,12.00""",
    "temps": """day,temp_c
Mon,11.0
Tue,13.5
Wed,9.0
Thu,12.0
Fri,15.5
Sat,14.0
Sun,10.5""",
    "students": """name,score
Gus,64
Hana,55
Ivo,88
Jun,61
Kai,40
Lia,77
Moe,52""",
    "orders": """month,amount
June,800.00
July,1450.50
August,990.25
September,2100.00
October,1200.75""",
    "inventory": """item,stock,reorder_level
cables,40,90
plugs,110,60
ties,25,50
clips,8,30
labels,60,80""",
}

QUESTIONS: dict[str, str] = {
    "sales-north": "What is the total revenue for the North region, where revenue is units times unit_price?",
    "sales-top-product": "Which product sold the most units in total? Answer with the product name.",
    "temps-mean": "What is the mean temperature in degrees Celsius, to one decimal place?",
    "students-pass-rate": "What fraction of students scored at least 60? Give a decimal.",
    "orders-biggest-month": "In which month was the largest single order placed? Answer with the month name.",
    "inventory-shortfall": (
        "For the item furthest below its reorder level, how many units short is it? "
        "Answer with the number of units."
    ),
}

#: Which table a family reads.  The family is what an oracle program is keyed by, so the same six
#: programs serve the training set and the held-out set - the schemas are identical, the rows are not.
FAMILY_TABLE = {
    "sales-north": "sales",
    "sales-top-product": "sales",
    "temps-mean": "temps",
    "students-pass-rate": "students",
    "orders-biggest-month": "orders",
    "inventory-shortfall": "inventory",
}


def rows(table: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(table)))


def answer(family: str, table: str) -> tuple[str, float]:
    """The expected answer, computed from the table - the only definition in the repository."""
    data = rows(table)
    if family == "sales-north":
        value = sum(float(r["units"]) * float(r["unit_price"]) for r in data if r["region"] == "North")
        return f"{value:.2f}", 1e-6
    if family == "sales-top-product":
        totals: dict[str, float] = {}
        for row in data:
            totals[row["product"]] = totals.get(row["product"], 0.0) + float(row["units"])
        return max(totals.items(), key=lambda kv: kv[1])[0], 1e-6
    if family == "temps-mean":
        temps = [float(r["temp_c"]) for r in data]
        return f"{sum(temps) / len(temps):.1f}", 0.05
    if family == "students-pass-rate":
        scores = [float(r["score"]) for r in data]
        return f"{sum(1 for s in scores if s >= 60) / len(scores):.4f}", 0.01
    if family == "orders-biggest-month":
        return max(data, key=lambda r: float(r["amount"]))["month"], 1e-6
    if family == "inventory-shortfall":
        shortfall = max(float(r["reorder_level"]) - float(r["stock"]) for r in data)
        return f"{shortfall:.0f}", 1e-6
    raise KeyError(f"no answer rule for family {family}")


def build(tables: dict[str, str], prefix: str) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for family in QUESTIONS:
        table = tables[FAMILY_TABLE[family]]
        expected, tolerance = answer(family, table)
        task: dict[str, object] = {
            "id": f"{prefix}{family}",
            "family": family,
            "table": table,
            "question": QUESTIONS[family],
            "expected": expected,
        }
        if tolerance != 1e-6:
            task["tolerance"] = tolerance
        tasks.append(task)
    return tasks


def render(tasks: list[dict[str, object]]) -> str:
    return "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the P4 task fixtures")
    parser.add_argument("--check", action="store_true", help="fail if either file differs")
    args = parser.parse_args(argv)

    payloads = {
        TRAIN_OUT: render(build(TABLES, "p4-")),
        HELDOUT_OUT: render(build(HELDOUT_TABLES, "p4h-")),
    }
    status = 0
    for path, text in payloads.items():
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != text:
                print(f"{path.name} is stale; regenerate with {Path(__file__).name}")
                status = 1
            else:
                print(f"{path.name} is current ({len(text.splitlines())} tasks)")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.name} ({len(text.splitlines())} tasks)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
