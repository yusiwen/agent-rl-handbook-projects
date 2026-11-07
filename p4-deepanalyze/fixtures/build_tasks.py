"""Build the P4 fixture tasks, and refuse to write any task whose answer is not recomputed.

The point of this script is not convenience.  If the expected answer in a fixture is typed by
hand, the fixture can be wrong in a way no test will catch: the model reports the right number,
the verifier says it is wrong, and the chapter's conclusion is backwards.  So every answer here
is computed from the CSV in this file, and the file is regenerated rather than edited.

    python3 p4-deepanalyze/fixtures/build_tasks.py            # writes tasks.jsonl
    python3 p4-deepanalyze/fixtures/build_tasks.py --check    # fails if tasks.jsonl is stale
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tasks.jsonl"

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


def rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(TABLES[name])))


def build() -> list[dict[str, object]]:
    sales = rows("sales")
    north = sum(float(r["units"]) * float(r["unit_price"]) for r in sales if r["region"] == "North")

    by_product: dict[str, float] = {}
    for row in sales:
        by_product[row["product"]] = by_product.get(row["product"], 0.0) + float(row["units"])
    top = max(by_product.items(), key=lambda kv: kv[1])

    temps = [float(r["temp_c"]) for r in rows("temps")]
    mean_temp = sum(temps) / len(temps)

    students = [float(r["score"]) for r in rows("students")]
    pass_rate = sum(1 for score in students if score >= 60) / len(students)

    orders = rows("orders")
    biggest = max(orders, key=lambda r: float(r["amount"]))

    inventory = rows("inventory")
    shortfalls = [(r["item"], float(r["reorder_level"]) - float(r["stock"])) for r in inventory]
    worst = max(shortfalls, key=lambda kv: kv[1])

    return [
        {
            "id": "p4-sales-north",
            "table": TABLES["sales"],
            "question": "What is the total revenue for the North region, where revenue is units times unit_price?",
            "expected": f"{north:.2f}",
        },
        {
            "id": "p4-sales-top-product",
            "table": TABLES["sales"],
            "question": "Which product sold the most units in total? Answer with the product name.",
            "expected": top[0],
        },
        {
            "id": "p4-temps-mean",
            "table": TABLES["temps"],
            "question": "What is the mean temperature in degrees Celsius, to one decimal place?",
            "expected": f"{mean_temp:.1f}",
            "tolerance": 0.05,
        },
        {
            "id": "p4-students-pass-rate",
            "table": TABLES["students"],
            "question": "What fraction of students scored at least 60? Give a decimal.",
            "expected": f"{pass_rate:.4f}",
            "tolerance": 0.01,
        },
        {
            "id": "p4-orders-biggest-month",
            "table": TABLES["orders"],
            "question": "In which month was the largest single order placed? Answer with the month name.",
            "expected": biggest["month"],
        },
        {
            "id": "p4-inventory-shortfall",
            "table": TABLES["inventory"],
            "question": (
                "For the item furthest below its reorder level, how many units short is it? "
                "Answer with the number of units."
            ),
            "expected": f"{worst[1]:.0f}",
        },
    ]


def render(tasks: list[dict[str, object]]) -> str:
    return "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the P4 task fixtures")
    parser.add_argument("--check", action="store_true", help="fail if tasks.jsonl differs")
    args = parser.parse_args(argv)

    text = render(build())
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"tasks.jsonl is stale; regenerate with {Path(__file__).name}")
            return 1
        print(f"tasks.jsonl is current ({len(text.splitlines())} tasks)")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text.splitlines())} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
