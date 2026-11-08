"""Does the right skill get found? Three routers, three library sizes, one query per instance.

    python3 p6-memento-skills/env/measure_retrieval.py [--out fixtures/retrieval.json]

The measurement is model-free on purpose: retrieval is the half of this project that can be
answered with arithmetic, so it is answered before a policy is involved. The other half - does the
retrieved skill change what the model does - needs the served policy and is measured separately.

Queries are `(family, ask, symptom)`: the task text, plus what the previous attempt said went wrong.
That second half is the whole design argument, so one of the routers is built to use it and the
other two are not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from p6_memento_skills.families import FAMILIES, make_instances  # noqa: E402
from p6_memento_skills.retrieval import (  # noqa: E402
    ROUTERS,
    SYMPTOMS,
    SkillDoc,
    filler_docs,
    recall_at_1,
)

# scale -> (on-topic decoys, word-salad fillers).  The decoys are the hard part: each is a
# plausible procedure about the *same* task that fixes the wrong thing, so a router that matches
# on topic words alone cannot tell them apart from the right one.
SCALES = {"ten": (0, 0), "fifty": (10, 30), "hundred": (10, 80)}
QUERY_SEEDS = range(200, 206)


def documents(decoys: int, fillers: int) -> list[SkillDoc]:
    real = [SkillDoc(name=family.id, family=family.id, trigger=family.title, procedure=family.skill) for family in FAMILIES]
    wrong = [
        SkillDoc(name=f"decoy-{family.id}", family=f"decoy:{family.id}", trigger=family.title, procedure=family.decoy)
        for family in FAMILIES[:decoys]
    ]
    return real + wrong + filler_docs(fillers, seed=11)


def queries() -> list[tuple[str, str, str]]:
    # seeds 0..N-1 are generated, and the query seeds are drawn from the tail of that range
    instances = make_instances(per_family=max(QUERY_SEEDS) + 1)
    rows = []
    for family in FAMILIES:
        wanted = {seed for seed in QUERY_SEEDS}
        for instance in instances:
            if instance.family == family.id and instance.seed in wanted:
                rows.append((family.id, instance.ask, SYMPTOMS[family.id]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="recall at 1 for three routers at three library sizes")
    parser.add_argument("--out", type=Path, default=HERE.parent / "fixtures" / "retrieval.json")
    args = parser.parse_args(argv)

    pool = queries()
    result: dict[str, dict[str, dict[str, float]]] = {}
    for scale, (decoys, fillers) in SCALES.items():
        docs = documents(decoys, fillers)
        result[scale] = {
            name: {
                "recall_at_1": round(recall_at_1(router, pool, docs), 3),
                "documents": len(docs),
                "hits": round(recall_at_1(router, pool, docs) * len(pool)),
            }
            for name, router in ROUTERS.items()
        }

    payload = {
        "queries": len(pool),
        "per_family": len(pool) // len(FAMILIES),
        "scales": result,
        "query_includes": "the task text and the failure symptom the previous attempt reported",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = f"{'scale':>8} {'documents':>10} " + " ".join(f"{name:>9}" for name in ROUTERS)
    print(header)
    for scale, rows in result.items():
        print(f"{scale:>8} {rows['keyword']['documents']:>10} " + " ".join(f"{rows[name]['recall_at_1']:>9.3f}" for name in ROUTERS))
    print(f"\n{len(pool)} queries, {payload['per_family']} per family -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
