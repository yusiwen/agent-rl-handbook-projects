"""Does the retrieved skill change what the model does? Four conditions, one policy, same instances.

    python3 p6-memento-skills/env/measure_outcome.py \
        --base-url http://127.0.0.1:8003 --model qwen3-1.7b-base --out fixtures/outcome.json

The four conditions are the experiment:

    bare             the question alone - the floor
    oracle-skill     the true procedure for that family, handed over without retrieval - the ceiling
    keyword          whatever the keyword router returns, injected
    idf              whatever the idf router returns, injected

Comparing `bare` to `oracle-skill` answers "could the model use this at all"; comparing `bare` to
`idf` answers "does the library help end to end", which is the only version of the claim the
chapter is allowed to make. The gap between the ceiling and the end-to-end number is the price of
retrieval errors, reported rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from p6_memento_skills.families import FAMILIES, make_instances, score  # noqa: E402
from p6_memento_skills.retrieval import ROUTERS, SYMPTOMS, SkillDoc, recall_at_1  # noqa: E402

CONDITIONS = ("bare", "oracle-skill", "keyword", "idf")


def documents() -> list[SkillDoc]:
    return [SkillDoc(name=f.id, family=f.id, trigger=f.title, procedure=f.skill) for f in FAMILIES]


THINK = re.compile(r"<think>.*?</think>", re.S)


class Truncated(Exception):
    """The reply ran out of budget inside its own reasoning; that is a harness failure, not a miss."""


def strip_reasoning(text: str) -> str:
    """Qwen3 writes its thinking first.  The answer is what survives it - and a reply cut off inside
    the thinking block is empty, which is a truncated budget, not a wrong answer."""
    if "<think>" in text and "</think>" not in text:
        raise Truncated("the reply ended inside its reasoning")
    return THINK.sub("", text).strip()


def chat(base_url: str, model: str, system: str, question: str, timeout: int = 120) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": f"{question}\n\nReply with the answer only, nothing else."})
    payload = json.dumps(
        # 96 tokens was the first value: the model spent all of it thinking and the run scored 0 of
        # 20 in every condition.  The budget has to fit the reasoning before it can measure anything.
        {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 2048}
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return strip_reasoning(body["choices"][0]["message"]["content"])


def system_for(condition: str, instance, docs: list[SkillDoc]) -> tuple[str, str | None]:
    """Returns (system prompt, retrieved family or None) for one condition and instance."""
    if condition == "bare":
        return "", None
    if condition == "oracle-skill":
        family = next(f for f in FAMILIES if f.id == instance.family)
        return f"Procedure to follow:\n- {family.skill}", instance.family
    ranked = ROUTERS[condition].rank(instance.ask, SYMPTOMS[instance.family], docs)
    top = ranked[0][0]
    return f"Procedure to follow:\n- {top.procedure}", top.family


def run_one(args, condition: str, instance, docs: list[SkillDoc]) -> dict:
    system, retrieved = system_for(condition, instance, docs)
    try:
        answer = chat(args.base_url, args.model, system, instance.ask)
        error = ""
    except Truncated as exc:
        answer, error = "", f"truncated: {exc}"
    except urllib.error.HTTPError as exc:
        answer, error = "", f"http {exc.code}"
    except Exception as exc:  # noqa: BLE001 - a failed request is data, not a crash
        answer, error = "", f"{type(exc).__name__}: {exc}"
    return {
        "condition": condition,
        "family": instance.family,
        "seed": instance.seed,
        "retrieved": retrieved,
        "solved": (not error) and score(instance, answer),
        "error": error,
        "answer": answer[:120],
        "expected": instance.answer[:120],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="with and without the skill, on the same instances")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="qwen3-1.7b-base")
    parser.add_argument("--instances", type=int, default=20, help="instances per family")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=HERE.parent / "fixtures" / "outcome.json")
    args = parser.parse_args(argv)

    test = make_instances(per_family=args.instances)
    docs = documents()
    rows: list[dict] = []
    for condition in CONDITIONS:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda instance: run_one(args, condition, instance, docs), test))
        solved = sum(1 for row in results if row["solved"])
        errors = sum(1 for row in results if row["error"])
        print(f"{condition:>13}  solved {solved:>3}/{len(results)}  errors {errors:>3}", flush=True)
        rows.extend(results)

    summary = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        per_family = {
            family.id: sum(1 for row in subset if row["family"] == family.id and row["solved"])
            for family in FAMILIES
        }
        summary[condition] = {
            "solved": sum(1 for row in subset if row["solved"]),
            "instances": len(subset),
            "errors": sum(1 for row in subset if row["error"]),
            "per_family": per_family,
        }

    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "instances_per_family": args.instances,
        "conditions": summary,
        "routing": {
            name: round(recall_at_1(ROUTERS[name], [(i.family, i.ask, SYMPTOMS[i.family]) for i in test], docs), 3)
            for name in ("keyword", "idf")
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key]["solved"] for key in CONDITIONS}), flush=True)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
