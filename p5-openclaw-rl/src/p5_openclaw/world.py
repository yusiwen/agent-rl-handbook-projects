"""A tool-calling world that resets in microseconds, so the loop can be about the loop.

ALFWorld and TextWorld are the honest environments for this chapter, and both cost more to stand up
than the measurement needs.  This is a smaller world with the properties the measurement needs and
nothing else:

* **tools are real**: `search` returns document ids, `open` returns text, `calc` evaluates an
  arithmetic expression, `submit` ends the episode.  Nothing is simulated by the harness that the
  model could not have done itself;
* **every task is solvable, and the oracle proves it** (`oracle_episode`), so a zero reward means the
  policy failed rather than the task being broken;
* **reset is free**: the world is a dict of strings, so a rollout costs microseconds plus the model's
  own latency, which is what makes an asynchronous loop worth measuring at all.

The corpus is three quarterly reports and a price list.  The questions need two or three tool calls:
find the report, open it, and sometimes compute a total.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

DOCS: dict[str, str] = {
    "q3-north": (
        "Q3 report, North region. Revenue 412000. Units 8400. Top product: widget. "
        "Growth versus Q2: 11 percent."
    ),
    "q3-south": (
        "Q3 report, South region. Revenue 298500. Units 6100. Top product: gadget. "
        "Growth versus Q2: 4 percent."
    ),
    "q3-east": (
        "Q3 report, East region. Revenue 356750. Units 7200. Top product: widget. "
        "Growth versus Q2: 9 percent."
    ),
    "prices": (
        "Price list. widget 4.50 per unit. gadget 12.00 per unit. gizmo 30.00 per unit. "
        "Discount on orders above 10000 units: 5 percent."
    ),
    "policy": "Returns accepted within 30 days. Shipping paid by the buyer unless the item is faulty.",
}

TITLES = {
    "q3-north": "Q3 report: North",
    "q3-south": "Q3 report: South",
    "q3-east": "Q3 report: East",
    "prices": "Price list",
    "policy": "Returns policy",
}


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    answer: str
    oracle_docs: tuple[str, ...] = ()
    oracle_expression: str = ""
    max_turns: int = 6


@dataclass
class Observation:
    text: str
    done: bool = False
    error: str = ""


@dataclass
class World:
    """One episode.  `history` is what the grader sees and what the model already knows."""

    task: Task
    calls: list[dict] = field(default_factory=list)
    submitted: str | None = None
    errors: int = 0

    def step(self, name: str, args: dict) -> Observation:
        self.calls.append({"tool": name, "args": args})
        if name == "search":
            query = str(args.get("query", "")).lower()
            hits = [doc for doc, text in DOCS.items() if query in (doc + " " + TITLES[doc] + " " + text).lower()]
            if not hits:
                return Observation(text="no documents matched")
            return Observation(text=json.dumps({doc: TITLES[doc] for doc in hits[:3]}))
        if name == "open":
            doc = str(args.get("doc", ""))
            if doc not in DOCS:
                self.errors += 1
                return Observation(text="", error=f"unknown document {doc!r}")
            return Observation(text=DOCS[doc])
        if name == "calc":
            expression = str(args.get("expression", ""))
            if not re.fullmatch(r"[0-9+\-*/(). ]{1,80}", expression):
                self.errors += 1
                return Observation(text="", error="calc accepts digits and + - * / ( ) only")
            try:
                value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - the regex above is the guard
            except Exception as exc:  # noqa: BLE001 - any failure is an observation
                self.errors += 1
                return Observation(text="", error=f"calc failed: {exc}")
            return Observation(text=str(round(float(value), 2)))
        if name == "submit":
            self.submitted = str(args.get("answer", "")).strip()
            return Observation(text="submitted", done=True)
        self.errors += 1
        return Observation(text="", error=f"unknown tool {name!r}")


def solved(world: World) -> bool:
    if world.submitted is None:
        return False
    return normalise(world.submitted) == normalise(world.task.answer)


def normalise(value: str) -> str:
    text = str(value).strip().strip(".").lower()
    text = text.replace(",", "").replace("%", "")
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


#: The six tasks the demonstrations cover.
DEMO_TASKS: tuple[Task, ...] = (
    Task(
        id="north-revenue",
        question="What was the North region's revenue in Q3? Answer with the number of currency units.",
        answer="412000",
        oracle_docs=("q3-north",),
    ),
    Task(
        id="best-region",
        question="Which region had the highest Q3 revenue? Answer with the region name.",
        answer="North",
        oracle_docs=("q3-north", "q3-south", "q3-east"),
    ),
    Task(
        id="widget-order",
        question="What is the price of 1200 widgets? Answer with the number, two decimals.",
        answer="5400.00",
        oracle_docs=("prices",),
        oracle_expression="1200 * 4.50",
    ),
    Task(
        id="units-gap",
        question="How many more units did North sell than South in Q3? Answer with the number.",
        answer="2300",
        oracle_docs=("q3-north", "q3-south"),
        oracle_expression="8400 - 6100",
    ),
    Task(
        id="growth-leader",
        question="Which region grew fastest versus Q2? Answer with the region name.",
        answer="North",
        oracle_docs=("q3-north", "q3-south", "q3-east"),
    ),
    Task(
        id="discounted-order",
        question=(
            "What does 12000 widgets cost after the large-order discount? "
            "Answer with the number, two decimals."
        ),
        answer="51300.00",
        oracle_docs=("prices",),
        oracle_expression="12000 * 4.50 * 0.95",
    ),
)


#: Three harder tasks that no demonstration covers: the answer needs two or three documents and a
#: calculation.  They exist because a loop trained on tasks a cold start already solves has no
#: gradient, and the honest way to say that is to train on a set that includes some it does not.
EXTRA_TASKS: tuple[Task, ...] = (
    Task(
        id="north-minus-east",
        question="By how much did North's Q3 revenue exceed East's? Answer with the number.",
        answer="55250",
        oracle_docs=("q3-north", "q3-east"),
        oracle_expression="412000 - 356750",
    ),
    Task(
        id="total-revenue",
        question="What was the total Q3 revenue across all three regions? Answer with the number.",
        answer="1067250",
        oracle_docs=("q3-north", "q3-south", "q3-east"),
        oracle_expression="412000 + 298500 + 356750",
    ),
    Task(
        id="bulk-discount",
        question="What do 9000 widgets cost after the large-order discount? Answer with two decimals.",
        answer="38475.00",
        oracle_docs=("prices",),
        oracle_expression="9000 * 4.50 * 0.95",
    ),
)

TASKS: tuple[Task, ...] = DEMO_TASKS + EXTRA_TASKS


def oracle_episode(task: Task) -> World:
    """The shortest correct trajectory, computed rather than written.

    Every task states which documents it needs, so the oracle is: open them, optionally compute,
    submit.  Keeping it here (instead of in a fixture) means the task file cannot describe an answer
    the world cannot reach.
    """
    world = World(task=task)
    for doc in task.oracle_docs:
        world.step("open", {"doc": doc})
    if task.oracle_expression:
        world.step("calc", {"expression": task.oracle_expression})
    world.step("submit", {"answer": task.answer})
    return world
