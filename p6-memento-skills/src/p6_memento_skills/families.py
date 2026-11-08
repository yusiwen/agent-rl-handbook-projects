"""Ten recurring task families, their instances, and the oracle that defines a correct answer.

The recurrence is the experiment. A skill library is only interesting when the same *shape* of
problem comes back with different data, and no public dataset is built to make that happen on
purpose - so the families are generated here, and each one carries a gotcha that a trace can
teach and a prompt cannot guess.

Every family is `ask` (the text a policy sees), `oracle` (the answer the checker compares
against) and `draw` (a seeded instance). Two skills are attached to each family: `skill`, the
procedure that is true, and `decoy`, a plausible procedure that is false in exactly one place.
The validation gate in `library.py` is what tells them apart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable


def _round(text: str | Decimal, places: int) -> str:
    value = Decimal(str(text))
    quantum = Decimal(1).scaleb(-places)
    return str(value.quantize(quantum, rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class Family:
    id: str
    title: str
    skill: str
    decoy: str
    draw: Callable[[random.Random], dict]
    ask: Callable[[dict], str]
    oracle: Callable[[dict], str]


@dataclass(frozen=True)
class Instance:
    family: str
    seed: int
    ask: str
    answer: str


# ---------------------------------------------------------------- 1. sum by group
def _draw_sum_by_group(rng: random.Random) -> dict:
    groups = rng.sample(["north", "south", "east", "west", "central"], k=4)
    rows = [(rng.choice(groups), rng.randrange(10, 400)) for _ in range(rng.randrange(9, 15))]
    rng.shuffle(rows)
    return {"rows": rows}


def _ask_sum_by_group(data: dict) -> str:
    body = "; ".join(f"{group} {amount}" for group, amount in data["rows"])
    return (
        "The rows below are sales by region. Report the total per region. "
        f"Answer as `region=total` pairs separated by `; `, sorted by region name. Rows: {body}."
    )


def _oracle_sum_by_group(data: dict) -> str:
    totals: dict[str, int] = {}
    for group, amount in data["rows"]:
        totals[group] = totals.get(group, 0) + amount
    return "; ".join(f"{group}={totals[group]}" for group in sorted(totals))


# ---------------------------------------------------------------- 2. dedupe, keep last
def _draw_dedupe(rng: random.Random) -> dict:
    ids = [rng.randrange(100, 140) for _ in range(rng.randrange(8, 13))]
    rows = [(f"id-{value}", rng.randrange(1, 9)) for value in ids]
    return {"rows": rows}


def _ask_dedupe(data: dict) -> str:
    body = "; ".join(f"{key} {count}" for key, count in data["rows"])
    return (
        "These records arrive twice with corrected counts. Keep the last record for each id and "
        f"report them in the order the ids first appear, as `id count` pairs separated by `; `. Records: {body}."
    )


def _oracle_dedupe(data: dict) -> str:
    first_seen: list[str] = []
    latest: dict[str, int] = {}
    for key, count in data["rows"]:
        if key not in latest:
            first_seen.append(key)
        latest[key] = count
    return "; ".join(f"{key} {latest[key]}" for key in first_seen)


# ---------------------------------------------------------------- 3. normalise dates
def _draw_dates(rng: random.Random) -> dict:
    month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    stamps = []
    for _ in range(rng.randrange(7, 11)):
        month, day = rng.randrange(1, 13), rng.randrange(1, 29)
        style = rng.choice(["us", "mon", "iso"])
        if style == "us":
            stamps.append(f"{month:02d}/{day:02d}/{rng.randrange(2021, 2025)}")
        elif style == "mon":
            stamps.append(f"{day} {month_names[month - 1].title()} {rng.randrange(21, 25)}")
        else:
            stamps.append(f"{rng.randrange(2021, 2025)}-{month:02d}-{day:02d}")
    return {"stamps": stamps}


def _ask_dates(data: dict) -> str:
    return (
        "These dates come from three different systems. Convert each to ISO `YYYY-MM-DD`, ascending, "
        f"separated by `; `. Dates: {'; '.join(data['stamps'])}."
    )


def _oracle_dates(data: dict) -> str:
    month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    parsed: list[tuple[int, int, int]] = []
    for stamp in data["stamps"]:
        if "/" in stamp:
            month, day, year = (int(part) for part in stamp.split("/"))
        elif "-" in stamp:
            year, month, day = (int(part) for part in stamp.split("-"))
        else:
            day_text, month_text, year_text = stamp.split()
            month, day = month_names.index(month_text.lower()[:3]) + 1, int(day_text)
            year = int(year_text.lstrip("'")) if year_text.startswith("'") else int(year_text)
            year = year + 2000 if year < 100 else year
        parsed.append((year, month, day))
    return "; ".join(f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in sorted(parsed))


# ---------------------------------------------------------------- 4. currency rounding
def _draw_currency(rng: random.Random) -> dict:
    items = [(f"item-{rng.randrange(10, 99)}", f"{rng.randrange(1, 90)}.{rng.randrange(0, 100):02d}") for _ in range(rng.randrange(6, 10))]
    tax = f"{rng.randrange(5, 12)}.{rng.randrange(0, 100):02d}"
    return {"items": items, "tax": tax}


def _ask_currency(data: dict) -> str:
    body = "; ".join(f"{name} {price}" for name, price in data["items"])
    return (
        "Sum the prices, add the tax rate as a percentage of that sum, and report the total to two "
        f"decimals rounded half up. Prices: {body}. Tax percent: {data['tax']}."
    )


def _oracle_currency(data: dict) -> str:
    total = sum(Decimal(price) for _, price in data["items"])
    taxed = total * (Decimal(1) + Decimal(data["tax"]) / Decimal(100))
    return _round(taxed, 2)


# ---------------------------------------------------------------- 5. top-k with ties
def _draw_top_k(rng: random.Random) -> dict:
    names = rng.sample(
        ["ada", "bo", "cy", "dee", "eli", "fay", "gus", "hal", "ivy", "jo", "kai", "lu"], k=7
    )
    rows = [(name, rng.randrange(1, 6)) for name in names]
    return {"rows": rows, "k": rng.randrange(2, 4)}


def _ask_top_k(data: dict) -> str:
    body = "; ".join(f"{name} {score}" for name, score in data["rows"])
    return (
        f"Rank these by score, highest first, and report the top {data['k']}. Break ties by name "
        f"ascending. Report as `name:score` separated by `; `. Entries: {body}."
    )


def _oracle_top_k(data: dict) -> str:
    ranked = sorted(data["rows"], key=lambda row: (-row[1], row[0]))
    return "; ".join(f"{name}:{score}" for name, score in ranked[: data["k"]])


# ---------------------------------------------------------------- 6. pivot counts
def _draw_pivot(rng: random.Random) -> dict:
    rows = [(rng.choice(["a", "b", "c"]), rng.choice(["x", "y"])) for _ in range(rng.randrange(7, 12))]
    return {"rows": rows}


def _ask_pivot(data: dict) -> str:
    body = "; ".join(f"{left}{right}" for left, right in data["rows"])
    return (
        "Count these pairs. Report every combination of the left values (a, b, c) and right values "
        f"(x, y), including pairs that never occur, as `left-right=count` separated by `; ` in that order. Pairs: {body}."
    )


def _oracle_pivot(data: dict) -> str:
    counts: dict[tuple[str, str], int] = {}
    for pair in data["rows"]:
        counts[pair] = counts.get(pair, 0) + 1
    return "; ".join(
        f"{left}-{right}={counts.get((left, right), 0)}"
        for left in ("a", "b", "c")
        for right in ("x", "y")
    )


# ---------------------------------------------------------------- 7. fill forward
def _draw_fill_forward(rng: random.Random) -> dict:
    cells = [str(rng.randrange(10, 99)) if rng.random() > 0.35 else "-" for _ in range(rng.randrange(8, 12))]
    if cells and cells[0] != "-":
        cells[0] = "-" if rng.random() > 0.5 else cells[0]
    return {"cells": cells}


def _ask_fill_forward(data: dict) -> str:
    return (
        "Fill each `-` with the value above it, but a `-` with nothing above it stays `-`. "
        f"Report the sequence separated by `; `. Sequence: {'; '.join(data['cells'])}."
    )


def _oracle_fill_forward(data: dict) -> str:
    filled: list[str] = []
    for cell in data["cells"]:
        if cell == "-" and filled and filled[-1] != "-":
            filled.append(filled[-1])
        else:
            filled.append(cell)
    return "; ".join(filled)


# ---------------------------------------------------------------- 8. unit conversion
def _draw_units(rng: random.Random) -> dict:
    pairs = [(f"leg-{index}", f"{rng.uniform(0.5, 40):.2f}") for index in range(1, rng.randrange(4, 7))]
    return {"pairs": pairs}


def _ask_units(data: dict) -> str:
    body = "; ".join(f"{name} {km}" for name, km in data["pairs"])
    return (
        "Convert kilometres to miles with the factor 0.621371, round to three decimals half up, and "
        f"report `name=miles` in the given order separated by `; `. Distances (km): {body}."
    )


def _oracle_units(data: dict) -> str:
    result = []
    for name, km in data["pairs"]:
        miles = Decimal(km) * Decimal("0.621371")
        result.append(f"{name}={_round(miles, 3)}")
    return "; ".join(result)


# ---------------------------------------------------------------- 9. split with remainder
def _draw_split(rng: random.Random) -> dict:
    return {"total": rng.randrange(1000, 9999), "parts": rng.choice([3, 4, 6])}


def _ask_split(data: dict) -> str:
    return (
        f"Split {data['total']} into {data['parts']} equal parts of whole units. Give the remainder "
        "to the last part. Report the parts separated by `; `, in order."
    )


def _oracle_split(data: dict) -> str:
    share, remainder = divmod(data["total"], data["parts"])
    parts = [share] * (data["parts"] - 1) + [share + remainder]
    return "; ".join(str(part) for part in parts)


# ---------------------------------------------------------------- 10. schema check
def _draw_schema(rng: random.Random) -> dict:
    """One planted violation at a random position, in a random rule, or none at all.

    The first draft planted violations by chance on the first row, so six instances produced two
    distinct answers and the family was a guess rather than a check.
    """
    rows = [
        (rng.choice(["alpha", "beta", "gamma", "delta"]), rng.choice(["2", "3", "4", "5", "8"]))
        for _ in range(rng.randrange(5, 9))
    ]
    plan = rng.choice(["none", "name", "quantity"])
    if plan != "none":
        index = rng.randrange(len(rows))
        name, qty = rows[index]
        if plan == "name":
            rows[index] = (rng.choice(["", "a-very-long-name", "over ten characters"]), qty)
        else:
            rows[index] = (name, rng.choice(["0", "-2", "twelve", "3.5"]))
    return {"rows": rows}


def _ask_schema(data: dict) -> str:
    body = "; ".join(f"[{name or ' '}] {qty}" for name, qty in data["rows"])
    return (
        "Check these records against two rules: name must be 1 to 10 characters, quantity must be a "
        "positive integer written in digits. Report the first violating record as `index:rule` with a "
        f"1-based index, or `ok` if none. Records: {body}."
    )


def _oracle_schema(data: dict) -> str:
    for index, (name, qty) in enumerate(data["rows"], start=1):
        if not (1 <= len(name) <= 10):
            return f"{index}:name"
        if not (qty.isdigit() and int(qty) > 0):
            return f"{index}:quantity"
    return "ok"


FAMILIES: tuple[Family, ...] = (
    Family(
        id="sum-by-group",
        title="sum a value per group",
        skill="Group by the label first, then add; sort the groups by name and join with `=`, not by first appearance.",
        decoy="Add the values in the order they appear and report one running total per group, keeping first-appearance order.",
        draw=_draw_sum_by_group,
        ask=_ask_sum_by_group,
        oracle=_oracle_sum_by_group,
    ),
    Family(
        id="dedupe-keep-last",
        title="collapse duplicate records",
        skill="Walk the records once, remember the latest value per id, and emit ids in first-appearance order.",
        decoy="Keep the first record for each id and sort the output by id, since later records are copies.",
        draw=_draw_dedupe,
        ask=_ask_dedupe,
        oracle=_oracle_dedupe,
    ),
    Family(
        id="normalise-dates",
        title="normalise mixed date formats",
        skill="Detect the format per string: month/day/year, year-month-day, or day Month year, where a two-digit year is 20xx. Sort the parsed dates.",
        decoy="Parse every date as day/month/year and sort the strings, which orders them correctly because the day comes first.",
        draw=_draw_dates,
        ask=_ask_dates,
        oracle=_oracle_dates,
    ),
    Family(
        id="round-currency",
        title="total with tax and rounding",
        skill="Sum in decimal, apply the percentage, and round half up to two places - Python's round() is half even and gives 0.01 less on ties.",
        decoy="Sum in floating point, multiply by one plus the rate, and round with the built-in round() function to two places.",
        draw=_draw_currency,
        ask=_ask_currency,
        oracle=_oracle_currency,
    ),
    Family(
        id="top-k-by",
        title="rank with a tie-break",
        skill="Sort by the score descending and the name ascending in the same key, then take the first k.",
        decoy="Sort by score descending, take the first k, and leave ties in the order they arrived.",
        draw=_draw_top_k,
        ask=_ask_top_k,
        oracle=_oracle_top_k,
    ),
    Family(
        id="pivot-counts",
        title="count every cell of a pair table",
        skill="Enumerate the full grid of left and right values and default the missing pairs to zero; a count table without the empty cells has the wrong shape.",
        decoy="Count the pairs that occur and report them in descending order, which is what a pivot table shows by default.",
        draw=_draw_pivot,
        ask=_ask_pivot,
        oracle=_oracle_pivot,
    ),
    Family(
        id="fill-forward",
        title="carry the last value down",
        skill="Walk once carrying the last non-dash value; a dash before the first real value stays a dash, and the carry never clears.",
        decoy="Replace every dash with the value above it, treating the first entry as a blank that the second value fills.",
        draw=_draw_fill_forward,
        ask=_ask_fill_forward,
        oracle=_oracle_fill_forward,
    ),
    Family(
        id="unit-convert",
        title="convert units with a rounding rule",
        skill="Multiply by the factor given, then round to the requested places with half-up rounding so .0005 goes away from zero.",
        decoy="Multiply by the factor and format the result with two decimals, since distances read better with fewer digits.",
        draw=_draw_units,
        ask=_ask_units,
        oracle=_oracle_units,
    ),
    Family(
        id="split-remainder",
        title="split an integer and place the remainder",
        skill="Use integer division for the share and put the whole remainder on the last part; dividing as a float leaves the parts unequal.",
        decoy="Divide the total by the number of parts and round each part to the nearest whole number so the parts stay equal.",
        draw=_draw_split,
        ask=_ask_split,
        oracle=_oracle_split,
    ),
    Family(
        id="schema-check",
        title="find the first record that breaks a rule",
        skill="Check the rules per record in order and report the 1-based index of the first failure with the rule name; an empty name is a length failure.",
        decoy="Collect every violation and report the count, since a count is more useful than a single index.",
        draw=_draw_schema,
        ask=_ask_schema,
        oracle=_oracle_schema,
    ),
)

FAMILY_BY_ID: dict[str, Family] = {family.id: family for family in FAMILIES}


def make_instances(per_family: int = 20, seed_base: int = 0, families=FAMILIES) -> list[Instance]:
    """Instances are pure functions of (family, seed), so the whole set is reproducible."""
    instances: list[Instance] = []
    for family in families:
        for index in range(per_family):
            seed = seed_base + index
            data = family.draw(random.Random(f"{family.id}:{seed}"))
            instances.append(
                Instance(
                    family=family.id,
                    seed=seed,
                    ask=family.ask(data),
                    answer=family.oracle(data),
                )
            )
    return instances


def score(instance: Instance, answer: str) -> bool:
    """One comparison, applied everywhere: exact text, whitespace collapsed."""
    return " ".join(answer.split()) == " ".join(instance.answer.split())
