"""Decontamination: is anything in the task set also in the evaluation set?

Chapter 7's fourth row.  A policy that has seen an evaluation item scores its memory, not
its ability, and the leak is usually textual rather than deliberate - the same problem
reworded, or a line copied while a task family was being written.  The check is cheap, so
there is no excuse for judging it by eye.

Two measures, both explainable:

* **13-gram containment decides.**  Normalise both texts, take every run of 13 consecutive
  words, and ask how many of *this task's* runs also occur in an evaluation item.  Thirteen
  is the span the common web-crawl decontamination pass uses: shorter runs match ordinary
  English by accident, and longer ones miss a problem that was reworded.  One shared run is
  a thirteen-word verbatim span, which is not a coincidence.
* **Character n-gram similarity is reported, never decides.**  A short item has no 13 words
  to match, so the report needs a second number to say whether the item was even comparable.
  It is not a verdict: `What is 17 * 23?` and `What is 17 + 23?` share most of their
  characters and are different problems - which is why a similarity threshold makes a bad
  gate, failing in both directions.

What it does not do, in its own words:

* It does not decide whether a task was *derived* from an evaluation item.  Absence of a
  shared run is evidence about wording, not about provenance.
* It does not find paraphrases it was not given, and it does not read the answer side.
* It does not fold number words into digits: ``sixteen eggs`` and ``16 eggs`` normalise
  differently.  With a 13-word span one such difference removes only the runs covering it, so
  a long copy is still caught, but a short item can slip through - which is why the similarity
  column is reported beside every verdict and why "clean" is not a claim about provenance.
* It compares against the file it was handed.  That file's digest travels with the report,
  because "we checked" has to be a statement about specific bytes.

Cost: one word-run intersection per (task, evaluation item) pair - about 28,000 of them for
21 tasks against a 1,319-row benchmark, a second of work.  Character similarity is computed
only where it is reported: for items too short to have runs, and for the closest few
candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

#: Words per run.  The decontamination span described in the module docstring.
DEFAULT_NGRAM = 13

#: Characters per run for the reported (non-deciding) similarity measure.
DEFAULT_CHAR_NGRAM = 5

#: How far two texts may differ in length and still be worth scoring for similarity.  A
#: 10-word prompt and a 300-word story are not near-duplicates of anything - but note that
#: this filter never touches containment, because a copied sentence sits inside a longer
#: problem just as happily as inside a short one.
LENGTH_RATIO = 3.0

#: How many of the closest candidates have their similarity reported.
CANDIDATES = 5

#: Below this many 13-word runs, the word measure is shallow: a task this short is one
#: reworded word away from sharing nothing.  Such a verdict is reported with a note rather
#: than quietly presented as evidence - the fixture's own prompts are 13 to 41 words, so this
#: case is the common one, not the corner one.
SHALLOW_RUNS = 5

#: Markup that is the author's, not the problem's: maths delimiters, markdown, commands.  Two
#: copies of one problem must not read as different because one was written in LaTeX.
_MARKUP = (
    (re.compile(r"\\[a-z]+", re.IGNORECASE), " "),
    (re.compile(r"[`*_~^#>|$\\{}()\[\]]+"), " "),
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def normalise(text: str) -> str:
    """Lowercase, strip markup, and split into word tokens.

    Everything that is not a letter or a digit becomes a separator, so ``17*23`` and
    ``17 * 23`` normalise alike, and so do ``$17 \\times 23$`` and ``17 times 23``.
    """
    lowered = text.lower()
    for pattern, replacement in _MARKUP:
        lowered = pattern.sub(replacement, lowered)
    return " ".join(_TOKEN.findall(lowered))


def word_runs(text: str, n: int = DEFAULT_NGRAM) -> set[tuple[str, ...]]:
    """Every run of ``n`` consecutive words in the normalised text."""
    words = normalise(text).split()
    if n <= 0 or len(words) < n:
        return set()
    return {tuple(words[start : start + n]) for start in range(len(words) - n + 1)}


def char_grams(text: str, k: int = DEFAULT_CHAR_NGRAM) -> set[str]:
    """Every run of ``k`` characters in the normalised text, separators included."""
    flat = normalise(text)
    if not flat:
        return set()
    if len(flat) < k:
        return {flat}
    return {flat[start : start + k] for start in range(len(flat) - k + 1)}


def jaccard(left: set, right: set) -> float:
    """Intersection over union; 0.0 when either side is empty."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class EvalItem:
    """One evaluation item, prepared once and reused for every task."""

    item_id: str
    runs: set
    length: int
    grams: set


@dataclass(frozen=True)
class TaskComparison:
    """One task's verdict, with the numbers that produced it."""

    task_id: str
    words: int
    runs: int
    shared_runs: int
    containment: float
    similarity: float
    best_match: str | None
    verdict: str
    note: str = ""

    @property
    def contaminated(self) -> bool:
        return self.verdict == "contaminated"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "words": self.words,
            "runs": self.runs,
            "shared_runs": self.shared_runs,
            "containment": round(self.containment, 4),
            "similarity": round(self.similarity, 4),
            "best_match": self.best_match,
            "verdict": self.verdict,
            "note": self.note,
        }

    def render(self) -> str:
        line = (
            f"{self.task_id:>7} {self.words:>4}w runs={self.runs:<4} shared={self.shared_runs:<3} "
            f"containment={self.containment:>4.0%} similarity={self.similarity:.2f}  "
            f"{self.verdict:<12} best={self.best_match or '-'}"
        )
        return f"{line}\n         {self.note}" if self.note else line


@dataclass
class ContaminationReport:
    """The check's answer: per task, plus the file it was compared against."""

    comparisons: list[TaskComparison] = field(default_factory=list)
    ngram: int = DEFAULT_NGRAM
    eval_source: str = "unknown"
    eval_digest: str = "unknown"
    eval_size: int = 0

    @property
    def contaminated(self) -> list[TaskComparison]:
        return [c for c in self.comparisons if c.contaminated]

    @property
    def undecided(self) -> list[TaskComparison]:
        return [c for c in self.comparisons if c.verdict == "undecided"]

    @property
    def shallow(self) -> list[TaskComparison]:
        """Clean verdicts reached on fewer than SHALLOW_RUNS runs: clean, but not strong."""
        return [
            c
            for c in self.comparisons
            if c.verdict == "clean" and 0 < c.runs < SHALLOW_RUNS
        ]

    @property
    def ok(self) -> bool:
        """True when nothing verifiably overlaps.  "Undecided" is not a pass - see render()."""
        return not self.contaminated

    def to_dict(self) -> dict:
        return {
            "ngram": self.ngram,
            "eval_source": self.eval_source,
            "eval_digest": self.eval_digest,
            "eval_size": self.eval_size,
            "tasks": len(self.comparisons),
            "contaminated": [c.task_id for c in self.contaminated],
            "undecided": [c.task_id for c in self.undecided],
            "shallow": [c.task_id for c in self.shallow],
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    def render(self) -> str:
        lines = [
            f"contamination check: {len(self.comparisons)} task(s) against {self.eval_source} "
            f"({self.eval_digest}, {self.eval_size} item(s)), {self.ngram}-word runs",
            f"contaminated: {len(self.contaminated)}  undecided: {len(self.undecided)}",
            "",
        ]
        lines.extend(comparison.render() for comparison in self.comparisons)
        if self.undecided:
            lines += [
                "",
                f"{len(self.undecided)} undecided: too short for {self.ngram}-word runs, so the "
                "word measure cannot speak.  That is size, not a pass - a short item is exactly "
                "what a hand-copied leak looks like.",
            ]
        if self.shallow:
            lines += [
                "",
                f"{len(self.shallow)} of {len(self.comparisons)} clean verdict(s) rest on fewer "
                f"than {SHALLOW_RUNS} run(s) ({', '.join(c.task_id for c in self.shallow[:8])}"
                + (" ..." if len(self.shallow) > 8 else ")")
                + ": a one-word rewrite would leave nothing to match.  Read their similarity column.",
            ]
        lines += ["", "verdict: " + self.verdict_line()]
        return "\n".join(lines)

    def verdict_line(self) -> str:
        """The verdict has to distinguish "nothing overlaps" from "nothing was comparable"."""
        if self.contaminated:
            return (
                f"{len(self.contaminated)} task(s) share a {self.ngram}-word run with the "
                "evaluation set"
            )
        if self.undecided and len(self.undecided) == len(self.comparisons):
            return (
                f"nothing comparable: every one of the {len(self.comparisons)} task(s) is shorter "
                f"than a {self.ngram}-word run"
            )
        if self.undecided:
            comparable = len(self.comparisons) - len(self.undecided)
            return (
                f"nothing overlaps among the {comparable} task(s) long enough to compare; "
                f"{len(self.undecided)} undecided"
            )
        return "nothing overlaps"


def compare(
    tasks: Sequence[tuple[str, str]],
    evaluations: Sequence[tuple[str, str]],
    *,
    ngram: int = DEFAULT_NGRAM,
    eval_source: str = "unknown",
    eval_digest: str = "unknown",
) -> ContaminationReport:
    """Compare every task against every evaluation item.

    ``tasks`` and ``evaluations`` are sequences of ``(id, text)``.  A task's verdict comes
    from the evaluation item with which it shares the most ``ngram``-word runs; ties are
    broken by character similarity, which is reported but never decides.
    """
    if not evaluations:
        raise ValueError(
            "the evaluation set is empty: a contamination check against nothing would report "
            "clean, which is the one answer it must never give for that reason"
        )

    prepared = [
        EvalItem(item_id, word_runs(text, ngram), len(normalise(text)), char_grams(text))
        for item_id, text in evaluations
    ]

    report = ContaminationReport(
        ngram=ngram, eval_source=eval_source, eval_digest=eval_digest, eval_size=len(prepared)
    )

    for task_id, text in tasks:
        normalised = normalise(text)
        words = len(normalised.split())
        task_runs = word_runs(text, ngram)

        if task_runs:
            # Containment decides, and it looks at every pair: the copied sentence may sit
            # inside a much longer problem, so length is not allowed to filter here.
            scored = sorted(
                ((len(task_runs & item.runs), item.item_id, item) for item in prepared),
                key=lambda entry: (-entry[0], entry[1]),
            )
            shared_runs = scored[0][0]
            task_grams = char_grams(text)
            candidates = scored[:CANDIDATES] if shared_runs else [
                (shared, item_id, item)
                for shared, item_id, item in scored
                if LENGTH_RATIO >= (len(normalised) + 1) / (item.length + 1) >= 1 / LENGTH_RATIO
            ]
            similarity = max(
                (jaccard(task_grams, item.grams) for _shared, _item_id, item in candidates),
                default=0.0,
            )
            note = ""
            if not shared_runs and len(task_runs) < SHALLOW_RUNS:
                note = (
                    f"clean on {len(task_runs)} run(s): the word measure is shallow at this length "
                    f"(fewer than {SHALLOW_RUNS}), so a one-word rewrite could hide a copy.  Closest "
                    f"evaluation item by character {DEFAULT_CHAR_NGRAM}-grams: {similarity:.2f}"
                )
            report.comparisons.append(
                TaskComparison(
                    task_id=task_id,
                    words=words,
                    runs=len(task_runs),
                    shared_runs=shared_runs,
                    containment=shared_runs / len(task_runs),
                    similarity=similarity,
                    # A tie-break on zero is not a match: naming an item here would dress a
                    # coincidence up as a finding.
                    best_match=scored[0][1] if shared_runs else None,
                    verdict="contaminated" if shared_runs else "clean",
                    note=note,
                )
            )
            continue

        # No runs: the item is shorter than the span, so the only honest verdict is that the
        # word measure could not speak about it.
        task_grams = char_grams(text)
        best_similarity = 0.0
        best_similarity_id: str | None = None
        for item in prepared:
            similarity = jaccard(task_grams, item.grams)
            if similarity > best_similarity:
                best_similarity, best_similarity_id = similarity, item.item_id
        report.comparisons.append(
            TaskComparison(
                task_id=task_id,
                words=words,
                runs=0,
                shared_runs=0,
                containment=0.0,
                similarity=best_similarity,
                best_match=best_similarity_id,
                verdict="undecided",
                note=(
                    f"only {words} word(s): no {ngram}-word run exists to compare.  Closest "
                    f"evaluation item by character {DEFAULT_CHAR_NGRAM}-grams: {best_similarity:.2f}"
                    + (f" ({best_similarity_id})" if best_similarity_id else "")
                ),
            )
        )

    return report
