"""Three routers over the same skill documents, so "the router works" is a comparison, not a claim.

The unit of retrieval is a *skill document*: a name, a trigger phrase, and a procedure. The query
is not only the task - it is the task plus the failure the previous attempt produced, because that
is the information a loop actually has and a plain question does not carry.

Three routers, in increasing sophistication, all stdlib:

    keyword  - token overlap (Jaccard) with the query, stopwords removed
    vector   - cosine over hashed character trigrams: a stand-in for an embedding index that costs
               nothing and is deterministic, so the comparison does not depend on a model download
    hybrid   - keyword over the task, plus a weighted share of the failure signature

`SYMPTOMS` is what a failed attempt would say about itself. It is metadata observed at run time,
not part of the family definition - which is why it lives here and not in `families.py`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from had has have if in into is it its of on or that the
    their them then there these they this to was were what when where which who why will with report
    each every below above all any not no than too so such as do does did done give keep use using
    """.split()
)

SYMPTOMS: dict[str, str] = {
    "sum-by-group": "the groups came back in first-appearance order and the separator was a colon",
    "dedupe-keep-last": "the first record was kept instead of the corrected later one",
    "normalise-dates": "the day and month were swapped for the slash dates and the order was wrong",
    "round-currency": "the total was off by one cent on a tie because the rounding was half even",
    "top-k-by": "the tie was settled by arrival order instead of by name",
    "pivot-counts": "the pairs that never occurred were missing from the table",
    "fill-forward": "the leading dash was filled with the first real value",
    "unit-convert": "the distance was formatted with two decimals instead of three",
    "split-remainder": "the parts were rounded so they no longer summed to the total",
    "schema-check": "a violation count was reported instead of the index of the first failure",
}


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def _trigrams(text: str) -> Counter:
    cleaned = re.sub(r"\s+", " ", text.lower())
    return Counter(cleaned[index : index + 3] for index in range(max(0, len(cleaned) - 2)))


@dataclass(frozen=True)
class SkillDoc:
    name: str
    family: str
    trigger: str
    procedure: str

    @property
    def text(self) -> str:
        return f"{self.trigger} {self.procedure}"


@dataclass(frozen=True)
class Route:
    name: str
    weights: dict[str, float] = field(default_factory=dict)

    def rank(self, query: str, symptom: str, docs: list[SkillDoc]) -> list[tuple[SkillDoc, float]]:
        tokens = Counter(tokenize(query))
        symptom_tokens = Counter(tokenize(symptom))
        query_vector = _trigrams(query)
        idf = {token: math.log((1 + len(docs)) / (1 + sum(1 for d in docs if token in tokenize(d.text)))) for token in set(tokens)}
        symptom_vector = _trigrams(symptom)
        scores: list[tuple[SkillDoc, float]] = []
        for doc in docs:
            score = 0.0
            if self.weights.get("keyword"):
                doc_tokens = Counter(tokenize(doc.text))
                union = set(tokens) | set(doc_tokens)
                overlap = sum((tokens & doc_tokens).values()) / len(union) if union else 0.0
                score += self.weights["keyword"] * overlap
            if self.weights.get("vector"):
                doc_vector = _trigrams(doc.text)
                dot = sum(count * doc_vector.get(gram, 0) for gram, count in query_vector.items())
                norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
                doc_norm = math.sqrt(sum(value * value for value in doc_vector.values())) or 1.0
                score += self.weights["vector"] * (dot / (norm * doc_norm))
            if self.weights.get("symptom"):
                doc_tokens = Counter(tokenize(doc.text))
                union = set(symptom_tokens) | set(doc_tokens)
                overlap = sum((symptom_tokens & doc_tokens).values()) / len(union) if union else 0.0
                score += self.weights["symptom"] * overlap
            if self.weights.get("idf"):
                # document frequency over the whole library, so a word every skill shares
                # ("report", "rows", "region") stops deciding the match
                doc_tokens = Counter(tokenize(doc.text))
                matched = set(tokens) & set(doc_tokens)
                weight = sum(idf.get(token, 0.0) for token in matched)
                norm = math.sqrt(sum(idf.get(token, 0.0) for token in doc_tokens)) or 1.0
                score += self.weights["idf"] * (weight / norm)
            if self.weights.get("symptom_vector"):
                doc_vector = _trigrams(doc.text)
                dot = sum(count * doc_vector.get(gram, 0) for gram, count in symptom_vector.items())
                norm = math.sqrt(sum(value * value for value in symptom_vector.values())) or 1.0
                doc_norm = math.sqrt(sum(value * value for value in doc_vector.values())) or 1.0
                score += self.weights["symptom_vector"] * (dot / (norm * doc_norm))
            scores.append((doc, score))
        return sorted(scores, key=lambda pair: (-pair[1], pair[0].name))


ROUTERS: dict[str, Route] = {
    "keyword": Route("keyword", {"keyword": 1.0}),
    "vector": Route("vector", {"vector": 1.0}),
    "hybrid": Route("hybrid", {"keyword": 0.35, "vector": 0.30, "symptom": 0.20, "symptom_vector": 0.15}),
    # added after the first three collapsed under on-topic decoys: a word's rarity across the
    # library is the cheapest signal that separates a procedure from a restatement of the task
    "idf": Route("idf", {"idf": 0.75, "symptom": 0.25}),
}

FILLER_WORDS = (
    "table column value row file path name source target number string list record field format "
    "count order group label check convert split fill rank filter merge append escape encode decode "
    "normalise validate compare truncate expand collect summarise select sort join dedupe rename map"
)


def filler_docs(count: int, seed: int = 0) -> list[SkillDoc]:
    """Near-miss skills: plausible procedures about data handling that fix the wrong thing.

    They exist to answer "what happens to retrieval when there are a hundred skills rather than
    five" without inventing a second, friendlier task set.
    """
    docs: list[SkillDoc] = []
    words = FILLER_WORDS.split()
    for index in range(count):
        offset = (seed + index * 7) % len(words)
        window = [words[(offset + step * 3) % len(words)] for step in range(9)]
        trigger = " ".join(window[:4])
        procedure = f"When the {window[4]} does not match, rewrite the {window[5]} before touching the {window[6]}."
        docs.append(SkillDoc(name=f"filler-{index:03d}", family="filler", trigger=trigger, procedure=procedure))
    return docs


def recall_at_1(route: Route, queries: list[tuple[str, str, str]], docs: list[SkillDoc]) -> float:
    """queries are (family, ask, symptom); a hit is a top-ranked document from the same family."""
    if not queries:
        return 0.0
    hits = 0
    for family, ask, symptom in queries:
        ranked = route.rank(ask, symptom, docs)
        if ranked and ranked[0][0].family == family:
            hits += 1
    return hits / len(queries)
