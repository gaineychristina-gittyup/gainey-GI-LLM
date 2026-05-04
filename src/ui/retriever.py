"""Retriever facade for the UI.

Phases 3 and 4 are not built yet. This module exposes the contract the UI
expects (`answer(question) -> (answer_text, citations)`) and ships with a
stub backend so the UI is runnable today. When the real retrieval +
generation pipelines land, swap `_stub_answer` for the production call.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    n: int
    society: str
    title: str
    year: int
    recommendation_id: str | None
    grade_evidence: str | None
    grade_strength: str | None
    page_start: int | None
    page_end: int | None
    text: str

    @property
    def short_label(self) -> str:
        rec = f", {self.recommendation_id}" if self.recommendation_id else ""
        return f"{self.society} {self.year}{rec}"


_STUB_CITATIONS: list[Citation] = [
    Citation(
        n=1,
        society="AGA",
        title="AGA Clinical Practice Update on Gastroparesis",
        year=2025,
        recommendation_id="Best Practice Advice 4",
        grade_evidence="moderate",
        grade_strength="conditional",
        page_start=6,
        page_end=7,
        text=(
            "Gastric scintigraphy performed over 4 hours with a low-fat, "
            "egg-white meal remains the reference standard for diagnosing "
            "gastroparesis. Studies of shorter duration significantly "
            "underestimate the prevalence of delayed emptying."
        ),
    ),
    Citation(
        n=2,
        society="ACG",
        title="ACG Clinical Guideline: Gastroparesis",
        year=2022,
        recommendation_id="Recommendation 7",
        grade_evidence="low",
        grade_strength="conditional",
        page_start=12,
        page_end=12,
        text=(
            "We suggest metoclopramide as first-line prokinetic therapy for "
            "gastroparesis, with the lowest effective dose for the shortest "
            "duration given the risk of tardive dyskinesia."
        ),
    ),
    Citation(
        n=3,
        society="ACG",
        title="ACG Clinical Guideline: Gastroparesis",
        year=2022,
        recommendation_id="Recommendation 12",
        grade_evidence="moderate",
        grade_strength="strong",
        page_start=15,
        page_end=16,
        text=(
            "Dietary modification with small, frequent, low-fat, low-fiber "
            "meals is recommended as initial management for symptomatic "
            "gastroparesis."
        ),
    ),
]


_STUB_ANSWER = (
    "For suspected gastroparesis, current society guidance recommends "
    "confirming delayed gastric emptying with 4-hour scintigraphy [1]. "
    "First-line management combines dietary modification with small, "
    "frequent, low-fat, low-fiber meals [3] and, when symptoms persist, "
    "a time-limited trial of metoclopramide at the lowest effective dose [2]."
)


def answer(question: str) -> tuple[str, list[Citation]]:
    """Return an answer plus its supporting citations.

    Replace the body with the Phase 3+4 pipeline once it lands. The UI only
    depends on this signature.
    """
    if not question.strip():
        return ("", [])
    return _stub_answer(question)


def _stub_answer(_question: str) -> tuple[str, list[Citation]]:
    return _STUB_ANSWER, _STUB_CITATIONS
