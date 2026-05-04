"""Unit tests for src.generate.verify."""

from __future__ import annotations

import pytest

from src.generate.verify import verify_answer


CHUNKS = [
    {
        "chunk_id": 1,
        "society": "ACG", "year": 2022,
        "title": "ACG Barrett's",
        "element_type": "table",
        "text": (
            "Baseline endoscopic finding | Suggested endoscopic surveillance "
            "Nondysplastic BE of <3 cm length | EGD every 5 yr "
            "Nondysplastic BE of >=3 cm length | EGD every 3 yr "
            "Confirmed LGD | EGD at 6 months from diagnosis, again at 12 months, "
            "and annually thereafter."
        ),
    },
    {
        "chunk_id": 2,
        "society": "AGA", "year": 2025,
        "title": "AGA Gastroparesis",
        "element_type": "recommendation",
        "text": (
            "Recommendation 2: We suggest metoclopramide over no metoclopramide "
            "in patients with gastroparesis. Conditional recommendation, very "
            "low certainty of evidence."
        ),
    },
]


def test_valid_citation_in_range_supported_passes():
    text = (
        "ACG 2022 recommends EGD at 6 months, again at 12 months, "
        "and annually thereafter for confirmed LGD [1]."
    )
    res = verify_answer(text, CHUNKS)
    assert res["ok"], res
    assert res["n_citations"] == 1


def test_out_of_range_citation_flagged():
    text = "AGA 2025 suggests metoclopramide [99]."
    res = verify_answer(text, CHUNKS)
    assert not res["ok"]
    assert 99 in res["out_of_range"]


def test_unsupported_claim_flagged():
    """A citation pointing to chunk 2 (gastroparesis), but the sentence
    talks about appendicitis — the partial-ratio match should fail."""
    text = (
        "We recommend laparoscopic appendectomy within 24 hours of presentation "
        "for uncomplicated appendicitis [2]."
    )
    res = verify_answer(text, CHUNKS, min_partial_ratio=80)
    assert not res["ok"]
    assert any(u["n"] == 2 for u in res["unsupported"])


def test_multiple_citations_grouped():
    """The [1, 2] form should be parsed and validated for each index."""
    text = "Recommendations span Barrett's [1] and gastroparesis [1, 2]."
    res = verify_answer(text, CHUNKS)
    # 3 citations counted (1, 1, 2). Both indices are in range; chunk 2 may
    # not match the sentence claim closely, so we just check counting here.
    assert res["n_citations"] == 3


def test_no_citations_returns_ok():
    text = "This is a refusal sentence with no citations."
    res = verify_answer(text, CHUNKS)
    assert res["ok"]
    assert res["n_citations"] == 0
    assert res["out_of_range"] == []
    assert res["unsupported"] == []
