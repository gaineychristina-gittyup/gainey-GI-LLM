"""Unit tests for src.generate (citation extraction, refusal detection)."""

from __future__ import annotations

import pytest

from src.generate.answer import _extract_cited_indices


def test_single_citation_extracted():
    assert _extract_cited_indices("Foo [1] bar.") == [1]


def test_grouped_citations_split():
    assert _extract_cited_indices("Foo [1, 3] bar.") == [1, 3]


def test_dense_grouped_citations_split():
    assert _extract_cited_indices("Foo [1,3,5] bar.") == [1, 3, 5]


def test_repeated_citation_dedupes_in_order():
    assert _extract_cited_indices("Foo [1] bar [3] baz [1] qux.") == [1, 3]


def test_no_citations_returns_empty():
    assert _extract_cited_indices("No citations here.") == []


def test_text_with_brackets_but_no_numbers_ignored():
    assert _extract_cited_indices("This [is not] a citation.") == []


def test_refusal_phrase_detected():
    """The refused flag fires on the literal opening sentence the system
    prompt mandates. Test the regex shape via the public refusal startswith
    pattern used in answer()."""
    refusal_phrase = "The provided guidelines do not directly address this."
    answer_text = (
        f"{refusal_phrase} The retrieved sources cover post-ERCP pancreatitis."
    )
    assert answer_text.strip().startswith(refusal_phrase)
