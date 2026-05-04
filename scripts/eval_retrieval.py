#!/usr/bin/env python
"""Regression harness for retrieval + answer quality.

Reads ``tests/eval/questions.yaml``, runs each question through the full
pipeline (``retrieve`` + ``answer``), and reports:

* top-1 hit-rate against expected society/year/recommendation_id
* element-type recall in top-6 (any-of expected types appears)
* refusal accuracy on out-of-corpus questions
* citation-verification pass-rate (any flagged citation = a fail)

By default we run *retrieval-only* checks (cheap, ~30s for the full set).
Pass ``--with-answers`` to also run the LLM and verification (~$0.50 per
full eval at current Opus pricing). The retrieval-only checks already cover
top-1 hit-rate and element-type recall; the full-pipeline mode adds refusal
accuracy + citation verification.

Output: per-question PASS/FAIL line, then a per-society summary table, then
overall numbers. Use this as the regression bar before any chunker /
retrieval / model change.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_QUESTIONS = REPO_ROOT / "tests" / "eval" / "questions.yaml"


def evaluate_retrieval(q: dict, rows: list[dict]) -> dict:
    """Score a single question's retrieval-only result."""
    top1 = rows[0] if rows else {}
    top6 = rows[:6]

    expected_societies = set(q.get("expect_society") or [])
    expected_year_min = q.get("expect_year_min")
    expected_rec_id_substr = (q.get("expect_recommendation_id") or "").lower()
    expected_etypes = set(q.get("expect_element_types") or [])

    society_hit = (
        not expected_societies
        or top1.get("society") in expected_societies
        or any(r.get("society") in expected_societies for r in top6)
    )
    year_hit = (
        expected_year_min is None
        or any(
            (r.get("year") or 0) >= expected_year_min for r in top6
        )
    )
    rec_id_hit = (
        not expected_rec_id_substr
        or any(
            expected_rec_id_substr in (r.get("recommendation_id") or "").lower()
            for r in top6
        )
    )
    etype_hit = (
        not expected_etypes
        or bool(expected_etypes & {r.get("element_type") for r in top6})
    )

    return {
        "society_hit": society_hit,
        "year_hit": year_hit,
        "rec_id_hit": rec_id_hit,
        "etype_hit": etype_hit,
        "top1_society": top1.get("society"),
        "top1_year": top1.get("year"),
        "top1_etype": top1.get("element_type"),
        "n_results": len(rows),
    }


def evaluate_answer(q: dict, out: dict) -> dict:
    """Score answer-side: refusal accuracy + citation verification."""
    in_corpus = q.get("in_corpus", True)
    refused = bool(out.get("refused"))
    verification = out.get("verification") or {}
    return {
        "refusal_correct": (refused == (not in_corpus)),
        "verification_ok": bool(verification.get("ok", True)),
        "n_citations": int(verification.get("n_citations", 0)),
        "n_unsupported": len(verification.get("unsupported") or []),
        "n_out_of_range": len(verification.get("out_of_range") or []),
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--with-answers", action="store_true",
        help="Also run answer() and verification (more expensive).",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--only", type=str, default=None,
        help="Only run questions whose id contains this substring.",
    )
    parser.add_argument(
        "--rate-limit-sleep", type=float, default=6.5,
        help="Seconds to sleep between questions to stay under Cohere's "
             "10/min trial-key rate limit. Set to 0 with a production key.",
    )
    args = parser.parse_args(argv)

    from src.retrieve import retrieve
    if args.with_answers:
        from src.generate import answer

    with open(args.questions) as f:
        questions = yaml.safe_load(f)
    if args.only:
        questions = [q for q in questions if args.only in q["id"]]

    print(f"Running {len(questions)} questions  "
          f"(answers={'on' if args.with_answers else 'off'})")
    print()

    per_society_hits: dict[str, list[bool]] = defaultdict(list)
    overall_retrieval_hits: list[bool] = []
    refusal_correct: list[bool] = []
    verification_pass: list[bool] = []

    errors: list[str] = []
    for i, q in enumerate(questions):
        in_corpus = q.get("in_corpus", True)
        rows = []
        try:
            if in_corpus:
                rows = retrieve(q["question"], top_k=args.top_k)
        except Exception as e:
            errors.append(f"{q['id']}: retrieval crashed: {e!r}")
            print(f"  ERROR  {q['id']}  {type(e).__name__}: {e}")
            continue

        rscore = evaluate_retrieval(q, rows) if in_corpus else {
            "society_hit": True, "year_hit": True, "rec_id_hit": True,
            "etype_hit": True, "n_results": 0,
        }

        ascore = {}
        if args.with_answers:
            try:
                out = answer(q["question"], top_k=args.top_k)
            except Exception as e:
                errors.append(f"{q['id']}: answer() crashed: {e!r}")
                print(f"  ERROR  {q['id']}  answer: {type(e).__name__}: {e}")
                continue
            ascore = evaluate_answer(q, out)
            refusal_correct.append(ascore["refusal_correct"])
            if in_corpus and not out.get("refused"):
                verification_pass.append(ascore["verification_ok"])

        retrieval_pass = (
            rscore["society_hit"] and rscore["year_hit"]
            and rscore["rec_id_hit"] and rscore["etype_hit"]
        )
        overall_retrieval_hits.append(retrieval_pass)
        for s in (q.get("expect_society") or []):
            per_society_hits[s].append(retrieval_pass)

        flag = "PASS" if retrieval_pass else "fail"
        if args.with_answers and not ascore.get("refusal_correct", True):
            flag = "fail (refusal)"
        if args.with_answers and not ascore.get("verification_ok", True):
            flag += " (verify)"

        bits = [flag, q["id"]]
        if in_corpus:
            bits.append(
                f"top1={rscore.get('top1_society')} {rscore.get('top1_year')} "
                f"{rscore.get('top1_etype')}"
            )
        else:
            bits.append("(out-of-corpus)")
            if args.with_answers:
                bits.append(f"refused={ascore.get('refusal_correct')}")
        print("  " + "  ".join(str(b) for b in bits))

        # Stay under Cohere trial-key 10 RPM. Skip the sleep on the last
        # question to keep the eval crisp.
        if args.rate_limit_sleep and i < len(questions) - 1:
            time.sleep(args.rate_limit_sleep)

    print()
    print("=" * 60)
    print("Retrieval pass rate (top-6 hits):")
    print(
        f"  overall: {sum(overall_retrieval_hits)}/{len(overall_retrieval_hits)} "
        f"({100 * sum(overall_retrieval_hits) / max(len(overall_retrieval_hits), 1):.0f}%)"
    )
    for soc, hits in sorted(per_society_hits.items()):
        print(
            f"  {soc:6s}: {sum(hits)}/{len(hits)} "
            f"({100 * sum(hits) / len(hits):.0f}%)"
        )

    if args.with_answers:
        print()
        print("Refusal accuracy (in-corpus + out-of-corpus):")
        print(
            f"  {sum(refusal_correct)}/{len(refusal_correct)} "
            f"({100 * sum(refusal_correct) / max(len(refusal_correct), 1):.0f}%)"
        )
        if verification_pass:
            print()
            print("Citation verification pass rate:")
            print(
                f"  {sum(verification_pass)}/{len(verification_pass)} "
                f"({100 * sum(verification_pass) / len(verification_pass):.0f}%)"
            )

    if errors:
        print()
        print(f"⚠️  {len(errors)} question(s) errored:")
        for e in errors:
            print(f"  - {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
