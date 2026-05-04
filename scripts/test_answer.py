#!/usr/bin/env python
"""Grounded answer probe — Phase 4 MVP.

  python scripts/test_answer.py "what's the recommended H. pylori regimen for a penicillin-allergic patient?"

Pulls top-k chunks via the Phase 3 hybrid retriever, hands them to Claude
with the strict-grounding system prompt from src/generate/prompt.py, and
prints the cited answer plus a citations footer with society/year/title/
page span/recommendation id/GRADE for each [N] tag the model used.

Filter flags mirror scripts/test_retrieval.py:

    --society ACG --year-min 2023 --topic "h. pylori"
    --element-types recommendation,table

Other flags:

    --top-k N        override config.yaml retrieval.top_k
    --no-rerank      skip Cohere rerank, take RRF top-k directly
    --show-sources   echo the SOURCES block sent to the model (debug)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src._env import clear_empty_creds  # noqa: E402


def _csv_list(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _format_citation(c: dict) -> str:
    bits = [f"[{c['n']}]", f"{c.get('society') or '?'} {c.get('year') or '?'}"]
    if (rid := c.get("recommendation_id")):
        bits.append(rid)
    gs = c.get("grade_strength")
    ge = c.get("grade_evidence")
    if gs or ge:
        bits.append(f"GRADE: {' / '.join(x for x in [gs, ge] if x)}")
    ps, pe = c.get("page_start"), c.get("page_end")
    if ps and pe and pe != ps:
        bits.append(f"pp {ps}-{pe}")
    elif ps:
        bits.append(f"p {ps}")
    if (etype := c.get("element_type")) and etype != "prose":
        bits.append(f"({etype})")
    head = "  ".join(bits)
    title = (c.get("title") or "").strip()
    if title:
        title = title if len(title) <= 110 else title[:107] + "..."
        head = f"{head}\n      {title}"
    return head


def main(argv: list[str] | None = None) -> int:
    clear_empty_creds()
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="+")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--show-sources", action="store_true",
                        help="Print the SOURCES block sent to Claude (for debugging).")

    parser.add_argument("--society", type=str, default=None)
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--element-types", type=str, default=None)
    args = parser.parse_args(argv)

    from src.generate import answer
    from src.generate.prompt import build_user_message

    question = " ".join(args.question)
    filters: dict = {}
    if (s := _csv_list(args.society)):
        filters["society"] = [x.upper() for x in s]
    if args.year_min is not None:
        filters["year_min"] = args.year_min
    if args.year_max is not None:
        filters["year_max"] = args.year_max
    if (t := _csv_list(args.topic)):
        filters["topic"] = t
    if (e := _csv_list(args.element_types)):
        filters["element_types"] = e

    print(f"Q: {question}")
    if filters:
        print(f"Filters: {filters}")
    print()

    out = answer(
        question,
        filters=filters or None,
        top_k=args.top_k,
        rerank=not args.no_rerank,
    )

    if args.show_sources:
        print("=" * 88)
        print("SOURCES sent to model:")
        print("=" * 88)
        print(build_user_message(question, out["chunks"]))
        print("=" * 88)
        print()

    print("ANSWER")
    print("-" * 88)
    print(out["answer"])
    print()

    print("CITATIONS")
    print("-" * 88)
    if out["citations"]:
        for c in out["citations"]:
            print(_format_citation(c))
            print()
    else:
        print("(model did not cite any sources)")
        print()

    print("USAGE")
    print("-" * 88)
    u = out["usage"]
    print(
        f"model={out['model']}  "
        f"in={u.get('input_tokens')}  "
        f"out={u.get('output_tokens')}  "
        f"cache_create={u.get('cache_creation_input_tokens')}  "
        f"cache_read={u.get('cache_read_input_tokens')}  "
        f"refused={out['refused']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
