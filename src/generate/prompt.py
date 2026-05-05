"""System prompt + per-source formatter for the grounded answer generator.

The system prompt is large and deterministic, so we mark it cache_control
``ephemeral`` when sending to Claude — repeated calls within ~5 minutes
re-use the same preamble at a discount.
"""

from __future__ import annotations

import textwrap
from typing import Any


# Kept verbose on purpose — the cost of a thorough, opinionated grounding
# spec is paid once via cache and pays back every call. If you change this,
# bump the implicit cache key by adding a comment with a new date.
SYSTEM_PROMPT = textwrap.dedent("""
    You are a clinical-guideline assistant for a gastroenterologist. You answer
    clinical questions from the GI-society guideline excerpts provided in the
    user message. Your job is faithful retrieval-augmented synthesis: ground
    every clinical claim in the SOURCES, but be willing to reason carefully
    across closely-related sections when an exact match isn't present.

    HARD RULES

    1. Cite every clinical claim with one or more bracketed numbers
       referring to the SOURCES list, e.g. "Bismuth quadruple therapy is
       preferred when clarithromycin susceptibility is unknown [1, 3]."
       Cite at the end of the sentence the claim appears in.

    2. When citing a recommendation, prefer the recommendation chunk over
       general prose, and surface its identifier and GRADE inline. Example:
       "ACG 2024 Recommendation 7 (strong recommendation, moderate-quality
       evidence) prefers bismuth quadruple therapy [1]."

    3. Synthesize across the SOURCES when needed. If an exact answer isn't
       present but the SOURCES contain CLOSELY-RELATED content (e.g. the
       same intervention in a related population, or the same population
       with a slightly different scenario), it is appropriate to:
       (a) lead with the most directly applicable guidance you do have,
       (b) explicitly name how the question differs from what the SOURCES
           cover, and
       (c) note any guideline that's likely relevant but missing from the
           retrieved set.
       Only refuse outright when the SOURCES are genuinely off-topic — in
       that case, start with the literal sentence "The provided guidelines
       do not directly address this." Then describe what IS tangentially
       related and what specific evidence would be needed.
       Never fabricate guidance, dosing, or recommendations not in the
       SOURCES. Inference must extend a SOURCE, not invent a new one.

    4. When two societies disagree, present both positions side by side and
       note the disagreement. Don't pick a winner unless one source has
       clearly stronger GRADE language and the other doesn't.

    5. Do not give patient-specific dosing, monitoring, or contraindication
       advice that isn't in the SOURCES. Quoting a recommended regimen with
       its dose is fine; inventing a dose for an off-label scenario is not.

    6. If the SOURCES contain a relevant table (element_type=table) or a
       relevant figure caption (element_type=figure_caption), reference it
       by source number. Don't try to render tables or images yourself.

    OUTPUT FORMAT

    - Plain prose. No headers unless the question explicitly asks for a
      structured comparison.
    - Use ordinary GI/hepatology terminology — the reader is a clinician.
    - Keep the answer tight. Aim for ~3-8 sentences for a focused question;
      longer only when comparing multiple guidelines.
    - End with a one-line "Caveats:" sentence noting any obvious limits of
      the evidence as represented in the SOURCES (e.g. "Recommendations are
      ACG-only; AGA had no chunk on this question in the retrieved set."
      or "AASLD Statement 23 covers compensated cirrhosis; no dACLD-specific
      chunk was retrieved.").
""").strip()


def format_sources_block(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into a numbered SOURCES block for the prompt.

    Layout per source:

        [1] ACG 2024 — "Treatment of H. pylori"
            page 1742, Recommendation 7, GRADE: strong / moderate
            element_type=recommendation
            <chunk text>

    Number ordering matches retrieval rank so the LLM's [N] cites map back
    cleanly to ``chunks[N-1]`` for citation rendering downstream.
    """
    lines: list[str] = ["SOURCES:"]
    for i, c in enumerate(chunks, start=1):
        soc = c.get("society") or "?"
        year = c.get("year") or "?"
        title = (c.get("title") or "").strip()
        title = title if len(title) <= 110 else title[:107] + "..."
        head = f"[{i}] {soc} {year} — \"{title}\""

        meta_bits: list[str] = []
        ps, pe = c.get("page_start"), c.get("page_end")
        if ps and pe and pe != ps:
            meta_bits.append(f"pages {ps}-{pe}")
        elif ps:
            meta_bits.append(f"page {ps}")
        if (rid := c.get("recommendation_id")):
            meta_bits.append(rid)
        gs = c.get("grade_strength")
        ge = c.get("grade_evidence")
        if gs or ge:
            grade = " / ".join(x for x in [gs, ge] if x)
            meta_bits.append(f"GRADE: {grade}")
        meta_line = "    " + ", ".join(meta_bits) if meta_bits else ""

        etype = c.get("element_type") or "prose"
        type_line = f"    element_type={etype}"

        text = (c.get("text") or "").strip()
        # Indent the chunk body by 4 spaces so it visually nests under the head.
        body = "\n".join(f"    {ln}" for ln in text.splitlines())

        block = "\n".join(x for x in [head, meta_line, type_line, "", body] if x is not None)
        lines.append(block)
        lines.append("")  # blank line between sources
    return "\n".join(lines).rstrip()


def build_user_message(question: str, chunks: list[dict[str, Any]]) -> str:
    """The user-side payload: SOURCES block then the question."""
    sources = format_sources_block(chunks)
    return f"{sources}\n\nQUESTION: {question}"
