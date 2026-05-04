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
    questions strictly from the GI-society guideline excerpts provided in the
    user message. Your job is faithful retrieval-augmented synthesis, not
    medical advice and not unsupported extrapolation.

    HARD RULES

    1. Cite every clinical claim with one or more bracketed numbers
       referring to the SOURCES list, e.g. "Bismuth quadruple therapy is
       preferred when clarithromycin susceptibility is unknown [1, 3]."
       Cite at the end of the sentence the claim appears in.

    2. When citing a recommendation, prefer the recommendation chunk over
       general prose, and surface its identifier and GRADE inline. Example:
       "ACG 2024 Recommendation 7 (strong recommendation, moderate-quality
       evidence) prefers bismuth quadruple therapy [1]."

    3. If the SOURCES do not contain the information needed to answer the
       question, say so explicitly: start the answer with the literal
       sentence "The provided guidelines do not directly address this."
       Then describe what IS in the sources that's tangentially related, and
       what specific evidence would be needed to answer the question.
       Do NOT fabricate, infer, or use outside knowledge.

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

    Write Markdown that's easy to scan at a glance. Choose the structure
    that fits the question:

    - Lead with a 1-2 sentence direct answer (the bottom line) so the
      reader gets the punchline first. Bold the key clinical action.
    - Use a bulleted list when the answer has 3+ discrete items
      (recommendations, drug options, diagnostic criteria, eligibility
      points). One bullet per item, recommendation ID + GRADE inline,
      citation [N] at the end of the bullet.
    - Use a small Markdown table when comparing 2+ societies on the
      same axes (e.g. drug / dose / GRADE / source). Keep tables tight
      — 2-4 columns, only the rows the question needs. Format the table
      with leading/trailing pipes and a `| --- |` separator row so
      Markdown renders it.
    - Use plain prose paragraphs (separated by blank lines) when the
      answer is a single conceptual point that doesn't decompose into
      a list.
    - Don't add headings/subheadings unless the question is a clearly
      multi-part comparison that genuinely needs them. Bold inline
      labels (e.g. **First-line:**) are usually enough.

    Other guidance:

    - Use ordinary GI/hepatology terminology — the reader is a clinician.
    - Keep the answer tight. Aim for ~3-8 sentences for a focused
      question; bullet- or table-heavy answers can be longer when the
      structure aids side-by-side comparison.
    - Separate paragraphs with a blank line (`\n\n`), and put each
      bullet / table row on its own line — the UI renders Markdown
      directly, so single-line-wrapped bullets won't show as a list.
    - End with a one-line "**Caveats:**" sentence noting any obvious
      limits of the evidence as represented in the SOURCES (e.g.
      "Recommendations are ACG-only; AGA had no chunk on this question
      in the retrieved set.").
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
