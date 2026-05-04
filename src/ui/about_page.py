"""About / methodology page rendered via st.navigation.

Shows the methodology summary, corpus composition, and an honest
limitations list.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
import streamlit as st

from src.ui.components import read_corpus_snapshot, render_corpus_snapshot_header


REPO_ROOT = Path(__file__).resolve().parents[2]


def _corpus_stats() -> dict[str, Any]:
    """Pull live counts from the live DB. Best-effort — falls back to a
    stub if Postgres is unreachable."""
    url = os.environ.get(
        "DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines"
    )
    try:
        with psycopg.connect(url, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            n_docs = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            n_chunks = cur.fetchone()[0]
            cur.execute("""
                SELECT society, count(*), min(year), max(year)
                  FROM documents
                 GROUP BY society
                 ORDER BY society
            """)
            by_society = cur.fetchall()
            cur.execute("SELECT element_type, count(*) FROM chunks GROUP BY element_type")
            by_etype = dict(cur.fetchall())
            cur.execute("SELECT count(DISTINCT topic) FROM documents WHERE topic IS NOT NULL")
            n_topics = cur.fetchone()[0]
        return {
            "n_docs": n_docs,
            "n_chunks": n_chunks,
            "by_society": by_society,
            "by_etype": by_etype,
            "n_topics": n_topics,
        }
    except Exception:
        return {
            "n_docs": None, "n_chunks": None,
            "by_society": [], "by_etype": {}, "n_topics": None,
        }


def render() -> None:
    # set_page_config lives in streamlit_app.py at module top — calling it
    # from inside a page render fails because navigation chrome has already
    # been emitted by the time st.navigation routes to this function.
    render_corpus_snapshot_header(
        title="About this assistant",
        subtitle="Methodology, corpus composition, and limitations.",
    )

    st.header("Methodology")
    st.markdown(
        """
        This assistant retrieves passages from a curated set of clinical
        guidelines published by four U.S. gastroenterology and hepatology
        societies — AGA, ACG, ASGE, and AASLD — and asks Claude (Anthropic's
        Opus 4.7) to answer your question using **only those passages**, with
        explicit citations to each claim. The retrieval pipeline is hybrid
        (dense vector search via pgvector + Postgres full-text BM25) followed
        by a Cohere rerank stage with society-aware diversity floors. Tables
        and figure captions are extracted as standalone retrieval units so
        clinical recommendations stored in summary tables surface alongside
        prose. The model is instructed to refuse cleanly when the retrieved
        excerpts don't address the question, and every answer's citations
        are post-hoc verified by fuzzy-matching the cited sentence against
        the cited passage.
        """
    )

    st.header("Corpus composition")
    stats = _corpus_stats()
    if stats["n_docs"] is None:
        st.warning(
            "Corpus statistics unavailable — the database connection failed. "
            "If you're a developer, run `docker compose ps` to check the "
            "Postgres container."
        )
    else:
        snap = read_corpus_snapshot()
        cols = st.columns(4)
        cols[0].metric("Documents", stats["n_docs"])
        cols[1].metric("Chunks", f"{stats['n_chunks']:,}")
        cols[2].metric("Topics", stats["n_topics"] or "?")
        cols[3].metric("Snapshot", snap)

        st.subheader("By society")
        if stats["by_society"]:
            soc_rows = []
            for soc, count, ymin, ymax in stats["by_society"]:
                yr = f"{ymin}–{ymax}" if ymin != ymax else str(ymin)
                soc_rows.append({"Society": soc, "Documents": count, "Year range": yr})
            st.dataframe(soc_rows, hide_index=True, use_container_width=True)

        st.subheader("Chunk types")
        if stats["by_etype"]:
            etype_rows = [
                {"Element type": k or "(none)", "Count": f"{v:,}"}
                for k, v in sorted(stats["by_etype"].items(), key=lambda x: -x[1])
            ]
            st.dataframe(etype_rows, hide_index=True, use_container_width=True)

    st.header("Limitations")
    st.markdown(
        """
        **What this tool will not do:**

        - **Cover anything outside the indexed guidelines.** Topics not
          represented in the four-society corpus (pediatric GI, surgical
          decision-making, GI manifestations of systemic disease, ACR
          radiology criteria, NCCN oncology, etc.) will produce a refusal.
        - **Replace clinical judgment.** Recommendations are extracted
          verbatim from published society guidelines and may not apply to
          the patient in front of you. Comorbidities, contraindications,
          patient preferences, and local resource constraints are out of
          scope.
        - **Reflect very recent literature.** The corpus snapshot is fixed
          (see header). Guidelines published after that date are not
          included until the next refresh.
        - **Resolve disagreements between societies.** When AGA and ACG
          disagree on a recommendation, the assistant surfaces both and
          will say so. It does not adjudicate which society is correct.
        - **Validate the GRADE evidence.** GRADE strength and certainty
          labels reflect what the guideline panel reported, not the
          underlying truth of the science. A "strong recommendation,
          high-quality evidence" line is faithful to the source — but the
          source itself can be wrong, outdated, or biased.
        - **Handle PHI.** A simple regex screen blocks obvious patterns
          (DOBs, SSNs, MRN-like strings, named patients), but the user is
          responsible for de-identification. This is a research tool, not
          a HIPAA-compliant clinical system.
        """
    )

    st.header("Citation")
    st.markdown(
        """
        If you use this tool in research, please cite:

        > *(citation placeholder — methods paper in preparation)*
        """
    )
