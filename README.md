# GI Guidelines RAG

A research-grade retrieval-augmented question-answering system for GI clinical
guidelines (AGA, ACG, ASGE, AASLD). Strictly extractive: the system answers
only from retrieved passages and cites every claim to a society guideline with
year and recommendation identifier. When the corpus does not cover a question,
it refuses or defers rather than confabulating.

This is a research prototype, not a clinical decision support tool. Do not use
to make patient care decisions.

## Stack

- **Python** 3.11+
- **LlamaIndex** as the RAG framework
- **Postgres 16 + pgvector** for the vector store (Docker Compose)
- **Voyage AI** `voyage-3-large` for embeddings (1024-dim)
- **Cohere Rerank v3** for reranking (Phase 3)
- **Anthropic Claude Sonnet 4.5** for generation (Phase 4)
- **unstructured** with `hi_res` PDF parsing
- **FastAPI** + **Streamlit** for the prototype interface (later phases)

## Prerequisites

1. Docker Desktop running.
2. Python 3.11+.
3. System libraries for `unstructured[pdf]` hi_res parsing:
   - macOS: `brew install poppler tesseract`
   - Ubuntu: `sudo apt-get install poppler-utils tesseract-ocr libmagic1`
4. API keys exported in your shell:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   export VOYAGE_API_KEY="..."
   export COHERE_API_KEY="..."        # used in Phase 3
   ```
   Or copy `.env.example` to `.env` and fill it in.

## Setup

```bash
# 1. Create a virtualenv and install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# 2. Start Postgres + pgvector
docker compose up -d
# schema.sql is auto-applied on first start. Confirm:
docker compose exec postgres psql -U gi -d gi_guidelines -c "\dt"

# 3. Copy env template
cp .env.example .env
# then edit .env with your keys
```

## Repo layout

```
gi-guidelines-rag/
├── docker-compose.yml         # Postgres 16 + pgvector
├── config.yaml                # tunable chunking / retrieval / generation knobs
├── sql/schema.sql             # documents + chunks tables, HNSW index, FTS index
├── data/
│   ├── raw_pdfs/{AGA,ACG,ASGE,AASLD}/   # drop PDFs here (gitignored)
│   ├── parsed/                          # cached parser output (gitignored)
│   └── corpus_metadata.csv              # one row per PDF: society, title, year, ...
├── src/
│   ├── ingest/    # parse → chunk → embed → load (Phase 2)
│   ├── retrieve/  # hybrid dense+BM25 + rerank (Phase 3)
│   ├── generate/  # strict-grounded answer synthesis (Phase 4)
│   ├── api/       # FastAPI backend (Phase 5)
│   ├── ui/        # Streamlit prototype (Phase 5)
│   └── eval/      # evaluation harness (Phase 6)
├── tests/         # pytest, currently chunker only
└── scripts/
```

## Running the ingestion pipeline (Phase 2)

Single-file mode (good for first-time sanity check):

```bash
python -m src.ingest.build_index --pdf data/raw_pdfs/AGA/AGA_2025_Gastroparesis_Staller.pdf \
    --society AGA --title "AGA Clinical Practice Update on Gastroparesis" --year 2025 \
    --topic gastroparesis
```

Bulk mode (full corpus):

```bash
python -m src.ingest.build_index --csv data/corpus_metadata.csv
```

The pipeline:

1. Parses the PDF with `unstructured` (`hi_res` strategy, falls back to `fast`
   if it takes too long — see `config.yaml`).
2. Chunks the parsed elements section-aware (target 400-600 tokens, never
   splitting a recommendation mid-sentence). Extracts recommendation IDs and
   GRADE evidence/strength via regex where possible.
3. Embeds with Voyage `voyage-3-large` in batches of 100.
4. Inserts into `documents` + `chunks` (idempotent on `(document_id,
   section_title, page_start, md5(text))`).

## Running the chunker tests

```bash
pytest tests/test_chunk.py -v
```

(LLM and embedding APIs are not unit-tested; we validate them end-to-end via
the Phase 6 eval harness.)

## Running the clinician UI (Phase 5)

The browser UI is two processes — a FastAPI backend and a Streamlit
front-end. Start them in two terminals:

```bash
# Terminal 1 — backend
.venv/bin/uvicorn src.api.app:app --port 8000 --reload

# Terminal 2 — UI
bash scripts/run_ui.sh
# then open http://localhost:8501
```

Things to try once it loads:

- **Specific recommendation**: *"What's the recommended endoscopic surveillance
  interval for low-grade dysplasia in Barrett's esophagus, and what's the
  GRADE?"* — verifies citation rendering + table chunk inline.
- **Cross-society**: *"What's the first-line pharmacologic treatment for
  diabetic gastroparesis, and do AGA and ACG agree?"* — should pull both
  AGA 2025 and ACG 2022 chunks.
- **Out-of-corpus refusal**: *"What's the optimal anesthesia regimen for
  ERCP in pregnancy?"* — friendly "didn't cover this directly" banner.
- **Figure**: *"What does the management algorithm for Barrett's surveillance
  look like?"* — the figure_caption chunk for Figure 2 should render its
  PNG inline.
- **PHI guard**: *"Patient John Smith, DOB 03/15/1965, presented with..."* —
  the screen blocks before the question reaches the model.

Each Q&A is appended to `logs/qa_log.jsonl` for offline review.

## Roadmap

- ✅ Phase 1 — scaffolding
- ✅ Phase 2 — ingestion pipeline
- ✅ Phase 3 — hybrid retrieval + Cohere rerank
- ✅ Phase 4 — strict-grounded generation with Claude Opus 4.7
- ✅ Phase 5 — FastAPI + Streamlit clinician UI
- ⬜ Phase 6 — evaluation harness against CDS vignettes
