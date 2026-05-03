# Codex Task: Download all corpus PDFs + normalize `corpus_metadata.csv`

You are working in the `gainey-GI-LLM` repo. The repo ships a one-row stub
`data/corpus_metadata.csv` and an empty `data/raw_pdfs/` tree. The ingestion
pipeline (`src/ingest/build_index.py --csv data/corpus_metadata.csv`) expects
each row's `pdf_path` to exist on disk before it runs.

Your job has three parts:

1. **Expand and normalize `data/corpus_metadata.csv`** so every row has a
   working `source_url` and a deterministic `pdf_path`.
2. **Add a downloader script** (`scripts/download_pdfs.py`) that reads the CSV
   and fetches every PDF into the path declared by its `pdf_path` column.
3. **Run the downloader** and confirm the files land where the ingestion
   pipeline expects them.

Do all work on the current branch. Do **not** commit any PDF — `data/raw_pdfs/`
is gitignored and must stay that way.

---

## 1. Normalize `data/corpus_metadata.csv`

Keep the existing header exactly:

```
society,title,year,topic,doi,source_url,pdf_path
```

Rules for every row:

- `society` ∈ {`AGA`, `ACG`, `ASGE`, `AASLD`}.
- `year` is a 4-digit integer.
- `topic` is a short lowercase slug (e.g. `gastroparesis`, `ibd`, `barretts`).
- `doi` is the bare DOI (no `https://doi.org/` prefix). Leave blank only if
  the guideline genuinely has no DOI.
- `source_url` must be a **direct PDF link** (Content-Type `application/pdf`)
  from the society's official site or the publisher's open-access URL. If only
  a landing page is available, find the embedded PDF link — do not put a
  landing-page URL in this column.
- `pdf_path` must follow the convention
  `data/raw_pdfs/{SOCIETY}/{SOCIETY}_{YEAR}_{TopicSlug}_{FirstAuthor}.pdf`
  matching the existing example row. Use PascalCase for the topic slug and
  the first author's last name. No spaces.

Add coverage for the four societies the README lists (AGA, ACG, ASGE, AASLD).
Aim for the most-cited recent (2019–2025) clinical practice guidelines and
clinical practice updates from each society. If you can't verify a direct PDF
URL for a candidate row, **omit the row** rather than guess — a broken URL
will fail the downloader.

Preserve the existing AGA gastroparesis row; just backfill its `doi` and
`source_url`.

---

## 2. Add `scripts/download_pdfs.py`

Create a new file at `scripts/download_pdfs.py`. Requirements:

- CLI: `python scripts/download_pdfs.py [--csv data/corpus_metadata.csv] [--force]`.
- Reads the CSV with `csv.DictReader`.
- For each row:
  - Resolve `pdf_path` relative to the repo root.
  - Skip if the file already exists and `--force` was not passed.
  - `mkdir -p` the parent directory.
  - GET `source_url` with `requests`, `stream=True`, a 60s timeout, and a
    descriptive `User-Agent` (e.g. `gainey-GI-LLM/0.1 (+research)`).
  - Verify the response `Content-Type` starts with `application/pdf`. If not,
    log a warning and skip — do not write a non-PDF to disk.
  - Verify the body starts with the bytes `%PDF-` before renaming into place.
  - Write to a `.part` temp file, then `os.replace` to the final path so a
    failed download never leaves a half-written PDF.
  - Print one line per row: `OK  AGA 2025 gastroparesis -> data/raw_pdfs/AGA/...`
    or `FAIL <reason>`.
- At the end, print a summary: `N downloaded, M skipped (existing), K failed`.
- Exit code 0 if no failures, 1 otherwise.
- Use only the stdlib plus `requests` (already a transitive dep via the
  existing `pyproject.toml`; if not, add it under `[project] dependencies`).

Do not parallelize the downloads — society sites rate-limit aggressively, and
sequential is fast enough for a corpus this size.

---

## 3. Run the downloader and verify

```bash
python scripts/download_pdfs.py --csv data/corpus_metadata.csv
```

Then verify:

- Every row's `pdf_path` exists: `awk -F, 'NR>1 {print $7}' data/corpus_metadata.csv | xargs -I{} test -f {} && echo OK`
- `find data/raw_pdfs -name '*.pdf' | wc -l` matches the number of CSV rows.
- A spot-check `file data/raw_pdfs/AGA/AGA_2025_Gastroparesis_Staller.pdf`
  reports `PDF document`.

If any download failed, fix the offending row's `source_url` (or remove the
row) and re-run. Do not commit a CSV that points at URLs the downloader
couldn't fetch.

---

## 4. Commit

Commit only:

- `data/corpus_metadata.csv`
- `scripts/download_pdfs.py`
- `pyproject.toml` (only if you had to add `requests`)

Do **not** commit anything under `data/raw_pdfs/`. Confirm with
`git status` before committing — if any `.pdf` shows up as untracked, the
`.gitignore` is correct; just don't `git add` it.

Suggested commit message:

```
Add PDF downloader and expand corpus metadata

- scripts/download_pdfs.py fetches every PDF declared in the CSV,
  validates Content-Type and the %PDF- magic bytes, writes atomically.
- data/corpus_metadata.csv: backfill source_url/doi on the existing
  row and add coverage for AGA/ACG/ASGE/AASLD recent guidelines.
```
