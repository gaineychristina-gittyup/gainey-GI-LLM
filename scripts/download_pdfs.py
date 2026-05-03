"""Download guideline PDFs listed in data/corpus_metadata.csv."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import requests


USER_AGENT = "gainey-GI-LLM/0.1 (+research)"
CHUNK_SIZE = 8192


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/corpus_metadata.csv", help="metadata CSV path")
    parser.add_argument("--force", action="store_true", help="redownload existing PDFs")
    return parser.parse_args()


def row_label(row: dict[str, str]) -> str:
    society = row.get("society", "").strip()
    year = row.get("year", "").strip()
    topic = row.get("topic", "").strip()
    return f"{society} {year} {topic}"


def first_content_chunk(response: requests.Response) -> tuple[bytes | None, Iterator[bytes]]:
    chunks = response.iter_content(chunk_size=CHUNK_SIZE)
    for chunk in chunks:
        if chunk:
            return chunk, chunks
    return None, chunks


def download_pdf(row: dict[str, str], root: Path, force: bool) -> tuple[str, str]:
    label = row_label(row)
    source_url = row.get("source_url", "").strip()
    pdf_path = row.get("pdf_path", "").strip()

    if not source_url:
        return "failed", f"FAIL missing source_url {label}"
    if not pdf_path:
        return "failed", f"FAIL missing pdf_path {label}"

    destination = root / pdf_path
    if destination.exists() and not force:
        return "skipped", f"SKIP exists {label} -> {pdf_path}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")

    try:
        with requests.get(
            source_url,
            headers={"User-Agent": USER_AGENT},
            stream=True,
            timeout=60,
        ) as response:
            if response.status_code >= 400:
                return "failed", f"FAIL HTTP {response.status_code} {label}"

            content_type = response.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/pdf"):
                return "failed", f"FAIL non-PDF content-type {content_type!r} {label}"

            first_chunk, remaining_chunks = first_content_chunk(response)
            if first_chunk is None:
                return "failed", f"FAIL empty body {label}"
            if not first_chunk.startswith(b"%PDF-"):
                return "failed", f"FAIL body does not start with %PDF- {label}"

            try:
                with part_path.open("wb") as handle:
                    handle.write(first_chunk)
                    for chunk in remaining_chunks:
                        if chunk:
                            handle.write(chunk)
                os.replace(part_path, destination)
            except Exception:
                part_path.unlink(missing_ok=True)
                raise
    except requests.RequestException as exc:
        part_path.unlink(missing_ok=True)
        return "failed", f"FAIL request error {label}: {exc}"
    except OSError as exc:
        part_path.unlink(missing_ok=True)
        return "failed", f"FAIL write error {label}: {exc}"

    return "downloaded", f"OK {label} -> {pdf_path}"


def main() -> int:
    args = parse_args()
    root = repo_root()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    downloaded = 0
    skipped = 0
    failed = 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status, message = download_pdf(row, root, args.force)
            print(message)
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

    print(f"Summary: {downloaded} downloaded, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
