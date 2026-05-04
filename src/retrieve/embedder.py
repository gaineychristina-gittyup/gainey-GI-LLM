"""Query-side embedder, picked from ``config.yaml`` ``embedding.provider``.

The embedder must match what produced the chunk vectors in the database —
swapping providers without re-ingesting will silently produce nonsense
similarities. The provider is therefore read from the same config the ingest
pipeline used.

We cache the embedder at module level so importing :func:`embed_query` from
multiple call sites doesn't repeatedly construct API clients (Voyage/OpenAI)
or reload local models (BGE).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# (provider, model) -> callable(text) -> list[float]
QueryEmbedder = Callable[[str], list[float]]


@lru_cache(maxsize=1)
def _load_config(path: str = str(DEFAULT_CONFIG)) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def _get_embedder(provider: str, model: str) -> QueryEmbedder:
    if provider == "voyage":
        return _make_voyage_embedder(model)
    if provider == "openai":
        return _make_openai_embedder(model)
    if provider == "bge":
        return _make_bge_embedder(model)
    raise ValueError(
        f"Unknown embedding provider {provider!r}. "
        "Set embedding.provider in config.yaml to 'voyage', 'openai', or 'bge'."
    )


def embed_query(text: str) -> list[float]:
    """Embed a query using the provider configured for the corpus.

    Reads ``config.yaml`` once on first call and caches the embedder
    instance. Returns a vector whose dimensionality matches the chunks
    table — 1024 for Voyage ``voyage-3-large`` (the project default) or
    BGE ``bge-large-en-v1.5``; 3072 for OpenAI ``text-embedding-3-large``.
    """
    cfg = _load_config()
    ecfg = cfg.get("embedding", {})
    provider = ecfg.get("provider", "voyage")
    if provider == "voyage":
        model = ecfg.get("voyage_model", "voyage-3-large")
    elif provider == "openai":
        model = ecfg.get("openai_model", "text-embedding-3-large")
    elif provider == "bge":
        model = ecfg.get("bge_model", "BAAI/bge-large-en-v1.5")
    else:
        model = ""
    return _get_embedder(provider, model)(text)


# --- providers --------------------------------------------------------------


def _make_voyage_embedder(model: str) -> QueryEmbedder:
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set; cannot embed queries with Voyage."
        )
    client = voyageai.Client(api_key=api_key)

    def embed(text: str) -> list[float]:
        # input_type="query" tells Voyage to use the query-side encoder,
        # which is paired with input_type="document" used at ingest time.
        resp = client.embed([text], model=model, input_type="query")
        return resp.embeddings[0]

    return embed


def _make_openai_embedder(model: str) -> QueryEmbedder:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot embed queries with OpenAI.")
    client = openai.OpenAI(api_key=api_key)

    def embed(text: str) -> list[float]:
        resp = client.embeddings.create(model=model, input=[text])
        return resp.data[0].embedding

    return embed


def _make_bge_embedder(model: str) -> QueryEmbedder:
    """Local BGE embedder via sentence-transformers. CPU-only by default.

    Not used by the current corpus (which was ingested with Voyage). Provided
    so that switching ``embedding.provider`` to ``bge`` and re-ingesting
    works end-to-end without changing this layer.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "BGE embedder requires sentence-transformers. "
            "Run: pip install 'sentence-transformers~=3.0' — or set "
            "embedding.provider to 'voyage' / 'openai' in config.yaml."
        ) from e

    logger.info("Loading BGE embedder %s (first run downloads ~1.3GB)...", model)
    st = SentenceTransformer(model)

    def embed(text: str) -> list[float]:
        # bge-* models expect a query prefix at retrieval time:
        # "Represent this sentence for searching relevant passages:"
        prefix = "Represent this sentence for searching relevant passages: "
        vec = st.encode([prefix + text], normalize_embeddings=True)[0]
        return vec.tolist()

    return embed
