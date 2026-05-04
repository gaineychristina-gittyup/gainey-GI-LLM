"""Tiny env-var hygiene helper used at startup before load_dotenv()."""

from __future__ import annotations

import os

_CRED_KEYS = (
    "ANTHROPIC_API_KEY",
    "VOYAGE_API_KEY",
    "COHERE_API_KEY",
    "OPENAI_API_KEY",
    "DATABASE_URL",
)


def clear_empty_creds() -> None:
    """Drop empty-string credential env vars from os.environ.

    python-dotenv's load_dotenv() defaults to override=False, so a parent
    shell that exports e.g. ANTHROPIC_API_KEY="" silently shadows the value
    in .env. Treating empty as unset preserves the "shell wins if truly set"
    contract while fixing the empty-string footgun.
    """
    for k in _CRED_KEYS:
        if os.environ.get(k) == "":
            os.environ.pop(k, None)
