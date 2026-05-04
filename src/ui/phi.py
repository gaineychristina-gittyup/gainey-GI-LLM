"""Lightweight PHI screen for the clinician UI.

This is a *guardrail*, not a HIPAA-compliant DLP system. It's meant to
catch the most common ways a busy clinician might paste a snippet of
chart text into the question box without de-identifying:

- Dates of birth (MM/DD/YYYY, MM-DD-YYYY)
- Social Security Numbers (XXX-XX-XXXX)
- MRN-like long digit strings (8-12 consecutive digits)
- Explicit personal references ("patient John ...", "Mr./Mrs./Ms. ...",
  "the patient is named ...")

False positives are expected — a question containing a date for a
publication ("the 2022 ACG guideline ... 06/15/2022") will trip the
date check. The intent is to make the clinician pause and rephrase, not
to be perfectly accurate.

The user is responsible for de-identification. The warning text says so
explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# --- patterns ---------------------------------------------------------------

# Date of birth: MM/DD/YYYY or MM-DD-YYYY with reasonable bounds.
# Years 1900-2025 catch real birthdates; years 2026+ would be guideline
# publication dates and are deliberately excluded.
_DOB_RE = re.compile(
    r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12][0-9]|3[01])[/\-](19\d{2}|20[0-2][0-5])\b"
)

# US Social Security Number — strict format with required dashes/spaces.
_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[\s\-]\d{2}[\s\-]\d{4}\b")

# MRN-like: 8-12 consecutive digits (US MRN ranges typically fall here).
# Excludes shorter numbers (could be ages, doses) and very long numbers
# (likely citation IDs / DOI prefixes / page ranges).
_MRN_RE = re.compile(r"\b\d{8,12}\b")

# Explicit personal references — the model gave these a clear pattern in
# typical chart-copy: "Mr./Mrs./Ms./Dr. <Cap-Word>"
_HONORIFIC_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Mx|Dr|Miss)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)

# "patient X" / "the patient is named X" / "named X" patterns. Case-
# insensitive on the trigger word so "Patient John ..." catches.
_PATIENT_NAME_RE = re.compile(
    r"\b(?:patient(?:'s)?|named|name\s+is)\s+(?!with\b|who\b|presents\b|reports\b|opting\b|undergoing\b|is\s+a\b)([A-Z][a-z]+)\b",
    re.IGNORECASE,
)


@dataclass
class PhiHit:
    pattern_name: str
    excerpt: str  # the matched substring (kept short; never logged in full)


def screen_for_phi(text: str) -> list[PhiHit]:
    """Return a list of PHI hits found in ``text``. Empty list = clean.

    Each hit names the pattern that fired and a redacted ~30-char excerpt
    of the surrounding context. Caller decides whether to block.
    """
    hits: list[PhiHit] = []

    for m in _DOB_RE.finditer(text):
        hits.append(PhiHit("date_of_birth", _excerpt(text, m.start(), m.end())))
    for m in _SSN_RE.finditer(text):
        hits.append(PhiHit("ssn", _excerpt(text, m.start(), m.end())))
    for m in _MRN_RE.finditer(text):
        hits.append(PhiHit("mrn_like", _excerpt(text, m.start(), m.end())))
    for m in _HONORIFIC_RE.finditer(text):
        hits.append(PhiHit("honorific_name", _excerpt(text, m.start(), m.end())))
    for m in _PATIENT_NAME_RE.finditer(text):
        hits.append(PhiHit("patient_name", _excerpt(text, m.start(), m.end())))

    return hits


def _excerpt(text: str, start: int, end: int, pad: int = 8) -> str:
    """Short surrounding window with the matched span partially masked."""
    a = max(0, start - pad)
    b = min(len(text), end + pad)
    span = text[start:end]
    masked = span[0] + "•" * max(0, len(span) - 2) + (span[-1] if len(span) > 1 else "")
    return text[a:start] + masked + text[end:b]


def phi_warning_message(hits: list[PhiHit]) -> str:
    """User-facing warning block. Generic on purpose — we don't echo the
    PHI-like content back at the user beyond the redacted excerpt."""
    kinds = sorted({h.pattern_name for h in hits})
    pretty = {
        "date_of_birth": "date of birth",
        "ssn": "Social Security Number",
        "mrn_like": "long numeric ID (possible MRN)",
        "honorific_name": "named individual (Mr./Mrs./Dr. ...)",
        "patient_name": "named patient",
    }
    bullets = "\n".join(f"- {pretty.get(k, k)}" for k in kinds)
    return (
        "Your question may contain protected health information. This is a "
        "research tool — please rephrase using only de-identified clinical "
        "details.\n\n"
        f"Patterns detected:\n{bullets}\n\n"
        "You are responsible for de-identification. Removing the patterns above "
        "is necessary but not always sufficient — review for indirect identifiers "
        "(rare conditions + small populations, exact dates of service, etc.) "
        "before resubmitting."
    )
