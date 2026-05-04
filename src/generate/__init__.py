"""Phase 4 (MVP): grounded answer generation over retrieved guideline chunks.

Public API
----------

    from src.generate import answer

    out = answer(
        "what's the recommended H. pylori regimen for a penicillin-allergic patient?",
        filters={"society": ["ACG"]},
        top_k=6,
    )
    print(out["answer"])
    for c in out["citations"]:
        print(c)

The generator calls :func:`src.retrieve.retrieve` to pull top-k chunks,
formats them with stable [N] tags, and asks Claude (model from
``config.yaml`` ``generation.model``) to answer using only those chunks.
The system prompt is cache-controlled so repeated calls re-use the same
~1k-token preamble without re-billing.
"""

from __future__ import annotations

from src.generate.answer import answer

__all__ = ["answer"]
