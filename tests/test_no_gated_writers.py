"""No gated writer models (operator, 2026-11-20: "no gated models").

Every local writer in the dropdown must download with no Hugging Face account.
google/gemma-2-2b-it was the one gated row: a Mac user who picked it to try a
small writer stopped at GatedModelError on the first pass.
"""
from __future__ import annotations

from nodes import _otr_model_catalog as C


def test_no_curated_writer_requires_a_hugging_face_login():
    gated = [m.repo_id for m in C.CURATED_LLM_MODELS if m.requires_auth]
    assert gated == [], "gated writer rows are not allowed: %s" % gated
    assert C.GATED_CURATED_MODELS == frozenset()


def test_the_removed_gated_row_is_gone_from_every_table():
    removed = "google/gemma-2-2b-it"
    assert all(m.repo_id != removed for m in C.CURATED_LLM_MODELS)
    for name, table in vars(C).items():
        if isinstance(table, dict) and name.isupper():
            assert removed not in table, "%s still lists %s" % (name, removed)
