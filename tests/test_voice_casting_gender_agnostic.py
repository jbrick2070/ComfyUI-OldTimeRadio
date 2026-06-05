"""Regression: a character whose gender is not in the bank (e.g. the writer
emitting gender='other') must still get a REAL indextts2 reference, never a
silent bark fallback. Pins the gender-agnostic last resort in
`_resolve_clone_ref_path`. Live observation 2026-06-05: cast row 'SEAN FLANDERS'
gender='other' dropped to bark; this guards that it does not recur.

Requires the indextts2 reference WAVs on disk (skipped otherwise).
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_voice_node_common import (  # noqa: E402
    _resolve_clone_ref_path, _resolve_ref_to_disk,
)
from nodes._otr_voice_bank import load_voice_bank  # noqa: E402

_bank, _ = load_voice_bank()
_idx = [e for e in _bank if e.engine == "indextts2"]


def _refs_present():
    for e in _idx:
        p = _resolve_ref_to_disk(e.ref_path)
        if not (p and os.path.exists(p)):
            return False
    return bool(_idx)


needs_refs = pytest.mark.skipif(
    not _refs_present(),
    reason="indextts2 reference WAVs not installed on disk",
)


@needs_refs
def test_other_gender_resolves_to_index_not_bark():
    path = _resolve_clone_ref_path("indextts2", {"char_id": "cX", "gender": "other"}, 42)
    assert path is not None and path.lower().endswith(".wav")


@needs_refs
@pytest.mark.parametrize("g", ["", None, "nonbinary", "unknown"])
def test_unservable_gender_still_resolves_to_index(g):
    path = _resolve_clone_ref_path("indextts2", {"char_id": "cY", "gender": g}, 7)
    assert path is not None and path.lower().endswith(".wav")


@needs_refs
def test_gender_agnostic_pick_is_deterministic():
    a = _resolve_clone_ref_path("indextts2", {"char_id": "cZ", "gender": "other"}, 99)
    b = _resolve_clone_ref_path("indextts2", {"char_id": "cZ", "gender": "other"}, 99)
    assert a is not None and a == b


@needs_refs
def test_known_gender_still_resolves():
    for g in ("male", "female"):
        path = _resolve_clone_ref_path("indextts2", {"char_id": "cM", "gender": g}, 1)
        assert path is not None and path.lower().endswith(".wav")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
