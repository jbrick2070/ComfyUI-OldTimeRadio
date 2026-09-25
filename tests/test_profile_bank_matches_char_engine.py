"""Every profile that pins a character voice engine sits on a voice bank that
engine's profile allows (kokoro-onnx r1, 2026-09-02).

Four profiles paired `voice_bank: default` with `char_voice_engine: kokoro`
(`char_kokoro_v1.allowed_voice_banks` is `[kokoro_builtin]`) and one paired
`default` with `bark` (`[bark_legacy]`); CastLock raised VoiceCastingError for
all five at the first episode -- the exact Mac / AMD / CPU rows the "ship all
audio lanes on kokoro" ruling is about. This keeps the pairing honest for every
workflow, so a new matrix row cannot ship the same trap.
"""
from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("OTR_TEST_MODE", "1")

import pytest

from nodes._otr_engine_profiles import load_resolver

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _pinned_profiles():
    """Every workflow that names a character voice.

    This used to glob `config/profiles/*.json`; that folder is gone, and every
    workflow is now a row in `config/workflow_matrix.json` where
    `char_voice_engine` is a declared key indicator, so all 24 state one. The
    five configurations whose broken pairing this test exists for were shipped
    rows -- the graphs a stranger opens.

    Resolved through `load_profile`, which reads matrix rows and nothing else.
    """
    from nodes._otr_shared.capability_profiles import (
        ProfileError, known_profile_ids, load_profile)

    rows = []
    for pid in known_profile_ids():
        try:
            data = load_profile(pid)
        except ProfileError:
            continue
        so = data.get("slot_overrides") or {}
        engine = str(so.get("char_voice_engine") or "").strip()
        if engine and engine != "auto":
            rows.append((pid, engine, str(so.get("voice_bank") or "")))

    assert rows, "nothing pins a character voice engine -- this test is checking nothing"
    # NON-VACUITY, sharpened: the shipped workflows must be in here.
    # `char_voice_engine` is a key indicator, so every matrix row states one; if none
    # of them appear, the enumeration has silently stopped reaching the matrix.
    import json as _json
    matrix = _json.loads(
        (pathlib.Path(_REPO) / "config" / "workflow_matrix.json").read_text(
            encoding="utf-8"))
    shipped = {r["id"] for r in matrix["rows"] if r.get("ships")}
    covered = {name for name, _, _ in rows} & shipped
    assert len(covered) >= 20, (
        "only %d of %d shipped workflows are covered; the enumeration is not reaching "
        "the matrix" % (len(covered), len(shipped)))
    return rows


@pytest.mark.parametrize("name,engine,bank", _pinned_profiles())
def test_profile_voice_bank_is_allowed_by_its_char_engine(name, engine, bank):
    resolver = load_resolver()
    profile = resolver.profile_for("char_voice", engine)
    if profile is None:
        pytest.skip("%s: engine %r has no char_voice profile row" % (name, engine))
    allowed = list(profile.allowed_voice_banks or [])
    if not allowed:
        return              # an engine with no bank restriction (cloud lanes)
    if not bank:
        # 2026-09-16: profiles no longer pin voice_bank. CastLock derives the
        # bank from the engine, so an absent override is not the leftover
        # default/kokoro trap this test was written to catch.
        return
    assert bank in allowed, (
        "%s pairs voice_bank %r with char_voice_engine %r, but %s allows only %s -- "
        "CastLock raises VoiceCastingError for this profile at the first episode"
        % (name, bank, engine, profile.profile_id, allowed))
