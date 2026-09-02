"""Every profile that pins a character voice engine sits on a voice bank that
engine's profile allows (kokoro-onnx r1, 2026-09-02).

Four profiles paired `voice_bank: default` with `char_voice_engine: kokoro`
(`char_kokoro_v1.allowed_voice_banks` is `[kokoro_builtin]`) and one paired
`default` with `bark` (`[bark_legacy]`); CastLock raised VoiceCastingError for
all five at the first episode -- the exact Mac / AMD / CPU rows the "ship all
audio lanes on kokoro" ruling is about. This keeps the pairing honest for every
profile, so a new lab preset cannot ship the same trap.
"""
from __future__ import annotations

import glob
import json
import os

os.environ.setdefault("OTR_TEST_MODE", "1")

import pytest

from nodes._otr_engine_profiles import load_resolver

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILES = sorted(glob.glob(os.path.join(_HERE, "..", "config", "profiles", "*.json")))


def _pinned_profiles():
    rows = []
    for path in _PROFILES:
        if os.path.basename(path) == "widget_mapping.json":
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        so = data.get("slot_overrides") or {}
        engine = str(so.get("char_voice_engine") or "").strip()
        if engine and engine != "auto":
            rows.append((os.path.basename(path), engine, str(so.get("voice_bank") or "")))
    assert rows, "no profile pins a character voice engine"
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
    assert bank in allowed, (
        "%s pairs voice_bank %r with char_voice_engine %r, but %s allows only %s -- "
        "CastLock raises VoiceCastingError for this profile at the first episode"
        % (name, bank, engine, profile.profile_id, allowed))
