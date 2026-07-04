"""Wave 2a -- OTR_CastLock single ledger authority (plan E.0-E.5).

Headless. Exercises the cheap char_id validator, the preserve_ledger byte-safe
default, and the auto_registry caster path (stamps voice_ref_id / voice_engine /
commercial_clean, deterministic, announcer pinned per engine).
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "cast_lock.py"

_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})


def _serialized_slots(it: dict) -> list:
    names = []
    for bucket in ("required", "optional"):
        for name, spec in (it.get(bucket) or {}).items():
            t = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            is_widget = isinstance(t, (list, tuple)) or (
                isinstance(t, str) and t.upper() in _WIDGET_TYPES
            )
            opts = (
                spec[1]
                if isinstance(spec, (list, tuple)) and len(spec) > 1
                and isinstance(spec[1], dict)
                else {}
            )
            if not is_widget or opts.get("forceInput"):
                continue
            names.append(name)
    return names


def _ledger(cast, meta=None):
    return json.dumps({"meta": meta or {"episode_seed": 42}, "cast": cast, "lines": []})


_CHAR_CAST = [
    {"char_id": "c1", "name": "MONTY", "gender": "male", "voice_preset": "v2/en_speaker_1"},
    {"char_id": "c2", "name": "VERA", "gender": "female", "voice_preset": "v2/en_speaker_9"},
    {"char_id": "a1", "name": "ANNOUNCER", "gender": "male", "voice_preset": "v2/en_speaker_6"},
]


# ----------------------------------------------------------------------------
# INPUT_TYPES / widget surface (E.4)
# ----------------------------------------------------------------------------
def test_input_types_widget_surface():
    from nodes.cast_lock import CastLock

    it = CastLock.INPUT_TYPES()
    for key in ("script_json", "gate_in"):
        spec = it.get("required", {}).get(key) or it.get("optional", {}).get(key)
        assert spec is not None and spec[1].get("forceInput") is True, key
    # delivery_profile surface removed 2026-07-04 (widget-audit Batch 1); single
    # option "neutral" -- the lock() kwarg still defaults + is validated/stamped.
    assert _serialized_slots(it) == [
        "voice_bank", "cast_voice_policy", "allow_voice_reuse",
    ]
    for name in ("ledger_json", "cast_lock_revision", "cast_report", "done"):
        assert name in CastLock.RETURN_NAMES


def test_rejects_forbidden_widgets():
    from nodes.cast_lock import CastLock

    it = CastLock.INPUT_TYPES()
    keys = set(it.get("required", {})) | set(it.get("optional", {}))
    for forbidden in ("voice_engine_mode", "deterministic_inference", "model_id", "seed"):
        assert forbidden not in keys, forbidden


# ----------------------------------------------------------------------------
# Validator + preserve_ledger
# ----------------------------------------------------------------------------
def test_duplicate_char_id_fails_before_cast():
    from nodes.cast_lock import CastLock

    dup = [{"char_id": "x", "name": "A"}, {"char_id": "x", "name": "B"}]
    with pytest.raises(ValueError):
        CastLock().lock(script_json=_ledger(dup))


def test_preserve_ledger_is_byte_safe_and_increments_revision():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(script_json=_ledger(_CHAR_CAST), cast_voice_policy="preserve_ledger")
    ledger_json, revision, report, done = out
    assert revision == 1
    led = json.loads(ledger_json)
    # No voice_ref_id stamped on any cast entry (nothing re-cast).
    for entry in led["cast"]:
        assert "voice_ref_id" not in entry
    assert led["meta"]["cast_lock_revision"] == 1
    assert done.startswith("cast_lock:done")


# ----------------------------------------------------------------------------
# STEP 3 (2026-06-22, revised per operator): node-80 OUTPUT voice resolution is
# FAIL-SOFT -- never abort the episode; resolve a voice relative to the selected
# model, re-route mis-stamped announcer lines, reassign true orphans. Engine-
# agnostic + future-proof for all approved voice models.
# ----------------------------------------------------------------------------
def _ledger_with_lines(cast, lines, meta=None):
    return json.dumps({"meta": meta or {"episode_seed": 42}, "cast": cast,
                       "lines": lines})


def _last_ledger(out):
    return json.loads(out[0])


def test_voice_passes_when_character_lines_have_presets():
    from nodes.cast_lock import CastLock

    lines = [
        {"line_id": "l001", "speaker_role": "character", "char_id": "c1",
         "text": "We move at dawn."},
        {"line_id": "l002", "speaker_role": "character", "char_id": "c2",
         "text": "Not without proof."},
    ]
    out = CastLock().lock(script_json=_ledger_with_lines(_CHAR_CAST, lines),
                          cast_voice_policy="preserve_ledger")
    assert out[1] == 1  # locked, no repair needed


def test_seedless_missing_preset_fails_loud():
    """NO-FALLBACK (2026-07-03): a character cast row with no voice_preset is a
    casting defect -- CastLock RAISES VoiceCastingError instead of synthesizing a
    deterministic v2/en_speaker_* fallback identity."""
    import pytest

    from nodes.cast_lock import CastLock
    from nodes._otr_voice_bank import VoiceCastingError

    cast = [
        {"char_id": "c1", "name": "MONTY", "gender": "male"},  # no voice_preset
        {"char_id": "a1", "name": "ANNOUNCER", "gender": "male"},
    ]
    lines = [
        {"line_id": "l001", "speaker_role": "character", "char_id": "c1",
         "text": "We move at dawn."},
    ]
    with pytest.raises(VoiceCastingError, match="no voice_preset"):
        CastLock().lock(script_json=_ledger_with_lines(cast, lines),
                        cast_voice_policy="preserve_ledger")


def test_announcer_lines_and_cue_rows_untouched():
    """Announcer lines (engine-resolved at node 82) and music/sfx cue rows
    never need a cast-row voice_preset and are left untouched."""
    from nodes.cast_lock import CastLock

    cast = [
        {"char_id": "a1", "name": "ANNOUNCER", "gender": "male"},  # no preset
    ]
    lines = [
        {"line_id": "l001", "speaker_role": "announcer", "char_id": "announcer",
         "text": "Tonight, on..."},
        {"line_id": "l002", "speaker_role": "music_open", "char_id": "music_open",
         "text": ""},
        {"line_id": "l003", "speaker_role": "sfx", "char_id": "sfx",
         "text": "door slams"},
    ]
    out = CastLock().lock(script_json=_ledger_with_lines(cast, lines),
                          cast_voice_policy="preserve_ledger")
    assert out[1] == 1  # locked, no repair


def test_misstamped_announcer_line_reroutes_to_announcer():
    """The live-crash case (b018): a speaker_role='character' line whose char_id
    is the announcer marker is a MIS-STAMPED announcer line -> re-routed to
    speaker_role='announcer' (no crash, voiced by the announcer model)."""
    from nodes.cast_lock import CastLock

    lines = [
        {"line_id": "b018", "speaker_role": "character", "char_id": "announcer",
         "text": "As the dust settles, the colony endures."},
    ]
    out = CastLock().lock(script_json=_ledger_with_lines(_CHAR_CAST, lines),
                          cast_voice_policy="preserve_ledger")
    assert out[1] == 1  # locked, no crash
    led = _last_ledger(out)
    b018 = next(ln for ln in led["lines"] if ln["line_id"] == "b018")
    assert b018["speaker_role"] == "announcer"   # re-routed


def test_orphan_character_line_fails_loud():
    """NO-FALLBACK (2026-07-03): a character line whose char_id matches no voiced
    cast row is a true orphan -- CastLock RAISES VoiceCastingError instead of
    reassigning it to another voiced character."""
    import pytest

    from nodes.cast_lock import CastLock
    from nodes._otr_voice_bank import VoiceCastingError

    lines = [
        {"line_id": "l009", "speaker_role": "character", "char_id": "c99",
         "text": "Who am I?"},
    ]
    with pytest.raises(VoiceCastingError, match="orphan"):
        CastLock().lock(script_json=_ledger_with_lines(_CHAR_CAST, lines),
                        cast_voice_policy="preserve_ledger")


def test_revision_increments_from_existing_meta():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_CHAR_CAST, meta={"episode_seed": 1, "cast_lock_revision": 7}),
    )
    assert out[1] == 8


# ----------------------------------------------------------------------------
# auto_registry caster path
# ----------------------------------------------------------------------------
def test_auto_registry_stamps_voice_refs():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_CHAR_CAST), voice_bank="default",
        cast_voice_policy="auto_registry",
    )
    led = json.loads(out[0])
    cast = {e["char_id"]: e for e in led["cast"]}
    # Characters get indextts2 references (the promoted rank-1 char_voice default).
    for cid in ("c1", "c2"):
        assert cast[cid].get("voice_ref_id"), cid
        assert cast[cid]["voice_engine"] == "indextts2"
        assert isinstance(cast[cid]["commercial_clean"], bool)
    # Distinct references (no reuse across the two characters).
    assert cast["c1"]["voice_ref_id"] != cast["c2"]["voice_ref_id"]
    # Announcer is pinned to the kokoro reference.
    assert cast["a1"]["voice_ref_id"] == "bm_george"
    assert cast["a1"]["voice_engine"] == "kokoro"


def test_auto_registry_is_deterministic():
    from nodes.cast_lock import CastLock

    a = CastLock().lock(script_json=_ledger(_CHAR_CAST), cast_voice_policy="auto_registry")[0]
    b = CastLock().lock(script_json=_ledger(_CHAR_CAST), cast_voice_policy="auto_registry")[0]
    assert a == b


def test_auto_registry_skips_genderless_character():
    from nodes.cast_lock import CastLock

    cast = [{"char_id": "c1", "name": "MYSTERY", "voice_preset": "v2/en_speaker_2"}]
    out = CastLock().lock(script_json=_ledger(cast), cast_voice_policy="auto_registry")
    led = json.loads(out[0])
    assert "voice_ref_id" not in led["cast"][0]  # preserved, not cast
    assert "no gender" in out[2]


def test_auto_registry_bark_legacy_preserves_characters():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_CHAR_CAST), voice_bank="bark_legacy",
        cast_voice_policy="auto_registry",
    )
    led = json.loads(out[0])
    cast = {e["char_id"]: e for e in led["cast"]}
    # bark has no reference entries -> characters preserved...
    assert "voice_ref_id" not in cast["c1"]
    # ...but the announcer still resolves its kokoro pin.
    assert cast["a1"]["voice_ref_id"] == "bm_george"


def test_unknown_delivery_profile_fails_closed():
    from nodes.cast_lock import CastLock

    with pytest.raises(Exception):
        CastLock().lock(script_json=_ledger(_CHAR_CAST), delivery_profile="dramatic")


def test_ledger_json_round_trips():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(script_json=_ledger(_CHAR_CAST), cast_voice_policy="auto_registry")
    assert isinstance(json.loads(out[0]), dict)  # valid canonical JSON


# ----------------------------------------------------------------------------
# Registration + hygiene
# ----------------------------------------------------------------------------
def test_node_class_matches_class_registry():
    from nodes._otr_class_registry import NEW_NODE_SPECS, expected_category
    from nodes.cast_lock import CastLock

    spec = NEW_NODE_SPECS["OTR_CastLock"]
    assert spec.class_name == "CastLock"
    assert CastLock.CATEGORY == expected_category("OTR_CastLock")
    assert CastLock.FUNCTION == "lock"


def test_node_wired_into_init_by_table_not_literal_key():
    text = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
    literal_keys = set(re.findall(r'"(OTR_\w+)"\s*:', text))
    assert "OTR_CastLock" not in literal_keys


def test_importing_node_triggers_no_engine_lib_imports():
    import importlib
    import sys

    importlib.import_module("nodes.cast_lock")
    for lib in ("chatterbox", "indextts", "stable_audio_tools", "kokoro"):
        assert lib not in sys.modules, f"engine lib {lib} imported at node import"


def test_source_is_ascii_no_em_dash():
    src = NODE_SRC.read_text(encoding="utf-8")
    assert "—" not in src
    src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
