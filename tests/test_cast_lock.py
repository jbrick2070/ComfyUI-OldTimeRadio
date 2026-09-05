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

# The curated kokoro announcer pool (eng_kokoro.ANNOUNCER_VOICE_POOL). Before
# 2026-08-05 only bm_george carried the announcer_voice role, so every episode
# opened with the same voice; the announcer is now drawn from all four per
# episode seed.
_KOKORO_ANNOUNCERS = frozenset({"bm_george", "bm_fable", "bf_emma", "bf_lily"})


def test_announcer_entry_prefers_canonical_char_id_over_display_name():
    from nodes.cast_lock import _is_announcer_entry

    assert _is_announcer_entry({
        "char_id": "announcer", "name": "Anita", "voice_preset": "bf_lily",
    })


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
    # S5 platform-portability (2026-07-10): voice_device appended at slot 5.
    assert _serialized_slots(it) == [
        "voice_bank", "cast_voice_policy", "allow_voice_reuse",
        "char_voice_engine", "announcer_voice_engine", "voice_device",
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
    # Announcer draws from the curated kokoro pool (seeded per episode, no
    # longer pinned to bm_george on every single episode).
    assert cast["a1"]["voice_ref_id"] in _KOKORO_ANNOUNCERS
    assert cast["a1"]["voice_engine"] == "kokoro"


def test_auto_registry_announcer_varies_across_episode_seeds():
    """The audible half of the announcer unpin, asserted through the real
    CastLock path rather than the bank helper: different episodes must not all
    open with the same voice. Fails on the pre-2026-08-05 code, which returned
    bm_george for every seed.
    """
    from nodes.cast_lock import CastLock

    seen = set()
    for seed in range(24):
        out = CastLock().lock(
            script_json=_ledger(_CHAR_CAST, meta={"episode_seed": seed}),
            cast_voice_policy="auto_registry",
        )
        cast = {e["char_id"]: e for e in json.loads(out[0])["cast"]}
        seen.add(cast["a1"]["voice_ref_id"])
    assert seen <= _KOKORO_ANNOUNCERS
    assert len(seen) > 1, f"announcer still pinned to a single voice: {seen}"


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
    # ...but the announcer still resolves a kokoro reference.
    assert cast["a1"]["voice_ref_id"] in _KOKORO_ANNOUNCERS


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


# --------------------------------------------------------------------------- #
# Content-owned lanes own their cast: CastLock must VERIFY, never REPLAY.
#
# A content-owned lane (scifi_codex / gemini / sonnet / scifi_news_pro) builds its own
# cast rows and stamps their voice_preset in the lane runner. The writer's
# seeded picker never ran, so there is no sequence to replay and no
# num_characters_request to replay it with. Live regression (2026-07-11): the
# tail stamped meta.cast_contract.cast_seed as a credits receipt, CastLock took
# that as "replay me", and assemble_pre_locked_rows died with
# "num_characters must be 1-6, got 0" AFTER 14 minutes of generation.
# --------------------------------------------------------------------------- #
def _content_owned_cast():
    return [
        {"char_id": "announcer", "name": "ANNOUNCER",
         "tts_model": "kokoro", "voice_preset": "bm_george"},
        {"char_id": "c01", "name": "Dr. Hart",
         "tts_model": "bark", "voice_preset": "v2/en_speaker_6"},
        {"char_id": "c02", "name": "Elena Ruiz",
         "tts_model": "bark", "voice_preset": "v2/en_speaker_3"},
    ]


def test_content_owned_lane_preserves_its_own_voices_without_replay():
    from nodes.cast_lock import CastLock

    cast = _content_owned_cast()
    # A cast_seed in the contract must NOT drag a lane-owned cast into the
    # writer-picker replay -- not even when num_characters_request is absent,
    # which is exactly the live crash.
    meta = {"source_bank": "scifi_news_pro", "episode_seed": 1234,
            "cast_contract": {"cast_seed": 1234}}
    report: list = []

    CastLock._assign_bark_voices(cast, meta, report)

    assert [row["voice_preset"] for row in cast] == [
        "bm_george", "v2/en_speaker_6", "v2/en_speaker_3",
    ]
    assert any("content-owned" in line for line in report)


def test_content_owned_lane_still_fails_on_colliding_bark_voices():
    from nodes._otr_casting import CastingFailedError
    from nodes.cast_lock import CastLock

    cast = _content_owned_cast()
    cast[2]["voice_preset"] = cast[1]["voice_preset"]  # duplicate bark voice
    meta = {"source_bank": "scifi_news_pro", "episode_seed": 1234}

    # Skipping the replay must not skip Gate 1: a lane-owned cast can never
    # ship two bark rows on the same voice.
    with pytest.raises(CastingFailedError):
        CastLock._assign_bark_voices(cast, meta, [])


def test_legacy_lane_without_cast_seed_still_preserves_presets():
    from nodes.cast_lock import CastLock

    cast = _content_owned_cast()
    report: list = []

    # A legacy / untagged ledger with no persisted cast_seed keeps its old
    # behavior: nothing to replay, presets untouched, no invariant run.
    CastLock._assign_bark_voices(cast, {}, report)

    assert [row["voice_preset"] for row in cast] == [
        "bm_george", "v2/en_speaker_6", "v2/en_speaker_3",
    ]
    assert any("no cast_seed" in line for line in report)


# ----------------------------------------------------------------------------
# Bark as an announcer engine (2026-08-24)
# ----------------------------------------------------------------------------
def _bark_announcer_cast():
    return [
        {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
         "tts_model": "kokoro", "voice_engine": "kokoro",
         "voice_ref_id": "bm_george", "voice_preset": "bm_george",
         "commercial_clean": True},
        {"char_id": "c1", "name": "MONTY", "gender": "male",
         "voice_preset": "v2/en_speaker_1"},
        {"char_id": "c2", "name": "VERA", "gender": "female",
         "voice_preset": "v2/en_speaker_9"},
    ]


def test_bark_announcer_is_stamped_with_a_v2_preset_and_engine_fields():
    from nodes.cast_lock import CastLock

    cast = _bark_announcer_cast()
    meta = {"episode_seed": 42}
    report: list = []

    CastLock._assign_bark_voices(cast, meta, report,
                                 announcer_voice_engine="bark")

    row = next(r for r in cast if r["name"] == "ANNOUNCER")
    assert str(row["voice_preset"]).startswith("v2/")
    assert row["tts_model"] == "bark"
    assert row["voice_engine"] == "bark"
    assert any("announcer stamped" in line for line in report)


def test_bark_announcer_clears_stale_kokoro_identity_on_relock():
    """A re-locked episode that was previously cast with Kokoro must not
    leave `voice_ref_id` / `commercial_clean` around -- the credits roll
    reads `voice_ref_id` AHEAD of `voice_preset`, so a stale kokoro ref
    would credit the wrong engine for audio Bark actually rendered."""
    from nodes.cast_lock import CastLock

    cast = _bark_announcer_cast()  # starts with a real kokoro stamp
    CastLock._assign_bark_voices(cast, {"episode_seed": 42}, [],
                                 announcer_voice_engine="bark")

    row = next(r for r in cast if r["name"] == "ANNOUNCER")
    assert row["voice_ref_id"] == ""
    assert row["commercial_clean"] is False


def test_bark_announcer_never_shares_a_preset_with_a_character():
    from nodes.cast_lock import CastLock

    cast = _bark_announcer_cast()
    CastLock._assign_bark_voices(cast, {"episode_seed": 7}, [],
                                 announcer_voice_engine="bark")

    presets = [r["voice_preset"] for r in cast]
    assert len(presets) == len(set(presets)), presets


def test_bark_announcer_draw_is_deterministic_per_episode_seed():
    from nodes.cast_lock import CastLock

    a = _bark_announcer_cast()
    b = _bark_announcer_cast()
    CastLock._assign_bark_voices(a, {"episode_seed": 99}, [],
                                 announcer_voice_engine="bark")
    CastLock._assign_bark_voices(b, {"episode_seed": 99}, [],
                                 announcer_voice_engine="bark")

    a_preset = next(r for r in a if r["name"] == "ANNOUNCER")["voice_preset"]
    b_preset = next(r for r in b if r["name"] == "ANNOUNCER")["voice_preset"]
    assert a_preset == b_preset


def test_bark_announcer_draw_does_not_perturb_character_voice_draws():
    """The announcer draw runs on an isolated RNG (a distinct sha1
    discriminator, not a shared stream with the character replay) -- adding
    it must not change what a character's own voice_preset already was."""
    from nodes.cast_lock import CastLock

    without = _bark_announcer_cast()
    CastLock._assign_bark_voices(without, {"episode_seed": 55}, [],
                                 announcer_voice_engine="kokoro")
    with_bark = _bark_announcer_cast()
    CastLock._assign_bark_voices(with_bark, {"episode_seed": 55}, [],
                                 announcer_voice_engine="bark")

    char_presets_without = {r["char_id"]: r["voice_preset"]
                             for r in without if r["name"] != "ANNOUNCER"}
    char_presets_with = {r["char_id"]: r["voice_preset"]
                          for r in with_bark if r["name"] != "ANNOUNCER"}
    assert char_presets_without == char_presets_with


def test_bark_announcer_runs_under_preserve_ledger_the_default_policy():
    """THE SILENT-NO-OP THIS GUARDS. `_auto_registry` never runs under
    `preserve_ledger` (the default cast_voice_policy) -- a fix placed only
    inside `_auto_registry` would leave the writer's Kokoro row on the
    announcer with `announcer_voice_engine="bark"` requested, and the render
    would only fail later at Gate 3. `_assign_bark_voices` runs
    unconditionally in `lock()`, before the policy fork, so this must pass
    end-to-end through `lock()` itself, not just the helper."""
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_bark_announcer_cast()),
        cast_voice_policy="preserve_ledger",
        announcer_voice_engine="bark",
    )
    led = json.loads(out[0])
    row = next(r for r in led["cast"] if r["name"] == "ANNOUNCER")
    assert str(row["voice_preset"]).startswith("v2/")
    assert row["tts_model"] == "bark"


def test_bark_announcer_runs_under_auto_registry_too():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_bark_announcer_cast()),
        cast_voice_policy="auto_registry",
        announcer_voice_engine="bark",
    )
    led = json.loads(out[0])
    row = next(r for r in led["cast"] if r["name"] == "ANNOUNCER")
    assert str(row["voice_preset"]).startswith("v2/")
    assert row["tts_model"] == "bark"
    # The old swallow-and-log-"announcer NOT cast" path must never fire once
    # _auto_registry's explicit bark skip is in place.
    assert not any("NOT cast" in line for line in out[2].splitlines())


def test_bark_announcer_fails_loud_on_content_owned_ledger():
    """kibitz r3 MUST-FIX (codex + cursor, independently). A coordinated
    `slot_overrides.announcer_voice_engine=bark` reaches BOTH OTR_CastLock
    and OTR_AnnouncerVoice in one pass -- if this ledger is content-owned
    (scifi_news_pro builds its own announcer row and never wires bark),
    silently preserving the Kokoro row would defer the failure to a
    confusing crash minutes later at eng_bark.py's Gate 3. Must refuse HERE,
    with a message that names the actual cause."""
    from nodes._otr_voice_bank import VoiceCastingError
    from nodes.cast_lock import CastLock

    cast = _content_owned_cast()
    meta = {"source_bank": "scifi_news_pro", "episode_seed": 42}

    with pytest.raises(VoiceCastingError, match="CONTENT_OWNED"):
        CastLock._assign_bark_voices(cast, meta, [],
                                     announcer_voice_engine="bark")


def test_bark_announcer_stamps_presentation_gender_from_delivered_preset():
    from nodes.cast_lock import CastLock

    cast = _bark_announcer_cast()  # ANNOUNCER row starts with gender="male"
    CastLock._assign_bark_voices(cast, {"episode_seed": 42}, [],
                                 announcer_voice_engine="bark")

    row = next(r for r in cast if r["name"] == "ANNOUNCER")
    from nodes._otr_voice_bank import bark_preset_gender

    assert row["presentation_gender"] == bark_preset_gender(row["voice_preset"])


def test_bark_announcer_gender_exhaustion_stamps_the_ACTUAL_delivered_gender():
    """kibitz r3 MUST-FIX (codex). python_assign_voice_preset falls back to
    the FULL pool when the requested gender's column is exhausted -- proven
    historically real by FIX-3 (3-female character casts against the
    4-female VOICE_PROFILES pool). A female-requesting announcer row whose
    gender column is exhausted by characters must be credited for whatever
    gender it ACTUALLY got, never the stale request -- Bug Bible 10.08."""
    from nodes.cast_lock import CastLock

    cast = [
        {"char_id": "a1", "name": "ANNOUNCER", "gender": "female",
         "tts_model": "kokoro", "voice_preset": "bm_george"},
        # Every female VOICE_PROFILES preset already taken by characters.
        {"char_id": "c1", "name": "A", "voice_preset": "v2/en_speaker_2"},
        {"char_id": "c2", "name": "B", "voice_preset": "v2/en_speaker_4"},
        {"char_id": "c3", "name": "C", "voice_preset": "v2/en_speaker_9"},
        {"char_id": "c4", "name": "D", "voice_preset": "v2/en_speaker_7"},
    ]
    CastLock._assign_bark_voices(cast, {"episode_seed": 42}, [],
                                 announcer_voice_engine="bark")

    row = next(r for r in cast if r["name"] == "ANNOUNCER")
    from nodes._otr_voice_bank import bark_preset_gender

    delivered = bark_preset_gender(row["voice_preset"])
    assert delivered == "male", "every female preset was taken; must fall back"
    assert row["presentation_gender"] == "male", (
        "presentation_gender must reflect the ACTUAL delivered preset, not "
        "the exhausted 'female' request")


def test_non_bark_announcer_engine_leaves_the_bark_helper_a_no_op():
    """`announcer_voice_engine` defaulting/resolving to anything other than
    bark must not touch the announcer row at all -- today's byte-identical
    behavior for the common case."""
    from nodes.cast_lock import CastLock

    cast = _bark_announcer_cast()
    before = dict(next(r for r in cast if r["name"] == "ANNOUNCER"))
    CastLock._assign_bark_voices(cast, {"episode_seed": 42}, [],
                                 announcer_voice_engine="kokoro")
    after = next(r for r in cast if r["name"] == "ANNOUNCER")
    assert after == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
