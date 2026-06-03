"""Wave 1 / 1d -- voice reference bank validator + deterministic caster (E.1/E.2).

Headless. Exercises the shipped bank against its schema, the caster's scoring
ladder + seeded stability + RNG disjointness (I-4), and the per-engine announcer
pin. Ladder precision is tested with synthetic banks injected via the ``bank``
parameter so the assertions do not depend on the shipped bank's exact contents.
"""
from __future__ import annotations

import pathlib
import random

import pytest

from nodes._otr_voice_bank import (
    CASTING_POLICY_VERSION,
    VoiceBankEntry,
    VoiceCastingError,
    announcer_voice_ref,
    assign_voice_for_slot,
    get_all_registered_voices,
    load_voice_bank,
    stable_cast_seed,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "_otr_voice_bank.py"


def _entry(vid, *, engine="chatterbox", gender="male", timbre=(),
           roles=("char_voice",), age="adult", clean=True):
    return VoiceBankEntry(
        voice_ref_id=vid, engine=engine, gender=gender, timbre=tuple(timbre),
        roles=tuple(roles), age_band=age, ref_path=f"refs/{vid}.wav",
        ref_sha256="pending", commercial_clean=clean,
    )


# ----------------------------------------------------------------------------
# Shipped bank validates + loads
# ----------------------------------------------------------------------------
def test_shipped_bank_loads_and_validates():
    entries, sha = load_voice_bank()
    assert entries and isinstance(sha, str) and len(sha) == 64
    ids = [e.voice_ref_id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate voice_ref_id in shipped bank"
    # Every entry has a registered engine name.
    from nodes._otr_audio_engines import is_registered
    for e in entries:
        assert is_registered(e.engine), f"bank engine {e.engine!r} not registered"


def test_get_all_registered_voices_stable_sorted():
    voices = get_all_registered_voices()
    ids = [v.voice_ref_id for v in voices]
    assert ids == sorted(ids)


def test_bank_entries_code_to_schema():
    import json
    schema = json.loads(
        (REPO_ROOT / "config" / "voice_bank_entry_schema.json").read_text(encoding="utf-8")
    )
    raw = json.loads(
        (REPO_ROOT / "config" / "voice_reference_bank.json").read_text(encoding="utf-8")
    )
    required = set(schema["required"])
    for row in raw["voices"]:
        assert required <= set(row), f"{row.get('voice_ref_id')}: missing fields"
        assert isinstance(row["timbre"], list)
        assert isinstance(row["roles"], list) and row["roles"]
        assert isinstance(row["commercial_clean"], bool)


def test_duplicate_voice_ref_id_rejected(tmp_path):
    import json
    from nodes._otr_voice_bank import VoiceBankError

    bad = {
        "voices": [
            {
                "voice_ref_id": "dup", "engine": "chatterbox", "gender": "male",
                "timbre": ["warm"], "roles": ["char_voice"], "age_band": "adult",
                "ref_path": "a.wav", "ref_sha256": "x", "commercial_clean": True,
            },
            {
                "voice_ref_id": "dup", "engine": "chatterbox", "gender": "female",
                "timbre": ["bright"], "roles": ["char_voice"], "age_band": "adult",
                "ref_path": "b.wav", "ref_sha256": "y", "commercial_clean": True,
            },
        ]
    }
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(VoiceBankError):
        load_voice_bank(str(p))


# ----------------------------------------------------------------------------
# Caster: stability, scoring, ladder
# ----------------------------------------------------------------------------
def _cast(bank, **kw):
    base = dict(
        role="char_voice", engine="chatterbox", char_id="c1", gender="male",
        timbre=("warm",), age_band="adult", episode_seed=12345, bank=bank,
    )
    base.update(kw)
    return assign_voice_for_slot(**base)


def test_caster_is_per_slot_stable():
    bank = (
        _entry("a", timbre=["warm"], age="adult"),
        _entry("b", timbre=["warm"], age="adult"),
        _entry("c", timbre=["warm"], age="adult"),
    )
    first = _cast(bank)
    for _ in range(5):
        assert _cast(bank).voice_ref_id == first.voice_ref_id


def test_caster_uses_own_rng_not_global():
    bank = (_entry("a", timbre=["warm"]), _entry("b", timbre=["warm"]),
            _entry("c", timbre=["warm"]))
    random.seed(1)
    one = _cast(bank).voice_ref_id
    random.seed(999999)
    two = _cast(bank).voice_ref_id
    assert one == two, "caster must use its own seeded RNG, not global random (I-4)"


def test_caster_gender_floor_holds():
    bank = (_entry("f1", gender="female", timbre=["warm"]),
            _entry("m1", gender="male", timbre=["xyz"]))
    chosen = _cast(bank)  # male slot
    assert chosen.gender == "male"


def test_caster_ladder_drops_age_then_role_then_timbre():
    # Drop age: warm+char_voice but wrong age beats a bright adult.
    bank_age = (_entry("warm_elder", timbre=["warm"], age="elder"),
                _entry("bright_adult", timbre=["bright"], age="adult"))
    assert _cast(bank_age).voice_ref_id == "warm_elder"

    # Drop role: only a gender+timbre match exists, wrong role.
    bank_role = (_entry("ann", timbre=["warm"], roles=["announcer_voice"], age="adult"),)
    assert _cast(bank_role).voice_ref_id == "ann"

    # Gender-only: no timbre match anywhere, but gender matches.
    bank_gender = (_entry("g_only", timbre=["nope"], age="elder"),)
    assert _cast(bank_gender).voice_ref_id == "g_only"


def test_caster_raises_when_no_gender_match():
    bank = (_entry("f1", gender="female", timbre=["warm"]),)
    with pytest.raises(VoiceCastingError):
        _cast(bank)  # male slot, only female available


def test_caster_no_reuse_then_reuse():
    bank = (_entry("a", timbre=["warm"]),)
    # 'a' already used, no other candidate -> raise without reuse.
    with pytest.raises(VoiceCastingError):
        _cast(bank, used_voice_ref_ids={"a"})
    # ...but allow_voice_reuse re-walks the ladder permitting the used ref.
    chosen = _cast(bank, used_voice_ref_ids={"a"}, allow_voice_reuse=True)
    assert chosen.voice_ref_id == "a"


def test_caster_carries_commercial_status():
    bank = (_entry("gated", timbre=["warm"], clean=False),)
    chosen = _cast(bank)
    assert chosen.commercial_clean is False


def test_caster_keys_on_char_id_not_name():
    import inspect
    sig = inspect.signature(assign_voice_for_slot)
    assert "char_id" in sig.parameters
    assert "name" not in sig.parameters, "caster keys on char_id, never name (I-9)"
    # Different char_id with otherwise identical dims can diverge.
    bank = tuple(_entry(f"v{i}", timbre=["warm"]) for i in range(6))
    a = assign_voice_for_slot(role="char_voice", engine="chatterbox", char_id="A",
                              gender="male", timbre=("warm",), age_band="adult",
                              episode_seed=7, bank=bank)
    b = assign_voice_for_slot(role="char_voice", engine="chatterbox", char_id="B",
                              gender="male", timbre=("warm",), age_band="adult",
                              episode_seed=7, bank=bank)
    # Same char_id is deterministic; the seed incorporates char_id.
    a2 = assign_voice_for_slot(role="char_voice", engine="chatterbox", char_id="A",
                               gender="male", timbre=("warm",), age_band="adult",
                               episode_seed=7, bank=bank)
    assert a.voice_ref_id == a2.voice_ref_id
    assert isinstance(a.voice_ref_id, str) and isinstance(b.voice_ref_id, str)


def test_stable_cast_seed_null_safe():
    s1 = stable_cast_seed(episode_seed=None, casting_policy_version=None,
                          char_id=None, gender=None, timbre=None, role=None,
                          age_band=None)
    s2 = stable_cast_seed(episode_seed="", casting_policy_version="",
                          char_id="", gender="", timbre=[], role="", age_band="")
    assert s1 == s2  # null -> "" canonicalization
    assert isinstance(s1, int) and s1 >= 0


# ----------------------------------------------------------------------------
# Announcer pin (E.1)
# ----------------------------------------------------------------------------
def test_announcer_voice_ref_resolves_for_active_engine():
    assert announcer_voice_ref("kokoro").voice_ref_id == "bm_george"
    assert announcer_voice_ref("chatterbox").voice_ref_id == "cc_announcer_male"


def test_announcer_voice_ref_raises_for_engine_without_ref():
    with pytest.raises(VoiceCastingError):
        announcer_voice_ref("musicgen")  # music engine, no announcer ref


def test_casting_policy_version_is_pinned():
    assert CASTING_POLICY_VERSION == "1"


def test_source_is_ascii_no_em_dash():
    src = NODE_SRC.read_text(encoding="utf-8")
    assert "—" not in src
    src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
