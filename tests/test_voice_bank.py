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

from nodes._otr_voice_bank import CASTING_POLICY_VERSION, VoiceBankEntry, VoiceCastingError, announcer_voice_ref, assign_voice_for_slot, load_voice_bank, stable_cast_seed, voice_ref_usage_keys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "_otr_voice_bank.py"


def _entry(vid, *, engine="chatterbox", gender="male", timbre=(),
           roles=("char_voice",), age="adult", clean=True, ref_path=None):
    return VoiceBankEntry(
        voice_ref_id=vid, engine=engine, gender=gender, timbre=tuple(timbre),
        roles=tuple(roles), age_band=age, ref_path=ref_path or f"refs/{vid}.wav",
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
    #
    # TWO warm elders, not one. With a single warm entry this tier is size 1 and
    # the min-tier-pool floor skips it, so the test would fall through to the
    # gender-only tier and still pass by coincidence -- silently no longer
    # exercising the age-drop it is named for. Two candidates clear the floor.
    bank_age = (_entry("warm_elder", timbre=["warm"], age="elder"),
                _entry("warm_elder_2", timbre=["warm"], age="elder"),
                _entry("bright_adult", timbre=["bright"], age="adult"))
    assert _cast(bank_age).voice_ref_id in {"warm_elder", "warm_elder_2"}

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


def test_caster_blocks_aliases_with_same_ref_path():
    used_ref = _entry("primary", timbre=["warm"], ref_path="refs/shared.wav")
    alternate = _entry("voice_alt", timbre=["warm"], ref_path="refs/shared.wav")
    with pytest.raises(VoiceCastingError):
        _cast((alternate,), used_voice_ref_ids=voice_ref_usage_keys(used_ref))

    chosen = _cast(
        (alternate,),
        used_voice_ref_ids=voice_ref_usage_keys(used_ref),
        allow_voice_reuse=True,
    )
    assert chosen.voice_ref_id == "voice_alt"


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
_KOKORO_ANNOUNCERS = {"bm_george", "bm_fable", "bf_emma", "bf_lily"}


def test_announcer_voice_ref_resolves_for_active_engine():
    assert announcer_voice_ref("kokoro").voice_ref_id in _KOKORO_ANNOUNCERS
    assert announcer_voice_ref("chatterbox").voice_ref_id in {
        "cb_announcer_male", "cb_announcer_female",
    }
    assert announcer_voice_ref("dia").voice_ref_id in {
        "dia_announcer_male", "dia_announcer_female",
    }


def test_kokoro_announcer_rotates_across_the_whole_curated_pool():
    """The announcer was pinned to bm_george on EVERY episode: kokoro carried a
    single announcer_voice row, so the legacy `sorted(...)[0]` branch had one
    candidate. Every published episode opened with the same voice.

    This asserts the seeded draw reaches all four curated announcers AND that
    bm_george is still among them -- a data-only fix (adding the role to the
    other three without the seeded pick) retires bm_george entirely, because
    the sorted head within each gender is bm_fable / bf_emma.
    """
    seen = {
        announcer_voice_ref("kokoro", episode_seed=s).voice_ref_id
        for s in range(64)
    }
    assert seen == _KOKORO_ANNOUNCERS
    assert "bm_george" in seen


def test_seeded_announcer_draw_leaves_single_candidate_engines_unmoved():
    """chatterbox / dia / google_tts each expose exactly ONE preferred announcer
    per gender, so the seeded draw runs over a one-element list and is the
    identity. Their pick sequences must not move -- this is what keeps the
    change scoped to kokoro.
    """
    for engine, expected in (
        ("chatterbox", {"cb_announcer_male", "cb_announcer_female"}),
        ("dia", {"dia_announcer_male", "dia_announcer_female"}),
        ("google_tts", {"gt_charon_announcer", "gt_sulafat_announcer"}),
    ):
        seq = [
            announcer_voice_ref(engine, episode_seed=s).voice_ref_id
            for s in range(60)
        ]
        assert set(seq) == expected, engine


def test_kokoro_announcer_pick_is_deterministic_for_a_seed():
    for seed in (0, 7, 42, 100003):
        picks = {
            announcer_voice_ref("kokoro", episode_seed=seed).voice_ref_id
            for _ in range(5)
        }
        assert len(picks) == 1, seed


def test_chatterbox_voice_bank_has_british_gendered_announcers():
    entries, _ = load_voice_bank()
    announcers = [
        e for e in entries
        if e.engine == "chatterbox" and "announcer_voice" in e.roles
        and "preferred_announcer" in e.style_tags
    ]
    genders = {e.gender for e in announcers}
    assert {"male", "female"} <= genders
    assert all("british_leaning" in e.style_tags for e in announcers)


def test_chatterbox_announcer_selection_mixes_gender_by_episode_seed():
    picks = [
        announcer_voice_ref("chatterbox", episode_seed=i)
        for i in range(80)
    ]
    genders = [p.gender for p in picks]
    assert set(genders) == {"male", "female"}
    delta = abs(genders.count("male") - genders.count("female"))
    assert delta <= 24
    assert {p.voice_ref_id for p in picks} <= {
        "cb_announcer_male", "cb_announcer_female",
    }
    assert announcer_voice_ref("chatterbox", episode_seed=17) == announcer_voice_ref(
        "chatterbox", episode_seed=17)


def test_dia_voice_bank_has_british_gendered_announcers():
    entries, _ = load_voice_bank()
    announcers = [
        e for e in entries
        if e.engine == "dia" and "announcer_voice" in e.roles
        and "preferred_announcer" in e.style_tags
    ]
    genders = {e.gender for e in announcers}
    assert {"male", "female"} <= genders
    assert all("british_leaning" in e.style_tags for e in announcers)


def test_dia_announcer_selection_mixes_gender_by_episode_seed():
    picks = [
        announcer_voice_ref("dia", episode_seed=i)
        for i in range(80)
    ]
    genders = [p.gender for p in picks]
    assert set(genders) == {"male", "female"}
    delta = abs(genders.count("male") - genders.count("female"))
    assert delta <= 24
    assert {p.voice_ref_id for p in picks} <= {
        "dia_announcer_male", "dia_announcer_female",
    }
    assert announcer_voice_ref("dia", episode_seed=17) == announcer_voice_ref(
        "dia", episode_seed=17)


def test_google_voice_bank_has_gendered_chars_and_announcers():
    entries, _ = load_voice_bank()
    google = [e for e in entries if e.engine == "google_tts"]
    chars = [e for e in google if "char_voice" in e.roles]
    announcers = [e for e in google if "announcer_voice" in e.roles]
    char_genders = {g: sum(1 for e in chars if e.gender == g) for g in ("male", "female")}
    ann_genders = {e.gender for e in announcers if "preferred_announcer" in e.style_tags}
    assert char_genders["male"] >= 2
    assert char_genders["female"] >= 2
    assert {"male", "female"} <= ann_genders
    assert all("british_leaning" in e.style_tags for e in announcers)


def test_google_announcer_selection_mixes_gender_by_episode_seed():
    picks = [
        announcer_voice_ref("google_tts", episode_seed=i)
        for i in range(80)
    ]
    genders = [p.gender for p in picks]
    assert set(genders) == {"male", "female"}
    delta = abs(genders.count("male") - genders.count("female"))
    assert delta <= 24
    assert announcer_voice_ref("google_tts", episode_seed=17) == announcer_voice_ref(
        "google_tts", episode_seed=17)


def test_google_castlock_assigns_gendered_chars_and_separate_announcer():
    from nodes.cast_lock import CastLock

    led = {"meta": {"episode_seed": "google-seed"}}
    cast = [
        {"char_id": "ann", "name": "ANNOUNCER", "speaker_role": "announcer"},
        {"char_id": "m1", "name": "CAPTAIN", "gender": "male"},
        {"char_id": "f1", "name": "DOCTOR", "gender": "female"},
    ]
    report = []
    CastLock()._auto_registry(
        led, cast, "google_tts", False, report,
        char_voice_engine="google_tts",
        announcer_voice_engine="google_tts",
    )
    entries, _ = load_voice_bank()
    by_id = {e.voice_ref_id: e for e in entries}
    ann, male, female = cast
    assert ann["voice_engine"] == "google_tts"
    assert male["voice_engine"] == "google_tts"
    assert female["voice_engine"] == "google_tts"
    assert ann["voice_ref_id"] not in {male["voice_ref_id"], female["voice_ref_id"]}
    assert by_id[male["voice_ref_id"]].gender == "male"
    assert by_id[female["voice_ref_id"]].gender == "female"
    assert ann["provider_voice_id"] not in {
        male["provider_voice_id"], female["provider_voice_id"],
    }


def test_chatterbox_castlock_uses_british_announcer_pair_and_separate_chars():
    from nodes.cast_lock import CastLock

    led = {"meta": {"episode_seed": "chatterbox-seed"}}
    cast = [
        {"char_id": "ann", "name": "ANNOUNCER", "speaker_role": "announcer"},
        {"char_id": "m1", "name": "CAPTAIN", "gender": "male"},
        {"char_id": "f1", "name": "DOCTOR", "gender": "female"},
    ]
    report = []
    CastLock()._auto_registry(
        led, cast, "default", False, report,
        char_voice_engine="chatterbox",
        announcer_voice_engine="chatterbox",
    )
    entries, _ = load_voice_bank()
    by_id = {e.voice_ref_id: e for e in entries}
    ann, male, female = cast
    assert ann["voice_engine"] == "chatterbox"
    assert male["voice_engine"] == "chatterbox"
    assert female["voice_engine"] == "chatterbox"
    assert ann["voice_ref_id"] in {"cb_announcer_male", "cb_announcer_female"}
    assert "british_leaning" in by_id[ann["voice_ref_id"]].style_tags
    assert ann["voice_ref_id"] not in {male["voice_ref_id"], female["voice_ref_id"]}
    assert by_id[ann["voice_ref_id"]].ref_path not in {
        by_id[male["voice_ref_id"]].ref_path,
        by_id[female["voice_ref_id"]].ref_path,
    }
    assert by_id[male["voice_ref_id"]].gender == "male"
    assert by_id[female["voice_ref_id"]].gender == "female"


def test_dia_castlock_uses_british_announcer_pair_and_separate_chars():
    from nodes.cast_lock import CastLock

    led = {"meta": {"episode_seed": "dia-seed"}}
    cast = [
        {"char_id": "ann", "name": "ANNOUNCER", "speaker_role": "announcer"},
        {"char_id": "m1", "name": "CAPTAIN", "gender": "male"},
        {"char_id": "f1", "name": "DOCTOR", "gender": "female"},
    ]
    report = []
    CastLock()._auto_registry(
        led, cast, "default", False, report,
        char_voice_engine="dia",
        announcer_voice_engine="dia",
    )
    entries, _ = load_voice_bank()
    by_id = {e.voice_ref_id: e for e in entries}
    ann, male, female = cast
    assert ann["voice_engine"] == "dia"
    assert male["voice_engine"] == "dia"
    assert female["voice_engine"] == "dia"
    assert ann["voice_ref_id"] in {"dia_announcer_male", "dia_announcer_female"}
    assert "british_leaning" in by_id[ann["voice_ref_id"]].style_tags
    assert ann["voice_ref_id"] not in {male["voice_ref_id"], female["voice_ref_id"]}
    assert by_id[ann["voice_ref_id"]].ref_path not in {
        by_id[male["voice_ref_id"]].ref_path,
        by_id[female["voice_ref_id"]].ref_path,
    }
    assert by_id[male["voice_ref_id"]].gender == "male"
    assert by_id[female["voice_ref_id"]].gender == "female"


def test_announcer_voice_ref_raises_for_engine_without_ref():
    with pytest.raises(VoiceCastingError):
        announcer_voice_ref("musicgen")  # music engine, no announcer ref


def test_casting_policy_version_is_pinned():
    # v2 = the 2026-06-11 whiny-fix weighted-lottery re-baseline (operator-
    # directed): scores WEIGHT the in-tier draw instead of only sorting it.
    # Bumping this constant is a deliberate C7 re-baseline of casting draws.
    assert CASTING_POLICY_VERSION == "3"


# ----------------------------------------------------------------------------
# Whiny-fix: weighted lottery (casting v2) + quality tiers
# ----------------------------------------------------------------------------
def _tiered(vid, tier="", tags=(), **kw):
    base = _entry(vid, **kw)
    return VoiceBankEntry(**{**base.__dict__, "quality_tier": tier,
                             "style_tags": tuple(tags)})


def test_weighted_lottery_favors_higher_scores():
    """v2: within ONE ladder tier the higher-scored entry must win the draw
    more often. Both entries land in the gender-only tier (no timbre match for
    either, so no earlier tier fires); 'scored' additionally matches role+age
    (score 130) vs 'bare' (100) -- weights 131 vs 101."""
    bank = (
        _entry("scored", timbre=("nasal",), roles=("char_voice",), age="adult"),
        _entry("bare", timbre=("nasal",), roles=("announcer_voice",), age="elder"),
    )
    wins = {"scored": 0, "bare": 0}
    for i in range(400):
        e = _cast(bank, char_id=f"c{i}", allow_voice_reuse=True)
        wins[e.voice_ref_id] += 1
    assert wins["scored"] > wins["bare"], wins
    assert wins["bare"] > 0, "lower-score entries must keep a small chance"


def test_weighted_lottery_deterministic_per_slot():
    bank = (_entry("a"), _entry("b"), _entry("c"))
    picks = {_cast(bank, allow_voice_reuse=True).voice_ref_id for _ in range(5)}
    assert len(picks) == 1, "same slot identity must always draw the same ref"


def test_uniform_v1_draw_reachable_via_env(monkeypatch):
    bank = (
        _entry("full_match", timbre=("nasal",), roles=("char_voice",), age="adult"),
        _entry("bare_gender", timbre=("nasal",), roles=("announcer_voice",), age="elder"),
    )
    monkeypatch.setenv("OTR_CAST_WEIGHTED", "0")
    uniform = [_cast(bank, char_id=f"u{i}", allow_voice_reuse=True).voice_ref_id
               for i in range(200)]
    share = uniform.count("bare_gender") / len(uniform)
    assert 0.3 < share < 0.7, f"uniform draw expected near 50/50, got {share}"


def test_reject_tier_never_casts():
    bank = (
        _tiered("good_ref", tier="a", tags=("lead_safe",)),
        _tiered("bad_ref", tier="reject"),
    )
    for i in range(50):
        e = _cast(bank, char_id=f"r{i}", allow_voice_reuse=True)
        assert e.voice_ref_id == "good_ref"


def test_filter_by_quality_tier_stages():
    from nodes._otr_voice_bank import filter_by_quality_tier
    a_lead = _tiered("a_lead", tier="a", tags=("lead_safe",))
    a_plain = _tiered("a_plain", tier="a")
    b_ref = _tiered("b_ref", tier="b")
    rej = _tiered("rej", tier="reject")
    untiered = _tiered("untiered")
    pool = (a_lead, a_plain, b_ref, rej, untiered)
    # reject dropped always; non-lead keeps everything else
    assert rej not in filter_by_quality_tier(pool)
    assert set(filter_by_quality_tier(pool)) == {a_lead, a_plain, b_ref, untiered}
    # leads: tier-a lead_safe wins when present
    assert filter_by_quality_tier(pool, lead=True) == [a_lead]
    # leads with no lead_safe: all tier-a
    pool2 = (a_plain, b_ref, untiered)
    assert filter_by_quality_tier(pool2, lead=True) == [a_plain]
    # un-audited bank: passes through unchanged (no tiers stamped anywhere)
    pool3 = (untiered, _tiered("untiered2"))
    assert filter_by_quality_tier(pool3, lead=True) == list(pool3)


def test_additive_tier_fields_parse_from_bank_rows():
    from nodes._otr_voice_bank import _entry_from_dict
    row = {
        "voice_ref_id": "x", "engine": "indextts2", "gender": "male",
        "timbre": ["warm"], "roles": ["char_voice"], "age_band": "adult",
        "ref_path": "refs/x.wav", "ref_sha256": "s", "commercial_clean": True,
        "quality_tier": "a", "style_tags": ["lead_safe"],
    }
    e = _entry_from_dict(row)
    assert e.quality_tier == "a" and e.style_tags == ("lead_safe",)
    # absent fields default empty (un-audited bank unchanged)
    del row["quality_tier"], row["style_tags"]
    e2 = _entry_from_dict(row)
    assert e2.quality_tier == "" and e2.style_tags == ()


def test_source_is_ascii_no_em_dash():
    src = NODE_SRC.read_text(encoding="utf-8")
    assert "—" not in src
    src.encode("ascii")


def test_shipped_bank_has_gendered_indextts2_voices():
    # Sprint 3: indextts2 carries gender-mapped reference voices (kokoro-seeded)
    # so a male slot gets a male voice and a female slot a female one.
    entries, _ = load_voice_bank()
    idx = [e for e in entries if e.engine == "indextts2"]
    genders = {e.gender for e in idx}
    assert "male" in genders and "female" in genders, "indextts2 needs male + female refs"
    male = assign_voice_for_slot(role="char_voice", engine="indextts2", char_id="M",
                                 gender="male", timbre=("warm",), age_band="adult", episode_seed=1)
    female = assign_voice_for_slot(role="char_voice", engine="indextts2", char_id="F",
                                   gender="female", timbre=("bright",), age_band="adult", episode_seed=1)
    assert male.engine == "indextts2" and male.gender == "male"
    assert female.engine == "indextts2" and female.gender == "female"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
