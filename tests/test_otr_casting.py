"""tests/test_otr_casting.py -- unit tests for the cast contract caller.

Covers:
  - assemble_pre_locked_rows: determinism, slot accounting, LEMMY hit/miss
  - ANNOUNCER 50/50 male/female distribution
  - Voice pre-filter (pre-locked voices excluded from open slot pool)
  - CastingResponse / DescriptionResponse schema validation
  - Sprint 3D three-stage split:
      * precompute_ensemble_slots -- Python owns gender/timbre/role
      * llm_write_description     -- LLM writes description only
      * python_assign_voice_preset -- Python picks the voice
  - cast_one_character: composes the three stages, reroll on bad JSON,
    repair on attempt 3
  - lock_cast end-to-end: pre-locked rows + Python-assembled rows
  - Prompt shape: description-only, no Voices: block, no balance line
  - Voice uniqueness still holds (the existing _assert_unique_bark_voices)

Hermetic: no GPU, no I/O, no ComfyUI imports. All LLM calls go
through stub generate_fn callables.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

# Make `nodes/` importable + put the repo root on sys.path so
# `config.cast_pools` resolves. Mirrors test_otr_ledger_consumers.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for p in (_REPO_ROOT, _NODES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _otr_casting as _OTRC  # noqa: E402
from config import cast_pools as _POOLS  # noqa: E402


# ---------------------------------------------------------------------------
# Stub generate_fn helpers
# ---------------------------------------------------------------------------


def _make_canned_generate_fn(responses):
    """Return a generate_fn that yields each response in order. Once
    the list is exhausted, raises RuntimeError('canned exhausted').
    """
    iterator = iter(responses)

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        try:
            return next(iterator)
        except StopIteration as exc:
            raise RuntimeError("canned exhausted") from exc

    return gen_fn


def _desc_response(
    description="Sharp-eyed scientist, 30s, anxious about the data.",
):
    """Sprint 3D: the LLM produces description ONLY. Gender and
    voice_preset are no longer LLM concerns."""
    return json.dumps({"character_description": description})


# Back-compat: some legacy-shaped responses still carry gender +
# voice_preset. The new DescriptionResponse schema ignores the extra
# keys (pydantic by default), so a fat response still parses.
def _good_response(name="ALICE",  # noqa: ARG001
                   description="Sharp-eyed scientist, 30s, anxious about the data.",
                   gender="female",  # noqa: ARG001
                   voice_preset="v2/en_speaker_4"):  # noqa: ARG001
    return _desc_response(description)


# ---------------------------------------------------------------------------
# assemble_pre_locked_rows
# ---------------------------------------------------------------------------


def test_assemble_pre_locked_rows_no_lemmy_basic_shape():
    rng = random.Random("test-seed-1")
    pre_locked, open_slots, lemmy_hit = _OTRC.assemble_pre_locked_rows(
        num_characters=3, rng=rng, force_lemmy=False,
    )
    assert lemmy_hit is False
    # ANNOUNCER only in pre_locked when LEMMY misses
    assert len(pre_locked) == 1
    assert pre_locked[0]["name"] == "ANNOUNCER"
    assert pre_locked[0]["char_id"] == "c01"
    # ANNOUNCER's voice is a Kokoro ID (bm_/bf_/am_/af_ prefix), NOT
    # a Bark preset. Kokoro and Bark are separate TTS namespaces.
    assert pre_locked[0]["voice_preset"].startswith(
        ("bm_", "bf_", "am_", "af_")
    )
    # Open slots fill ALL the requested num_characters when LEMMY misses
    assert len(open_slots) == 3
    assert [s.char_id for s in open_slots] == ["c02", "c03", "c04"]
    # Names from the pool (uppercase, two words)
    for s in open_slots:
        assert s.name.isupper(), f"slot name not uppercase: {s.name!r}"
        assert " " in s.name, f"slot name not 'FIRST LAST': {s.name!r}"


def test_assemble_pre_locked_rows_lemmy_consumes_a_slot():
    rng = random.Random("test-seed-2")
    pre_locked, open_slots, lemmy_hit = _OTRC.assemble_pre_locked_rows(
        num_characters=3, rng=rng, force_lemmy=True,
    )
    assert lemmy_hit is True
    # ANNOUNCER + LEMMY pre-locked when LEMMY hits
    assert len(pre_locked) == 2
    assert pre_locked[0]["name"] == "ANNOUNCER"
    assert pre_locked[1]["name"] == "LEMMY"
    assert pre_locked[1]["char_id"] == "c02"
    assert pre_locked[1]["gender"] == "male"
    assert pre_locked[1]["voice_preset"] == "v2/en_speaker_8"
    # Open slots = num_characters - 1 (LEMMY consumed one)
    assert len(open_slots) == 2
    assert [s.char_id for s in open_slots] == ["c03", "c04"]


def test_assemble_pre_locked_rows_determinism():
    """Same seed + same force_lemmy must yield identical results."""
    pre1, open1, hit1 = _OTRC.assemble_pre_locked_rows(
        num_characters=4, rng=random.Random("det-test"), force_lemmy=False,
    )
    pre2, open2, hit2 = _OTRC.assemble_pre_locked_rows(
        num_characters=4, rng=random.Random("det-test"), force_lemmy=False,
    )
    assert hit1 == hit2
    assert [r["name"] for r in pre1] == [r["name"] for r in pre2]
    assert [r["voice_preset"] for r in pre1] == [r["voice_preset"] for r in pre2]
    assert [s.name for s in open1] == [s.name for s in open2]


def test_assemble_pre_locked_rows_announcer_pool_has_2m_2f():
    """Determinism-friendly check: the ANNOUNCER pool itself contains
    exactly 2 male + 2 female Kokoro presets. The 50/50 distribution
    is then a property of the pool composition, not of statistical
    luck across runs.

    Kokoro convention: bm_* = British male, bf_* = British female,
    am_* = American male, af_* = American female. (Today's pool is
    BBC voices only -- bm_* and bf_*.)

    Replaces the prior 200-trial statistical test (CI flake bomb per
    round-robin synthesis 2026-05-10).
    """
    male_count = sum(1 for vid, _ in _POOLS.ANNOUNCER_PRESETS
                     if vid.startswith(("bm_", "am_")))
    female_count = sum(1 for vid, _ in _POOLS.ANNOUNCER_PRESETS
                       if vid.startswith(("bf_", "af_")))
    assert male_count == 2, \
        f"ANNOUNCER pool male count drifted: {male_count} (expected 2)"
    assert female_count == 2, \
        f"ANNOUNCER pool female count drifted: {female_count} (expected 2)"
    assert male_count + female_count == len(_POOLS.ANNOUNCER_PRESETS), \
        f"ANNOUNCER pool has entries that are neither bm_/am_ nor " \
        f"bf_/af_: {_POOLS.ANNOUNCER_PRESETS!r}"


def test_announcer_row_carries_tts_model_kokoro():
    """ANNOUNCER cast row stamps tts_model="kokoro" so downstream
    consumers can route by reading the field directly."""
    rng = random.Random("tts-model-announcer")
    pre_locked, _, _ = _OTRC.assemble_pre_locked_rows(
        num_characters=2, rng=rng, force_lemmy=False,
    )
    announcer = pre_locked[0]
    assert announcer["name"] == "ANNOUNCER"
    assert announcer["tts_model"] == "kokoro"


def test_lemmy_row_carries_tts_model_bark():
    """LEMMY cast row stamps tts_model="bark"."""
    rng = random.Random("tts-model-lemmy")
    pre_locked, _, _ = _OTRC.assemble_pre_locked_rows(
        num_characters=3, rng=rng, force_lemmy=True,
    )
    lemmy = pre_locked[1]
    assert lemmy["name"] == "LEMMY"
    assert lemmy["tts_model"] == "bark"
    assert lemmy["voice_preset"] == "v2/en_speaker_8"


def test_voice_registry_exposes_bark_and_kokoro():
    """VOICE_REGISTRY is the unified per-model voice catalog."""
    assert "bark" in _POOLS.VOICE_REGISTRY
    assert "kokoro" in _POOLS.VOICE_REGISTRY
    assert _POOLS.KNOWN_TTS_MODELS == ("bark", "kokoro")
    # Each entry exposes presets + params_spec
    for model, entry in _POOLS.VOICE_REGISTRY.items():
        assert "presets" in entry, f"{model}: missing presets"
        assert "params_spec" in entry, f"{model}: missing params_spec"
        assert isinstance(entry["presets"], list)
        assert isinstance(entry["params_spec"], dict)
        # presets is list of (preset_id, short_desc) tuples
        for preset_entry in entry["presets"]:
            assert isinstance(preset_entry, tuple)
            assert len(preset_entry) == 2


def test_voice_registry_bark_presets_match_voice_profiles():
    """VOICE_REGISTRY['bark']['presets'] is a flat view of VOICE_PROFILES;
    the preset IDs must match exactly (the descriptions in the registry
    are flattened from quality tags)."""
    bark_presets = {p for p, _ in _POOLS.VOICE_REGISTRY["bark"]["presets"]}
    profile_presets = {p for p, _, _, _ in _POOLS.VOICE_PROFILES}
    assert bark_presets == profile_presets


def test_announcer_row_carries_voice_params_none():
    """voice_params default is None until the casting LLM is wired
    to pick model-specific knobs (Phase 2)."""
    rng = random.Random("voice-params-announcer")
    pre_locked, _, _ = _OTRC.assemble_pre_locked_rows(
        num_characters=2, rng=rng, force_lemmy=False,
    )
    assert pre_locked[0]["voice_params"] is None


def test_lemmy_row_carries_voice_params_none():
    rng = random.Random("voice-params-lemmy")
    pre_locked, _, _ = _OTRC.assemble_pre_locked_rows(
        num_characters=2, rng=rng, force_lemmy=True,
    )
    lemmy = pre_locked[1]
    assert lemmy["voice_params"] is None


def test_lock_cast_open_character_rows_carry_voice_params_none():
    rng = random.Random("voice-params-open")
    gen = _make_canned_generate_fn([_desc_response()])
    cast, _ = _OTRC.lock_cast(creative_fn=gen,
        num_characters=1,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=False,
    )
    open_row = cast[1]  # cast[0] is ANNOUNCER
    assert open_row["voice_params"] is None


def test_lock_cast_open_character_rows_carry_tts_model_bark():
    """Every Python-assembled open-character row must stamp
    tts_model="bark" since open characters are drawn from the Bark
    VOICE_PROFILES pool by construction."""
    rng = random.Random("tts-model-open")
    gen = _make_canned_generate_fn([_desc_response(), _desc_response()])
    cast, _ = _OTRC.lock_cast(creative_fn=gen,
        num_characters=2,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=False,
    )
    # cast[0] is ANNOUNCER (Kokoro), cast[1:] are open characters (Bark).
    for row in cast[1:]:
        assert row["tts_model"] == "bark", \
            f"open character {row['name']!r} missing tts_model=bark: {row!r}"


def test_announcer_voice_is_kokoro_namespace_not_bark():
    """The announcer's voice_preset must come from the Kokoro
    namespace (bm_*, bf_*, am_*, af_*), NOT the Bark namespace
    (v2/en_speaker_*). Per Jeffrey 2026-05-10: announcer renders
    through Kokoro, characters render through Bark, two separate
    TTS pools so they cannot collide."""
    bark_presets = {p for p, _, _, _ in _POOLS.VOICE_PROFILES}
    for voice_id, _ in _POOLS.ANNOUNCER_PRESETS:
        assert voice_id not in bark_presets, \
            f"ANNOUNCER preset {voice_id!r} collides with Bark pool"
        assert voice_id.startswith(("bm_", "bf_", "am_", "af_")), \
            f"ANNOUNCER preset {voice_id!r} not in Kokoro namespace"


def test_assemble_pre_locked_rows_announcer_picks_each_preset_with_fixed_seeds():
    """Lock down the deterministic mapping from seed -> announcer pick.

    With a fixed-seed random.Random(), the announcer pick is fully
    deterministic. We assert that across a bank of seeds we hit BOTH
    a male announcer and a female announcer at least once -- so the
    pool is exercised in both directions, without any statistical
    tolerance band.
    """
    seen_genders = set()
    for i in range(20):
        rng = random.Random(f"announcer-seed-{i}")
        pre_locked, _, _ = _OTRC.assemble_pre_locked_rows(
            num_characters=2, rng=rng, force_lemmy=False,
        )
        seen_genders.add(pre_locked[0]["gender"])
    assert seen_genders == {"male", "female"}, \
        f"announcer pool not exercised in both directions: {seen_genders!r}"


def test_assemble_pre_locked_rows_num_characters_bounds():
    rng = random.Random("bounds")
    with pytest.raises(ValueError):
        _OTRC.assemble_pre_locked_rows(num_characters=0, rng=rng)
    with pytest.raises(ValueError):
        _OTRC.assemble_pre_locked_rows(num_characters=7, rng=rng)
    # Boundaries OK
    _OTRC.assemble_pre_locked_rows(num_characters=1, rng=rng)
    _OTRC.assemble_pre_locked_rows(num_characters=6, rng=rng)


def test_assemble_pre_locked_rows_no_name_collisions_in_pool_fill():
    """Multiple open slots must all have distinct names."""
    rng = random.Random("collision-test")
    _, open_slots, _ = _OTRC.assemble_pre_locked_rows(
        num_characters=6, rng=rng, force_lemmy=False,
    )
    names = [s.name for s in open_slots]
    assert len(set(names)) == len(names), f"name collision: {names}"


# ---------------------------------------------------------------------------
# Sprint 3D Stage 1 -- precompute_ensemble_slots (Python owns balance)
# ---------------------------------------------------------------------------


def _slots(n):
    return [_OTRC.CastSlot(char_id=f"c{i+2:02d}", name=f"NAME{i}")
            for i in range(n)]


def test_precompute_ensemble_slots_returns_one_slot_per_open():
    ensemble = _OTRC.precompute_ensemble_slots(
        _slots(4), prior_cast=[], rng=random.Random("ens-1"),
    )
    assert len(ensemble) == 4
    for ens in ensemble:
        assert isinstance(ens, _OTRC.EnsembleSlot)
        assert ens.gender in _OTRC._VALID_GENDERS
        assert ens.timbre in _OTRC._TIMBRE_VOCAB
        assert ens.role in _OTRC._ROLE_VOCAB


def test_precompute_ensemble_slots_preserves_char_id_and_name():
    open_slots = _slots(3)
    ensemble = _OTRC.precompute_ensemble_slots(
        open_slots, prior_cast=[], rng=random.Random("ens-2"),
    )
    assert [e.char_id for e in ensemble] == [s.char_id for s in open_slots]
    assert [e.name for e in ensemble] == [s.name for s in open_slots]


def test_precompute_ensemble_slots_is_deterministic_for_a_seed():
    """C7 byte-identity: same open slots + same seed yield an
    identical ensemble plan (gender shuffle uses the seeded rng)."""
    a = _OTRC.precompute_ensemble_slots(
        _slots(6), prior_cast=[], rng=random.Random("det"),
    )
    b = _OTRC.precompute_ensemble_slots(
        _slots(6), prior_cast=[], rng=random.Random("det"),
    )
    assert [(e.gender, e.timbre, e.role) for e in a] == \
           [(e.gender, e.timbre, e.role) for e in b]


def test_precompute_ensemble_slots_python_owns_gender_balance():
    """Python -- not the LLM -- decides the ~40/40/20 gender split.
    Over a 6-slot ensemble the largest-remainder allocation lands at
    least one male and one female; 'other' is the 20% minority."""
    ensemble = _OTRC.precompute_ensemble_slots(
        _slots(6), prior_cast=[], rng=random.Random("balance"),
    )
    counts = Counter(e.gender for e in ensemble)
    assert counts["male"] >= 1, f"no male slots in {counts!r}"
    assert counts["female"] >= 1, f"no female slots in {counts!r}"
    # 6 slots * 40% = 2.4 -> male and female each land near 2-3.
    assert counts["male"] + counts["female"] >= counts["other"], \
        f"'other' should be the minority share: {counts!r}"
    assert sum(counts.values()) == 6


def test_precompute_ensemble_slots_accounts_for_prior_cast_gender():
    """The whole-ensemble balance offsets genders already locked in
    the prior cast (LEMMY is male), so the open slots compensate."""
    # 5 open slots + 1 prior male (LEMMY). Whole ensemble = 6.
    prior = [{"name": "LEMMY", "gender": "male",
              "character_description": "engineer"}]
    ensemble = _OTRC.precompute_ensemble_slots(
        _slots(5), prior_cast=prior, rng=random.Random("prior-gender"),
    )
    counts = Counter(e.gender for e in ensemble)
    # Whole ensemble (6) ideal: ~2.4 M / ~2.4 F / ~1.2 X. LEMMY already
    # supplies one M, so the open slots should carry <= 2 male.
    assert counts["male"] <= 2, \
        f"open slots over-allocated male given LEMMY prior: {counts!r}"


def test_precompute_ensemble_slots_timbre_spans_the_range():
    """timbre rotates through _TIMBRE_VOCAB so the ensemble spans the
    vocal range rather than clustering on one timbre."""
    ensemble = _OTRC.precompute_ensemble_slots(
        _slots(6), prior_cast=[], rng=random.Random("timbre"),
    )
    timbres = [e.timbre for e in ensemble]
    # 6 slots, 6 timbre words -> every timbre used exactly once.
    assert set(timbres) == set(_OTRC._TIMBRE_VOCAB)


def test_precompute_ensemble_slots_empty_open_slots():
    assert _OTRC.precompute_ensemble_slots(
        [], prior_cast=[], rng=random.Random("empty"),
    ) == []


# ---------------------------------------------------------------------------
# Sprint 3D Stage 2 -- llm_write_description (description-only LLM call)
# ---------------------------------------------------------------------------


def _ens_slot(name="ALICE", gender="female", timbre="warm", role="lead"):
    return _OTRC.EnsembleSlot(
        char_id="c02", name=name, gender=gender, timbre=timbre, role=role,
    )


def test_llm_write_description_returns_description_response():
    gen = _make_canned_generate_fn([_desc_response("A keen-eyed pilot.")])
    r = _OTRC.llm_write_description(
        gen, slot=_ens_slot(), news_seed="story", style="noir",
        prior_cast=[],
    )
    assert isinstance(r, _OTRC.DescriptionResponse)
    assert r.character_description == "A keen-eyed pilot."


def test_llm_write_description_output_has_no_gender_or_voice():
    """Sprint 3D core invariant: the LLM's response object exposes
    ONLY character_description -- gender and voice_preset are NOT
    fields the LLM produces anymore."""
    gen = _make_canned_generate_fn([_desc_response()])
    r = _OTRC.llm_write_description(
        gen, slot=_ens_slot(), news_seed="story", style="noir",
        prior_cast=[],
    )
    fields = set(type(r).model_fields.keys())
    assert fields == {"character_description"}, \
        f"DescriptionResponse must carry description only, got {fields!r}"
    assert not hasattr(r, "gender")
    assert not hasattr(r, "voice_preset")


def test_llm_write_description_ignores_extra_gender_voice_keys():
    """If the model emits a legacy fat object with gender/voice_preset,
    the DescriptionResponse schema silently drops the extra keys --
    the LLM's voice/gender opinion is discarded, not honoured."""
    fat = json.dumps({
        "character_description": "Weathered detective, 50s, gravel voice.",
        "gender": "male",
        "voice_preset": "v2/en_speaker_0",
    })
    gen = _make_canned_generate_fn([fat])
    r = _OTRC.llm_write_description(
        gen, slot=_ens_slot(gender="female"), news_seed="s", style="noir",
        prior_cast=[],
    )
    assert r.character_description.startswith("Weathered detective")
    assert not hasattr(r, "voice_preset")


def test_llm_write_description_rerolls_on_invalid_json():
    gen = _make_canned_generate_fn([
        "no json here at all, sorry",
        _desc_response(),
    ])
    r = _OTRC.llm_write_description(
        gen, slot=_ens_slot(), news_seed="story", style="noir",
        prior_cast=[],
    )
    assert r.character_description


def test_llm_write_description_max_attempts_zero_rejected():
    gen = _make_canned_generate_fn([_desc_response()])
    with pytest.raises(ValueError, match="max_attempts"):
        _OTRC.llm_write_description(
            gen, slot=_ens_slot(), news_seed="s", style="noir",
            prior_cast=[], max_attempts=0,
        )


def test_llm_write_description_raises_after_all_attempts_fail():
    gen = _make_canned_generate_fn(["bad", "still bad", "yet again bad"])
    with pytest.raises(_OTRC.CastingFailedError) as exc:
        _OTRC.llm_write_description(
            gen, slot=_ens_slot(), news_seed="s", style="noir",
            prior_cast=[], max_attempts=3,
        )
    assert "ALICE" in str(exc.value)
    assert len(exc.value.attempts) == 3


# ---------------------------------------------------------------------------
# Sprint 3D Stage 3 -- python_assign_voice_preset (Python picks voice)
# ---------------------------------------------------------------------------


def _mixed_voices():
    """A gendered pool: 3 female + 3 male voices with descriptive shorts."""
    return [
        ("v2/en_speaker_4", "female warm bright 30s"),
        ("v2/en_speaker_2", "female sharp precise 30s"),
        ("v2/en_speaker_9", "female deep authoritative 50s"),
        ("v2/en_speaker_0", "male deep authoritative 50s"),
        ("v2/en_speaker_3", "male sharp energetic 20s"),
        ("v2/en_speaker_8", "male gravelly confident 40s"),
    ]


def test_python_assign_voice_preset_picks_from_pool():
    voice = _OTRC.python_assign_voice_preset(
        _ens_slot(gender="female"), available_voices=_mixed_voices(),
        rng=random.Random("v1"),
    )
    assert voice in {p for p, _ in _mixed_voices()}


def test_python_assign_voice_preset_matches_slot_gender():
    """A female slot draws a female-column voice when one exists."""
    voice = _OTRC.python_assign_voice_preset(
        _ens_slot(gender="female"), available_voices=_mixed_voices(),
        rng=random.Random("v2"),
    )
    female_presets = {p for p, s in _mixed_voices()
                      if s.startswith("female")}
    assert voice in female_presets, \
        f"female slot got a non-female voice: {voice!r}"


def test_python_assign_voice_preset_matches_slot_timbre():
    """When a timbre-matched voice exists in the gender column it is
    preferred -- a 'gravelly' male slot lands the gravelly voice."""
    voice = _OTRC.python_assign_voice_preset(
        _ens_slot(gender="male", timbre="gravelly"),
        available_voices=_mixed_voices(),
        rng=random.Random("v3"),
    )
    assert voice == "v2/en_speaker_8", \
        f"gravelly male slot should land the gravelly voice, got {voice!r}"


def test_python_assign_voice_preset_is_deterministic_for_a_seed():
    a = _OTRC.python_assign_voice_preset(
        _ens_slot(), available_voices=_mixed_voices(),
        rng=random.Random("det"),
    )
    b = _OTRC.python_assign_voice_preset(
        _ens_slot(), available_voices=_mixed_voices(),
        rng=random.Random("det"),
    )
    assert a == b


def test_python_assign_voice_preset_empty_pool_raises():
    with pytest.raises(_OTRC.CastingFailedError) as exc:
        _OTRC.python_assign_voice_preset(
            _ens_slot(), available_voices=[], rng=random.Random("e"),
        )
    assert "empty" in str(exc.value).lower()


def test_python_assign_voice_preset_other_gender_falls_back_to_full_pool():
    """The Bark pool is binary male/female; an 'other'-gender slot has
    no column of its own and draws from the whole pre-filtered pool."""
    voice = _OTRC.python_assign_voice_preset(
        _ens_slot(gender="other"), available_voices=_mixed_voices(),
        rng=random.Random("other"),
    )
    assert voice in {p for p, _ in _mixed_voices()}


def test_python_assign_voice_preset_gender_column_exhausted_fallback():
    """If a gender column is empty (more slots of one gender than the
    pool supplies) python_assign_voice_preset falls back to the whole
    pre-filtered pool rather than failing -- a voiced character beats
    a hard cast failure."""
    female_only = [(p, s) for p, s in _mixed_voices()
                   if s.startswith("female")]
    voice = _OTRC.python_assign_voice_preset(
        _ens_slot(gender="male"), available_voices=female_only,
        rng=random.Random("exhausted"),
    )
    assert voice in {p for p, _ in female_only}


# ---------------------------------------------------------------------------
# Voice pre-filter
# ---------------------------------------------------------------------------


def test_open_voice_pool_excludes_taken():
    taken = {"v2/en_speaker_8", "v2/en_speaker_4"}
    pool = _POOLS.open_voice_pool(taken)
    presets = {p for p, _ in pool}
    assert "v2/en_speaker_8" not in presets
    assert "v2/en_speaker_4" not in presets
    # Other voices still in
    assert "v2/en_speaker_0" in presets


def test_open_voice_pool_short_descriptions_have_gender():
    """Each entry is (preset, short) where short starts with the
    gender so python_assign_voice_preset can read it at a glance."""
    pool = _POOLS.open_voice_pool(set())
    assert len(pool) == len(_POOLS.VOICE_PROFILES)
    for preset, short in pool:
        assert short.startswith("male") or short.startswith("female"), \
            f"short missing gender: {preset!r} -> {short!r}"


def test_open_voice_pool_short_descriptions_are_deterministic():
    """C7 byte-identity guard: open_voice_pool must produce IDENTICAL
    short-descriptions across runs. Set iteration order in Python is
    hash-randomization dependent, so without sorted() the rendered
    short can vary process-to-process. Per round-robin synthesis
    2026-05-10."""
    pool_a = _POOLS.open_voice_pool(set())
    pool_b = _POOLS.open_voice_pool(set())
    assert pool_a == pool_b, \
        "open_voice_pool not deterministic across calls in same process"
    by_preset_a = {p: s for p, s in pool_a}
    by_preset_b = {p: s for p, s in pool_b}
    for preset in by_preset_a:
        assert by_preset_a[preset] == by_preset_b[preset]


# ---------------------------------------------------------------------------
# DescriptionResponse / CastingResponse schema
# ---------------------------------------------------------------------------


def test_description_response_accepts_valid():
    r = _OTRC.DescriptionResponse(
        character_description="Female, 40s, weary broadcaster",
    )
    assert r.character_description.startswith("Female")


def test_description_response_rejects_short_description():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _OTRC.DescriptionResponse(character_description="too short")


def test_casting_response_accepts_valid():
    """CastingResponse is the assembled result (description + gender +
    voice_preset). Sprint 3D: gender/voice come from Python, not the
    LLM, but the combined shape is unchanged for the writer."""
    r = _OTRC.CastingResponse(
        character_description="Female, 40s, weary broadcaster",
        gender="female",
        voice_preset="v2/en_speaker_4",
    )
    assert r.gender == "female"
    assert r.voice_preset == "v2/en_speaker_4"


def test_casting_response_rejects_bad_gender():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _OTRC.CastingResponse(
            character_description="Female, 40s, weary",
            gender="non-binary",  # not in {male,female,other}
            voice_preset="v2/en_speaker_4",
        )


def test_casting_response_rejects_short_description():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _OTRC.CastingResponse(
            character_description="too short",  # < 10 chars
            gender="male",
            voice_preset="v2/en_speaker_0",
        )


def test_casting_response_normalizes_gender_case():
    r = _OTRC.CastingResponse(
        character_description="Some valid description text",
        gender="MALE",
        voice_preset="v2/en_speaker_0",
    )
    assert r.gender == "male"


# ---------------------------------------------------------------------------
# cast_one_character -- composes the three Sprint 3D stages
# ---------------------------------------------------------------------------


def _three_voices():
    return [
        ("v2/en_speaker_4", "female bright 30s"),
        ("v2/en_speaker_6", "female throaty 40s"),
        ("v2/en_speaker_9", "female commanding 50s"),
    ]


def test_cast_one_character_first_attempt_success():
    gen = _make_canned_generate_fn([_desc_response()])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        rng=random.Random("c1"),
    )
    # gender comes from Python's ensemble plan; voice_preset is now empty here
    # (OTR_CastLock stamps the bark voice after the freeze).
    assert r.gender in _OTRC._VALID_GENDERS
    assert r.voice_preset == ""  # Sprint 2 (a): CastLock assigns the bark voice post-freeze; cast_one_character no longer does
    assert r.character_description


def test_cast_one_character_voice_always_in_pool():
    """Sprint 2 (a): the writer no longer assigns the bark voice --
    cast_one_character leaves voice_preset EMPTY and OTR_CastLock stamps it
    (from the same pre-filtered pool) after the freeze. Test name kept for
    nodeid stability; the in-pool guarantee now lives in the CastLock parity +
    replay tests."""
    gen = _make_canned_generate_fn([_desc_response()])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        rng=random.Random("c-pool"),
    )
    assert r.voice_preset == ""  # Sprint 2 (a): CastLock assigns the bark voice post-freeze; cast_one_character no longer does


def test_cast_one_character_honours_precomputed_ensemble_slot():
    """When lock_cast passes a precomputed EnsembleSlot, cast_one_character
    uses that slot's gender (Python-decided) on the cast row -- the LLM
    has no say."""
    gen = _make_canned_generate_fn([_desc_response()])
    ens = _ens_slot(name="ALICE", gender="male", timbre="deep", role="lead")
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_mixed_voices(),
        ensemble_slot=ens,
        rng=random.Random("ens-honour"),
    )
    assert r.gender == "male", "cast row gender must follow the Python plan"


def test_cast_one_character_max_attempts_zero_rejected():
    """max_attempts=0 is invalid (no attempts to make)."""
    gen = _make_canned_generate_fn([_desc_response()])
    with pytest.raises(ValueError, match="max_attempts"):
        _OTRC.cast_one_character(
            gen,
            name="ALICE", news_seed="story", style="noir",
            prior_cast=[], available_voices=_three_voices(),
            max_attempts=0,
        )


def test_cast_one_character_max_attempts_one_single_shot_no_repair():
    """max_attempts=1 is allowed (single-shot, no repair branch)."""
    gen = _make_canned_generate_fn([_desc_response()])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        max_attempts=1, rng=random.Random("c-single"),
    )
    assert r.voice_preset == ""  # Sprint 2 (a): CastLock assigns the bark voice post-freeze; cast_one_character no longer does


def test_cast_one_character_repair_truncates_huge_raw():
    """A 4000-char garbage response must NOT be embedded whole into the
    repair prompt -- it would bloat the KV cache against the 14.5 GB
    VRAM ceiling.

    The repair attempt is structured_call's Attempt 3, whose
    default_repair_prompt_factory embeds only a truncated slice of the
    prior failed output. Two bad responses then a good one so the
    repair (third) attempt fires and recovers."""
    huge_garbage = "x" * 4000
    captured_messages: list = []

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        captured_messages.append(messages)
        if len(captured_messages) <= 2:
            return huge_garbage
        return _desc_response()

    r = _OTRC.cast_one_character(
        gen_fn,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        max_attempts=3, rng=random.Random("c-huge"),
    )
    assert r.voice_preset == ""  # Sprint 2 (a): CastLock assigns the bark voice post-freeze; cast_one_character no longer does
    repair_text = "".join(m["content"] for m in captured_messages[-1])
    assert huge_garbage not in repair_text, (
        "repair prompt embedded the full 4000-char garbage -- the "
        "structured_call repair factory must truncate prior output"
    )
    assert len(repair_text) < len(huge_garbage), (
        f"repair prompt ({len(repair_text)} chars) is not smaller than "
        f"the 4000-char garbage it was supposed to truncate"
    )


def test_lock_cast_preflight_fails_fast_when_voice_pool_too_small(
    monkeypatch,
):
    """Patch open_voice_pool to return only 1 voice, then ask for 3
    open slots. lock_cast must raise BEFORE any LLM call fires."""
    llm_call_count = [0]

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        llm_call_count[0] += 1
        return _desc_response()

    monkeypatch.setattr(
        _POOLS, "open_voice_pool",
        lambda taken: [("v2/en_speaker_4", "female bright 30s")],
    )

    rng = random.Random("preflight-test")
    with pytest.raises(_OTRC.CastingFailedError, match="too small"):
        _OTRC.lock_cast(creative_fn=gen_fn,
            num_characters=3,
            news_seed="story", style="noir",
            rng=rng,
            force_lemmy=False,
        )
    assert llm_call_count[0] == 0, \
        "preflight should fail BEFORE any LLM call"


def test_cast_one_character_rerolls_on_invalid_description_json():
    """Attempt 1 returns garbage; attempt 2 returns clean JSON."""
    gen = _make_canned_generate_fn([
        "no json here at all, sorry",
        _desc_response(),
    ])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        rng=random.Random("c-reroll"),
    )
    assert r.voice_preset == ""  # Sprint 2 (a): CastLock assigns the bark voice post-freeze; cast_one_character no longer does


def test_cast_one_character_survives_trailing_object_after_json():
    """BUG-LOCAL-261: the LLM emits a valid object, then a SECOND
    top-level JSON object. The shared _otr_json extractor takes the
    first complete object and ignores the trailing one."""
    first = _desc_response("a sharp-eyed scientist")
    trailing = '\n{"character_description":"a stray hallucinated second"}'
    gen = _make_canned_generate_fn([first + trailing])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        rng=random.Random("c-trailing"),
    )
    assert r.character_description.startswith("a sharp-eyed scientist")


def test_cast_one_character_survives_trailing_prose_after_json():
    """Trailing prose with no braces after the object is tolerated."""
    gen = _make_canned_generate_fn([
        _desc_response("a weary broadcaster")
        + "\n\nHope this casting fits your radio drama!",
    ])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        rng=random.Random("c-prose"),
    )
    assert r.character_description.startswith("a weary broadcaster")


def test_cast_one_character_repair_attempt_used():
    """All three attempts: two bad, third (repair) succeeds."""
    gen = _make_canned_generate_fn([
        "garbage 1",
        "garbage 2",
        _desc_response(),
    ])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        max_attempts=3, rng=random.Random("c-repair"),
    )
    assert r.character_description


def test_cast_one_character_raises_after_all_attempts_fail():
    gen = _make_canned_generate_fn(["bad", "still bad", "yet again bad"])
    with pytest.raises(_OTRC.CastingFailedError) as exc_info:
        _OTRC.cast_one_character(
            gen,
            name="ALICE", news_seed="story", style="noir",
            prior_cast=[], available_voices=_three_voices(),
            max_attempts=3,
        )
    assert "ALICE" in str(exc_info.value)
    assert len(exc_info.value.attempts) == 3


def test_cast_one_character_strips_markdown_fences():
    """Some models wrap JSON in ```json ... ```."""
    fenced = "```json\n" + _desc_response() + "\n```"
    gen = _make_canned_generate_fn([fenced])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        rng=random.Random("c-fence"),
    )
    assert r.voice_preset == ""  # Sprint 2 (a): CastLock assigns the bark voice post-freeze; cast_one_character no longer does


def test_cast_one_character_handles_prose_preamble():
    """Some models add 'Sure! Here is the cast: { ... }'."""
    prefixed = "Sure! Here is the cast: " + _desc_response()
    gen = _make_canned_generate_fn([prefixed])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        rng=random.Random("c-preamble"),
    )
    assert r.voice_preset == ""  # Sprint 2 (a): CastLock assigns the bark voice post-freeze; cast_one_character no longer does


def test_cast_one_character_empty_voice_pool_raises_immediately():
    gen = _make_canned_generate_fn([_desc_response()])
    with pytest.raises(_OTRC.CastingFailedError) as exc_info:
        _OTRC.cast_one_character(
            gen,
            name="ALICE", news_seed="story", style="noir",
            prior_cast=[], available_voices=[],
        )
    assert "available_voices is empty" in str(exc_info.value).lower() \
        or "nothing to pick" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Prompt shape -- Sprint 3D: description-only, no Voices: / balance line
# ---------------------------------------------------------------------------


def test_prompt_has_story_and_style_adjacent_no_period_literal():
    """news_seed + style must be on adjacent lines (continuity rule)
    and the prompt must NOT contain '1940s' or '1903'."""
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="A real science story about black holes.",
        style="noir mystery",
        prior_cast=[],
    )
    lines = prompt.splitlines()
    story_idx = next(i for i, l in enumerate(lines) if l.startswith("Story:"))
    style_idx = next(i for i, l in enumerate(lines) if l.startswith("Style:"))
    assert style_idx == story_idx + 1, \
        f"Story and Style must be adjacent (got {story_idx}, {style_idx})"
    assert "1940" not in prompt
    assert "1903" not in prompt


def test_prompt_omits_cast_so_far_block_when_prior_empty():
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=[],
    )
    assert "Cast so far:" not in prompt


def test_prompt_includes_cast_so_far_when_prior_nonempty():
    prior = [
        {"name": "LEMMY", "gender": "male",
         "character_description": "Grizzled wrench-wielding engineer, 50s"},
    ]
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=prior,
    )
    assert "Cast so far:" in prompt
    assert "LEMMY (M, " in prompt


def test_prompt_truncates_news_seed_at_500_chars():
    huge_seed = "x" * 5000
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed=huge_seed, style="noir",
        prior_cast=[],
    )
    story_line = next(l for l in prompt.splitlines() if l.startswith("Story:"))
    assert len(story_line) <= len("Story: ") + 500


def test_prompt_is_description_only_no_voice_block_no_balance_line():
    """Sprint 3D: the prompt no longer lists voices and no longer
    carries the '~40/40/20' balance line -- voice selection and
    ensemble balance left the LLM."""
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=[],
        gender="female", timbre="warm", role="lead",
    )
    assert "Voices:" not in prompt, \
        "the voice list must not be shown to the LLM anymore"
    assert "40%" not in prompt and "Aim" not in prompt, \
        "the ensemble-balance instruction must be gone from the prompt"
    # The JSON contract asks for description only.
    assert '"character_description"' in prompt
    assert '"voice_preset"' not in prompt
    assert '"gender":"male|female|other"' not in prompt


def test_prompt_states_python_decided_gender_timbre_role_as_facts():
    """The Python-decided gender / timbre / role are handed to the LLM
    as fixed facts to write into, not choices to make."""
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=[],
        gender="female", timbre="bright", role="foil",
    )
    assert "Gender: female" in prompt
    assert "Voice: bright" in prompt
    assert "Role: foil" in prompt


def test_prompt_has_no_system_prompt_baked():
    """Per the lean-prompt rule, _build_user_prompt returns a single
    user-message body."""
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=[],
    )
    assert "system:" not in prompt.lower()
    assert "<|system|>" not in prompt.lower()
    assert "you are a casting director" not in prompt.lower()


# ---------------------------------------------------------------------------
# lock_cast end-to-end
# ---------------------------------------------------------------------------


def test_lock_cast_no_lemmy_end_to_end():
    """3-character episode, no LEMMY. ANNOUNCER c01, three open slots.

    ANNOUNCER's voice (Kokoro bm_/bf_) is in a different TTS namespace
    from Bark v2/en_speaker_*, so any Bark voice is fair game for an
    open character -- no announcer-collision risk.
    """
    rng = random.Random("e2e-no-lemmy")
    gen = _make_canned_generate_fn([
        _desc_response(), _desc_response(), _desc_response(),
    ])
    cast, meta = _OTRC.lock_cast(creative_fn=gen,
        num_characters=3,
        news_seed="A science article about deep-sea hydrothermal vents.",
        style="noir mystery",
        rng=rng,
        force_lemmy=False,
    )
    # Cast: ANNOUNCER + 3 open characters = 4 rows
    assert len(cast) == 4
    assert cast[0]["name"] == "ANNOUNCER"
    assert cast[0]["char_id"] == "c01"
    assert [r["char_id"] for r in cast[1:]] == ["c02", "c03", "c04"]
    # Sprint 2 (a): the writer no longer assigns open-character voices --
    # OTR_CastLock stamps them post-freeze, so the open rows carry empty
    # voice_preset here. Uniqueness is asserted in the CastLock + parity tests.
    assert all(r["voice_preset"] == "" for r in cast[1:])
    assert meta["lemmy_hit"] is False
    assert meta["num_characters_locked"] == 3


def test_lock_cast_with_lemmy_end_to_end():
    """3-character episode with LEMMY. ANNOUNCER c01, LEMMY c02, two open."""
    rng = random.Random("e2e-with-lemmy")
    gen = _make_canned_generate_fn([_desc_response(), _desc_response()])
    cast, meta = _OTRC.lock_cast(creative_fn=gen,
        num_characters=3,
        news_seed="A science article.",
        style="noir mystery",
        rng=rng,
        force_lemmy=True,
    )
    # Cast: ANNOUNCER + LEMMY + 2 open = 4 rows total
    assert len(cast) == 4
    assert cast[0]["name"] == "ANNOUNCER"
    assert cast[1]["name"] == "LEMMY"
    assert cast[1]["char_id"] == "c02"
    assert cast[1]["voice_preset"] == "v2/en_speaker_8"
    assert [r["char_id"] for r in cast[2:]] == ["c03", "c04"]
    assert meta["lemmy_hit"] is True
    assert meta["num_characters_locked"] == 3
    # Sprint 2 (a): open-character voices are stamped by OTR_CastLock post-freeze
    # (empty here); LEMMY (pre-locked) keeps v2/en_speaker_8.
    assert all(r["voice_preset"] == "" for r in cast[2:])


def test_lock_cast_python_owns_distribution_not_the_llm():
    """Sprint 3D core proof: lock_cast's stub LLM emits a fixed
    description with NO gender and NO voice_preset, yet every cast row
    still carries a valid gender and an in-pool, unique voice -- proof
    that Python (not the LLM) owns the ensemble distribution."""
    rng = random.Random("python-owns")
    # The LLM only ever returns a description; it never names a gender
    # or a voice.
    gen = _make_canned_generate_fn([_desc_response()] * 5)
    cast, _ = _OTRC.lock_cast(creative_fn=gen,
        num_characters=5,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=False,
    )
    open_rows = cast[1:]  # drop ANNOUNCER
    for row in open_rows:
        assert row["gender"] in _OTRC._VALID_GENDERS, \
            f"Python must stamp a valid gender: {row!r}"
        # Sprint 2 (a): the bark voice is no longer stamped by the writer -- it's
        # empty here; OTR_CastLock (also pure Python, never the LLM) assigns it
        # post-freeze. The LLM still never decides the voice.
        assert row["voice_preset"] == "", \
            f"writer must NOT stamp a voice (CastLock owns it now): {row!r}"


def test_lock_cast_one_llm_call_per_open_slot():
    """Sprint 3D: the LLM call count did not go up. Exactly one
    successful description call per open slot -- no extra call site."""
    call_count = [0]

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        call_count[0] += 1
        return _desc_response()

    rng = random.Random("call-count")
    _OTRC.lock_cast(creative_fn=gen_fn,
        num_characters=4,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=False,
    )
    # 4 open slots, each fills on its first attempt -> 4 calls, no more.
    assert call_count[0] == 4, \
        f"expected one LLM call per open slot, got {call_count[0]}"


def test_lock_cast_open_slots_see_lemmy_in_prior_cast_but_not_announcer():
    """When LEMMY hits, the per-character LLM prompt should include
    LEMMY in 'Cast so far' (he's ensemble) but NOT ANNOUNCER."""
    captured_prompts: list[str] = []

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        captured_prompts.append(messages[0]["content"])
        return _desc_response()

    rng = random.Random("prior-cast-test")
    _OTRC.lock_cast(creative_fn=gen_fn,
        num_characters=2,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=True,
    )
    assert len(captured_prompts) == 1
    p = captured_prompts[0]
    assert "LEMMY (M, " in p
    assert "ANNOUNCER" not in p


def test_lock_cast_voice_pool_pre_filtered_excludes_lemmy_voice():
    """When LEMMY hits, his v2/en_speaker_8 must never be assigned to
    an open character (the pool pre-filter drops it)."""
    rng = random.Random("voice-filter-test")
    gen = _make_canned_generate_fn([_desc_response()])
    cast, _ = _OTRC.lock_cast(creative_fn=gen,
        num_characters=2,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=True,
    )
    open_voices = [r["voice_preset"] for r in cast[2:]]  # drop ANNOUNCER+LEMMY
    assert "v2/en_speaker_8" not in open_voices, \
        "LEMMY's voice must be pre-filtered out of the open-slot pool"


def test_assert_unique_bark_voices_accepts_clean_cast():
    """Happy path: a cast with all-distinct Bark voices + Kokoro
    announcer passes the invariant cleanly."""
    cast = [
        {"char_id": "c01", "name": "ANNOUNCER",
         "voice_preset": "bm_george"},
        {"char_id": "c02", "name": "LEMMY",
         "voice_preset": "v2/en_speaker_8"},
        {"char_id": "c03", "name": "ALICE",
         "voice_preset": "v2/en_speaker_4"},
        {"char_id": "c04", "name": "BOB",
         "voice_preset": "v2/en_speaker_6"},
    ]
    _OTRC._assert_unique_bark_voices(cast)


def test_assert_unique_bark_voices_catches_collision():
    """A regression scenario where two Bark cast rows accidentally
    share a voice. The invariant must catch it and raise
    CastingFailedError with a descriptive duplicate report."""
    cast = [
        {"char_id": "c01", "name": "ANNOUNCER",
         "voice_preset": "bm_george"},
        {"char_id": "c02", "name": "LEMMY",
         "voice_preset": "v2/en_speaker_8"},
        {"char_id": "c03", "name": "ALICE",
         "voice_preset": "v2/en_speaker_4"},
        {"char_id": "c04", "name": "BOB",
         "voice_preset": "v2/en_speaker_4"},
    ]
    with pytest.raises(_OTRC.CastingFailedError) as exc_info:
        _OTRC._assert_unique_bark_voices(cast)
    msg = str(exc_info.value)
    assert "v2/en_speaker_4" in msg, \
        f"error must name the duplicate voice: {msg!r}"
    assert "c04" in msg and "c03" in msg, \
        f"error must name both colliding char_ids: {msg!r}"


def test_assert_unique_bark_voices_ignores_announcer():
    """ANNOUNCER's voice (Kokoro namespace) is exempt from the
    uniqueness check."""
    cast = [
        {"char_id": "c01", "name": "ANNOUNCER",
         "voice_preset": "v2/en_speaker_4"},
        {"char_id": "c02", "name": "ALICE",
         "voice_preset": "v2/en_speaker_4"},
    ]
    _OTRC._assert_unique_bark_voices(cast)


def test_lock_cast_invariant_fires_at_end_if_voices_collide():
    """Fail-safe: the Bark voice-uniqueness invariant -- relocated to OTR_CastLock
    in Sprint 2 (a) (formerly the post-cast guard inside lock_cast) -- must raise
    if two non-ANNOUNCER rows ever carry the SAME Bark voice. lock_cast no longer
    assigns voices, so the invariant function is exercised directly here; it's
    exactly what OTR_CastLock._assign_bark_voices calls after it stamps. (Test
    name kept for nodeid stability.)
    """
    colliding = [
        {"char_id": "c01", "name": "ANNOUNCER", "voice_preset": "bm_george"},
        {"char_id": "c02", "name": "ALICE", "voice_preset": "v2/en_speaker_4"},
        {"char_id": "c03", "name": "BOB", "voice_preset": "v2/en_speaker_4"},
    ]
    with pytest.raises(_OTRC.CastingFailedError) as exc_info:
        _OTRC._assert_unique_bark_voices(colliding)
    assert "POST-CAST INVARIANT" in str(exc_info.value), \
        f"relocated invariant must fire, got: {exc_info.value!r}"


def test_lock_cast_announcer_kokoro_voice_does_not_filter_bark_pool():
    """When the announcer takes a Kokoro voice, the open-slot Bark pool
    must remain unaffected. Kokoro and Bark are separate TTS namespaces.

    With num_characters=6 and no LEMMY there are 6 open slots and 9
    Bark voices; the announcer's Kokoro voice never reduces that pool.
    All 6 assigned voices must be distinct Bark presets."""
    rng = random.Random("kokoro-bark-separation")
    gen = _make_canned_generate_fn([_desc_response()] * 6)
    cast, meta = _OTRC.lock_cast(creative_fn=gen,
        num_characters=6,
        news_seed="story", style="noir",
        rng=rng, cast_seed=777001,
        force_lemmy=False,
    )
    # Sprint 2 (a): the writer leaves open voices empty; OTR_CastLock stamps them
    # via the byte-identical replay. The announcer's Kokoro voice never reduces
    # the Bark pool (separate namespaces), so all 6 open characters resolve to
    # distinct Bark presets after the CastLock replay.
    voices = _OTRC.replay_voice_assignment(
        cast_seed=777001, num_characters=6, lemmy_hit=meta["lemmy_hit"])
    open_voices = [voices[r["char_id"]] for r in cast[1:]]
    all_bark_presets = {p for p, _, _, _ in _POOLS.VOICE_PROFILES}
    assert len(open_voices) == 6
    assert len(set(open_voices)) == 6, \
        f"open-character voices collided: {open_voices!r}"
    for v in open_voices:
        assert v in all_bark_presets, \
            f"assigned voice {v!r} is not a Bark preset"


# ---------------------------------------------------------------------------
# BUG-LOCAL-260: LEMMY cameo decoupled from the seed
# ---------------------------------------------------------------------------


class _FixedRandom:
    """Stand-in RNG whose .random() always returns a fixed value."""

    def __init__(self, value):
        self._value = value

    def random(self):
        return self._value


def test_roll_lemmy_takes_no_seed_argument():
    """roll_lemmy() no longer accepts an rng -- a caller cannot
    re-couple the cameo to a seed by passing one (BUG-LOCAL-260)."""
    import inspect
    assert list(inspect.signature(_POOLS.roll_lemmy).parameters) == [], (
        "roll_lemmy must take no arguments so the cameo cannot be "
        "re-tied to a seeded RNG"
    )


def test_roll_lemmy_uses_os_entropy(monkeypatch):
    """roll_lemmy() rolls against the module-level SystemRandom."""
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.05))
    assert _POOLS.roll_lemmy() is True   # 0.05 < 0.11
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.50))
    assert _POOLS.roll_lemmy() is False  # 0.50 >= 0.11


def test_assemble_lemmy_hit_independent_of_the_cast_seed(monkeypatch):
    """assemble_pre_locked_rows with the natural roll: the SAME cast
    seed yields a DIFFERENT lemmy_hit when OS entropy differs --
    proof the cameo is decoupled from the seed (BUG-LOCAL-260)."""
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.05))
    _pre_a, _open_a, hit_a = _OTRC.assemble_pre_locked_rows(
        num_characters=3, rng=random.Random(42),
    )
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.90))
    _pre_b, _open_b, hit_b = _OTRC.assemble_pre_locked_rows(
        num_characters=3, rng=random.Random(42),
    )
    assert hit_a is True and hit_b is False, (
        "same cast seed must NOT determine lemmy_hit -- the cameo "
        "roll is decoupled from the seed (BUG-LOCAL-260)"
    )


def test_assemble_cast_names_still_seed_deterministic(monkeypatch):
    """Decoupling LEMMY must NOT weaken C7 for the rest of the cast:
    the announcer pick and open-slot names stay seed-deterministic."""
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.90))
    pre_a, open_a, _ = _OTRC.assemble_pre_locked_rows(
        num_characters=4, rng=random.Random("c7-seed"),
    )
    pre_b, open_b, _ = _OTRC.assemble_pre_locked_rows(
        num_characters=4, rng=random.Random("c7-seed"),
    )
    assert [r["name"] for r in pre_a] == [r["name"] for r in pre_b]
    assert pre_a[0]["voice_preset"] == pre_b[0]["voice_preset"]
    assert [s.name for s in open_a] == [s.name for s in open_b]


def test_force_lemmy_still_overrides_the_os_entropy_roll(monkeypatch):
    """force_lemmy must still win over the OS-entropy roll, so tests
    and the lemmy_cameo writer widget keep deterministic control."""
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.0))
    _pre, _open, hit = _OTRC.assemble_pre_locked_rows(
        num_characters=2, rng=random.Random(1), force_lemmy=False,
    )
    assert hit is False, "force_lemmy=False must override an entropy hit"
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.99))
    _pre2, _open2, hit2 = _OTRC.assemble_pre_locked_rows(
        num_characters=2, rng=random.Random(1), force_lemmy=True,
    )
    assert hit2 is True, "force_lemmy=True must override an entropy miss"
