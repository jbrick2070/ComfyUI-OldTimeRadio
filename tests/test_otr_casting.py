"""tests/test_otr_casting.py -- unit tests for the cast contract LLM caller.

Covers:
  - assemble_pre_locked_rows: determinism, slot accounting, LEMMY hit/miss
  - ANNOUNCER 50/50 male/female distribution
  - Voice pre-filter (pre-locked voices excluded from open slot pool)
  - CastingResponse schema validation (gender enum, length bounds)
  - cast_one_character validator + reroll: bad JSON, schema fail,
    voice not in pool, repair on attempt 3
  - lock_cast end-to-end: pre-locked rows + LLM-cast rows in order
  - Prompt shape: news_seed + style adjacent, no 1940s literal,
    Cast-so-far block omitted on first call

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


def _good_response(name="ALICE",
                   description="Sharp-eyed scientist, 30s, anxious about the data.",
                   gender="female",
                   voice_preset="v2/en_speaker_4"):
    return json.dumps({
        "character_description": description,
        "gender":                gender,
        "voice_preset":          voice_preset,
    })


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
    gen = _make_canned_generate_fn([
        _good_response(voice_preset="v2/en_speaker_3"),
    ])
    cast, _ = _OTRC.lock_cast(creative_fn=gen, technical_fn=gen,
        num_characters=1,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=False,
    )
    open_row = cast[1]  # cast[0] is ANNOUNCER
    assert open_row["voice_params"] is None


def test_lock_cast_open_character_rows_carry_tts_model_bark():
    """Every LLM-cast open-character row must stamp tts_model="bark"
    since open characters are drawn from the Bark VOICE_PROFILES
    pool by construction."""
    rng = random.Random("tts-model-open")
    gen = _make_canned_generate_fn([
        _good_response(voice_preset="v2/en_speaker_3"),
        _good_response(voice_preset="v2/en_speaker_5"),
    ])
    cast, _ = _OTRC.lock_cast(creative_fn=gen, technical_fn=gen,
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


def test_cast_one_character_max_attempts_zero_rejected():
    """max_attempts=0 is invalid (no attempts to make)."""
    gen = _make_canned_generate_fn([_good_response()])
    with pytest.raises(ValueError, match="max_attempts"):
        _OTRC.cast_one_character(
            gen,
            name="ALICE", news_seed="story", style="noir",
            prior_cast=[], available_voices=_three_voices(),
            max_attempts=0,
        )


def test_cast_one_character_max_attempts_one_single_shot_no_repair():
    """max_attempts=1 is allowed (single-shot, no repair branch)."""
    gen = _make_canned_generate_fn([_good_response()])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        max_attempts=1,
    )
    assert r.voice_preset == "v2/en_speaker_4"


def test_cast_one_character_repair_truncates_huge_raw():
    """A 4000-char garbage response on attempt 1 must NOT bloat the
    repair prompt's KV cache on attempt 2 (max_attempts=2 -> repair
    fires on attempt 1 since attempt_idx>0 required for repair, so
    we use max_attempts=3 to make sure repair fires)."""
    huge_garbage = "x" * 4000
    captured_messages: list = []

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        captured_messages.append(messages)
        if len(captured_messages) <= 2:
            return huge_garbage
        return _good_response()

    _OTRC.cast_one_character(
        gen_fn,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        max_attempts=3,
    )
    # Third call is the repair attempt. Its messages should include
    # the truncated assistant turn.
    repair_messages = captured_messages[-1]
    assistant_turn = next(m for m in repair_messages
                          if m["role"] == "assistant")
    assert len(assistant_turn["content"]) <= _OTRC._REPAIR_RAW_CAP_CHARS, \
        f"repair prompt did not truncate huge raw: " \
        f"{len(assistant_turn['content'])} chars"


def test_lock_cast_preflight_fails_fast_when_voice_pool_too_small(
    monkeypatch,
):
    """Patch open_voice_pool to return only 1 voice, then ask for 3
    open slots. lock_cast must raise BEFORE any LLM call fires."""
    llm_call_count = [0]

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        llm_call_count[0] += 1
        return _good_response()

    monkeypatch.setattr(
        _POOLS, "open_voice_pool",
        lambda taken: [("v2/en_speaker_4", "female bright 30s")],
    )

    rng = random.Random("preflight-test")
    with pytest.raises(_OTRC.CastingFailedError, match="too small"):
        _OTRC.lock_cast(creative_fn=gen_fn, technical_fn=gen_fn,
            num_characters=3,
            news_seed="story", style="noir",
            rng=rng,
            force_lemmy=False,
        )
    assert llm_call_count[0] == 0, \
        "preflight should fail BEFORE any LLM call"


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
    # The short-description for each preset must always render the
    # same way (sorted-tag invariant).
    by_preset_a = {p: s for p, s in pool_a}
    by_preset_b = {p: s for p, s in pool_b}
    for preset in by_preset_a:
        assert by_preset_a[preset] == by_preset_b[preset]


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
    gender so the LLM can read it at a glance."""
    pool = _POOLS.open_voice_pool(set())
    assert len(pool) == len(_POOLS.VOICE_PROFILES)
    for preset, short in pool:
        assert short.startswith("male") or short.startswith("female"), \
            f"short missing gender: {preset!r} -> {short!r}"


# ---------------------------------------------------------------------------
# CastingResponse schema
# ---------------------------------------------------------------------------


def test_casting_response_accepts_valid():
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
# cast_one_character (the LLM call)
# ---------------------------------------------------------------------------


def _three_voices():
    return [
        ("v2/en_speaker_4", "female bright 30s"),
        ("v2/en_speaker_6", "female throaty 40s"),
        ("v2/en_speaker_9", "female commanding 50s"),
    ]


def test_cast_one_character_first_attempt_success():
    gen = _make_canned_generate_fn([_good_response()])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
    )
    assert r.gender == "female"
    assert r.voice_preset == "v2/en_speaker_4"


def test_cast_one_character_rerolls_on_invalid_json():
    """Attempt 1 returns garbage; attempt 2 returns clean JSON."""
    gen = _make_canned_generate_fn([
        "no json here at all, sorry",
        _good_response(),
    ])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
    )
    assert r.voice_preset == "v2/en_speaker_4"


def test_cast_one_character_survives_trailing_object_after_json():
    """BUG-LOCAL-261: the casting LLM emits a valid cast object, then a
    SECOND top-level JSON object after it. The old first-'{'-to-last-'}'
    extractor concatenated both into {...}{...} and json.loads rejected
    the result as 'Extra data'. The shared _otr_json extractor takes the
    first complete object and ignores the trailing one -- attempt 1
    succeeds, no reroll needed."""
    first = _good_response(voice_preset="v2/en_speaker_4")
    trailing = (
        '\n{"character_description":"a stray hallucinated second cast",'
        '"gender":"other","voice_preset":"v2/en_speaker_9"}'
    )
    gen = _make_canned_generate_fn([first + trailing])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
    )
    # The FIRST object wins; the trailing object is ignored.
    assert r.voice_preset == "v2/en_speaker_4"


def test_cast_one_character_survives_trailing_prose_after_json():
    """Trailing prose with no braces after the cast object is tolerated
    too -- the first complete object parses, the chatter is ignored."""
    gen = _make_canned_generate_fn([
        _good_response(voice_preset="v2/en_speaker_6")
        + "\n\nHope this casting fits your radio drama!",
    ])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
    )
    assert r.voice_preset == "v2/en_speaker_6"


def test_cast_one_character_rerolls_on_voice_not_in_pool():
    """Attempt 1 picks a voice not in the available list; attempt 2 fixes it."""
    gen = _make_canned_generate_fn([
        _good_response(voice_preset="v2/en_speaker_8"),  # not in _three_voices()
        _good_response(voice_preset="v2/en_speaker_6"),
    ])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
    )
    assert r.voice_preset == "v2/en_speaker_6"


def test_cast_one_character_repair_attempt_used():
    """All three attempts: two bad, third (repair) succeeds."""
    gen = _make_canned_generate_fn([
        "garbage 1",
        "garbage 2",
        _good_response(),
    ])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
        max_attempts=3,
    )
    assert r.gender == "female"


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
    fenced = "```json\n" + _good_response() + "\n```"
    gen = _make_canned_generate_fn([fenced])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
    )
    assert r.voice_preset == "v2/en_speaker_4"


def test_cast_one_character_handles_prose_preamble():
    """Some models add 'Sure! Here is the cast: { ... }'."""
    prefixed = "Sure! Here is the cast: " + _good_response()
    gen = _make_canned_generate_fn([prefixed])
    r = _OTRC.cast_one_character(
        gen,
        name="ALICE", news_seed="story", style="noir",
        prior_cast=[], available_voices=_three_voices(),
    )
    assert r.voice_preset == "v2/en_speaker_4"


def test_cast_one_character_empty_voice_pool_raises_immediately():
    gen = _make_canned_generate_fn([_good_response()])
    with pytest.raises(_OTRC.CastingFailedError) as exc_info:
        _OTRC.cast_one_character(
            gen,
            name="ALICE", news_seed="story", style="noir",
            prior_cast=[], available_voices=[],
        )
    assert "available_voices is empty" in str(exc_info.value).lower() \
        or "nothing to pick" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Prompt shape
# ---------------------------------------------------------------------------


def test_prompt_has_story_and_style_adjacent_no_period_literal():
    """news_seed + style must be on adjacent lines (continuity rule)
    and the prompt must NOT contain '1940s' or '1903'."""
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="A real science story about black holes.",
        style="noir mystery",
        prior_cast=[],
        available_voices=_three_voices(),
    )
    lines = prompt.splitlines()
    story_idx = next(i for i, l in enumerate(lines) if l.startswith("Story:"))
    style_idx = next(i for i, l in enumerate(lines) if l.startswith("Style:"))
    assert style_idx == story_idx + 1, \
        f"Story and Style must be adjacent (got {story_idx}, {style_idx})"
    # No baked period literals
    assert "1940" not in prompt
    assert "1903" not in prompt


def test_prompt_omits_cast_so_far_block_when_prior_empty():
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=[],
        available_voices=_three_voices(),
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
        available_voices=_three_voices(),
    )
    assert "Cast so far:" in prompt
    assert "LEMMY (M, " in prompt


def test_prompt_truncates_news_seed_at_500_chars():
    huge_seed = "x" * 5000
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed=huge_seed, style="noir",
        prior_cast=[],
        available_voices=_three_voices(),
    )
    story_line = next(l for l in prompt.splitlines() if l.startswith("Story:"))
    # "Story: " + up to 500 chars
    assert len(story_line) <= len("Story: ") + 500


def test_prompt_keeps_voices_in_pool():
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=[],
        available_voices=_three_voices(),
    )
    for preset, _ in _three_voices():
        assert preset in prompt


def test_prompt_has_no_system_prompt_baked():
    """Per the lean-prompt rule, _build_user_prompt returns a single
    user-message body. The role-folding lives at the model loader,
    NOT here."""
    prompt = _OTRC._build_user_prompt(
        name="ALICE",
        news_seed="story", style="noir",
        prior_cast=[],
        available_voices=_three_voices(),
    )
    # Prompt should not contain any role markers
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
        _good_response(voice_preset="v2/en_speaker_3"),
        _good_response(voice_preset="v2/en_speaker_5"),
        _good_response(voice_preset="v2/en_speaker_7"),
    ])
    cast, meta = _OTRC.lock_cast(creative_fn=gen, technical_fn=gen,
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
    # Open slots c02, c03, c04
    assert [r["char_id"] for r in cast[1:]] == ["c02", "c03", "c04"]
    # All voices distinct
    voices = [r["voice_preset"] for r in cast]
    assert len(set(voices)) == len(voices)
    assert meta["lemmy_hit"] is False
    assert meta["num_characters_locked"] == 3


def test_lock_cast_with_lemmy_end_to_end():
    """3-character episode with LEMMY. ANNOUNCER c01, LEMMY c02, two open.

    Bark voice exclusion when LEMMY hits: LEMMY's v2/en_speaker_8 is
    added to taken_voices, so the open slots draw from the remaining
    Bark pool. ANNOUNCER's Kokoro voice never touches the Bark pool.
    """
    rng = random.Random("e2e-with-lemmy")
    gen = _make_canned_generate_fn([
        _good_response(voice_preset="v2/en_speaker_3"),
        _good_response(voice_preset="v2/en_speaker_6"),
    ])
    cast, meta = _OTRC.lock_cast(creative_fn=gen, technical_fn=gen,
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
    # Open slots are c03, c04
    assert [r["char_id"] for r in cast[2:]] == ["c03", "c04"]
    assert meta["lemmy_hit"] is True
    assert meta["num_characters_locked"] == 3  # LEMMY counts


def test_lock_cast_open_slots_see_lemmy_in_prior_cast_but_not_announcer():
    """When LEMMY hits, the per-character LLM prompt should include
    LEMMY in 'Cast so far' (he's ensemble) but NOT ANNOUNCER (he's
    narrator)."""
    captured_prompts: list[str] = []

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        # Capture the user-message content so the test can inspect it
        captured_prompts.append(messages[0]["content"])
        # Any Bark voice works (Kokoro announcer can't collide); pick
        # one that isn't LEMMY's reserved v2/en_speaker_8.
        return _good_response(voice_preset="v2/en_speaker_6")

    rng = random.Random("prior-cast-test")
    _OTRC.lock_cast(creative_fn=gen_fn, technical_fn=gen_fn,
        num_characters=2,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=True,
    )
    # One open slot when num_characters=2 and LEMMY hits.
    assert len(captured_prompts) == 1
    p = captured_prompts[0]
    # LEMMY visible
    assert "LEMMY (M, " in p
    # ANNOUNCER hidden (not in the cast-so-far block)
    assert "ANNOUNCER" not in p


def test_lock_cast_voice_pool_pre_filtered_excludes_lemmy_voice():
    """When LEMMY hits, his v2/en_speaker_8 must not appear in the
    'Voices:' block of the open-slot prompt."""
    captured_prompts: list[str] = []

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        captured_prompts.append(messages[0]["content"])
        # Any non-LEMMY Bark voice works (Kokoro announcer doesn't
        # touch the Bark pool).
        return _good_response(voice_preset="v2/en_speaker_6")

    rng = random.Random("voice-filter-test")
    _OTRC.lock_cast(creative_fn=gen_fn, technical_fn=gen_fn,
        num_characters=2,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=True,
    )
    p = captured_prompts[0]
    assert "v2/en_speaker_8" not in p, \
        "LEMMY's voice should be pre-filtered out of the open-slot pool"


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
    # Should not raise
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
        # BOB collides with ALICE on en_speaker_4 -- a future
        # refactor bug that today's pre-filter + validator + reroll
        # would have prevented.
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
    uniqueness check. A cast where the announcer's Kokoro voice
    happens to lexically match a Bark voice in another row would
    NOT trigger the invariant -- but in practice the namespaces
    don't overlap, so this is a defensive test of the exclusion
    logic itself.
    """
    cast = [
        # Made-up matching string just to prove the exclusion holds;
        # in production an announcer would never have a Bark ID.
        {"char_id": "c01", "name": "ANNOUNCER",
         "voice_preset": "v2/en_speaker_4"},
        {"char_id": "c02", "name": "ALICE",
         "voice_preset": "v2/en_speaker_4"},
    ]
    # ANNOUNCER is excluded from the check; the second en_speaker_4
    # (on ALICE) appears only once among non-announcer rows, so the
    # invariant passes.
    _OTRC._assert_unique_bark_voices(cast)


def test_lock_cast_invariant_fires_at_end_if_voices_collide(
    monkeypatch,
):
    """End-to-end fail-safe: simulate a future refactor where the
    pre-filter no longer excludes already-taken voices AND the
    validator no longer rejects duplicates. Two open characters
    pick the SAME Bark voice; the final invariant in lock_cast
    must raise.

    To get past the new preflight capacity check, the stubbed pool
    has to be large enough for the requested open-slot count.
    """
    # Stub: pool always returns BOTH voices regardless of `taken`,
    # simulating a refactor that broke the pre-filter.
    monkeypatch.setattr(
        _POOLS, "open_voice_pool",
        lambda taken: [
            ("v2/en_speaker_4", "female bright 30s"),
            ("v2/en_speaker_6", "female throaty 40s"),
        ],
    )

    # Stub: cast_one_character always picks the SAME voice -- a
    # refactor that broke the validator's "voice in available_presets"
    # check. Returns a valid CastingResponse so the row goes into
    # the cast.
    def stub_cast_one_character(generate_fn, **kwargs):  # noqa: ARG001
        return _OTRC.CastingResponse(
            character_description="stub character description text",
            gender="female",
            voice_preset="v2/en_speaker_4",
        )

    monkeypatch.setattr(_OTRC, "cast_one_character", stub_cast_one_character)

    rng = random.Random("invariant-fires")
    with pytest.raises(_OTRC.CastingFailedError) as exc_info:
        _OTRC.lock_cast(creative_fn=(lambda *_a, **_kw: ""), technical_fn=(lambda *_a, **_kw: ""),  # generate_fn unused after stub
            num_characters=2,
            news_seed="story", style="noir",
            rng=rng,
            force_lemmy=False,
        )
    assert "POST-CAST INVARIANT" in str(exc_info.value), \
        f"final invariant must fire, got: {exc_info.value!r}"


def test_lock_cast_announcer_kokoro_voice_does_not_filter_bark_pool():
    """When the announcer takes a Kokoro voice, the open-slot Bark
    pool must remain unaffected. Kokoro and Bark are separate TTS
    namespaces -- no collision possible. Per Jeffrey 2026-05-10:
    'announcer is in Kokoro so there can be no cast overlaps.'

    Regression guard: if a future refactor accidentally adds the
    announcer's Kokoro voice to the Bark exclusion set, the
    open-slot prompt will be missing one Bark voice -- not a
    correctness bug today (Kokoro IDs don't match Bark presets so
    the filter is no-op), but a sign of confused intent."""
    captured_prompts: list[str] = []

    def gen_fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        captured_prompts.append(messages[0]["content"])
        return _good_response(voice_preset="v2/en_speaker_3")

    rng = random.Random("kokoro-bark-separation")
    _OTRC.lock_cast(creative_fn=gen_fn, technical_fn=gen_fn,
        num_characters=1,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=False,
    )
    p = captured_prompts[0]
    # The full Bark VOICE_PROFILES list (9 voices) should all appear
    # in the open-slot prompt -- nothing has been pre-filtered out by
    # the Kokoro announcer.
    all_bark_presets = [p for p, _, _, _ in _POOLS.VOICE_PROFILES]
    for preset in all_bark_presets:
        assert preset in p, \
            f"Bark preset {preset!r} missing from open-slot prompt -- " \
            f"announcer's Kokoro voice was wrongly excluding Bark voices"


# ---------------------------------------------------------------------------
# BUG-LOCAL-260: LEMMY cameo decoupled from the seed
#
# A 2026-05-10 change routed the 11% LEMMY roll through the cast
# contract's seeded random.Random. The writer's seed widget ships a
# fixed value, and a fixed seed reproduces ONE roll forever -- so a
# LEMMY-positive seed (42 was one) cast LEMMY on 100% of runs. The
# roll is decoupled again: cast_pools.roll_lemmy() always uses OS
# entropy. force_lemmy stays as the deterministic override.
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
    # Pin OS entropy to a miss so the LEMMY cameo never perturbs the
    # comparison.
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
    # OS entropy pinned to a guaranteed hit -- force_lemmy=False wins.
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.0))
    _pre, _open, hit = _OTRC.assemble_pre_locked_rows(
        num_characters=2, rng=random.Random(1), force_lemmy=False,
    )
    assert hit is False, "force_lemmy=False must override an entropy hit"
    # OS entropy pinned to a guaranteed miss -- force_lemmy=True wins.
    monkeypatch.setattr(_POOLS, "_LEMMY_RNG_SYSTEM", _FixedRandom(0.99))
    _pre2, _open2, hit2 = _OTRC.assemble_pre_locked_rows(
        num_characters=2, rng=random.Random(1), force_lemmy=True,
    )
    assert hit2 is True, "force_lemmy=True must override an entropy miss"
