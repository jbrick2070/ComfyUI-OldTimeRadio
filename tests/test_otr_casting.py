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
    assert pre_locked[0]["voice_preset"].startswith("v2/en_speaker_")
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


def test_assemble_pre_locked_rows_announcer_5050_balance():
    """Roll 200 episodes; ANNOUNCER gender should land ~50/50."""
    counts = Counter()
    for i in range(200):
        rng = random.Random(f"balance-{i}")
        pre_locked, _, _ = _OTRC.assemble_pre_locked_rows(
            num_characters=2, rng=rng, force_lemmy=False,
        )
        counts[pre_locked[0]["gender"]] += 1
    male = counts["male"]
    female = counts["female"]
    # 4 announcer presets (2 male + 2 female), each picked uniformly
    # at random by the seeded RNG. Tolerance ~10pp on 200 trials.
    assert 80 <= male <= 120, f"male skew: {male}/200 (expected ~100)"
    assert 80 <= female <= 120, f"female skew: {female}/200 (expected ~100)"
    assert male + female == 200, f"unknown announcer gender values: {counts!r}"


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
    """3-character episode, no LEMMY. ANNOUNCER c01, three open slots."""
    rng = random.Random("e2e-no-lemmy")
    # Exactly 3 LLM calls (one per open slot). Each picks a different
    # voice. ALL three are NEVER in ANNOUNCER_PRESETS ({0,1,4,9}) and
    # NEVER LEMMY's voice (8), so the responses are seed-safe -- no
    # matter which announcer voice the seeded RNG picks, none of these
    # three collide with it.
    gen = _make_canned_generate_fn([
        _good_response(voice_preset="v2/en_speaker_3"),
        _good_response(voice_preset="v2/en_speaker_5"),
        _good_response(voice_preset="v2/en_speaker_7"),
    ])
    cast, meta = _OTRC.lock_cast(
        gen,
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
    """3-character episode with LEMMY. ANNOUNCER c01, LEMMY c02, two open."""
    rng = random.Random("e2e-with-lemmy")
    # Both responses use voices NEVER in ANNOUNCER_PRESETS so the
    # test is seed-safe regardless of which announcer voice is picked.
    gen = _make_canned_generate_fn([
        _good_response(voice_preset="v2/en_speaker_3"),
        _good_response(voice_preset="v2/en_speaker_6"),
    ])
    cast, meta = _OTRC.lock_cast(
        gen,
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
        # Use en_speaker_6: never in ANNOUNCER_PRESETS, not LEMMY's
        # voice, so it survives both pre-filters regardless of seed.
        return _good_response(voice_preset="v2/en_speaker_6")

    rng = random.Random("prior-cast-test")
    _OTRC.lock_cast(
        gen_fn,
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
        # en_speaker_6 is never in ANNOUNCER_PRESETS and is not
        # LEMMY's reserved voice; safe pick regardless of seed.
        return _good_response(voice_preset="v2/en_speaker_6")

    rng = random.Random("voice-filter-test")
    _OTRC.lock_cast(
        gen_fn,
        num_characters=2,
        news_seed="story", style="noir",
        rng=rng,
        force_lemmy=True,
    )
    p = captured_prompts[0]
    assert "v2/en_speaker_8" not in p, \
        "LEMMY's voice should be pre-filtered out of the open-slot pool"
