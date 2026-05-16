"""tests/test_musicgen_style_palette.py

P1 guardrails for the voice-path-cleanbreak — MusicGen ledger-aware
rewrite. These tests pin the new deterministic style-palette contract
that replaces the legacy ``CUE_DEFAULTS`` / Director-derived plan.

Contract under test (per docs/voice-path-cleanbreak-plan.md §4):
  - Module exports ``_STYLE_PALETTE`` covering every active style slug.
  - Module exports ``_PROMPT_TAIL`` appended to every cue prompt.
  - Module exports ``_resolve_cue_from_style(cue_id, style, mood_suffix)``.
  - Unknown style slug raises ValueError.
  - Every (style, cue_id) pair lands a non-empty prompt.
  - ``_PROMPT_TAIL`` is the literal tail of every resolved prompt.
  - No hardcoded era literals ("1940s", "vintage", "old time radio",
    "warm brass", "upright bass") survive in any palette entry.
"""
from __future__ import annotations

import re

import pytest


# The 10 canonical style slugs per OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL.
CANONICAL_STYLE_SLUGS = (
    "closed_room_suspense",
    "detective_case_file",
    "pulp_serial_cliffhanger",
    "mission_control_procedural",
    "deep_space_distress_call",
    "noir_interrogation",
    "small_town_uncanny",
    "radio_newsroom_emergency",
    "haunted_broadcast_signal",
    "laboratory_containment",
)


# Forbidden literals — must not appear in any palette entry. Anchors the
# Standing Directive no-era-literals invariant for the music plane.
FORBIDDEN_LITERALS = (
    "1940",
    "vintage",
    "old time radio",
    "warm brass",
    "upright bass",
    "tube saturation",  # post-rewrite the palette is era-neutral
    "AM radio",
)


@pytest.fixture
def musicgen_mod():
    """Import the module under test once per test."""
    from nodes import musicgen_theme as mod
    return mod


def test_palette_covers_every_canonical_style(musicgen_mod):
    """Every active style slug must have a palette entry."""
    palette = musicgen_mod._STYLE_PALETTE
    missing = [s for s in CANONICAL_STYLE_SLUGS if s not in palette]
    assert not missing, f"_STYLE_PALETTE missing slugs: {missing!r}"


def test_palette_entry_covers_every_cue_id(musicgen_mod):
    """Every palette entry must define opening / closing / interstitial."""
    palette = musicgen_mod._STYLE_PALETTE
    cue_ids = musicgen_mod.CUE_IDS
    for slug, entry in palette.items():
        for cue_id in cue_ids:
            assert cue_id in entry, (
                f"palette[{slug!r}] missing cue_id {cue_id!r}"
            )
            prompt = entry[cue_id]
            assert isinstance(prompt, str) and prompt.strip(), (
                f"palette[{slug!r}][{cue_id!r}] must be non-empty str, "
                f"got {prompt!r}"
            )


def test_no_era_literals_in_any_palette_prompt(musicgen_mod):
    """Hardcoded period literals are a Standing Directive violation."""
    palette = musicgen_mod._STYLE_PALETTE
    for slug, entry in palette.items():
        for cue_id, prompt in entry.items():
            low = prompt.lower()
            for bad in FORBIDDEN_LITERALS:
                assert bad.lower() not in low, (
                    f"_STYLE_PALETTE[{slug!r}][{cue_id!r}] contains "
                    f"forbidden era literal {bad!r}: {prompt!r}"
                )


def test_prompt_tail_constant_exists_and_nonempty(musicgen_mod):
    """_PROMPT_TAIL is the deterministic suffix appended to every cue."""
    tail = musicgen_mod._PROMPT_TAIL
    assert isinstance(tail, str) and tail.strip(), (
        f"_PROMPT_TAIL must be a non-empty string, got {tail!r}"
    )


def test_resolve_cue_from_style_lands_tail(musicgen_mod):
    """Every resolved prompt ends with _PROMPT_TAIL verbatim."""
    tail = musicgen_mod._PROMPT_TAIL
    for slug in CANONICAL_STYLE_SLUGS:
        for cue_id in musicgen_mod.CUE_IDS:
            prompt, dur = musicgen_mod._resolve_cue_from_style(
                cue_id, slug, ""
            )
            assert prompt.endswith(tail), (
                f"resolve({cue_id!r}, {slug!r}) missing _PROMPT_TAIL: "
                f"{prompt!r}"
            )
            assert isinstance(dur, int) and dur > 0, (
                f"resolve({cue_id!r}, {slug!r}) duration must be positive "
                f"int, got {dur!r}"
            )


def test_resolve_cue_from_style_raises_on_unknown_slug(musicgen_mod):
    """Unknown style is a hard fail — no default palette."""
    with pytest.raises(ValueError, match="unknown style"):
        musicgen_mod._resolve_cue_from_style(
            "opening", "totally_made_up_slug", ""
        )


def test_mood_suffix_concatenates_when_keyword_present(musicgen_mod):
    """Mood overlay appends matching tags."""
    # Choose a keyword from _MOOD_TAGS so the contract is symmetric.
    mood_tags = musicgen_mod._MOOD_TAGS
    assert mood_tags, "_MOOD_TAGS must be non-empty"
    keyword = next(iter(mood_tags))
    expected_tag = mood_tags[keyword]
    suffix = musicgen_mod._mood_suffix(f"a story about {keyword} on the air")
    assert expected_tag in suffix, (
        f"mood suffix for keyword {keyword!r} should contain "
        f"{expected_tag!r}; got {suffix!r}"
    )
    # Suffix starts with a comma so it concatenates cleanly.
    assert suffix.startswith(", "), (
        f"mood suffix must start with ', '; got {suffix!r}"
    )


def test_mood_suffix_empty_when_no_keyword_match(musicgen_mod):
    """No mood keyword → empty suffix."""
    assert musicgen_mod._mood_suffix("") == ""
    assert musicgen_mod._mood_suffix(
        "this brief contains no matching mood vocabulary at all"
    ) == ""


def test_resolve_cue_from_style_concatenates_mood_before_tail(musicgen_mod):
    """Resolved prompt order: base → mood suffix → _PROMPT_TAIL."""
    mood_tags = musicgen_mod._MOOD_TAGS
    keyword = next(iter(mood_tags))
    expected_tag = mood_tags[keyword]
    suffix = musicgen_mod._mood_suffix(f"about {keyword}")

    slug = CANONICAL_STYLE_SLUGS[0]
    prompt, _ = musicgen_mod._resolve_cue_from_style("opening", slug, suffix)

    # Mood tag must land somewhere between the base and the tail.
    tail = musicgen_mod._PROMPT_TAIL
    assert prompt.endswith(tail)
    body = prompt[: -len(tail)]
    assert expected_tag in body, (
        f"mood tag {expected_tag!r} missing from prompt body {body!r}"
    )


def test_director_secondary_path_is_gone(musicgen_mod):
    """The legacy Director-derived helpers must not survive the rewrite."""
    forbidden_attrs = (
        "CUE_DEFAULTS",        # legacy era-anchored defaults
        "_resolve_cue",        # legacy plan-based resolver
    )
    for attr in forbidden_attrs:
        assert not hasattr(musicgen_mod, attr), (
            f"musicgen_theme.{attr} must be deleted (Standing Directive)"
        )


def test_palette_covers_writer_style_pool(musicgen_mod):
    """Sprint 1.3 drift guardrail (voice-path-cleanbreak follow-up).

    The MusicGen palette MUST stay in lockstep with the writer's
    ``_STYLE_PICKER_SEED_POOL``. Any slug the writer can emit for the
    style widget must have a palette entry; any palette entry that
    doesn't appear in the writer's pool is dead vocabulary.

    Catches the failure mode that Q2 of the QA doc named: a writer-side
    pool extension lands without a palette extension and MusicGen
    raises ValueError("unknown style slug 'X'") at render time. With
    this test in CI, the drift is caught at commit instead of soak.
    """
    from nodes.OTR_LedgerScriptWriter import _STYLE_PICKER_SEED_POOL
    palette_slugs = set(musicgen_mod._STYLE_PALETTE)
    pool_slugs = set(_STYLE_PICKER_SEED_POOL)
    in_palette_only = palette_slugs - pool_slugs
    in_pool_only = pool_slugs - palette_slugs
    assert palette_slugs == pool_slugs, (
        f"Style palette drifted from writer style pool. "
        f"In palette only (dead vocab): {sorted(in_palette_only)}. "
        f"In pool only (writer can emit but MusicGen will hard-fail): "
        f"{sorted(in_pool_only)}."
    )


def test_canonical_slugs_in_test_match_writer_pool(musicgen_mod):
    """Defensive: the CANONICAL_STYLE_SLUGS constant in this test
    file is a snapshot of the writer's pool. If the writer's pool
    drifts and someone forgets to update CANONICAL_STYLE_SLUGS, the
    other tests in this file would silently still pass against the
    stale snapshot. This test catches that case independently of the
    palette-vs-pool drift test above."""
    from nodes.OTR_LedgerScriptWriter import _STYLE_PICKER_SEED_POOL
    assert set(CANONICAL_STYLE_SLUGS) == set(_STYLE_PICKER_SEED_POOL), (
        "CANONICAL_STYLE_SLUGS in this test file is stale relative to "
        "OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL. Update both "
        "in lockstep."
    )


# Sprint C C3 (2026-05-15): the six genre-table tests that previously
# lived here -- test_genre_table_strict_equality_with_style_pool,
# test_resolve_genre_known_styles_use_table, test_resolve_genre_empty_raises,
# test_resolve_genre_unknown_raises, test_preview_genre_is_isolated_from_writer_path,
# test_preview_genre_keeps_legacy_fallback_behavior -- were deleted alongside
# the _GENRE_BY_STYLE / _resolve_genre / _preview_genre symbols they pinned.
# The `meta.visual_plan.genre` stamp is retired; downstream consumers fall
# back to `meta.style` directly. Per the no-legacy-back-compat standing
# directive: tests for deleted contracts are deleted, not preserved.
# Replacement deletion-locked assertions live in
# `tests/test_no_genre_by_style_c3.py`.
