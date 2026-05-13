"""S13.1 -- Cast contract structural-token rejection (IMP-4).

Pre-S13.1 audit: each of the five tokens TITLE / NOTE / TARGET /
STYLE / NARRATOR slipped through the cast contract when present
as a character ``name`` in ``ledger.cast[]``. The
``_assert_voice_preset_invariant`` only checks for empty / non-v2
voice_preset; it has no opinion on the *name* shape.

This module's parametrized test pins the expected post-S13.1
behavior: every structural token name MUST raise
``CastingFailedError`` from
``_otr_casting._assert_no_structural_tokens_in_cast``.

Risk asymmetry
  False-positive: a real character named "Style" gets rejected.
  Cost: writer reroll with a different name.
  False-negative: an LLM hallucination renders as a voice line in
  production.
  Cost: a broken episode that ships before anyone notices.

The plan calls this asymmetry "far lower" for the false-positive,
which is why the patterns in
``_otr_casting._NON_CHARACTER_CAST_PATTERNS`` are aggressive.
"""
from __future__ import annotations

import pytest

from nodes._otr_casting import (
    CastingFailedError,
    _assert_no_structural_tokens_in_cast,
    _looks_like_non_character_cast_name,
)


STRUCTURAL_TOKENS = ("TITLE", "NOTE", "TARGET", "STYLE", "NARRATOR")


@pytest.mark.parametrize("token", STRUCTURAL_TOKENS)
def test_cast_contract_rejects_structural_token_as_character(token):
    """Each of the five S13.1 structural tokens MUST raise when
    present as a character name in the cast list."""
    cast = [
        {"name": "ANNOUNCER", "voice_preset": "bm_george",
         "character_description": "..."},
        {"name": token, "voice_preset": "v2/en_speaker_0",
         "character_description": "..."},
    ]
    with pytest.raises(CastingFailedError) as excinfo:
        _assert_no_structural_tokens_in_cast(cast)
    assert token in str(excinfo.value), (
        f"CastingFailedError message must name the offending "
        f"structural token {token!r} so the diagnostic is actionable."
    )


def test_cast_contract_accepts_real_character_names():
    """Sanity check: real character names PASS the structural-token
    guard. The patterns are aggressive but they shouldn't fire on
    obviously-character strings."""
    cast = [
        {"name": "ANNOUNCER", "voice_preset": "bm_george",
         "character_description": "..."},
        {"name": "OSCAR SIRIKIT", "voice_preset": "v2/en_speaker_0",
         "character_description": "..."},
        {"name": "WILL SMITHERS", "voice_preset": "v2/en_speaker_1",
         "character_description": "..."},
        {"name": "JIMBO HALPERT", "voice_preset": "v2/en_speaker_2",
         "character_description": "..."},
    ]
    # Should NOT raise.
    _assert_no_structural_tokens_in_cast(cast)


def test_cast_contract_helper_classifies_each_token():
    """``_looks_like_non_character_cast_name`` returns True for each
    of the five structural tokens. Direct unit test of the helper
    so a pattern regression surfaces here even before the assert
    fires."""
    for token in STRUCTURAL_TOKENS:
        assert _looks_like_non_character_cast_name(token), (
            f"_looks_like_non_character_cast_name({token!r}) returned "
            f"False; pattern coverage broken."
        )


def test_cast_contract_helper_passes_real_names():
    """The helper does NOT fire on real character names."""
    real = ["OSCAR SIRIKIT", "WILL SMITHERS", "JIMBO HALPERT",
            "WENDY HUDSON", "STANLEY CRANSTON", "MINA SPENDER",
            "BABA YAGA", "LEMMY KILMISTER"]
    for name in real:
        assert not _looks_like_non_character_cast_name(name), (
            f"_looks_like_non_character_cast_name({name!r}) fired on "
            f"a legitimate character name. Tighten the patterns or "
            f"add a case-sensitive whitelist."
        )


def test_announcer_slot_passes_structural_guard():
    """ANNOUNCER is the canonical narrator slot, not a structural
    artefact. The guard MUST exempt it explicitly."""
    cast = [
        {"name": "ANNOUNCER", "voice_preset": "bm_george",
         "character_description": "..."},
    ]
    _assert_no_structural_tokens_in_cast(cast)


def test_legacy_sfx_cue_artefacts_still_caught():
    """The structural-token guard ports + extends the deleted
    story_orchestrator._looks_like_non_character_cast_name. This
    test pins the legacy SFX-cue patterns (BUG-LOCAL-090) still
    fire post-port. Original BLOCKLIST patterns must still match."""
    legacy_artefacts = [
        "ALARM BLARING",
        "SFX EXPLOSION",
        "MUSIC QUEUE",
        "KEVIN VOICEOVER",  # BUG-LOCAL-097
        "JOHN V.O.",
        "OFF SCREEN",
    ]
    for artefact in legacy_artefacts:
        assert _looks_like_non_character_cast_name(artefact), (
            f"Legacy artefact {artefact!r} no longer caught. Pattern "
            f"port from story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS "
            f"is incomplete."
        )
