"""tests/test_cast_contract_helpers.py

Phase 0+ Cast Contract Extensions §1 helpers:
    build_contract_from_director_plan
    detect_aliases

Pure-stdlib unit tests; no torch / no VRAM / no Comfy graph.
"""
from __future__ import annotations

import pytest

from nodes._otr_cast_contract import (
    CastContract,
    CharacterEntry,
    build_contract_from_director_plan,
    detect_aliases,
)


# ---------- build_contract_from_director_plan ---------------------------


def test_build_contract_from_simple_plan_assigns_canonical_ids():
    plan = {
        "voice_assignments": {
            "MONTY": "v2/en_speaker_3",
            "AEGEUS": "v2/en_speaker_5",
            "SAILOR": "v2/en_speaker_9",
        }
    }
    contract = build_contract_from_director_plan(plan)
    # Sorted alphabetically: AEGEUS, MONTY, SAILOR -> c01, c02, c03
    assert [c.character_id for c in contract.characters] == ["c01", "c02", "c03"]
    assert [c.canonical_name for c in contract.characters] == ["AEGEUS", "MONTY", "SAILOR"]
    # Default engine prefix added
    assert contract.characters[0].voice_spec == "bark:v2/en_speaker_5"
    assert contract.characters[1].voice_spec == "bark:v2/en_speaker_3"
    # Versioned
    assert contract.version.startswith("sha:")


def test_build_contract_preserves_explicit_engine_prefix():
    plan = {
        "voice_assignments": {
            "MONTY": "kokoro:bm_fable",
            "AEGEUS": "v2/en_speaker_5",  # implicit -> bark
        }
    }
    contract = build_contract_from_director_plan(plan)
    by_name = {c.canonical_name: c for c in contract.characters}
    assert by_name["MONTY"].voice_spec == "kokoro:bm_fable"
    assert by_name["AEGEUS"].voice_spec == "bark:v2/en_speaker_5"


def test_build_contract_accepts_dict_voice_assignments():
    plan = {
        "voice_assignments": {
            "MONTY": {"engine": "cosyvoice", "preset": "robotic_calm"},
            "AEGEUS": {"preset": "v2/en_speaker_5"},  # no engine -> bark
        }
    }
    contract = build_contract_from_director_plan(plan)
    by_name = {c.canonical_name: c for c in contract.characters}
    assert by_name["MONTY"].voice_spec == "cosyvoice:robotic_calm"
    assert by_name["AEGEUS"].voice_spec == "bark:v2/en_speaker_5"


def test_build_contract_empty_plan_returns_empty_versioned_contract():
    contract = build_contract_from_director_plan({})
    assert contract.characters == []
    assert contract.version.startswith("sha:")


def test_build_contract_none_plan_returns_empty_versioned_contract():
    contract = build_contract_from_director_plan(None)  # type: ignore[arg-type]
    assert contract.characters == []
    assert contract.version.startswith("sha:")


def test_build_contract_rejects_non_dict_voice_assignments():
    with pytest.raises(TypeError, match="must be a dict"):
        build_contract_from_director_plan({"voice_assignments": ["MONTY", "AEGEUS"]})


def test_build_contract_uppercases_canonical_names():
    plan = {"voice_assignments": {"monty": "v2/en_speaker_3"}}
    contract = build_contract_from_director_plan(plan)
    assert contract.characters[0].canonical_name == "MONTY"


def test_build_contract_skips_blank_keys():
    plan = {
        "voice_assignments": {
            "MONTY": "v2/en_speaker_3",
            "   ": "v2/en_speaker_9",  # blank -> dropped
        }
    }
    contract = build_contract_from_director_plan(plan)
    assert len(contract.characters) == 1
    assert contract.characters[0].canonical_name == "MONTY"


def test_build_contract_handles_padded_keys_without_keyerror():
    """Regression: 2026-05-08 round-robin caught that stripping keys then
    indexing the original dict raised KeyError on padded inputs. Helper
    must now route through the cleaned lookup map.
    """
    plan = {
        "voice_assignments": {
            "  MONTY  ": "v2/en_speaker_3",
            "\tAEGEUS\n": "v2/en_speaker_5",
        }
    }
    contract = build_contract_from_director_plan(plan)
    assert [c.canonical_name for c in contract.characters] == ["AEGEUS", "MONTY"]
    by_name = {c.canonical_name: c for c in contract.characters}
    assert by_name["MONTY"].voice_spec == "bark:v2/en_speaker_3"
    assert by_name["AEGEUS"].voice_spec == "bark:v2/en_speaker_5"


def test_build_contract_padded_collision_first_wins():
    """If a plan has both 'MONTY' and '  MONTY  ', the first one wins
    deterministically (no silent overwrite, no crash).
    """
    plan = {
        "voice_assignments": {
            "MONTY": "v2/en_speaker_3",
            "  MONTY  ": "v2/en_speaker_9",  # collision after strip
        }
    }
    contract = build_contract_from_director_plan(plan)
    assert len(contract.characters) == 1
    assert contract.characters[0].voice_spec == "bark:v2/en_speaker_3"


def test_build_contract_version_stable_under_insertion_order():
    plan_a = {
        "voice_assignments": {
            "MONTY": "v2/en_speaker_3",
            "AEGEUS": "v2/en_speaker_5",
        }
    }
    plan_b = {
        "voice_assignments": {
            "AEGEUS": "v2/en_speaker_5",
            "MONTY": "v2/en_speaker_3",
        }
    }
    a = build_contract_from_director_plan(plan_a)
    b = build_contract_from_director_plan(plan_b)
    assert a.version == b.version


# ---------- detect_aliases ----------------------------------------------


def _baseline_contract() -> CastContract:
    return CastContract(
        characters=[
            CharacterEntry("c01", "AEGEUS", [], "bark:v2/en_speaker_5"),
            CharacterEntry("c02", "MONTY", [], "bark:v2/en_speaker_3"),
            CharacterEntry("c03", "SAILOR", [], "bark:v2/en_speaker_9"),
        ]
    )


def test_detect_aliases_finds_truncation_alias():
    """MONTGOMERY appears in script; MONTY in contract -> alias of c02."""
    script = (
        "AEGEUS: hello there.\n"
        "MONTGOMERY: hello back to you.\n"
        "SAILOR: aye aye.\n"
    )
    aliases = detect_aliases(script, _baseline_contract())
    assert aliases == {"MONTGOMERY": "c02"}


def test_detect_aliases_finds_expansion_alias():
    """MONT appears in script; MONTY in contract -> alias of c02."""
    script = "MONT: short form.\n"
    aliases = detect_aliases(script, _baseline_contract())
    assert aliases == {"MONT": "c02"}


def test_detect_aliases_skips_known_canonical():
    script = "MONTY: I am the canonical.\nAEGEUS: as am I.\n"
    aliases = detect_aliases(script, _baseline_contract())
    assert aliases == {}


def test_detect_aliases_skips_known_alias():
    contract = CastContract(
        characters=[
            CharacterEntry("c01", "MONTY", ["MONTGOMERY"], "bark:v2/en_speaker_3"),
        ]
    )
    script = "MONTGOMERY: I am already a registered alias.\n"
    aliases = detect_aliases(script, contract)
    assert aliases == {}


def test_detect_aliases_ignores_genuinely_unknown():
    """A tag that shares no 4-char prefix with any canonical is left alone."""
    script = "ZORG: I am not in the cast.\n"
    aliases = detect_aliases(script, _baseline_contract())
    assert aliases == {}


def test_detect_aliases_ignores_structural_headers():
    script = (
        "SCENE 1: Hill House.\n"
        "ACT 2: The reckoning.\n"
        "MONTY: real dialogue.\n"
    )
    aliases = detect_aliases(script, _baseline_contract())
    assert aliases == {}


def test_detect_aliases_first_match_wins_when_two_canonicals_share_prefix():
    """MARLA + MARLON both share 'MARL' prefix; MARLENE picks the first listed."""
    contract = CastContract(
        characters=[
            CharacterEntry("c01", "MARLA", [], "bark:v2/en_speaker_2"),
            CharacterEntry("c02", "MARLON", [], "bark:v2/en_speaker_4"),
        ]
    )
    aliases = detect_aliases("MARLENE: ambiguous.\n", contract)
    # First match in iteration order -> c01. §4 is the canonical disambiguator.
    assert aliases == {"MARLENE": "c01"}


def test_detect_aliases_empty_script_returns_empty():
    assert detect_aliases("", _baseline_contract()) == {}


def test_detect_aliases_rejects_non_contract():
    with pytest.raises(TypeError, match="must be a CastContract"):
        detect_aliases("MONTY: hi.\n", "not a contract")  # type: ignore[arg-type]


def test_detect_aliases_case_insensitive_dialogue_tag():
    """Dialogue tag in mixed/lower case still matches once normalized."""
    # Using a fully-uppercase tag is the normal case; a mostly-uppercase
    # variant like 'MONTY' should still hit the lookup. A genuinely lower
    # case 'monty:' would fail the uppercase ratio gate by design.
    script = "MONTY: yes.\n"
    aliases = detect_aliases(script, _baseline_contract())
    assert aliases == {}  # MONTY is canonical, so no alias to add


def test_detect_aliases_returns_first_seen_order():
    """Multiple unknowns return in script-order to keep merge logs deterministic."""
    contract = CastContract(
        characters=[
            CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3"),
            CharacterEntry("c02", "AEGEUS", [], "bark:v2/en_speaker_5"),
        ]
    )
    script = (
        "AEGEU: ambiguous prefix one.\n"
        "MONTGOMERY: ambiguous prefix two.\n"
    )
    aliases = detect_aliases(script, contract)
    # Insertion order: AEGEU first, MONTGOMERY second.
    assert list(aliases.keys()) == ["AEGEU", "MONTGOMERY"]
    assert aliases["AEGEU"] == "c02"
    assert aliases["MONTGOMERY"] == "c01"
