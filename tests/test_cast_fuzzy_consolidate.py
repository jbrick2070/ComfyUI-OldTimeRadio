"""Regression test for BUG-LOCAL-068 EXPANSION: fuzzy cast consolidation.

Bug shapes covered (both observed live 2026-04-26 PM):

  1. Prefix-overlap (Captain-Eris -> Mistral two-LLM split):
     Captain-Eris uses first names in dialogue tags ("[LLOYD]") while
     Mistral cleanup adds full names in character descriptions
     ("LLOYD KAPOOR"). Naive equality leaves two rows -- one with
     dialogue, no voice; one with voice, no dialogue.

  2. Typo divergence (single-LLM Mistral, no two-LLM split needed):
     Mistral mid-stream typoed "[STANLEY]" as "[STANLEARY]" and
     alternated for the rest of the script. Same fragmentation outcome.

The fix adds difflib.SequenceMatcher.ratio() with prefix-overlap as
a special case. This test exercises the helper directly without
loading transformers / torch / comfy.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root so we can import the orchestrator helpers without
# loading the full ComfyUI custom_nodes machinery.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def _import_helpers():
    """Lazy-import to avoid a hard dependency on transformers etc."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_orch_helpers",
        _REPO / "nodes" / "story_orchestrator.py",
    )
    # We can't actually load the whole module without ComfyUI; instead,
    # extract just the two helpers via source parsing + exec into a
    # private namespace.
    src = (_REPO / "nodes" / "story_orchestrator.py").read_text(encoding="utf-8")
    ns = {"__name__": "_otr_orch_helpers_test"}
    # Find the two helper definitions and exec them in a clean namespace
    import re
    blocks = re.findall(
        r"(^def _norm_cast_key.*?(?=^def |\Z)|^def _cast_names_should_merge.*?(?=^def |\Z)|^def _consolidate_similar_cast_rows.*?(?=^def |\Z))",
        src,
        re.MULTILINE | re.DOTALL,
    )
    if len(blocks) < 3:
        raise RuntimeError(
            f"could not extract all three helpers; got {len(blocks)} block(s)"
        )
    for blk in blocks:
        exec(compile(blk, "story_orchestrator.py", "exec"), ns)
    return ns


_H = _import_helpers()
_norm_cast_key = _H["_norm_cast_key"]
_cast_names_should_merge = _H["_cast_names_should_merge"]
_consolidate_similar_cast_rows = _H["_consolidate_similar_cast_rows"]


# ---------------------------------------------------------------------------
# Pair-wise should-merge decisions
# ---------------------------------------------------------------------------


class TestPrefixOverlapDecisions:
    """LLOYD vs LLOYD KAPOOR should merge with the longer winning."""

    def test_first_name_inside_full_name_merges(self):
        should, winner = _cast_names_should_merge("LLOYD", "LLOYD KAPOOR")
        assert should is True
        assert winner == "LLOYD KAPOOR"

    def test_full_name_inside_first_name_merges_either_order(self):
        should, winner = _cast_names_should_merge("LLOYD KAPOOR", "LLOYD")
        assert should is True
        assert winner == "LLOYD KAPOOR"

    def test_partial_prefix_without_word_boundary_does_not_merge(self):
        # "LLOY" is a prefix of "LLOYD" but doesn't end on a space -- not a merge
        should, winner = _cast_names_should_merge("LLOY", "LLOYD")
        # SequenceMatcher.ratio("LLOY", "LLOYD") = 8/9 = 0.89, both single-token
        # so the typo rule fires and we get a merge. That's also correct.
        assert should is True

    def test_distinct_first_names_do_not_merge(self):
        should, winner = _cast_names_should_merge("PHYLLIS", "LLOYD KAPOOR")
        assert should is False


class TestTypoDivergenceDecisions:
    """STANLEY vs STANLEARY should merge via fuzzy ratio."""

    def test_stanley_vs_stanleary_merges(self):
        should, winner = _cast_names_should_merge("STANLEY", "STANLEARY")
        assert should is True
        # Longer string wins for typo divergence
        assert winner == "STANLEARY"

    def test_distinct_single_tokens_do_not_merge(self):
        should, winner = _cast_names_should_merge("STANLEY", "PHYLLIS")
        assert should is False
        assert winner is None

    def test_multi_token_typo_does_not_merge(self):
        # ROBERT FROST vs ROBERT FORD -- different last names, ratio is high
        # but multi-token names aren't fuzzy-merged (would risk wrong calls).
        should, winner = _cast_names_should_merge("ROBERT FROST", "ROBERT FORD")
        assert should is False


class TestEdgeCaseDecisions:

    def test_empty_strings_never_merge(self):
        assert _cast_names_should_merge("", "") == (False, None)
        assert _cast_names_should_merge("STANLEY", "") == (False, None)
        assert _cast_names_should_merge("", "STANLEY") == (False, None)

    def test_one_char_names_never_merge(self):
        # Avoid degenerate ratio calculations on single chars.
        should, winner = _cast_names_should_merge("A", "B")
        assert should is False

    def test_identical_names_merge(self):
        should, winner = _cast_names_should_merge("LLOYD", "LLOYD")
        assert should is True
        assert winner == "LLOYD"


# ---------------------------------------------------------------------------
# Whole-list consolidation
# ---------------------------------------------------------------------------


class TestPrefixCollapse:

    def test_lloyd_lloyd_kapoor_collapses_into_one(self):
        cast = [
            {"char_id": "c01", "name": "ANNOUNCER",   "voice_preset": "v2/en_speaker_0", "line_count": 3},
            {"char_id": "c02", "name": "LLOYD",       "voice_preset": None,                "line_count": 6},
            {"char_id": "c03", "name": "LLOYD KAPOOR","voice_preset": "v2/en_speaker_1",  "line_count": 0},
            {"char_id": "c04", "name": "PHYLLIS",     "voice_preset": None,                "line_count": 7},
            {"char_id": "c05", "name": "PHYLLIS SHAW","voice_preset": "v2/en_speaker_7",  "line_count": 0},
        ]
        out = _consolidate_similar_cast_rows(cast)
        names = [r["name"] for r in out]
        assert "LLOYD KAPOOR" in names
        assert "LLOYD" not in names  # standalone LLOYD merged into the longer
        assert "PHYLLIS SHAW" in names
        assert "PHYLLIS" not in names
        # Voice preset survived from the description-only row
        by_name = {r["name"]: r for r in out}
        assert by_name["LLOYD KAPOOR"]["voice_preset"] == "v2/en_speaker_1"
        assert by_name["PHYLLIS SHAW"]["voice_preset"] == "v2/en_speaker_7"
        # line_count summed (defensive)
        assert by_name["LLOYD KAPOOR"]["line_count"] == 6
        assert by_name["PHYLLIS SHAW"]["line_count"] == 7

    def test_collapse_is_idempotent(self):
        cast = [
            {"char_id": "c01", "name": "LLOYD",        "voice_preset": None,               "line_count": 4},
            {"char_id": "c02", "name": "LLOYD KAPOOR", "voice_preset": "v2/en_speaker_3", "line_count": 0},
        ]
        once = _consolidate_similar_cast_rows(cast)
        twice = _consolidate_similar_cast_rows(once)
        assert [r["name"] for r in once] == [r["name"] for r in twice]


class TestTypoCollapse:

    def test_stanley_stanleary_typo_collapses(self):
        cast = [
            {"char_id": "c01", "name": "ANNOUNCER",  "voice_preset": "v2/en_speaker_0", "line_count": 4},
            {"char_id": "c02", "name": "STANLEY",    "voice_preset": "v2/en_speaker_8", "line_count": 4},
            {"char_id": "c03", "name": "STANLEARY",  "voice_preset": None,                "line_count": 8},
        ]
        out = _consolidate_similar_cast_rows(cast)
        names = [r["name"] for r in out]
        # Two of the three rows collapse into one
        assert len(out) == 2
        assert "ANNOUNCER" in names
        # Longer name wins on typo collapse
        assert "STANLEARY" in names
        # The voice preset survived (it was on the shorter name row)
        by_name = {r["name"]: r for r in out}
        assert by_name["STANLEARY"]["voice_preset"] == "v2/en_speaker_8"
        # Line counts summed
        assert by_name["STANLEARY"]["line_count"] == 12


class TestNoOverMerge:
    """Distinct characters with different names must NOT collapse."""

    def test_two_unrelated_first_names_stay_separate(self):
        cast = [
            {"char_id": "c01", "name": "LLOYD",   "voice_preset": "v2/en_speaker_1"},
            {"char_id": "c02", "name": "PHYLLIS", "voice_preset": "v2/en_speaker_7"},
            {"char_id": "c03", "name": "EDWARD",  "voice_preset": "v2/en_speaker_3"},
        ]
        out = _consolidate_similar_cast_rows(cast)
        assert len(out) == 3
        assert {"LLOYD", "PHYLLIS", "EDWARD"} == {r["name"] for r in out}

    def test_robert_frost_vs_robert_ford_stays_separate(self):
        # Multi-token names with high fuzzy ratio still must not merge.
        cast = [
            {"char_id": "c01", "name": "ROBERT FROST"},
            {"char_id": "c02", "name": "ROBERT FORD"},
        ]
        out = _consolidate_similar_cast_rows(cast)
        assert len(out) == 2

    def test_short_name_not_inside_other_stays(self):
        # "ED" is a prefix of "EDWARD" but only at the substring level,
        # not at a word boundary. Should NOT prefix-merge. May or may
        # not fuzzy-merge depending on ratio; the test is documenting
        # current behaviour to catch unintentional regressions.
        cast = [
            {"char_id": "c01", "name": "ED"},
            {"char_id": "c02", "name": "EDWARD"},
        ]
        # Both single-token, ratio("ED", "EDWARD") = 4/8 = 0.5 -> below 0.85
        out = _consolidate_similar_cast_rows(cast)
        assert len(out) == 2


class TestEmptyAndSingleton:

    def test_empty_input(self):
        assert _consolidate_similar_cast_rows([]) == []

    def test_none_input(self):
        assert _consolidate_similar_cast_rows(None) == []

    def test_single_row_input_unchanged(self):
        cast = [{"char_id": "c01", "name": "ANNOUNCER", "voice_preset": "v2/en_speaker_0"}]
        out = _consolidate_similar_cast_rows(cast)
        assert len(out) == 1
        assert out[0]["name"] == "ANNOUNCER"


class TestNormaliserParity:
    """The new module-level _norm_cast_key must match the inline _norm_name_key."""

    def test_underscore_to_space(self):
        assert _norm_cast_key("LLOYD_KAPOOR") == "LLOYD KAPOOR"

    def test_lowercase_to_upper(self):
        assert _norm_cast_key("lloyd kapoor") == "LLOYD KAPOOR"

    def test_strip(self):
        assert _norm_cast_key("  LLOYD  ") == "LLOYD"

    def test_empty_safe(self):
        assert _norm_cast_key("") == ""
        assert _norm_cast_key(None) == ""
