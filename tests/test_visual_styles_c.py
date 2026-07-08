"""tests/test_visual_styles_c.py

Visual-style TOTAL COVERAGE chunk C (STAGE3_TOTAL_COVERAGE_SUBPLAN v5 FINAL,
the LAST chunk) -- still_word typography / backdrop / title-mood PACK
OWNERSHIP. The composer (`compose_still_word_prompt`) + the derive_image_prompts
provenance stamp now read the VALUES from the resolved VisualStyle pack; the
per-episode genre SELECTOR (`_still_word_genre`) and the per-episode lettering
LOCK stay Python (operator lettering-consistency directive 2026-07-04).

Coverage:
  1. Raw-field deltas -- every non-default pack authors all 3 still_word fields
     (5 typography keys + 5 backdrop keys + title-mood) away from sci_fi.
  2. sci_fi byte-identity through the re-route (composer output still carries the
     extraction-fixture constants for the default lane).
  3. Composed delta through the REAL composer per genre.
  4. Provenance stamps (still_word_typography:<genre> / still_word_title_mood_style)
     + lettering_style sourced from the pack, through derive_image_prompts.
  5. Per-episode lettering LOCK preserved (determinism).
  6. Negative-vocab: no pack's own forbidden term survives into a styled card.
  7. AST guard: the composer body reads the pack attributes, NOT the module
     extraction constants (which survive only as sci_fi fixtures).

This chunk RETIRES the still_word half of
test_visual_styles_a1.TestDormantDefaults (those fields are now consumed).
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from nodes import otr_meta_brief_image_prompt as ip
from nodes import _otr_visual_styles as vs

_REPO = Path(__file__).resolve().parent.parent
_STYLES_DIR = _REPO / "nodes" / "visual_styles"

_NON_DEFAULT_IDS = ("anime", "archival_documentary", "cartoon",
                    "paper_origami", "shakespeare_stage_realism",
                    "storybook_engraving")
_GENRE_KEYS = ("noir", "sci-fi", "western", "pulp", "default")

#: brief text that DRIVES each genre through _still_word_genre (status "ok" +
#: a keyword in the radio_form haystack). "default" carries no genre keyword.
_GENRE_STYLE = {
    "noir": "a noir detective case",
    "sci-fi": "an orbital space station",
    "western": "a frontier western town",
    "pulp": "a pulp adventure serial",
    "default": "a quiet village conversation",
}


def _raw(style_id: str) -> dict:
    return json.loads(
        (_STYLES_DIR / f"{style_id}.json").read_text(encoding="utf-8"))


def _meta(style_id: str, genre: str, *, title: str = "The Signal") -> dict:
    return {
        "visual_style": style_id,
        "episode_title": title,
        "story_brief_status": "ok",
        "style": _GENRE_STYLE[genre],
        "story_brief_terms": {"setting": [_GENRE_STYLE[genre]],
                              "atmosphere": ["tense"]},
    }


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


# --------------------------------------------------------------------------- #
# 1. Raw-field deltas -- every still_word field authored away from sci_fi
# --------------------------------------------------------------------------- #
class TestStillWordFieldDeltas:
    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_typography_every_key_changed(self, style_id):
        sci, raw = _raw("sci_fi_radio"), _raw(style_id)
        for k in _GENRE_KEYS:
            assert raw["still_word_typography"][k] != \
                sci["still_word_typography"][k], (style_id, "typography", k)

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_backdrop_every_key_changed(self, style_id):
        sci, raw = _raw("sci_fi_radio"), _raw(style_id)
        for k in _GENRE_KEYS:
            assert raw["still_word_backdrop"][k] != \
                sci["still_word_backdrop"][k], (style_id, "backdrop", k)

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_title_mood_changed_and_wordless(self, style_id):
        sci, raw = _raw("sci_fi_radio"), _raw(style_id)
        assert raw["still_word_title_mood_style"] != \
            sci["still_word_title_mood_style"]
        # the music card is WORDLESS by design -- the pack must not invite text.
        assert "no lettering" in raw["still_word_title_mood_style"]


# --------------------------------------------------------------------------- #
# 2. sci_fi byte-identity through the re-route (the default lane is unchanged)
# --------------------------------------------------------------------------- #
class TestSciFiByteIdentical:
    @pytest.mark.parametrize("genre", _GENRE_KEYS)
    def test_default_word_card_carries_fixture_constants(self, genre):
        # No visual_style -> sci_fi_radio; the pack values equal the extraction
        # fixtures, so the composed card is byte-identical to pre-chunk-C.
        out = ip.compose_still_word_prompt(
            _meta("sci_fi_radio", genre), "character_video", {"text": "Go."})
        assert ip._STILL_WORD_TYPOGRAPHY[genre] in out
        assert ip._STILL_WORD_BACKDROP[genre] in out

    def test_default_music_card_carries_fixture_title_mood(self):
        out = ip.compose_still_word_prompt(
            _meta("sci_fi_radio", "default"), "music_visual", {})
        assert ip._STILL_WORD_TITLE_MOOD_STYLE in out

    def test_absent_visual_style_equals_explicit_sci_fi(self):
        no_style = {k: v for k, v in _meta("sci_fi_radio", "noir").items()
                    if k != "visual_style"}
        with_style = _meta("sci_fi_radio", "noir")
        assert ip.compose_still_word_prompt(
            no_style, "character_video", {"text": "Hi."}) == \
            ip.compose_still_word_prompt(
                with_style, "character_video", {"text": "Hi."})


# --------------------------------------------------------------------------- #
# 3. Composed delta through the REAL composer, per genre
# --------------------------------------------------------------------------- #
class TestComposedDelta:
    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    @pytest.mark.parametrize("genre", _GENRE_KEYS)
    def test_word_card_carries_pack_values_and_differs(self, style_id, genre):
        s = vs.resolve_visual_style(style_id)
        styled = ip.compose_still_word_prompt(
            _meta(style_id, genre), "character_video", {"text": "Go."})
        default = ip.compose_still_word_prompt(
            _meta("sci_fi_radio", genre), "character_video", {"text": "Go."})
        assert styled != default
        assert s.still_word_typography[genre] in styled
        assert s.still_word_backdrop[genre] in styled

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_music_card_carries_pack_title_mood(self, style_id):
        s = vs.resolve_visual_style(style_id)
        out = ip.compose_still_word_prompt(
            _meta(style_id, "default"), "music_visual", {})
        assert s.still_word_title_mood_style in out


# --------------------------------------------------------------------------- #
# 4. Provenance through derive_image_prompts (the operator-QA stamps)
# --------------------------------------------------------------------------- #
def _cast():
    return [{"char_id": "c01",
             "character_description": "a weathered inspector in an oilskin coat"}]


def _lines():
    return [
        {"line_id": "b001", "speaker_role": "character", "char_id": "c01",
         "text": "We have to go back.", "start_s": 1.0, "dur_s": 2.0},
        {"line_id": "b002", "speaker_role": "announcer", "char_id": "announcer",
         "text": "And now, our story.", "start_s": 3.0, "dur_s": 1.0},
    ]


class TestProvenanceStamps:
    def test_word_and_title_provenance_from_pack(self):
        meta = _meta("anime", "noir")
        s = vs.resolve_visual_style("anime")
        genre = ip._still_word_genre(meta)
        payload, _warn = ip.derive_image_prompts(
            _cast(), meta, llm_fn=None, lines=_lines(),
            still_word_roles={"character_video", "announcer_visual",
                              "music_visual"})
        word_objs = [o for o in payload["objects"]
                     if o.get("source") == "still_word"]
        assert word_objs
        for o in word_objs:
            assert o["visual_style"] == "anime"
            if o.get("role") == "music_visual":
                assert o["prompt_field_source"] == "still_word_title_mood_style"
            else:
                assert o["prompt_field_source"] == \
                    "still_word_typography:%s" % genre
                # lettering_style is the PACK value, not the Python fixture.
                assert o["lettering_style"] == s.still_word_typography[genre]
                assert o["backdrop_family"] == genre


# --------------------------------------------------------------------------- #
# 5. Per-episode lettering LOCK preserved (determinism; operator directive)
# --------------------------------------------------------------------------- #
class TestLetteringLock:
    def test_same_episode_locks_one_lettering_set(self):
        s = vs.resolve_visual_style("anime")
        meta = _meta("anime", "western")
        genre = ip._still_word_genre(meta)          # locked per episode
        a = ip.compose_still_word_prompt(meta, "character_video", {"text": "A."})
        b = ip.compose_still_word_prompt(meta, "character_video", {"text": "B."})
        # every card in the episode reads the SAME locked lettering + backdrop.
        assert s.still_word_typography[genre] in a
        assert s.still_word_typography[genre] in b
        assert s.still_word_backdrop[genre] in a
        assert s.still_word_backdrop[genre] in b


# --------------------------------------------------------------------------- #
# 6. Negative-vocab: no pack's own forbidden term survives a styled card
# --------------------------------------------------------------------------- #
class TestNegativeVocab:
    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_no_own_forbidden_term_in_word_or_title_cards(self, style_id):
        s = vs.resolve_visual_style(style_id)
        cards = [ip.compose_still_word_prompt(
                    _meta(style_id, g), "character_video", {"text": "Go."})
                 for g in _GENRE_KEYS]
        cards.append(ip.compose_still_word_prompt(
            _meta(style_id, "default"), "music_visual", {}))
        offenders = []
        for i, text in enumerate(cards):
            low = text.lower()
            for term in s.forbidden_terms:
                if term.lower() in low:
                    offenders.append(f"card[{i}] carries {term!r}: {text[:90]}")
        assert not offenders, "\n".join(offenders)


# --------------------------------------------------------------------------- #
# 7. AST guard: the composer reads the PACK, not the module fixtures
# --------------------------------------------------------------------------- #
def _compose_fn_node():
    src = inspect.getsource(ip)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "compose_still_word_prompt"):
            return node
    raise AssertionError("compose_still_word_prompt not found")


class TestReRouteAST:
    def test_reads_pack_attributes(self):
        attrs = {n.attr for n in ast.walk(_compose_fn_node())
                 if isinstance(n, ast.Attribute)}
        assert "still_word_typography" in attrs
        assert "still_word_backdrop" in attrs
        assert "still_word_title_mood_style" in attrs

    def test_does_not_read_extraction_constants(self):
        names = {n.id for n in ast.walk(_compose_fn_node())
                 if isinstance(n, ast.Name)}
        for banned in ("_STILL_WORD_TYPOGRAPHY", "_STILL_WORD_BACKDROP",
                       "_STILL_WORD_TITLE_MOOD_STYLE"):
            assert banned not in names, (
                f"{banned} is an extraction fixture -- production must read "
                f"the pack (VisualStyle.still_word_*), not the constant")
