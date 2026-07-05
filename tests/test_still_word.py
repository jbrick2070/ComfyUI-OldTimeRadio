"""still_word (Sprint B, 2026-07-03) -- the model-agnostic word/title still.

A still_flat SIBLING VIDEO engine whose base still is minted from a WORD/TITLE-
driven prompt (character/announcer beats -> the spoken line as a readable word
card; music beats -> an abstract picture of the episode title, no words). The
IMAGE model that mints it is chosen INDEPENDENTLY, so still_word is model-
agnostic. NO FALLBACKS: a missing base still fails LOUD (never the dark floor);
a blank spoken line / blank title fails LOUD in the composer.

CPU-only: no ffmpeg render is exercised here (the fail-LOUD path raises before
any encode). Covers registration in all sites, the pure composer (word + title
modes, edge cases, determinism, model-agnosticism), the policy reader, the
render_clip no-floor guard, and the derive_image_prompts word/title branch.
"""
from __future__ import annotations

import inspect
import json

import pytest

from nodes import otr_meta_brief_image_prompt as ip
from nodes import otr_video_director as vd
from nodes._otr_video_engines import cheap_families, registry as vreg
from nodes._otr_video_engines import render_driver


# --------------------------------------------------------------------------- #
# Registration -- all sites (else still_word never receives a still)
# --------------------------------------------------------------------------- #
def test_still_word_registered_engine():
    assert "still_word" in vreg.all_engine_names()
    eng = vreg.get_engine("still_word")
    assert eng.family == "static_image_gen"
    assert eng.default_roles == ()                 # selectable, never a default
    assert eng.accepts_still is True
    assert eng._still_motion is False              # flat hold (like still_flat)
    assert eng._require_still is True              # NO dark floor


def test_still_word_capabilities_row():
    assert "still_word" in vreg.CAPABILITIES
    row = vreg.CAPABILITIES["still_word"]
    assert row["cpu_ok"] is True
    # the registry-consistency invariant (CAPABILITIES == registered engines).
    assert set(vreg.CAPABILITIES) == set(vreg.all_engine_names())


def test_still_word_engine_family_map():
    assert render_driver.ENGINE_FAMILY["still_word"] == "static_image_gen"


def test_still_word_in_video_combo_and_parses():
    parsed = {vd._engine_id_from_pick(c) for c in vd._video_model_combo()}
    assert "still_word" in parsed                  # selectable in the dropdown


def test_still_flat_sibling_unchanged_no_require_still():
    # the sibling keeps its always-renders floor (byte-identical behavior).
    assert cheap_families.StillFlatFamily._require_still is False
    assert cheap_families.StillWordFamily._require_still is True


# --------------------------------------------------------------------------- #
# compose_still_word_prompt -- pure, deterministic, model-agnostic, fail-LOUD
# --------------------------------------------------------------------------- #
_META = {"episode_title": "The Signal From Deck Nine",
         "story_brief_terms": {"setting": ["a fog-bound harbor town"],
                               "lighting": ["moody"]}}


def test_word_mode_renders_the_spoken_line():
    out = ip.compose_still_word_prompt(_META, "character_video",
                                       "We have to go back.")
    assert "We have to go back." in out
    assert "title card" in out.lower()
    # WORD mode WANTS text -> the no-text clause must NOT be appended.
    assert "no on-screen text" not in out


def test_word_mode_scrubs_stage_direction_and_speaker_label():
    out = ip.compose_still_word_prompt(
        _META, "announcer_visual", "NARRATOR: [whispering] The signal is gone")
    assert "The signal is gone" in out
    assert "[whispering]" not in out
    assert "NARRATOR:" not in out


def test_word_mode_folds_inner_quotes_keeps_ellipsis_emdash():
    # §7 quote-fold: inner double-quotes fold to single so the ONLY double-quote
    # pair is the template's outer "%s" boundary (closes nested-quote ambiguity +
    # the prompt-injection). Ellipsis / em-dash are kept.
    line = 'He said "run" -- and then... silence'
    out = ip.compose_still_word_prompt(_META, "character_video", line)
    assert "'run'" in out
    assert '"run"' not in out
    # exactly one double-quote PAIR remains (the template boundary)
    assert out.count('"') == 2
    assert "--" in out
    assert "..." in out


def test_word_mode_quote_fold_blocks_injection():
    # a line that tries to CLOSE the span early and inject a sibling clause is
    # neutralized: after the fold there is exactly one double-quote pair.
    line = 'run", ominous horror style, blood'
    out = ip.compose_still_word_prompt(_META, "character_video", line)
    assert out.count('"') == 2


def test_word_mode_blank_line_fails_loud():
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt(_META, "character_video", "")
    # a line that is ONLY a stage direction scrubs to empty -> also LOUD.
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt(_META, "character_video", "[static hiss]")


def test_music_mode_abstract_title_no_words():
    out = ip.compose_still_word_prompt(_META, "music_visual", "")
    assert "The Signal From Deck Nine" in out
    assert "abstract" in out.lower()
    # TITLE mode is wordless -> the no-text clause IS appended.
    assert out.endswith("no on-screen text")


def test_music_mode_blank_title_fails_loud():
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt({"story_brief_terms": {}}, "music_visual", "")


def test_compose_is_deterministic():
    a = ip.compose_still_word_prompt(_META, "character_video", "Once more.")
    b = ip.compose_still_word_prompt(_META, "character_video", "Once more.")
    assert a == b


def test_compose_is_model_agnostic():
    # the composer takes NO engine argument -> the prompt is identical regardless
    # of which image engine mints it (operator priority: model-agnostic).
    params = list(inspect.signature(ip.compose_still_word_prompt).parameters)
    assert params == ["meta", "role", "beat_line"]
    assert "engine" not in params


# --------------------------------------------------------------------------- #
# still_word PROMPT v2 (2026-07-04): locked lettering + cool backdrop + beat mood
# allowlist + line-length reduction + mandatory era-tail scrub + dict|str sig.
# --------------------------------------------------------------------------- #
import re

_NOIR_OK = {"story_brief_status": "ok", "style": "1940s noir detective",
            "episode_title": "The Pier", "story_brief_terms":
                {"setting": ["a rain-slick city office"]}}


def _quoted_words(prompt):
    m = re.search(r'the words "([^"]*)"', prompt)
    return m.group(1) if m else ""


def test_typography_and_backdrop_locked_per_episode():
    # Two character word cards in ONE episode share the SAME lettering + backdrop
    # family; only the beat MOOD differs (the operator's consistency ask).
    a = ip.compose_still_word_prompt(
        _NOIR_OK, "character_video",
        {"text": "He never made it.", "traits": "tense, worried"})
    b = ip.compose_still_word_prompt(
        _NOIR_OK, "character_video",
        {"text": "She waited all night.", "traits": "somber"})
    lettering = ip._STILL_WORD_TYPOGRAPHY["noir"]
    backdrop = ip._STILL_WORD_BACKDROP["noir"]
    assert lettering in a and lettering in b
    assert backdrop in a and backdrop in b
    assert "tense mood" in a and "somber mood" in b
    # strip the quoted words + the mood token -> the remaining style is identical
    def _canon(p, mood):
        return p.replace('"%s"' % _quoted_words(p), "").replace(", %s mood" % mood, "")
    assert _canon(a, "tense") == _canon(b, "somber")


def test_word_mode_has_positive_text_guard_never_no_text_clause():
    out = ip.compose_still_word_prompt(_NOIR_OK, "character_video",
                                       {"text": "Come home."})
    assert out.endswith(ip._STILL_WORD_TEXT_GUARD)
    assert "no captions" in out
    assert "no on-screen text" not in out          # NO_TEXT_CLAUSE never in word mode


def test_word_mode_legibility_guard_present_and_ordered():
    out = ip.compose_still_word_prompt(_NOIR_OK, "character_video",
                                       {"text": "Come home."})
    assert ip._STILL_WORD_LEGIBILITY_GUARD in out
    # order: quoted words -> legibility -> lettering -> backdrop -> ... -> text guard
    assert out.index("Come home.") < out.index(ip._STILL_WORD_LEGIBILITY_GUARD)
    assert out.index(ip._STILL_WORD_LEGIBILITY_GUARD) < out.index(
        ip._STILL_WORD_TYPOGRAPHY["noir"])
    assert out.index(ip._STILL_WORD_TYPOGRAPHY["noir"]) < out.index(
        ip._STILL_WORD_BACKDROP["noir"])


def test_mood_allowlist_free_form_traits():
    f = ip._still_word_mood_from_line
    assert f({"traits": "urgent, gruff"}) == "urgent"    # compound: danger-forward
    assert f({"traits": "warm baritone"}) == ""          # voice-only -> no token
    assert f({"traits": "frantic"}) == "urgent"          # keyword maps to urgent
    assert f({"traits": "tense"}) == "tense"
    assert f({"traits": "grief-stricken and defeated"}) == "somber"
    assert f({"traits": "hopeful"}) == "hopeful"
    assert f({"traits": "calm, measured"}) == "calm"
    # the production DEFAULT filler + null/voice cases -> NO token (not calm)
    assert f({"traits": "neutral"}) == ""
    assert f({"traits": None}) == ""
    assert f({"traits": "cold booming baritone"}) == ""  # timbre words never map
    assert f("a bare string, not a dict") == ""
    assert f({}) == ""


def test_announcer_card_has_no_beat_mood():
    # ANNOUNCER cards carry the episode backdrop but NO beat mood adjective.
    out = ip.compose_still_word_prompt(
        _NOIR_OK, "announcer_visual",
        {"text": "And now, our story.", "traits": "urgent"})
    assert "mood" not in out
    assert ip._STILL_WORD_BACKDROP["noir"] in out


def test_line_length_word_boundary_reduction_not_abort():
    twelve = " ".join("aa bb cc dd ee ff gg hh ii jj kk ll".split())   # 12 words
    thirteen = twelve + " mm"                                          # 13 words
    out12 = ip.compose_still_word_prompt(_NOIR_OK, "character_video",
                                         {"text": twelve})
    out13 = ip.compose_still_word_prompt(_NOIR_OK, "character_video",
                                         {"text": thirteen})           # must NOT raise
    assert _quoted_words(out12) == twelve                             # <=12 unchanged
    reduced = _quoted_words(out13)
    assert len(reduced.split()) <= ip.STILL_WORD_MAX_WORDS
    assert reduced == twelve                                          # first 12 kept


def test_line_length_char_boundary_reduction():
    w = "abcdefghij"
    sixty = (" ".join([w] * 5) + " abcde")             # 5*10 + 4 spaces + 6 = 60
    assert len(sixty) == 60
    sixtyone = sixty + "f"                              # 61 chars, same word count
    out60 = ip.compose_still_word_prompt(_NOIR_OK, "character_video",
                                         {"text": sixty})
    out61 = ip.compose_still_word_prompt(_NOIR_OK, "character_video",
                                         {"text": sixtyone})
    assert _quoted_words(out60) == sixty               # <=60 chars unchanged
    reduced = _quoted_words(out61)
    assert len(reduced) <= ip.STILL_WORD_MAX_CHARS and reduced != sixtyone


def test_blank_line_still_fails_loud_after_reduction():
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt(_NOIR_OK, "character_video",
                                     {"text": "[static hiss]"})


def test_absent_and_failed_brief_use_neutral_defaults():
    # absent brief (no status) -> default lettering + backdrop
    out = ip.compose_still_word_prompt(
        {"story_brief_terms": {"setting": ["a rain-slick city office"]}},
        "character_video", {"text": "Hello."})
    assert ip._STILL_WORD_TYPOGRAPHY["default"] in out
    assert ip._STILL_WORD_BACKDROP["default"] in out
    # failed brief -> also default (never the noir map even with a noir style)
    failed = {"story_brief_status": "failed", "style": "1940s noir detective"}
    out2 = ip.compose_still_word_prompt(failed, "character_video", {"text": "Hi."})
    assert ip._STILL_WORD_TYPOGRAPHY["default"] in out2


def test_genre_typography_and_backdrop_maps():
    cases = {
        "noir": "1940s noir detective",
        "sci-fi": "a science fiction orbital docking story",
        "western": "a dusty old west frontier tale",
        "pulp": "a lurid pulp adventure with a monster",
    }
    for genre, style in cases.items():
        meta = {"story_brief_status": "ok", "style": style,
                "story_brief_terms": {"setting": [style]}}
        assert ip._still_word_genre(meta) == genre
        out = ip.compose_still_word_prompt(meta, "character_video", {"text": "Go."})
        assert ip._STILL_WORD_TYPOGRAPHY[genre] in out
        assert ip._STILL_WORD_BACKDROP[genre] in out
        # every lettering phrase locks the CASE
        assert ip._STILL_WORD_TYPOGRAPHY[genre].endswith("capitals") or \
            "capitals" in ip._STILL_WORD_TYPOGRAPHY[genre]


def test_era_tail_text_summoners_scrubbed_word_mode():
    # a "neon signs" atmosphere in the era tail is a text-summoner -> MANDATORY
    # scrub in word mode (never a QA-watch).
    meta = {"episode_title": "X",
            "story_brief_terms": {"lighting": ["flickering neon signs"]}}
    out = ip.compose_still_word_prompt(meta, "character_video", {"text": "Run."})
    assert "neon" not in out and "signs" not in out


# --------------------------------------------------------------------------- #
# _still_word_roles_from_policy -- reads video_models via role_slots
# --------------------------------------------------------------------------- #
def test_roles_from_policy_resolves_video_models():
    policy = json.dumps({"video_models": {
        "character_video_model": {"engine_id": "still_word"},
        "announcer_video_model": {"engine_id": "ltx_video"},
        "music_video_model": {"engine_id": "still_word"},
    }})
    roles = ip._still_word_roles_from_policy(policy)
    assert roles == {"character_video", "music_visual"}


def test_roles_from_policy_empty_when_absent():
    assert ip._still_word_roles_from_policy("{}") == set()
    assert ip._still_word_roles_from_policy(
        json.dumps({"video_models": {"character_video_model":
                                     {"engine_id": "ltx_video"}}})) == set()


# --------------------------------------------------------------------------- #
# render_clip -- NO dark floor (fail LOUD when the base still is missing)
# --------------------------------------------------------------------------- #
def test_render_clip_fails_loud_without_a_still():
    req = {"asset_refs": {}, "canvas": {"w": 832, "h": 480, "fps": 25},
           "timing": {"target_frame_count": 10}, "shot_id": "s_word"}
    with pytest.raises(RuntimeError, match="NO FALLBACKS"):
        cheap_families.StillWordFamily().render_clip(req)


# --------------------------------------------------------------------------- #
# derive_image_prompts -- the word/title branch mints still_word-sourced stills
# --------------------------------------------------------------------------- #
def _cast():
    return [{"char_id": "c01",
             "character_description": "a weathered dock inspector in an oilskin coat"}]


def _lines():
    return [
        {"line_id": "b001", "speaker_role": "character", "char_id": "c01",
         "text": "We have to go back.", "start_s": 1.0, "dur_s": 2.0},
        {"line_id": "b002", "speaker_role": "announcer", "char_id": "announcer",
         "text": "And now, our story.", "start_s": 3.0, "dur_s": 1.0},
    ]


def test_derive_image_prompts_word_and_title_stills():
    payload, _warn = ip.derive_image_prompts(
        _cast(), _META, llm_fn=None, lines=_lines(),
        still_word_roles={"character_video", "announcer_visual", "music_visual"})
    objs = payload["objects"]
    word_objs = [o for o in objs if o.get("source") == "still_word"]
    assert word_objs, "expected still_word-sourced scene stills"
    # every still_word beat is ONE scene-still object (picked up by _still_index).
    for o in word_objs:
        assert o["object_id"] == "still_%s" % o["beat_id"]
        assert o["prompt"] and o["prompt_hash"]
    # the character beat's spoken line drives its still.
    char = next((o for o in word_objs if o.get("char_id") == "c01"), None)
    assert char is not None
    assert "We have to go back." in char["prompt"]
    # a music beat (title mode) carries the episode title, no words.
    music = next((o for o in word_objs if o.get("role") == "music_visual"), None)
    if music is not None:                            # music open beat present
        assert "The Signal From Deck Nine" in music["prompt"]
        assert music["prompt"].endswith("no on-screen text")


def test_derive_image_prompts_no_still_word_roles_is_legacy():
    # with no still_word roles the branch is inert (no still_word source objects).
    payload, _warn = ip.derive_image_prompts(
        _cast(), _META, llm_fn=None, lines=_lines())
    assert not [o for o in payload["objects"] if o.get("source") == "still_word"]
