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
    assert "cpu" in row["device_backends"]
    # the registry-consistency invariant (CAPABILITIES == registered engines).
    assert set(vreg.CAPABILITIES) == set(vreg.all_engine_names())


def test_still_word_engine_family_map():
    assert render_driver.ENGINE_FAMILY["still_word"] == "static_image_gen"


def test_still_word_in_video_combo_and_parses():
    parsed = {vd._engine_id_from_pick(c) for c in vd._video_model_combo()}
    assert "still_word" in parsed                  # selectable in the dropdown


def test_still_word_was_the_FIRST_to_require_a_still_and_is_no_longer_alone():
    """Sprint B (2026-07-03) made `still_word` the ONLY family that refuses a
    missing still, and this test guarded that scoping by asserting its
    `still_flat` sibling still had the always-renders floor.

    It went RED in lane 17 (2026-08-11), which is the guard working: lanes 15-17
    gave `still_motion`, `still_pan` and `still_flat` the same refusal, one lane
    at a time and each on its own evidence, so ALL FOUR now require a still.
    Sprint B's reasoning was simply generalised -- "a silent black floor would
    swallow a mint failure exactly where it matters" turned out to matter
    everywhere, not only on a word card.

    The per-lane ledger of who ruled lives in
    `tests/test_video_cheap_render.py::test_which_still_families_REFUSE_a_missing_still_is_pinned_per_lane`;
    what THIS test keeps saying is that `still_word` was first and that the base
    default is still opt-in, so a new cheap family inherits nothing.
    """
    assert cheap_families.StillWordFamily._require_still is True
    assert cheap_families.StillFlatFamily._require_still is True     # lane 17
    assert cheap_families._CheapFamilyBase._require_still is False   # opt IN


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


# --------------------------------------------------------------------------- #
# BACKSLASH / QUOTE-SHIELD integrity on the composed card (banana-route QA)
#
# The card template wraps its text in the ONLY double-quote pair the prompt is
# allowed to contain, and the banana route reads that pair as a shielded span so
# the SPOKEN LINE passes through untransformed. A line ending in an ODD
# backslash run made the closing quote read as ESCAPED -- no span was found and
# the WHOLE card prompt transformed, putting a substituted word on the one
# audience-readable surface.
# --------------------------------------------------------------------------- #

def _dquote_pairs(prompt):
    return prompt.count('"')


@pytest.mark.parametrize("line", [
    "He drew his revolver \\",              # odd run -- the leak
    "He drew his revolver \\\\",            # even run -- was already fine
    "He drew his revolver \\\\\\",          # odd, longer
    "He drew his revolver\\ and holstered it \\",   # interior AND trailing
])
def test_word_card_backslash_never_breaks_the_quote_shield(line):
    from nodes import _otr_banana_route as banana
    out = ip.compose_still_word_prompt(_META, "character_video", line)
    assert "\\" not in out
    assert _dquote_pairs(out) == 2          # the template boundary, intact
    # the spoken words inside the card are SCRIPT and stay byte-identical
    result = banana.apply(out, variety_key="")
    assert "revolver" in result.text
    assert "banana" not in result.text.lower()


@pytest.mark.parametrize("line", ["\\", "\\\\", "[static hiss] \\", "\\ \\ \\"])
def test_backslash_only_line_still_composes_and_never_kills_the_render(line):
    """These compose fine today. Scrubbing backslashes turns them BLANK, and a
    blank line is a NO-FALLBACK raise -- so a naive scrub would invent a new way
    to kill an episode over a stray character. THE LAW: an audit may never fail
    a story for language or style."""
    out = ip.compose_still_word_prompt(_META, "character_video", line)
    assert "title card" in out.lower()
    assert "\\" not in out
    assert _dquote_pairs(out) == 2


@pytest.mark.parametrize("line", ["", "   ", "[static hiss]"])
def test_genuinely_blank_lines_still_fail_loud(line):
    """The blank raise is deliberate and must survive the backslash guard."""
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt(_META, "character_video", line)


def test_music_title_backslash_is_scrubbed_and_still_shields():
    """The music card calls the fold DIRECTLY and never passes through the
    word-card cleaner, so a fix placed one level lower leaves the episode-title
    path leaking: `The Revolver of Deck Nine \\` renders as `The Banana ...`."""
    from nodes import _otr_banana_route as banana
    meta = dict(_META, episode_title="The Revolver of Deck Nine \\")
    out = ip.compose_still_word_prompt(meta, "music_visual", "")
    assert "\\" not in out
    assert banana.apply(out, variety_key="").text.count("Revolver") == 1


def test_music_title_that_is_only_a_backslash_does_not_mint_an_empty_card():
    """A backslash-only title must not mint `evoking ""`.

    Note what actually guarantees this: the fold's degenerate-input rescue, not
    the fold-then-validate ORDER. Because the rescue makes the folded value
    truthy for every truthy raw title, validating before or after the fold gives
    the same answer today -- the reordering is defence in depth for the day
    someone removes the rescue, and this test cannot isolate it. What the test
    does pin is that the card is composed FROM THE FOLDED VALUE."""
    meta = dict(_META, episode_title="\\")
    out = ip.compose_still_word_prompt(meta, "music_visual", "")
    assert 'evoking ""' not in out
    assert _dquote_pairs(out) == 2
    # composed from the folded value, not the raw title
    assert 'evoking "%s"' % ip._fold_inner_dquotes("\\") in out


def test_era_tail_quotes_cannot_shield_a_weapon_from_the_route():
    """The era tail splices LLM-authored atmosphere/palette text into the same
    card. An authored quote PAIR there wrongly shielded whatever it wrapped, so
    `a "cold revolver" on the desk` kept its revolver -- the route silently not
    working. Folding era restores this composer's stated invariant: the
    template's outer pair is the ONLY double-quote pair in the card."""
    from nodes import _otr_banana_route as banana
    meta = dict(_META, atmosphere_line='a "cold revolver" on the desk')
    out = ip.compose_still_word_prompt(meta, "character_video", "We go back.")
    assert _dquote_pairs(out) == 2          # only the card boundary
    result = banana.apply(out, variety_key="")
    # the spoken card text is still shielded byte-identically
    assert "We go back." in result.text


def test_backslash_free_text_keeps_its_whitespace_through_the_fold():
    """The whitespace collapse is SCOPED to strings that actually carried a
    backslash, so a card that never had one is not re-seeded.

    This has to be asserted on the MUSIC and ERA paths, which call the fold
    DIRECTLY. Asserting it on the word card proves nothing: that path runs its
    own unconditional collapse afterwards, so an unscoped fold would produce
    byte-identical word cards and the regression would slip through."""
    for raw in ("The  Signal   From Deck Nine", "  padded title  ",
                "plain title"):
        assert ip._fold_inner_dquotes(raw) == raw, raw
    # ...and through the real music card, whose title is folded directly.
    meta = dict(_META, episode_title="The  Signal   From Deck Nine")
    out = ip.compose_still_word_prompt(meta, "music_visual", "")
    assert 'evoking "The  Signal   From Deck Nine"' in out


def test_backslash_bearing_text_IS_collapsed_by_the_fold():
    """The other half: where a backslash WAS present, the residue it leaves
    behind is collapsed rather than shipped as stray whitespace."""
    assert ip._fold_inner_dquotes("Deck Nine \\") == "Deck Nine"
    assert ip._fold_inner_dquotes("a\\\\b") == "a b"


def test_word_cards_are_deterministic():
    for line in ("We have to go back.", "A quiet, padded  line",
                 "He said 'run' -- and then... silence"):
        first = ip.compose_still_word_prompt(_META, "character_video", line)
        assert first == ip.compose_still_word_prompt(
            _META, "character_video", line)


def test_compose_is_deterministic():
    a = ip.compose_still_word_prompt(_META, "character_video", "Once more.")
    b = ip.compose_still_word_prompt(_META, "character_video", "Once more.")
    assert a == b


def test_compose_is_model_agnostic():
    # the composer takes NO engine argument -> the prompt is identical regardless
    # of which image engine mints it (operator priority: model-agnostic).
    # Chunk C adds an OPTIONAL resolved-style kwarg (resolve-once threading);
    # ``style`` is the visual STYLE, not the image ENGINE -- model-agnosticism
    # is preserved (no engine arg anywhere).
    params = list(inspect.signature(ip.compose_still_word_prompt).parameters)
    assert params[:3] == ["meta", "role", "beat_line"]
    assert params == ["meta", "role", "beat_line", "style"]
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


def test_era_tail_preserves_authored_world_terms_in_word_mode():
    meta = {
        "episode_title": "X",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "lighting": ["flickering neon signs"],
            "atmosphere": ["poster label", "smoking lounge"],
        },
    }

    out = ip.compose_still_word_prompt(
        meta, "character_video", {"text": "Run."},
    )

    assert "flickering neon signs" in out
    assert "poster label" in out
    assert "smoking lounge" in out


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


def test_force_map_into_still_word_included(monkeypatch):
    # r3 seam: a run that FORCES a role to still_word must mint word cards.
    policy = json.dumps({"video_models": {
        "character_video_model": {"engine_id": "ltx_video"}}})
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    assert ip._still_word_roles_from_policy(policy) == set()      # unforced
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "character_video=still_word")
    assert ip._still_word_roles_from_policy(policy) == {"character_video"}


def test_force_map_away_from_still_word_excluded(monkeypatch):
    policy = json.dumps({"video_models": {
        "character_video_model": {"engine_id": "still_word"}}})
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    assert ip._still_word_roles_from_policy(policy) == {"character_video"}
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "character_video=ltx_video")
    assert ip._still_word_roles_from_policy(policy) == set()


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
