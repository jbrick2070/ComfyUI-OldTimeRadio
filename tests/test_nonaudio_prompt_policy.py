"""THE NON-AUDIO PROMPT POLICY (2026-08-26).

A character lane that cannot SPEAK must not be handed the spoken line.

The live defect: the H3 Caretaker episode
(`signal_lost_the_caretakers_clause_20260826_155835`, every beat on
`minimax_h3_video`) received both `_subject_anchor`'s "face visible, speaking to
camera" and the beat's literal dialogue in its visual prompt. That lane decodes
no audio at all, so the only thing it could do with a line of dialogue was draw
a mouth in motion -- a character mouthing words the render can never deliver.

What these tests pin:
  * AUDIO-IN lanes and `still_word` keep their dialogue byte-for-byte;
  * every other character lane gets nonverbal acting, and neither the literal
    line nor any speaking/lip/mouth instruction survives into the prompt;
  * an engine the policy cannot CLASSIFY fails loudly instead of guessing --
    which is not an admission gate: every registered engine passes, and a
    public menu id resolves rather than being refused;
  * the LTX 2.5 foley/mime lanes get their audio requirement appended to the
    ONE positive string that conditions both picture and sound.

CPU-only. The M4 writer is injected as a fake callable throughout.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import otr_shot_lock as sl  # noqa: E402
from nodes._otr_video_engines import eng_ltx25 as ltx25  # noqa: E402
from nodes._otr_video_engines import render_driver as rd  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

_LINE = "we have to leave now"


def _ledger():
    return {
        "cast": [{"char_id": "c1", "name": "BABA",
                  "portrait_prompt": "a tall weathered spacer with a scar"}],
        "lines": [],
        "meta": {"story_brief_terms":
                 {"setting": ["a derelict orbital station"]}},
    }


def _beats(text=_LINE):
    return [{"beat_id": "b1", "role": "character_video", "char_id": "c1",
             "text": text, "samples": None, "sample_rate": None,
             "dur_s": 1.0}]


def _policy(character_engine, *, effective=None):
    """A production-shaped policy: the PICKED slot map, optionally with the
    frozen route OTR_VideoDirector stamps over it."""
    policy = {
        "policy_version": 2,
        "video_models": {
            "announcer_video_model": {"engine_id": "ltx_video"},
            "music_video_model": {"engine_id": "ltx_video"},
            "character_video_model": {"engine_id": character_engine},
        },
    }
    if effective is not None:
        policy["effective_video_models"] = {"character_video": effective}
    return policy


def _directive_llm(expression="grim resolve", motion="steps forward",
                   camera="slow push-in", **extra):
    row = {"beat_id": "b1", "expression": expression, "motion": motion,
           "camera": camera}
    row.update(extra)
    return lambda _prompt: json.dumps([row])


def _capturing_llm(sink, **kwargs):
    inner = _directive_llm(**kwargs)

    def _call(prompt):
        sink.append(prompt)
        return inner(prompt)
    return _call


# --------------------------------------------------------------------------- #
# the lanes that KEEP the dialogue
# --------------------------------------------------------------------------- #

def test_audio_in_lane_directive_bytes_are_unchanged():
    """HuMo is audio-driven: the line is what it lip-syncs to, so the whole
    spoken composition -- subject anchor, setting, LINE, directives -- stands
    exactly as it did before this policy existed."""
    led = _ledger()
    without = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm())[0]["b1"]
    with_policy = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("humo"))[0]["b1"]
    assert with_policy == without
    assert _LINE in with_policy["text_prompt"]
    assert "speaking to camera" in with_policy["text_prompt"]
    # ...and the CONTRAST, so this test fails if the policy is ever ripped out
    # and every lane silently goes back to receiving the line.
    silent = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert _LINE not in silent["text_prompt"]
    assert silent["text_prompt"] != with_policy["text_prompt"]


# --------------------------------------------------------------------------- #
# the style cue must never precede a lane's PINNED opener
# --------------------------------------------------------------------------- #

class _StyledPack:
    """A minimal stand-in for a non-default `VisualStyle` -- only the three
    attributes `compact_style_cue` reads."""
    style_id = "anime"
    positive_tail = "anime style, cel shaded, clean linework"
    portrait_look_talking = ""


class _DefaultPack:
    style_id = "sci_fi_radio"
    positive_tail = "whatever"
    portrait_look_talking = ""


def test_a_style_cue_never_precedes_the_H3_required_opener():
    """CONFIRMED LIVE (r3, 2026-08-28): the anime pack turned H3's required
    opener into "anime style. For the target video, ..." -- exactly the
    failure `eng_minimax_h3.py` already claimed was impossible. The claim was
    true only on the default pack, whose cue is empty."""
    from nodes._otr_video_engines.eng_minimax_h3 import H3_REFERENCE_OPENER
    composed = (H3_REFERENCE_OPENER + " the announcer leans toward the "
               "microphone, the movement carried at a clear steady tempo "
               "to a visible endpoint")
    out = rd._style_cue_after_pinned_opener(_StyledPack, composed)
    assert out.startswith(H3_REFERENCE_OPENER)
    assert "anime style" in out


def test_the_style_cue_still_reaches_an_H3_prompt_after_the_opener():
    """The fix SEATS the cue rather than dropping it -- an exemption would
    silently lose the pack's look on both H3 lanes."""
    from nodes._otr_video_engines.eng_minimax_h3 import H3_REFERENCE_OPENER
    composed = H3_REFERENCE_OPENER + " hands steady on the console"
    out = rd._style_cue_after_pinned_opener(_StyledPack, composed)
    assert out != composed
    assert "anime style" in out


def test_the_default_pack_is_byte_identical_on_an_H3_prompt():
    """The default pack's cue is empty, which is why this defect never
    surfaced in production -- and why it must stay a true no-op."""
    from nodes._otr_video_engines.eng_minimax_h3 import H3_REFERENCE_OPENER
    composed = H3_REFERENCE_OPENER + " hands steady on the console"
    assert rd._style_cue_after_pinned_opener(_DefaultPack, composed) == composed


def test_a_non_H3_prompt_still_gets_the_ordinary_prefix():
    """No pinned opener present -> ordinary behaviour, unaffected by the fix."""
    plain = "hands on the console, dust drifting"
    out = rd._style_cue_after_pinned_opener(_StyledPack, plain)
    assert out.startswith("anime style")


def test_still_word_keeps_the_dialogue_because_the_words_are_the_picture():
    """`still_word` is `static_image_gen`, not an audio family -- it is the one
    NON-audio lane that keeps the line on purpose, because it renders the words
    into the still as a word card."""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("still_word"))[0]["b1"]
    assert _LINE in c["text_prompt"]


# --------------------------------------------------------------------------- #
# the lanes that must NOT
# --------------------------------------------------------------------------- #

_SPEECH_ANCHORS = ("speaking to camera", "lip", "lip-sync", "mouth",
                   "talking", "subtitle", "caption")


def test_h3_gets_acting_and_never_the_line_or_a_speaking_anchor():
    """THE REGRESSION. `minimax_h3_video` is `image_to_video` and decodes no
    audio, so it receives the PERFORMANCE and none of the words."""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    prompt = c["text_prompt"]
    assert _LINE not in prompt
    for anchor in _SPEECH_ANCHORS:
        assert anchor not in prompt.lower(), (anchor, prompt)
    # the acting IS there
    assert "grim resolve" in prompt
    assert "steps forward" in prompt
    assert "slow push-in" in prompt
    # ...and so is the subject and the world
    assert "weathered spacer" in prompt
    assert "derelict orbital station" in prompt


def test_m4_is_still_told_the_line_privately():
    """The writer needs to know what the moment IS to act it. The line goes
    INTO the derivation prompt and does not come back out of it."""
    led = _ledger()
    seen = []
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_capturing_llm(seen),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert seen, "the writer was never called"
    assert _LINE in seen[0], "M4 was not told the line"
    assert "do_not_quote" in seen[0]
    assert _LINE not in c["text_prompt"]


def test_an_authored_text_prompt_is_ignored_on_a_silent_lane():
    """An adapter MAY author a whole rich prompt on the spoken path. Here it is
    dropped unread -- it is the most likely place for the line to smuggle
    itself back in -- and it does not by itself count as an LLM hit."""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led,
        llm_fn=_directive_llm(expression="", motion="", camera="",
                              text_prompt="BABA says '%s' to camera" % _LINE),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert _LINE not in c["text_prompt"]
    assert "says" not in c["text_prompt"]
    assert c["source"] == "template_after_llm_miss"


def test_the_exact_nonverbal_fallbacks():
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led,
        llm_fn=_directive_llm(expression="", motion="", camera=""),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert "a vivid, readable reaction" in c["text_prompt"]
    assert "decisive full-body movement" in c["text_prompt"]
    assert "slow push-in" in c["text_prompt"]


def test_the_composition_order_is_exact():
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    head = c["text_prompt"].split(", lantern")[0]  # before any era tail
    assert head.startswith("a tall weathered spacer with a scar, "
                           "a derelict orbital station, grim resolve, "
                           "steps forward, slow push-in"), head


# --------------------------------------------------------------------------- #
# the literal-line filter
# --------------------------------------------------------------------------- #

def test_a_field_that_quotes_the_whole_line_is_dropped():
    led = _ledger()
    creative, warns = sl.derive_creative_directives(
        _beats(), led["meta"], led,
        llm_fn=_directive_llm(expression="says we have to leave now"),
        video_policy=_policy("minimax_h3_video"))
    c = creative["b1"]
    assert _LINE not in c["text_prompt"]
    assert "a vivid, readable reaction" in c["text_prompt"]   # fell back
    # The message widened 2026-08-29 when the beat intent and the logline
    # joined the line under the same backstop.
    assert any("private context" in w and "back into" in w for w in warns)
    # the OTHER fields survived -- one bad field is not a collapsed beat
    assert "steps forward" in c["text_prompt"]


_LONG_LINE = (
    "You have to listen to me because the caretaker never left this building "
    "and every night since the war he has walked the third floor corridor "
    "counting the doors one by one and marking them in a ledger nobody has "
    "ever been allowed to read, and I have seen that ledger with my own eyes.")


def test_a_quote_of_the_shown_context_is_caught_on_a_long_line():
    """THE FILTER MUST COMPARE WHAT THE MODEL WAS ACTUALLY SHOWN.

    M4 receives a capped slice of the line. When the filter tokenised the FULL
    line instead, a model that quoted back exactly the slice it had been given
    produced a sequence that was NOT the complete line -- so the filter never
    matched and the dialogue sailed into a silent lane's prompt with no
    warning. Measured on the real corpus: 295 of 7096 ledger lines are over the
    cap, across 111 episodes, so this is a live path and not a curiosity.
    """
    assert len(_LONG_LINE) > sl._M4_LINE_CONTEXT_CHARS
    shown = sl._line_context(_LONG_LINE)
    led = _ledger()
    creative, warns = sl.derive_creative_directives(
        _beats(_LONG_LINE), led["meta"], led,
        llm_fn=_directive_llm(expression=shown),
        video_policy=_policy("minimax_h3_video"))
    prompt = creative["b1"]["text_prompt"]
    assert shown[:80] not in prompt
    assert "a vivid, readable reaction" in prompt          # fell back
    assert any("private context" in w and "back into" in w for w in warns)


def test_the_model_is_shown_exactly_what_the_filter_compares():
    """One value, two consumers. If these ever diverge the filter is hunting a
    sequence the model was never in a position to emit."""
    led = _ledger()
    seen = []
    sl.derive_creative_directives(
        _beats(_LONG_LINE), led["meta"], led, llm_fn=_capturing_llm(seen),
        video_policy=_policy("minimax_h3_video"))
    shown = sl._line_context(_LONG_LINE)
    assert shown in seen[0]
    assert _LONG_LINE not in seen[0]        # the raw line is never sent whole


def test_the_line_context_cuts_on_a_word_boundary():
    ctx = sl._line_context(_LONG_LINE)
    assert len(ctx) <= sl._M4_LINE_CONTEXT_CHARS
    assert _LONG_LINE.startswith(ctx)
    assert not ctx.endswith(" ")
    # a whole final token, never a half-word the filter could not match
    assert _LONG_LINE[len(ctx)] in (" ", "") or ctx == _LONG_LINE


def test_a_short_line_is_passed_through_untouched():
    assert sl._line_context(_LINE) == _LINE
    assert sl._line_context("") == ""
    assert sl._line_context(None) == ""


def test_a_non_latin_line_is_still_filtered():
    """The ASCII-only tokenizer produced NO tokens for a wholly non-Latin
    line, so the filter answered False for an exact quotation -- silently inert
    on exactly the material the adaptation lanes carry in the author's own
    language."""
    cyrillic = "Мы должны уйти"
    assert sl._word_tokens(cyrillic), "non-Latin line produced no tokens"
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(cyrillic), led["meta"], led,
        llm_fn=_directive_llm(expression=cyrillic),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert cyrillic not in c["text_prompt"]


def test_yes_does_not_match_inside_eyes():
    """The filter compares whole-word TOKENS. A substring test would delete a
    perfectly good expression because the line happened to be 'Yes.'"""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats("Yes."), led["meta"], led,
        llm_fn=_directive_llm(expression="narrowed eyes"),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert "narrowed eyes" in c["text_prompt"]


def test_a_partial_overlap_is_not_a_quotation():
    """Sharing words with the line is not quoting it -- only the COMPLETE
    contiguous run counts."""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led,
        llm_fn=_directive_llm(expression="we have to"),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert "we have to" in c["text_prompt"]


def test_an_empty_line_never_matches():
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(""), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert "grim resolve" in c["text_prompt"]


@pytest.mark.parametrize("field", ["expression", "motion", "camera"])
def test_every_field_is_filtered(field):
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led,
        llm_fn=_directive_llm(**{field: "BABA: %s" % _LINE}),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert _LINE not in c["text_prompt"]


# --------------------------------------------------------------------------- #
# source provenance
# --------------------------------------------------------------------------- #

def test_source_is_llm_when_one_field_survives():
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led,
        llm_fn=_directive_llm(expression="", motion="", camera="held wide"),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert c["source"] == "llm"


def test_source_is_template_when_no_writer_is_configured():
    """`llm_fn is None` and no writer resolvable: the fallbacks ARE the local
    lane, not a miss."""
    led = _ledger()
    creative, _ = sl.derive_creative_directives(
        _beats(), {}, led, video_policy=_policy("minimax_h3_video"))
    assert creative["b1"]["source"] == "template"


def test_prompt_hash_matches_the_finished_prompt():
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert c["prompt_hash"] == sl._content_hash(c["text_prompt"])


def test_request_hash_semantics_are_untouched():
    """`request_hash` excludes prompt bytes by design -- it keys the BEAT, not
    the wording. Changing the prompt policy must not have moved it."""
    led = _ledger()
    spoken = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("humo"))[0]["b1"]
    silent = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert spoken["request_hash"] == silent["request_hash"]
    assert spoken["prompt_hash"] != silent["prompt_hash"]


# --------------------------------------------------------------------------- #
# classification failures -- loud, never a guess
# --------------------------------------------------------------------------- #

def test_a_blank_character_engine_fails_loudly():
    led = _ledger()
    with pytest.raises(ValueError, match="OTR_ShotLock"):
        sl.derive_creative_directives(
            _beats(), led["meta"], led, llm_fn=_directive_llm(),
            video_policy=_policy(""))


def test_an_unknown_character_engine_fails_loudly():
    """An unregistered id must not classify itself. The family helpers answer
    'abstract' for an unknown engine, 'abstract' is not an audio family, and
    the lane would silently lose a line it may have needed."""
    led = _ledger()
    with pytest.raises(ValueError, match="OTR_ShotLock"):
        sl.derive_creative_directives(
            _beats(), led["meta"], led, llm_fn=_directive_llm(),
            video_policy=_policy("no_such_engine_v9"))


def test_a_public_menu_id_resolves_instead_of_being_refused():
    """NOT AN ADMISSION GATE (operator, 2026-08-26: models are not hidden
    behind gates). A hand-written policy may carry the menu name; it resolves
    for CLASSIFICATION and renders normally."""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("ltx25_high_foley_plus"))[0]["b1"]
    assert _LINE not in c["text_prompt"]          # foley is not an audio-in lane
    assert "grim resolve" in c["text_prompt"]


def test_the_frozen_route_outranks_the_picked_slot():
    """`effective_video_models` is what OTR_VideoDirector stamped after every
    redirect, so it is what the prompt policy must classify."""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("humo", effective="minimax_h3_video"))[0]["b1"]
    assert _LINE not in c["text_prompt"]


def test_a_policy_free_legacy_call_is_unchanged():
    """No policy at all means nothing to classify FROM. Those callers keep the
    historical spoken composition rather than silently losing their dialogue."""
    led = _ledger()
    c = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm())[0]["b1"]
    assert _LINE in c["text_prompt"]


def test_shotlock_and_the_execution_plan_resolve_the_same_engine():
    """One resolution, four consumers. If these ever disagree a lane renders a
    prompt written for a different adapter."""
    policy = _policy("humo", effective="minimax_h3_video")
    assert sl._policy_engine_for_role(policy, "character_video") == \
        "minimax_h3_video"
    assert sl._policy_engine_for_role(_policy("humo"), "character_video") == \
        "humo"


# --------------------------------------------------------------------------- #
# the LTX 2.5 joint-AV positive finisher
# --------------------------------------------------------------------------- #

#: The invariant terminator every joint-AV prompt must end with.
_NO_VOICE = "No speech, no voices."
_SOUND_FRAME = "close dry room tone"

#: THE TWO CATEGORY PHRASES THAT MUST NEVER COME BACK (operator, 2026-08-27).
#: Production used to append these instead of naming a sound, and with a human
#: in frame the model filled the unnamed request with VOICES. If either of
#: these strings reappears in a joint-AV prompt, the bug is back.
_BANNED_CATEGORIES = ("matched environmental foley", "ambient room tone",
                      "instrumental scene score", "non-speech ambience",
                      "scene-appropriate")

_CORE = "a captain repairing a smoking console"
#: "console" is the only lexicon cue in `_CORE`, so its suffix is single-sound
#: and can be asserted exactly. The phrase carries the 2026-08-29 intensity
#: wording (the golden recipes say HOW LOUD, and that is what raised the bed).
_CORE_FINISHED = (_CORE + ", a sharp switch snap and close mechanical clicks, "
                  + _SOUND_FRAME + ". " + _NO_VOICE)


def test_the_foley_suffix_names_the_sound_of_the_action():
    assert ltx25.finish_joint_av_positive("ltx25_foley_plus", _CORE) == \
        _CORE_FINISHED


def test_foley_and_mime_compose_the_IDENTICAL_PICTURE_too():
    """OPERATOR RULING, restated 2026-08-28: "foley and mime should have the
    same prompting, the only difference is the mux layer setting."

    An earlier pass unified only the AUDIO TAIL and left the picture clauses
    divergent -- foley said "working the objects within reach so every
    movement has a visible source", mime said "the gesture played out fully
    and carried to a clear held endpoint". That is not the same prompting.
    """
    inputs = {"appearance": "40s, rustic weaver", "setting": "a moonlit bower",
              "expression": "awed", "motion": "he reaches toward her",
              "camera": "static medium shot"}
    foley = ltx25.compose_ltx25_foley_plus(object(), inputs)
    mime = ltx25.compose_ltx25_mime(object(), inputs)
    assert foley == mime, "the lanes diverged again: %r vs %r" % (foley, mime)
    # and end to end, through the finisher
    assert (ltx25.finish_joint_av_positive("ltx25_foley_plus", foley)
            == ltx25.finish_joint_av_positive("ltx25_mime", mime))


def test_the_lanes_remain_INDEPENDENTLY_rewordable():
    """The OTHER standing ruling, which the sameness must not quietly undo:
    "I want each lane independent, so later one could have slow motion, some
    could have tulip motion." Same TEXT today, separate SEAMS always -- the
    dispatcher resolves compose_prompt from each class's own __dict__, so
    collapsing these into one shared method would make it stop seeing the
    children at all."""
    foley_fn = ltx25.Ltx25FoleyPlusEngine.__dict__.get("compose_prompt")
    mime_fn = ltx25.Ltx25MimeEngine.__dict__.get("compose_prompt")
    video_fn = ltx25.Ltx25VideoEngine.__dict__.get("compose_prompt")
    assert foley_fn is not None and mime_fn is not None and video_fn is not None
    assert foley_fn is not mime_fn
    assert mime_fn is not video_fn


def test_foley_and_mime_receive_the_IDENTICAL_string():
    """Operator, 2026-08-27: "foley / mime same thing, they use the new foley
    prompting" and "the only difference between foley and mime is the mux
    layer". Mime used to lead with the brief's mood terms and ask for
    "instrumental scene score" -- a category, and the same defect the foley
    tail had. One shape now, so the two cannot drift apart."""
    assert ltx25.finish_joint_av_positive("ltx25_mime", _CORE) == \
        ltx25.finish_joint_av_positive("ltx25_foley_plus", _CORE)


def test_the_finisher_takes_no_mood_argument_any_more():
    """The brief's `music_mood_terms` still drives the MUSIC bookends through
    `_otr_music_prompt`, which was always their real owner. Passing them here
    must be a hard error, not a silently ignored keyword."""
    with pytest.raises(TypeError):
        ltx25.finish_joint_av_positive("ltx25_mime", _CORE,
                                       music_mood_terms=["tense"])


@pytest.mark.parametrize("banned", _BANNED_CATEGORIES)
@pytest.mark.parametrize("engine", ["ltx25_foley_plus", "ltx25_mime"])
def test_no_joint_av_prompt_may_ask_for_a_CATEGORY(engine, banned):
    """The regression guard for the whole fix. A category leaves the model to
    choose the sound; with a face in frame it chooses voice."""
    out = ltx25.finish_joint_av_positive(engine, _CORE)
    assert banned not in out


@pytest.mark.parametrize("core,expected_sound", [
    ("she pushes the heavy door open", "a door latch clacking"),
    ("rain hammering the window", "rain drumming"),
    ("papers scattered across the desk", "papers rustling"),
    ("boots on the metal deck", "a hard metal clang"),
    ("he lifts the telephone", "a receiver clattering onto its cradle"),
])
def test_the_named_sounds_come_from_the_action_itself(core, expected_sound):
    assert expected_sound in ltx25.finish_joint_av_positive(
        "ltx25_foley_plus", core)


def test_an_action_with_no_cue_still_NAMES_sounds():
    """Falling back to a category here would reinstate the exact defect."""
    out = ltx25.finish_joint_av_positive(
        "ltx25_foley_plus", "a figure stands looking out")
    assert "cloth shifting" in out
    assert out.endswith(_NO_VOICE)
    for banned in _BANNED_CATEGORIES:
        assert banned not in out


def test_at_most_three_sounds_are_named():
    """One latent decodes the picture AND the audio, so a long sound list
    competes with the visual half of the very same string."""
    crowded = ("she runs through the rain past the fire, papers and glass "
               "underfoot, a door and a clock and a bell and an engine")
    out = ltx25.finish_joint_av_positive("ltx25_foley_plus", crowded)
    assert len(ltx25.named_sounds_for(crowded)) == 3
    assert out.endswith(_NO_VOICE)


# --------------------------------------------------------------------------- #
# the 2026-08-29 golden shape: sounds INSIDE the action, clause still last
# --------------------------------------------------------------------------- #

def test_the_composer_seats_sounds_inside_the_action_golden_shape():
    """Operator ruling 2026-08-29 ("prompt for foley like the golden recipe").
    The lab's golden prompts interleave the sound with the action that causes
    it, and they are the renders whose audio measures ~-25 dB mean while the
    append-a-tag-list shape measured 10-40 dB under. The composed order is
    setting, expression, motion, sounds, sound frame, camera -- and the
    no-voice clause still arrives LAST, from the finisher, downstream of the
    style-cue pass."""
    inputs = {"setting": "a shadowed office", "expression": "impatient",
              "motion": "slams his fist on the desk and snatches the papers",
              "camera": "fast lateral move"}
    core = ltx25.compose_ltx25_foley_plus(object(), inputs)
    thud = "a loud heavy wooden thud echoing"
    assert thud in core
    assert core.index("slams his fist") < core.index(thud)
    assert core.index(thud) < core.index("fast lateral move")
    assert core.index(_SOUND_FRAME) < core.index("fast lateral move")
    assert _NO_VOICE not in core, "the clause belongs to the finisher alone"
    fin = ltx25.finish_joint_av_positive("ltx25_foley_plus", core)
    assert fin.endswith(_NO_VOICE)
    assert fin == core.rstrip(" ,.;:") + ". " + _NO_VOICE, (
        "finishing a composed core must append ONLY the clause -- a second "
        "derived tail would name the sounds twice")
    # and it is idempotent through the new predicate
    assert ltx25.finish_joint_av_positive("ltx25_foley_plus", fin) == fin


def test_a_desk_slam_is_an_impact_never_furniture_scraping():
    """The 2026-08-29 lexicon fix: the old chair/table/desk row matched the
    NOUN, so a fist slam at a desk asked for wood scraping. The impact row
    owns it now, and scraping needs an actual drag verb."""
    inputs = {"setting": "a study", "expression": "furious",
              "motion": "slams his fist on the desk", "camera": "push in"}
    core = ltx25.compose_ltx25_foley_plus(object(), inputs)
    assert "a loud heavy wooden thud echoing" in core
    assert "wood scraping" not in core
    assert "wood scraping across the floor" in ltx25.named_sounds_for(
        "he drags the chair back and scrapes it across the boards")
    assert "wood scraping across the floor" not in ltx25.named_sounds_for(
        "he sits quietly at the desk reading")


def test_sounds_are_derived_from_the_action_never_the_camera():
    """A camera direction naming a lexicon word must not mint a sound -- the
    composer derives sounds from setting+expression+motion only."""
    inputs = {"setting": "an empty room", "expression": "calm",
              "motion": "stands and waits",
              "camera": "the camera tracks past the door"}
    core = ltx25.compose_ltx25_foley_plus(object(), inputs)
    assert "a door latch clacking" not in core


def test_finishedness_is_frame_present_plus_clause_last():
    """The r3 false-receipt finding survives the reordering: a bare no-voice
    tail with NO sound frame is still unfinished, and a frame with no tail
    clause is unfinished too. The frame-and-clause pair is no longer
    contiguous on a composed prompt, so the old endswith(terminator) check
    would call every correctly finished prompt unfinished."""
    assert not ltx25.joint_av_prompt_is_finished("x. " + _NO_VOICE)
    assert not ltx25.joint_av_prompt_is_finished(
        "x, " + _SOUND_FRAME + ", steady push in")
    good = "x, " + _SOUND_FRAME + ", steady push in. " + _NO_VOICE
    assert ltx25.joint_av_prompt_is_finished(good)
    assert ltx25.finish_joint_av_positive("ltx25_foley_plus", good) == good


def test_the_sounds_receipt_reads_phrases_off_the_finished_string():
    """`sounds_named_in` matches PHRASES, not cues, because the phrases
    contain cue words and a cue re-scan over a finished prompt would match
    its own output."""
    text = ("a figure at the console, a loud heavy wooden thud echoing, "
            + _SOUND_FRAME + ", push in. " + _NO_VOICE)
    assert ltx25.sounds_named_in(text) == ["a loud heavy wooden thud echoing"]
    assert ltx25.sounds_named_in("no sounds here at all") == []


def test_a_weapon_beat_names_NO_weapon_sound(capsys):
    """THE BANANA COLLISION (r3, 2026-08-28).

    `_otr_banana_route.apply` -- "transform every weapon noun" -- runs AFTER
    this tail is composed, on a pinned ordering. Proven live: "the captain
    raises his revolver toward the hatch" named "a hammer clicking back", and
    the banana route then rendered "raises his banana". The picture showed a
    banana while the audio asked for a revolver hammer.

    The weapon cue was the ONLY lexicon entry whose subject that route
    rewrites, so it is gone. This test is the guard against it returning.
    """
    core = "the captain raises his revolver toward the hatch"
    out = ltx25.finish_joint_av_positive("ltx25_foley_plus", core)
    for weapon_sound in ("hammer", "gunshot", "shot", "cocking"):
        assert weapon_sound not in out.split(core)[-1], \
            "a weapon sound came back into the lexicon: %r" % out
    # the beat is still finished, off its non-weapon cue
    assert out.endswith(_NO_VOICE)
    assert "a door latch clacking" in out          # "hatch"


@pytest.mark.parametrize("cue,forbidden", [
    ("archival documentary footage of the crew", "papers rustling"),
])
def test_a_style_cue_word_does_not_masquerade_as_an_action(cue, forbidden):
    """`document` matched `documentary`, and `archival_documentary` is a real
    style pack whose two-word video cue is literally "archival documentary" --
    so every beat of that pack was asking for rustling paper."""
    assert forbidden not in ltx25.finish_joint_av_positive(
        "ltx25_foley_plus", cue)


def test_a_bare_no_voice_clause_is_NOT_treated_as_finished():
    """THE FALSE RECEIPT (r3, 2026-08-28).

    A caller-supplied prompt merely ENDING in the no-voice clause used to come
    back unchanged with no sound named at all, while observability still
    reported joint_av_prompt=finished. Idempotency now requires the full
    canonical terminator, so such a prompt gets properly finished.
    """
    bare = "an operator override that happens to end here. No speech, no voices."
    out = ltx25.finish_joint_av_positive("ltx25_foley_plus", bare)
    assert out != bare
    assert out.rstrip(" ,.;:").endswith(
        ltx25.JOINT_AV_TERMINATOR.rstrip(" ,.;:"))


def test_a_properly_finished_prompt_IS_left_alone():
    """The other half of the same contract -- it must still be idempotent."""
    once = ltx25.finish_joint_av_positive(
        "ltx25_foley_plus", "hands on the console")
    assert ltx25.finish_joint_av_positive("ltx25_foley_plus", once) == once


def test_the_fallback_names_exactly_ONE_sound():
    """One latent decodes the picture AND the audio, so three simultaneous
    fallback events are three instructions to the PICTURE too -- on the beat
    least likely to contain them."""
    sounds = ltx25.named_sounds_for("a figure stands looking out")
    assert len(sounds) == 1, sounds
    for invented in ("footsteps", "knocking"):
        assert invented not in sounds[0]


def test_the_receipt_records_WHICH_sounds_were_named(_text_only_lane):
    """A receipt that cannot distinguish a good cue from a wrong one is not
    evidence. `joint_av_prompt=finished` alone could not."""
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    obs = req["observability"]
    assert obs["joint_av_prompt"] == "finished"
    assert obs["joint_av_sounds"]
    for sound in obs["joint_av_sounds"]:
        assert sound in req["text_prompt"]


def test_a_joint_av_prompt_carries_NO_character_identity():
    """THE MODEL SPEAKS ITS OWN PROMPT (proven live 2026-08-28).

    On `signal_lost_the_weaver_of_dreams_20260828_003427` a mime beat rendered
    a woman SAYING "Queen of the Fairies" aloud. Nobody wrote that line -- the
    ledger's `character_description` opens with that title and `_ltx25_parts`
    put `appearance` FIRST, so the joint audio-video latent read the identity
    label out loud. The mux was proven innocent by correlation (1.000 against
    the TTS master in the announcer window, ~0.00 in every mime window).

    Identity is carried by the conditioning STILL, whose scene_character row
    mints the face unobstructed, so this text was redundant as well as
    harmful."""
    inputs = {
        "appearance": "30s, Queen of the Fairies. Face: heart-shaped",
        "setting": "a moonlit bank of wild thyme",
        "motion": "she lifts one hand toward him",
        "camera": "static medium shot",
    }
    for engine in ("ltx25_foley_plus", "ltx25_mime"):
        formatter = getattr(ltx25, "compose_" + engine.replace("ltx25_", "ltx25_"))
        out = formatter(object(), inputs)
        assert "Queen of the Fairies" not in out, (engine, out)
        assert "heart-shaped" not in out, (engine, out)
        # the ACTION survives -- this must not become a prompt-gutting fix
        assert "lifts one hand" in out


def test_the_joint_av_LEGACY_fallback_does_not_reintroduce_identity():
    """THE HOLE THE FIRST FIX LEFT (kibitz r1, codex MUST-FIX 5).

    Dropping `appearance` fixed the COMPOSED path. A beat whose only
    structured leaf was appearance leaves an empty core, falls back to
    `_ltx25_legacy`, and that returned `text_prompt` unfiltered -- with the
    identity back in it."""
    look = "30s, Queen of the Fairies. Face: heart-shaped"
    inputs = {"appearance": look, "setting": "", "expression": "",
              "motion": "", "camera": "",
              "text_prompt": look + ", standing in a bower"}
    for fn in (ltx25.compose_ltx25_mime, ltx25.compose_ltx25_foley_plus):
        out = fn(object(), inputs)
        assert "Queen of the Fairies" not in out, out
        # THE LAW: what the writer authored survives; only the span our own
        # composer injected is removed.
        assert "standing in a bower" in out


def test_the_legacy_fallback_returns_the_ORIGINAL_when_nothing_would_remain():
    """A beat with no picture direction at all is worse than one that names a
    face, so this refuses to invent. The runtime guard reports the residue."""
    look = "30s, Queen of the Fairies. Face: heart-shaped"
    inputs = {"appearance": look, "setting": "", "expression": "",
              "motion": "", "camera": "", "text_prompt": look}
    assert ltx25.compose_ltx25_mime(object(), inputs) == look


def test_the_identity_guard_catches_the_REAL_shipped_prompt():
    """Replayed against the actual published defect. A guard that cannot catch
    the bug that motivated it is theatre.

    `signal_lost_the_weaver_of_dreams_20260828_003427`, beat b003: the prompt
    opened with TITANIA's character_description and the model said "Queen of
    the Fairies" out loud."""
    look = ("30s, Queen of the Fairies. Face: heart-shaped, high arched brows, "
            "thin straight nose, sharp-pointed chin")
    as_rendered = ("photorealistic Shakespearean. " + look +
                   ", a moonlit bank, she lifts one hand, static shot")
    leaks = ltx25.identity_leaks_in(as_rendered, appearance=look,
                                    names=["TITANIA", "BOTTOM"])
    assert leaks, "the guard missed the exact prompt that shipped the defect"
    assert any("appearance" in x for x in leaks)


def test_the_identity_guard_is_quiet_on_todays_prompt():
    """The other half: it must not fire on what the lanes compose now, or it
    is a guard nobody reads."""
    inputs = {
        "appearance": "30s, Queen of the Fairies. Face: heart-shaped",
        "setting": "a moonlit bank of wild thyme",
        "motion": "she lifts one hand toward him",
        "camera": "static medium shot",
    }
    for engine, fn in (("ltx25_mime", ltx25.compose_ltx25_mime),
                       ("ltx25_foley_plus", ltx25.compose_ltx25_foley_plus)):
        prompt = ltx25.finish_joint_av_positive(engine, fn(object(), inputs))
        assert not ltx25.identity_leaks_in(
            prompt, appearance=inputs["appearance"],
            names=["TITANIA", "BOTTOM"]), (engine, prompt)


def test_a_cast_name_matches_WHOLE_WORDS_only():
    """A bare substring pass flagged the cast name "LEAR" inside "clearly" --
    and every prompt on this lane says "full face clearly visible", so it fired
    on every beat of that episode. A guard that cries wolf gets ignored."""
    prompt = ("a moonlit bank, full face clearly visible, generous headroom. "
              "No speech, no voices.")
    assert ltx25.identity_leaks_in(prompt, names=["LEAR"]) == []
    # ...but a REAL occurrence still fires
    assert ltx25.identity_leaks_in("LEAR stands in the storm", names=["LEAR"])


def test_the_SILENT_ltx25_lane_keeps_its_identity():
    """The contrast that proves the fix is scoped. `ltx25_video` discards its
    audio latent entirely, so it has no mouth to protect and identity in the
    text is pure benefit there."""
    inputs = {
        "appearance": "30s, Queen of the Fairies. Face: heart-shaped",
        "motion": "she lifts one hand toward him",
    }
    out = ltx25.compose_ltx25_video(object(), inputs)
    assert "Queen of the Fairies" in out


def test_the_finisher_is_idempotent():
    once = ltx25.finish_joint_av_positive("ltx25_foley_plus", _CORE)
    twice = ltx25.finish_joint_av_positive("ltx25_foley_plus", once)
    # THREE passes, not two. The suffix is DERIVED FROM THE PROMPT, so once it
    # is appended the prompt carries sound words the lexicon would match on a
    # second read. A whole-suffix comparison would drift here; keying on the
    # invariant no-voice clause is what holds it stable.
    assert twice == once
    assert ltx25.finish_joint_av_positive("ltx25_foley_plus", twice) == once
    assert once.count(_NO_VOICE) == 1


def test_trailing_punctuation_is_trimmed_before_the_join():
    assert ltx25.finish_joint_av_positive(
        "ltx25_foley_plus", _CORE + " ,.;: ") == _CORE_FINISHED


@pytest.mark.parametrize("engine", ["ltx25_video", "minimax_h3_video", "humo",
                                    ""])
def test_other_engines_are_untouched(engine):
    assert ltx25.finish_joint_av_positive(engine, _CORE) == _CORE
    assert ltx25.finish_joint_av_positive(engine, "") == ""


@pytest.mark.parametrize("engine", ["ltx25_foley_plus", "ltx25_mime"])
@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_positive_raises_and_names_the_engine(engine, blank):
    with pytest.raises(ValueError, match=engine):
        ltx25.finish_joint_av_positive(engine, blank)


@pytest.mark.parametrize("engine,tail", [
    ("ltx25_foley_plus", "No speech, no voices."),
    ("ltx25_mime", "No speech, no voices."),
])
def test_the_non_speech_tail_is_last(engine, tail):
    """The tail is the whole point: it is what stops a lane that generates its
    own audio from generating a voice over dialogue it never received."""
    assert ltx25.finish_joint_av_positive(engine, _CORE).endswith(tail)


def test_a_curly_apostrophe_tokenises_like_an_ascii_one():
    """Gutenberg and the adaptation lanes are full of U+2019. Without folding,
    "I<U+2019>ll" splits into "i" + "ll" while "I'll" stays one token, so the
    same words spelled two ways would not compare equal and an exact quote
    would slip the filter."""
    assert sl._word_tokens("I’ll go") == sl._word_tokens("I'll go")
    led = _ledger()
    line = "I’ll never tell them"
    c = sl.derive_creative_directives(
        _beats(line), led["meta"], led,
        llm_fn=_directive_llm(expression="I'll never tell them"),
        video_policy=_policy("minimax_h3_video"))[0]["b1"]
    assert "never tell them" not in c["text_prompt"]


def test_a_non_latin_line_reaches_the_writer_unescaped():
    """The shared-value invariant has to hold in the PROMPT, not just in
    Python: escaped as \\uXXXX the model is shown something the filter, which
    tokenises the decoded string, could never match."""
    led = _ledger()
    seen = []
    cyrillic = "Мы должны уйти сейчас"
    sl.derive_creative_directives(
        _beats(cyrillic), led["meta"], led, llm_fn=_capturing_llm(seen),
        video_policy=_policy("minimax_h3_video"))
    assert cyrillic in seen[0], "the line was escaped before the model saw it"


@pytest.mark.parametrize("engine", ["ltx25_foley_plus", "ltx25_mime"])
@pytest.mark.parametrize("trailing", [".", ", ", " ,", ";", ":", ".."])
def test_stray_punctuation_after_the_suffix_does_not_stack_it(engine, trailing):
    """A prompt already carrying its suffix must not collect a second copy,
    even with stray punctuation after it.

    The obvious repair -- strip the core's punctuation and THEN test endswith
    -- is wrong, and was proposed as the fix: every suffix ends in a period of
    its own, so stripping the core removes it and an ordinary finished prompt
    fails the test. That would have turned this narrow edge into a defect on
    the common path.
    """
    once = ltx25.finish_joint_av_positive(engine, _CORE)
    again = ltx25.finish_joint_av_positive(engine, once + trailing)
    assert again.count("No speech") == 1, again


def test_the_ordinary_idempotent_case_still_holds():
    """The guard above must not have broken the common path it protects."""
    for engine in ("ltx25_foley_plus", "ltx25_mime"):
        once = ltx25.finish_joint_av_positive(engine, _CORE)
        assert ltx25.finish_joint_av_positive(engine, once) == once
        assert once.count("No speech") == 1


def test_the_helper_has_no_dialogue_argument():
    import inspect
    params = inspect.signature(ltx25.finish_joint_av_positive).parameters
    assert set(params) == {"engine_id", "positive"}


# --------------------------------------------------------------------------- #
# open health
# --------------------------------------------------------------------------- #

def _row(beat_id, role, engine_id, exists=True):
    return {"order": 0, "shot_id": "shot_" + beat_id, "beat_id": beat_id,
            "role": role, "engine_id": engine_id, "exists": exists}


# --------------------------------------------------------------------------- #
# the finisher AT THE RENDER SEAM -- the whole request, not the helper alone
# --------------------------------------------------------------------------- #

_BRIEF_META = {
    "story_brief_status": "ok",
    "story_brief": ("A lonely lighthouse crew battles a storm that speaks. "
                    "The sea glows green at midnight."),
    "story_brief_terms": {
        "setting": ["a storm-wracked lighthouse"],
        "lighting": ["lantern glow"],
        "atmosphere": ["uneasy"],
    },
}


@pytest.fixture
def _text_only_lane(monkeypatch):
    """These assert prompt composition only; they mint no stills."""
    monkeypatch.setenv("OTR_ENABLE_LTX_I2V", "0")
    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)


def _open_shot(engine_id):
    return {"shot_id": "shot_b001", "source_line_ids": ["b001"],
            "role": "announcer_visual", "engine_id": engine_id,
            "group_id": "grp_announcer_visual", "target_frame_count": 50,
            "creative": {}}


def _open_ledger(meta=None):
    return {
        "meta": dict(meta if meta is not None else _BRIEF_META),
        "lines": [{"line_id": "b001", "char_id": "announcer",
                   "speaker_role": "announcer", "text": "Tonight...",
                   "start_s": 0.0, "dur_s": 5.0}],
        "images": {"images": [
            {"object_id": "still_b001", "kind": "scene_open",
             "beat_id": "b001", "path": "X:/img/still_b001.png"}]},
    }


def test_a_foley_open_composes_a_real_prompt_and_finishes_it(_text_only_lane):
    """TWO defects in one request. Before this change `ltx25_foley_plus`
    matched no prompt branch at all -- `roles = ROLES` makes it legal on the
    announcer bookend, that role arrives with `text_prompt` cleared, and it was
    absent from the scene allowlist -- so the beat shipped `build_request`'s
    hardcoded "a 1940s radio studio" with no `prompt_source` stamped. Now it
    composes the role motion prompt AND carries its audio requirement."""
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    prompt = req["text_prompt"]
    assert "a 1940s radio studio" not in prompt
    assert req["observability"]["prompt_source"] == "motion_role"
    assert prompt.endswith(_NO_VOICE)
    assert req["observability"]["joint_av_prompt"] == "finished"


def test_a_mime_open_IGNORES_the_briefs_mood_terms(_text_only_lane):
    """The mood terms used to LEAD the mime tail. Under the operator's ruling
    mime takes the foley prompting unchanged, so a brief full of moods must
    make no difference to the string -- the moods still reach the music
    bookends, which is where they always belonged."""
    meta = dict(_BRIEF_META)
    meta["music_mood_terms"] = ["tense", "elegiac", "hushed"]
    with_moods = rd.build_request_from_shot(
        _open_shot("ltx25_mime"), _open_ledger(meta))["text_prompt"]
    without = rd.build_request_from_shot(
        _open_shot("ltx25_mime"), _open_ledger())["text_prompt"]
    assert with_moods == without
    assert with_moods.endswith(_NO_VOICE)
    for banned in _BANNED_CATEGORIES:
        assert banned not in with_moods


def test_a_mime_open_is_finished_with_named_sounds(_text_only_lane):
    req = rd.build_request_from_shot(_open_shot("ltx25_mime"), _open_ledger())
    assert _SOUND_FRAME in req["text_prompt"]
    assert req["text_prompt"].endswith(_NO_VOICE)
    # TAG FORM, NOT PROSE (2026-08-28): this lane reads its prompt aloud, and
    # a transcribed stem quoted the old "with ... close and dry in the room"
    # wording back at us. The sounds are still NAMED -- that is
    # PBUG-20260828-01's fix and must not regress -- they are simply no longer
    # wrapped in a sentence.
    assert ", with " not in req["text_prompt"]
    assert "close and dry in the room" not in req["text_prompt"]


def test_a_silent_ltx25_character_beat_is_not_told_to_hold_still(
        _text_only_lane):
    """MOTION BAKE-IN (2026-08-27). The LTX character append used to say
    "stable centered subject ... comfortably composed" on every ltx lane --
    an instruction to hold still on lanes with no mouth to protect. Silent
    ltx25 lanes now keep the framing safety (face visible, headroom) and ask
    for motion; the AUDIO-IN ltx lane keeps the steadying clause, because
    there the stability protects the lip-sync that lane sells."""
    led = _ledger()
    creative = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("ltx25_video"))[0]["b1"]
    shot = {"shot_id": "shot_b1", "source_line_ids": ["b1"],
            "role": "character_video", "engine_id": "ltx25_video",
            "group_id": "grp_character_video", "target_frame_count": 50,
            "creative": creative}
    req = rd.build_request_from_shot(shot, {
        "meta": led["meta"],
        "lines": [{"line_id": "b1", "char_id": "c1",
                   "speaker_role": "char_voice", "text": _LINE,
                   "start_s": 0.0, "dur_s": 1.0}],
        "images": {"images": [{"object_id": "still_b1", "kind": "scene_beat",
                               "beat_id": "b1",
                               "path": "X:/img/still_b1.png"}]},
    })
    prompt = req["text_prompt"]
    assert "stable centered subject" not in prompt
    assert "comfortably composed" not in prompt
    assert "full face clearly visible" in prompt      # framing safety kept
    assert "generous headroom" in prompt
    assert "the subject in real motion" in prompt


@pytest.mark.parametrize("spelling", ["ltx_audio_in", "ltx23_low_audio_in",
                                      "ltx23_16gb_audio_in"])
def test_an_audio_in_lane_keeps_its_framing_under_any_spelling(spelling):
    """THE LIP-SYNC PROTECTION MUST NOT DEPEND ON HOW THE ID IS SPELLED.

    `engine_family` is keyed on INTERNAL ids, so a shot row carrying the
    public (`ltx23_low_audio_in`) or legacy (`ltx23_16gb_audio_in`) spelling
    classified as `abstract` -- and once the LTX character append started
    keying the steadying clause on family, an audio-in lane misread that way
    would silently lose the framing that keeps its mouth in shot. Found by
    review before it reached a render; the id is resolved before classifying.
    """
    from nodes._otr_video_engines.render_driver import engine_family
    from nodes._otr_shared.public_engines import resolve_engine_id
    assert resolve_engine_id(spelling) == "ltx_audio_in"
    assert engine_family(resolve_engine_id(spelling), "") == \
        "audio_conditioned_video"


def test_ltx25_video_is_not_touched_at_the_seam(_text_only_lane):
    """The silent sibling DISCARDS its audio latent, so an audio clause there
    would steer the picture for a track nobody keeps."""
    req = rd.build_request_from_shot(_open_shot("ltx25_video"), _open_ledger())
    assert "No speech, no voices" not in req["text_prompt"]
    assert "joint_av_prompt" not in (req.get("observability") or {})


def test_the_operator_override_is_finished_too(monkeypatch, _text_only_lane):
    """OTR_LTX_RADIO_PROMPT is verbatim, and the audio requirement is not a
    rewrite of it -- it is appended, so the operator's words survive whole."""
    monkeypatch.setenv("OTR_LTX_RADIO_PROMPT", "OPERATOR SAYS EXACTLY THIS")
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    assert req["text_prompt"].startswith("OPERATOR SAYS EXACTLY THIS,")
    assert _SOUND_FRAME in req["text_prompt"]
    assert req["text_prompt"].endswith(_NO_VOICE)


def test_the_digest_and_length_describe_the_shipped_prompt(_text_only_lane):
    """Evidence here is cited by hash. A receipt describing the pre-tail text
    while a different string renders is a live defect, not a cosmetic one."""
    import hashlib
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    prompt = req["text_prompt"]
    obs = req["observability"]
    assert obs["prompt_chars"] == len(prompt)
    assert obs["prompt_sha8"] == \
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


def test_the_mandatory_tail_is_never_trimmed_by_a_prompt_budget(_text_only_lane):
    """The composing branches publish a char budget for the banana re-cap, and
    the non-speech sentence lands PAST it. The seam clears the budget for
    exactly this reason; without that, `_banana_cap` would trim away the one
    clause that stops the lane generating voices.

    The banana route is ON by default (`OTR_BANANA_VIDEO`, default True), so
    this request really does pass through the funnel.
    """
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    prompt = req["text_prompt"]
    assert len(prompt) > 188, len(prompt)      # past the scene branch's budget
    assert prompt.endswith(_NO_VOICE), prompt[-140:]


def test_an_already_finished_prompt_still_gets_its_budget_cleared(
        monkeypatch, _text_only_lane):
    """THE PROTECTION IS UNCONDITIONAL, THE MUTATION IS NOT.

    Clearing the prompt budget used to sit inside the "did the text change"
    branch, so a prompt that ALREADY carried its suffix took the no-change path
    and left the scene branch's published budget standing -- and the banana
    re-cap would then trim away the mandatory non-speech tail, the one clause
    this seam exists to protect. A lane that keeps its audio owes that tail
    whether or not this call is what appended it.
    """
    monkeypatch.setenv(
        "OTR_LTX_RADIO_PROMPT",
        "a quiet console at midnight, switches clicking and dials "
        "turning, " + _SOUND_FRAME + ". " + _NO_VOICE)
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    assert req["text_prompt"].count(_NO_VOICE) == 1          # not stacked
    assert req["observability"]["joint_av_prompt"] == "finished"
    assert req["text_prompt"].endswith(_NO_VOICE)


def test_a_public_menu_id_on_the_shot_row_still_gets_finished(_text_only_lane):
    """`is_foley_route` resolves ids before comparing, for the stated reason
    that a policy can hold a public menu string. The finisher compares the same
    way, or a beat that really is on the route silently skips its audio
    requirement."""
    from nodes._otr_shared.public_engines import resolve_engine_id
    assert resolve_engine_id("ltx25_high_foley_plus") == "ltx25_foley_plus"
    shot = _open_shot("ltx25_high_foley_plus")
    req = rd.build_request_from_shot(shot, _open_ledger())
    prompt = req["text_prompt"]
    assert req["observability"].get("joint_av_prompt") == "finished"
    assert prompt.endswith(_NO_VOICE)
    # THE ASSERTIONS ABOVE ARE NOT SUFFICIENT ON THEIR OWN, and an earlier cut
    # of this test proved it: with the id unresolved at the SCENE allowlist the
    # beat kept build_request's hardcoded default and then had the suffix
    # appended, so `endswith` and the receipt both passed while the prompt was
    # the exact degrade this change exists to close. Pin the composition too.
    assert "a 1940s radio studio" not in prompt
    assert req["observability"]["prompt_source"] == "motion_role"
    # the row itself is NOT rewritten -- resolution was for the comparison only
    assert shot["engine_id"] == "ltx25_high_foley_plus"


def test_a_public_menu_mime_id_is_also_finished(_text_only_lane):
    from nodes._otr_shared.public_engines import resolve_engine_id
    assert resolve_engine_id("ltx25_high_mime") == "ltx25_mime"
    req = rd.build_request_from_shot(_open_shot("ltx25_high_mime"),
                                     _open_ledger())
    assert "a 1940s radio studio" not in req["text_prompt"]
    assert req["text_prompt"].endswith(_NO_VOICE)
    assert req["observability"]["prompt_source"] == "motion_role"


def test_the_composing_branchs_provenance_survives_the_finisher(_text_only_lane):
    """The audio tail FINISHES the prompt; it does not author it. Overwriting
    prompt_source here would erase where the prompt actually came from."""
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    assert req["observability"]["prompt_source"] == "motion_role"


def test_the_empty_ltx25_negative_is_unchanged(_text_only_lane):
    """LTX 2.5 ships a locked empty negative. This change touches the POSITIVE
    only -- the non-speech instruction lives there because the audio half reads
    the same string the picture does."""
    for engine in ("ltx25_video", "ltx25_foley_plus", "ltx25_mime"):
        req = rd.build_request_from_shot(_open_shot(engine), _open_ledger())
        assert str(req.get("negative_prompt") or "") == ""


def test_a_character_beat_on_foley_gets_no_dialogue_end_to_end():
    """The two halves of this change meet here: ShotLock refuses the line to a
    foley lane, and the render seam appends the no-voices requirement."""
    led = _ledger()
    creative = sl.derive_creative_directives(
        _beats(), led["meta"], led, llm_fn=_directive_llm(),
        video_policy=_policy("ltx25_foley_plus"))[0]["b1"]
    assert _LINE not in creative["text_prompt"]
    shot = {"shot_id": "shot_b1", "source_line_ids": ["b1"],
            "role": "character_video", "engine_id": "ltx25_foley_plus",
            "group_id": "grp_character_video", "target_frame_count": 50,
            "creative": creative}
    # A scene-init engine must have its per-beat still minted upstream; there
    # is no portrait fallback, by design.
    req = rd.build_request_from_shot(shot, {
        "meta": led["meta"],
        "lines": [{"line_id": "b1", "char_id": "c1",
                   "speaker_role": "char_voice", "text": _LINE,
                   "start_s": 0.0, "dur_s": 1.0}],
        "images": {"images": [{"object_id": "still_b1", "kind": "scene_beat",
                               "beat_id": "b1",
                               "path": "X:/img/still_b1.png"}]},
    })
    assert _LINE not in req["text_prompt"]
    assert req["text_prompt"].endswith(_NO_VOICE)


@pytest.mark.parametrize("engine", ["ltx25_foley_plus", "ltx25_mime"])
def test_a_foley_or_mime_open_is_a_healthy_ltx_open(engine):
    """They render the LTX 2.5 picture graph unchanged, so an open on one is a
    real LTX open -- not the procgen floor BUG-LOCAL-413 hunts for."""
    m = {"episode_id": "e", "clips": [
        _row("b001", "announcer_visual", engine),
        _row("b000_music_open", "music_visual", engine)]}
    assert rd.check_ltx_open_health(m, strict=True) == []
