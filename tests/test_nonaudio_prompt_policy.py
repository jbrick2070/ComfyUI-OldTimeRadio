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
    assert "restrained visible reaction" in c["text_prompt"]
    assert "subtle natural body motion" in c["text_prompt"]
    assert "stable mid-shot" in c["text_prompt"]


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
    assert "restrained visible reaction" in c["text_prompt"]   # fell back
    assert any("spoken line back into" in w for w in warns)
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
    assert "restrained visible reaction" in prompt          # fell back
    assert any("spoken line back into" in w for w in warns)


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

_FOLEY = ("matched environmental foley for the visible action, ambient room "
          "tone. No speech, no voices, pure action.")
_MIME_TAIL = ("instrumental scene score and non-speech ambience matching the "
              "visible action. No speech, no voices.")
_CORE = "a captain repairing a smoking console"


def test_the_foley_suffix_is_exact():
    assert ltx25.finish_joint_av_positive("ltx25_foley_plus", _CORE) == \
        _CORE + ", " + _FOLEY


def test_the_mime_suffix_without_moods_is_exact():
    assert ltx25.finish_joint_av_positive("ltx25_mime", _CORE) == \
        _CORE + ", scene-appropriate " + _MIME_TAIL


def test_mime_mood_terms_are_stripped_deduped_and_capped_at_three():
    out = ltx25.finish_joint_av_positive(
        "ltx25_mime", _CORE,
        music_mood_terms=["  tense ", "", "tense", "melancholy", "brooding",
                          "dropped"])
    assert out == _CORE + ", tense, melancholy, brooding " + _MIME_TAIL


@pytest.mark.parametrize("moods", ["tense", None, 42, {"a": 1}, ("tense",)])
def test_a_non_list_mood_value_yields_no_moods(moods):
    """The brief declares `music_mood_terms: list[str]`. A bare string here is
    a caller mistake, and treating it as one long mood is worse than ignoring
    it."""
    assert ltx25.finish_joint_av_positive(
        "ltx25_mime", _CORE, music_mood_terms=moods) == \
        _CORE + ", scene-appropriate " + _MIME_TAIL


def test_the_finisher_is_idempotent():
    once = ltx25.finish_joint_av_positive("ltx25_foley_plus", _CORE)
    assert ltx25.finish_joint_av_positive("ltx25_foley_plus", once) == once


def test_trailing_punctuation_is_trimmed_before_the_join():
    assert ltx25.finish_joint_av_positive(
        "ltx25_foley_plus", _CORE + " ,.;: ") == _CORE + ", " + _FOLEY


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
    ("ltx25_foley_plus", "No speech, no voices, pure action."),
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
    assert set(params) == {"engine_id", "positive", "music_mood_terms"}


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
    assert prompt.endswith(_FOLEY)
    assert req["observability"]["joint_av_prompt"] == "finished"


def test_a_mime_open_reads_the_briefs_own_mood_terms(_text_only_lane):
    meta = dict(_BRIEF_META)
    meta["music_mood_terms"] = ["  tense ", "", "tense", "elegiac", "hushed",
                                "ignored"]
    req = rd.build_request_from_shot(_open_shot("ltx25_mime"),
                                     _open_ledger(meta))
    assert req["text_prompt"].endswith("tense, elegiac, hushed " + _MIME_TAIL)


def test_a_mime_open_without_mood_terms_uses_the_default_lead(_text_only_lane):
    req = rd.build_request_from_shot(_open_shot("ltx25_mime"), _open_ledger())
    assert req["text_prompt"].endswith("scene-appropriate " + _MIME_TAIL)


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
    assert req["text_prompt"] == "OPERATOR SAYS EXACTLY THIS, " + _FOLEY


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
    assert prompt.endswith(_FOLEY), prompt[-140:]


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
    monkeypatch.setenv("OTR_LTX_RADIO_PROMPT",
                       "a quiet console at midnight, " + _FOLEY)
    req = rd.build_request_from_shot(_open_shot("ltx25_foley_plus"),
                                     _open_ledger())
    assert req["text_prompt"].count("pure action.") == 1     # not stacked
    assert req["observability"]["joint_av_prompt"] == "finished"
    assert req["text_prompt"].endswith(_FOLEY)


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
    assert prompt.endswith(_FOLEY)
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
    assert req["text_prompt"].endswith(_MIME_TAIL)
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
    assert req["text_prompt"].endswith(_FOLEY)


@pytest.mark.parametrize("engine", ["ltx25_foley_plus", "ltx25_mime"])
def test_a_foley_or_mime_open_is_a_healthy_ltx_open(engine):
    """They render the LTX 2.5 picture graph unchanged, so an open on one is a
    real LTX open -- not the procgen floor BUG-LOCAL-413 hunts for."""
    m = {"episode_id": "e", "clips": [
        _row("b001", "announcer_visual", engine),
        _row("b000_music_open", "music_visual", engine)]}
    assert rd.check_ltx_open_health(m, strict=True) == []
