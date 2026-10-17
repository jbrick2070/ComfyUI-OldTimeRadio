"""A voice of the wrong gender is reassigned, not a reason to kill the episode.

THE DEFECT (4060 leg s03, 2026-10-16, profile otr_8gb_ltx25_audio_in,
comfy_native gemma4-e2b writer). Both attempts of
`scifi_news_pro_casting_voices` cast the female "Ziyin" on m05, a MALE voice:

    Ziyin: menu id m05 is male, but gender is 'female' -- pick one of the
    female ids instead: m07, m08, m09, m10

The typed retry repeated the pick word for word, the ladder exhausted after 2
attempts, and NewsProCastError killed a finished script 64 s in, before any
video. The validator already knew every legal id, so this has one checkable
right answer: keep the character's gender, move the voice to the first unused
id of that gender, and fail only when none is left.

CPU-only: no model, no GPU.
"""

from __future__ import annotations

import json

import pytest

from nodes import _otr_scifi_news_pro as F2


def _live_menu():
    """The live leg's menu shape: m01-m06 male, m07-m10 female."""
    return F2.VoiceMenu(entries=tuple(
        F2.VoiceMenuEntry(menu_id=f"m{i:02d}",
                          gender="male" if i <= 6 else "female",
                          description="d", preset=f"v2/en_speaker_{i}")
        for i in range(1, 11)
    ))


_SPEAKERS = ["K. Lisa Yang", "Ziyin"]


def _row(name, gender, timbre):
    return {"name": name, "role": "r", "character_description": "d",
            "gender": gender, "age_band": "30s", "timbre": timbre}


# The live answer, both attempts: Yang is legal on m10, Ziyin is not on m05.
_LIVE_RAW = json.dumps({"cast": [
    _row("K. Lisa Yang", "female", "m10"),
    _row("Ziyin", "female", "m05"),
]})


def test_the_live_answer_reproduces_the_exact_refusal():
    validator = F2._make_casting_validator(_live_menu(), _SPEAKERS)
    error = validator(F2.CastingVoices.model_validate_json(_LIVE_RAW))
    assert error == (
        "Ziyin: menu id m05 is male, but gender is 'female' -- pick one of "
        "the female ids instead: m07, m08, m09, m10")


def test_the_mismatched_voice_moves_and_the_gender_stays():
    menu = _live_menu()
    settled = F2._make_casting_settle(menu, _SPEAKERS)(_LIVE_RAW)

    assert settled is not None
    by_name = {c.name: c for c in settled.cast}
    assert by_name["Ziyin"].gender == "female", "the model's gender is kept"
    # First unused female id in menu order -- m10 is Yang's, so m07.
    assert by_name["Ziyin"].timbre == "m07"
    assert by_name["K. Lisa Yang"].timbre == "m10", "a legal row is untouched"
    assert F2._make_casting_validator(menu, _SPEAKERS)(settled) is None


def test_the_choice_is_stable_and_never_takes_a_voice_already_cast():
    raw = json.dumps({"cast": [
        _row("K. Lisa Yang", "female", "m07"),
        _row("Ziyin", "female", "m05"),
    ]})
    settle = F2._make_casting_settle(_live_menu(), _SPEAKERS)
    first = settle(raw)
    again = settle(raw)
    assert {c.name: c.timbre for c in first.cast}["Ziyin"] == "m08"
    assert first.model_dump() == again.model_dump()


def test_every_mismatched_row_is_settled_in_one_pass():
    """The validator reports only the FIRST bad row; the repair must not stop
    there, or the second bad row kills the episode anyway."""
    speakers = ["Ziyin", "Marcus", "Ada"]
    raw = json.dumps({"cast": [
        _row("Ziyin", "female", "m05"),
        _row("Marcus", "male", "m08"),
        _row("Ada", "female", "m06"),
    ]})
    menu = _live_menu()
    settled = F2._make_casting_settle(menu, speakers)(raw)
    assert settled is not None
    picks = {c.name: c.timbre for c in settled.cast}
    assert picks == {"Ziyin": "m07", "Marcus": "m01", "Ada": "m09"}
    assert F2._make_casting_validator(menu, speakers)(settled) is None


def test_no_voice_of_that_gender_left_fails_closed():
    """Two characters never share a larynx -- an exhausted stock declines."""
    menu = F2.VoiceMenu(entries=(
        F2.VoiceMenuEntry(menu_id="m01", gender="male", description="d",
                          preset="v2/en_speaker_1"),
        F2.VoiceMenuEntry(menu_id="m02", gender="male", description="d",
                          preset="v2/en_speaker_2"),
        F2.VoiceMenuEntry(menu_id="m03", gender="female", description="d",
                          preset="v2/en_speaker_3"),
    ))
    raw = json.dumps({"cast": [
        _row("K. Lisa Yang", "female", "m03"),
        _row("Ziyin", "female", "m01"),
    ]})
    assert F2._make_casting_settle(menu, _SPEAKERS)(raw) is None


def test_an_unknown_menu_id_is_not_guessed():
    raw = json.dumps({"cast": [
        _row("K. Lisa Yang", "female", "m10"),
        _row("Ziyin", "female", "m99"),
    ]})
    menu = _live_menu()
    settled = F2._make_casting_settle(menu, _SPEAKERS)(raw)
    # Left as the model wrote it, so the real validator still refuses it.
    assert settled is None or (
        F2._make_casting_validator(menu, _SPEAKERS)(settled) is not None)


def test_a_valid_cast_settles_to_itself():
    raw = json.dumps({"cast": [
        _row("K. Lisa Yang", "female", "m10"),
        _row("Ziyin", "female", "m08"),
    ]})
    settled = F2._make_casting_settle(_live_menu(), _SPEAKERS)(raw)
    assert settled == F2.CastingVoices.model_validate_json(raw)


def test_unparseable_output_fails_closed():
    assert F2._make_casting_settle(_live_menu(), _SPEAKERS)("{not json") is None


# --------------------------------------------------------------------------- #
# the live failure, end to end through the real retry ladder
# --------------------------------------------------------------------------- #
def _stub_prompt_inputs(monkeypatch):
    """`_pass_casting` builds its prompt from a sealed draft; none of that is
    under test here, so only the prompt-building helpers are stubbed. The
    retry ladder, the validator and the settle all run for real."""
    monkeypatch.setattr(F2, "_validate_selected_final_draft",
                        lambda *a, **k: None)
    monkeypatch.setattr(F2, "_script_view", lambda *a, **k: "SCRIPT")
    monkeypatch.setattr(F2, "_shapes_for_speakers", lambda *a, **k: [])
    monkeypatch.setattr(F2, "_seam", lambda *a, **k: "SYSTEM")


class _FinalDraft:
    parsed = None


def test_the_live_failure_now_ships_through_the_real_ladder(monkeypatch):
    _stub_prompt_inputs(monkeypatch)
    calls = []

    def slot(*args, **kwargs):
        calls.append(kwargs.get("temperature"))
        return _LIVE_RAW            # the model repeats itself, as it did live

    menu = _live_menu()
    casting = F2._pass_casting(slot, None, _FinalDraft(), None, menu,
                               speakers=list(_SPEAKERS))

    assert len(calls) == 2, "the model keeps both of its own attempts first"
    by_name = {c.name: c for c in casting.cast}
    assert (by_name["Ziyin"].gender, by_name["Ziyin"].timbre) == ("female",
                                                                  "m07")
    assert F2._make_casting_validator(menu, _SPEAKERS)(casting) is None


def test_an_answer_the_settle_cannot_fix_still_raises(monkeypatch):
    _stub_prompt_inputs(monkeypatch)
    raw = json.dumps({"cast": [
        _row("K. Lisa Yang", "female", "m10"),
        _row("Ziyin", "female", "m10"),      # a shared voice is not settled
    ]})
    with pytest.raises(F2.NewsProCastError, match="already taken"):
        F2._pass_casting(lambda *a, **k: raw, None, _FinalDraft(), None,
                         _live_menu(), speakers=list(_SPEAKERS))


def test_the_settle_is_wired_into_the_casting_pass():
    """An unwired settle is inert and the suite would still be green."""
    import inspect

    source = inspect.getsource(F2._pass_casting)
    assert "on_attempt_complete=_record" in source
    assert "_make_casting_settle(menu, speakers)" in source
