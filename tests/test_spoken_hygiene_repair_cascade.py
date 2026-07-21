"""Total spoken-line craft repair: A/B/C, dynamic LLM passes, and floor."""
from __future__ import annotations

from dataclasses import replace

import pytest

from nodes._otr_line_composer import (
    LineRequest,
    _quality_flags_for_line,
    compose_line,
    repair_existing_spoken_line,
    spoken_surface_flags_for_line,
)
from nodes._otr_line_hygiene import (
    detect_stage_business_for_reroll,
    deterministic_hygiene_floor,
    flag_anchor_stuffing,
    flag_cliche,
    flag_objective_literal,
    flag_on_the_nose,
    flag_one_breath,
    flag_personal_cost_boilerplate,
    flag_stage_business,
    flag_thesis_close,
    is_non_dialogue_stage_line,
    is_truncated,
)
from nodes._otr_readiness import normalize_for_delivery, stamp_text_for_tts_delivery


ANCHORS = ("copper key", "blue lamp", "north stair")
HEADER = (
    "TITLE: x\nSETTING: y\nTIME: z\n"
    "Specificity anchors (use selectively):\n"
    "- copper key\n- blue lamp\n- north stair\n\n"
    "PREMISE: w"
)


def _req() -> LineRequest:
    return LineRequest(
        speaker="ALICE",
        intent="force the guard to confess he hid the copper key",
        mood="tense",
        target_words=18,
        canon_header=HEADER,
        last_lines=[("BOB", "What did you find?")],
        beat_objective="force the guard to confess he hid the copper key",
        speaker_role="character",
    )


def test_floor_resolves_every_named_craft_gate_and_is_idempotent():
    cases = [
        (
            "You're playing with fire and there is no turning back.",
            ("cliche",),
            lambda text: flag_cliche(text)[0],
        ),
        (
            "I'll go check the west hall.",
            ("stage_business",),
            lambda text: flag_stage_business(text)[0],
        ),
        (
            "I'm really scared of the receiver.",
            ("on_the_nose",),
            lambda text: flag_on_the_nose(text)[0],
        ),
        (
            "The copper key, blue lamp, and north stair mark the route.",
            ("anchor_stuffing",),
            lambda text: flag_anchor_stuffing(text, ANCHORS)[0],
        ),
        (
            " ".join(["signal"] * 35),
            ("one_breath",),
            lambda text: flag_one_breath(text)[0],
        ),
        (
            "We know the trust they will lose either way.",
            ("personal_cost",),
            lambda text: flag_personal_cost_boilerplate(text)[0],
        ),
        (
            "Guard, confess you hid the copper key.",
            ("objective_literal",),
            lambda text: flag_objective_literal(
                text, "force the guard to confess he hid the copper key",
            )[0],
        ),
        (
            "And so the lesson is that patience matters.",
            ("thesis_close",),
            lambda text: flag_thesis_close(text)[0],
        ),
    ]
    for raw, gates, detector in cases:
        result = deterministic_hygiene_floor(
            raw,
            gates,
            beat_objective=_req().beat_objective,
            anchors=ANCHORS,
        )
        assert result.text
        assert detector(result.text) is False
        again = deterministic_hygiene_floor(
            result.text,
            gates,
            beat_objective=_req().beat_objective,
            anchors=ANCHORS,
        )
        assert again.text == result.text


def test_floor_repairs_stage_forms_and_preserves_true_empty_mechanical_row():
    leading = deterministic_hygiene_floor(
        "twirls his pen nervously We have one chance left.",
        ("stage_direction",),
    )
    assert leading.text == "We have one chance left."

    whole = deterministic_hygiene_floor(
        "snaps off the pen tip, jams it into the port, and turns the dial",
        ("stage_direction",),
    )
    assert whole.text == "It moved."

    assert deterministic_hygiene_floor("", ("empty",)).text == ""
    assert deterministic_hygiene_floor("...", ("non_lexical",)).text == ""


def test_spoken_format_floor_keeps_grounded_acronym_and_normalizes_other_caps():
    result = deterministic_hygiene_floor(
        'ANNOUNCER: [quietly] "NASA says STOP ... now."',
        ("spoken_format", "stage_direction", "all_caps", "non_lexical"),
        allowed_all_caps=("NASA",),
    )
    assert result.text == "NASA says Stop now."


def test_composer_uses_alternate_slot_before_floor_and_stamps_rung():
    dirty = "You're playing with fire, Watson."
    clean = "The seal cracked before midnight."
    creative_temperatures: list[float] = []
    alternate_temperatures: list[float] = []

    def creative(messages, *, temperature, max_new_tokens, stop=None):
        creative_temperatures.append(temperature)
        return dirty

    def alternate(messages, *, temperature, max_new_tokens, stop=None):
        alternate_temperatures.append(temperature)
        return clean

    result = compose_line(
        creative_fn=creative,
        repair_fn=alternate,
        req=_req(),
        source_bank_id="original",
    )
    assert result.text == clean
    assert creative_temperatures == [0.8, 0.8, 0.3]
    assert alternate_temperatures == [0.18]
    assert (
        "hygiene_repaired_after_reroll:cliche:repair_c_alternate_slot"
        in result.compose_flags
    )


def test_persistent_composer_defects_reach_nonempty_clean_floor():
    dirty = (
        "I'm scared and you're playing with fire beside the copper key, "
        "blue lamp, and north stair."
    )

    def same(messages, *, temperature, max_new_tokens, stop=None):
        return dirty

    result = compose_line(
        creative_fn=same,
        repair_fn=same,
        req=_req(),
        source_bank_id="original",
    )
    assert result.text
    assert _quality_flags_for_line(result.text, _req()) == []
    assert any(
        flag.endswith(":deterministic_floor")
        for flag in result.compose_flags
        if flag.startswith("hygiene_repaired_after_reroll:")
    )


def test_composer_keeps_repairing_past_a_b_c_until_dynamic_pass_is_clean():
    dirty = "You're playing with fire, Watson."
    clean = "The seal cracked before midnight."
    creative_calls = 0
    alternate_calls = 0

    def creative(messages, *, temperature, max_new_tokens, stop=None):
        nonlocal creative_calls
        creative_calls += 1
        # Initial draft + A + B remain dirty; the first post-C fresh call wins.
        return clean if creative_calls >= 4 else dirty

    def alternate(messages, *, temperature, max_new_tokens, stop=None):
        nonlocal alternate_calls
        alternate_calls += 1
        return dirty

    result = compose_line(
        creative_fn=creative,
        repair_fn=alternate,
        req=_req(),
        source_bank_id="original",
    )
    assert result.text == clean
    assert creative_calls == 4
    assert alternate_calls == 1
    assert (
        "hygiene_repaired_after_reroll:cliche:"
        "repair_dynamic_04_same_slot"
    ) in result.compose_flags


def test_final_row_scour_keeps_rotating_slots_until_dynamic_pass_is_clean():
    dirty = "twirls his pen nervously We have one chance left."
    clean = "The seal cracked before midnight."
    creative_calls = 0
    alternate_calls = 0

    def creative(messages, *, temperature, max_new_tokens, stop=None):
        nonlocal creative_calls
        creative_calls += 1
        return clean if creative_calls >= 2 else dirty

    def alternate(messages, *, temperature, max_new_tokens, stop=None):
        nonlocal alternate_calls
        alternate_calls += 1
        return dirty

    result = repair_existing_spoken_line(
        text=dirty,
        req=_req(),
        creative_fn=creative,
        repair_fn=alternate,
    )
    assert result.text == clean
    assert creative_calls == 2
    assert alternate_calls == 1
    assert (
        "hygiene_repaired_after_reroll:stage_direction:"
        "repair_dynamic_03_same_slot"
    ) in result.compose_flags


def test_final_row_scour_is_noop_on_clean_and_repairs_bypass_line():
    calls: list[float] = []

    def clean_writer(messages, *, temperature, max_new_tokens, stop=None):
        calls.append(temperature)
        return "The seal cracked before midnight."

    clean = repair_existing_spoken_line(
        text="The seal cracked before midnight.",
        req=_req(),
        creative_fn=clean_writer,
    )
    assert clean.text == "The seal cracked before midnight."
    assert calls == []

    repaired = repair_existing_spoken_line(
        text="twirls his pen nervously We have one chance left.",
        req=_req(),
        creative_fn=clean_writer,
    )
    assert repaired.text == "The seal cracked before midnight."
    assert calls == [0.3]
    assert (
        "hygiene_repaired_after_reroll:stage_direction:repair_b_same_slot"
        in repaired.compose_flags
    )


def test_bare_production_cues_become_spoken_words_after_exhaustion():
    cues = (
        "<pause>",
        "SFX: door slams",
        "Door slams.",
        "The telephone rings.",
        "Pause.",
        "Silence.",
        "Music swells.",
        "He laughs.",
        "laughs",
        "door creaks",
        "Sound of thunder.",
        "He opens the door.",
        "She crosses the room.",
        "They turn off the radio.",
        "Thunder rolls.",
        "Wind howls.",
        "A knock at the door.",
        "Footsteps approach.",
    )

    for cue in cues:
        assert is_non_dialogue_stage_line(cue)

        def stubborn_writer(
            messages, *, temperature, max_new_tokens, stop=None,
        ):
            return cue

        result = repair_existing_spoken_line(
            text=cue,
            req=_req(),
            creative_fn=stubborn_writer,
        )
        assert result.text.strip()
        assert not is_non_dialogue_stage_line(result.text)
        assert (
            "hygiene_repaired_after_reroll:stage_direction:"
            "deterministic_floor"
        ) in result.compose_flags

    assert not is_non_dialogue_stage_line("I heard the door slam.")
    assert not is_non_dialogue_stage_line("He knows the code.")
    assert not is_non_dialogue_stage_line("She says the code is wrong.")


def test_own_name_action_narration_is_scoured_and_floored_as_dialogue():
    req = replace(_req(), speaker="EDNA")
    dirty = "Edna stares at the sealed vial."
    flags = spoken_surface_flags_for_line(dirty, req)
    assert any(code == "stage_direction" for code, _reason, _flag in flags)
    assert detect_stage_business_for_reroll(dirty, "EDNA")[0]
    assert is_non_dialogue_stage_line(dirty, "EDNA")

    repaired = deterministic_hygiene_floor(
        dirty, ("stage_direction",), speaker_name="EDNA",
    )
    assert repaired.text
    assert not is_non_dialogue_stage_line(repaired.text, "EDNA")


def test_delivery_backstop_uses_cast_name_for_own_action_narration():
    ledger = {
        "cast": [{"char_id": "c01", "name": "Edna"}],
        "lines": [{
            "line_id": "l001",
            "char_id": "c01",
            "speaker_role": "character",
            "text": "Edna stares at the sealed vial.",
            "skip": False,
        }],
        "meta": {},
    }
    stamp_text_for_tts_delivery(ledger)
    line = ledger["lines"][0]
    assert line["text"] == "Edna stares at the sealed vial."
    assert line["text_for_tts"] != line["text"]
    assert line["text_for_tts_receipt"]["delivery_hygiene_emergency"] is True
    assert "stage_direction" in (
        line["text_for_tts_receipt"]["delivery_hygiene_gates"]
    )


@pytest.mark.parametrize(
    ("long_line", "expected"),
    (
        (
            "Your body is screaming for a reprieve, Shannon. If you keep "
            "pushing for this level of clarity, your muscles will collapse "
            "before dawn.",
            "Your body is screaming for a reprieve, Shannon.",
        ),
        (
            "But you're asking for a life of eternal vigilance. Is a sharp "
            "mind worth a body that feels like a prison every morning?",
            "But you're asking for a life of eternal vigilance.",
        ),
        (
            "Then let the furnace burn. I would rather see the world clearly "
            "than drown in a fog I cannot wake from before morning.",
            "Then let the furnace burn.",
        ),
    ),
)
def test_one_breath_floor_keeps_complete_sentences_never_token_fragments(
    long_line,
    expected,
):
    repaired = deterministic_hygiene_floor(long_line, ("one_breath",))
    assert repaired.text == expected
    assert not is_truncated(repaired.text)
    assert not flag_one_breath(repaired.text)[0]


def test_content_owned_projection_repairs_exact_normalized_tts_surface():
    raw = "Read 999999 999999 999999 999999."
    projected, _receipt = normalize_for_delivery(raw)
    assert len(raw.split()) < 28
    assert len(projected.split()) > 28
    assert flag_one_breath(projected)[0]

    def stubborn_writer(
        messages, *, temperature, max_new_tokens, stop=None,
    ):
        return raw

    result = repair_existing_spoken_line(
        text=raw,
        req=_req(),
        creative_fn=stubborn_writer,
        repair_fn=stubborn_writer,
        spoken_projection_fn=normalize_for_delivery,
    )
    final_delivery, _ = normalize_for_delivery(result.text)
    assert result.text != raw
    assert not flag_one_breath(final_delivery)[0]
    assert (
        "hygiene_repaired_after_reroll:one_breath:deterministic_floor"
        in result.compose_flags
    )


def test_clean_spoken_projection_is_byte_identical_and_makes_no_calls():
    calls = []

    def should_not_run(*args, **kwargs):
        calls.append((args, kwargs))
        return "wrong"

    text = "Doctor Vale heard two quiet signals."
    result = repair_existing_spoken_line(
        text=text,
        req=_req(),
        creative_fn=should_not_run,
        spoken_projection_fn=normalize_for_delivery,
    )
    assert result.text == text
    assert result.compose_flags == ()
    assert calls == []


@pytest.mark.parametrize(
    "source_bank_id",
    ("media_archive", "original", "public_domain", "shakespeare"),
)
def test_each_inline_bank_repairs_bare_cue_after_reroll_exhaustion(
    source_bank_id,
):
    cue = "Door slams."

    def stubborn_writer(
        messages, *, temperature, max_new_tokens, stop=None,
    ):
        return cue

    result = compose_line(
        creative_fn=stubborn_writer,
        repair_fn=stubborn_writer,
        req=_req(),
        source_bank_id=source_bank_id,
    )
    assert result.text.strip()
    assert not is_non_dialogue_stage_line(result.text)
    assert any(
        flag.startswith("hygiene_repaired_after_reroll:stage_direction:")
        for flag in result.compose_flags
    )
