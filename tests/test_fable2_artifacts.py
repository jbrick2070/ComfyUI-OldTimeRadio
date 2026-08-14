"""Positive contract tests for the fixed-topology scifi_news_pro lane.

The lane accepts the first structurally valid complete script. Requested and
actual length are telemetry; only malformed markup opens another script call.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_scifi_fable2 as F2  # noqa: E402


PACK_PATH = (
    REPO_ROOT
    / "nodes"
    / "story_packs"
    / "scifi_news_pro"
    / "scifi_news_pro.json"
)
PIPELINES_PATH = REPO_ROOT / "nodes" / "story_packs" / "pipelines.json"

MODEL_SEAMS = {
    "fable2_dossier_system",
    "fable2_pitch_system",
    "fable2_treatment_system",
    "fable2_news_read_system",
    "fable2_script_system",
    "fable2_casting_system",
}

FIXED_PASSES = [
    "dossier",
    "pitch",
    "treatment",
    "news_read",
    "script",
    "same_story_safety_cleanup",
    "final_draft_proof",
    "casting_voices",
    "assemble",
]


def _pack() -> SimpleNamespace:
    raw = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    return SimpleNamespace(
        story_model_id=raw["story_model_id"],
        prompt_stages=raw["prompt_stages"],
    )


def _treatment() -> F2.Treatment:
    return F2.Treatment.model_validate({
        "title": "The Quiet Relay",
        "dramatic_question": "Will Sela trust the relay before dawn?",
        "setting": "a mountain relay station at night",
        "cast_shapes": [{
            "name": "SELA",
            "role": "relay keeper",
            "want": "to recover the buried transmission",
            "pressure": "the station is losing power",
            "register": "plain, quick answers",
        }],
        "turn": "The silence is the message.",
        "priced_ending": {
            "choice": "Sela sends the record onward.",
            "cost_paid": "She gives up the last reserve charge.",
        },
        "news_thread": "A survey found measurable heat before tremors.",
        "news_close_read": (
            "The source survey measured heat before tremors at Mount Etna."
        ),
    })


def test_treatment_preserves_authored_cast_name_wording():
    authored = "Dr. Sela Cross-Smythe"
    payload = _treatment().model_dump()
    payload["cast_shapes"][0]["name"] = authored

    treatment = F2.Treatment.model_validate(payload)

    assert treatment.cast_shapes[0].name == authored


def _script(dialogue: str) -> str:
    return "\n".join([
        "TITLE: The Quiet Relay",
        "MUSIC: low glass tones",
        "ANNOUNCER: A relay waits above the sleeping valley.",
        "SCENE 1: the mountain relay room",
        f"SELA: {dialogue}",
        "ANNOUNCER: The relay returns to silence before dawn.",
        "CODA: The factual source beneath this fiction is:",
        "MUSIC: one low sustained note",
        "END.",
    ])


def _malformed_script() -> str:
    rows = _script("The signal is steady now.").splitlines()
    rows.insert(5, "this line has no legal markup label")
    return "\n".join(rows)


def test_pack_and_pipeline_expose_the_exact_fixed_topology():
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    registry = json.loads(PIPELINES_PATH.read_text(encoding="utf-8"))
    pipeline = next(
        row for row in registry["pipelines"]
        if row["story_pipeline_id"] == "scifi_news_pro_multipass"
    )

    assert set(pack["prompt_stages"]) == MODEL_SEAMS
    assert set(pipeline["declared_seams"]) == MODEL_SEAMS
    assert [row["pass_id"] for row in pipeline["passes"]] == FIXED_PASSES
    assert {
        seam
        for row in pipeline["passes"]
        for seam in row.get("seam_refs", [])
    } == MODEL_SEAMS
    assert [
        row["pass_id"] for row in pipeline["passes"]
        if row["pass_id"] == "script"
    ] == ["script"]


def test_one_pitch_is_dealt_for_every_act_count():
    """2026-08-14: was `..._for_every_requested_length`.

    The word half of the envelope is gone -- `total_words` and
    `scene_word_targets` no longer exist -- so the two assertions about them
    were deleted rather than reinterpreted. The surviving point is that
    exactly one pitch is dealt whatever the episode's shape.
    """
    deck = F2._load_frame_deck()
    for act_count in (1, 3, 5, 8):
        cards, stance = F2._deal(random.Random(act_count), deck)
        envelope = F2._build_envelope(act_count)
        assert len(cards) == 1
        assert stance["name"].strip()
        assert envelope.scene_count == act_count


@pytest.mark.parametrize(
    ("act_count", "dialogue"),
    [
        (8, "Go."),
        (1, ("Signal " * 700) + "holds."),
    ],
    ids=("tiny-script-many-acts", "huge-script-one-act"),
)
def test_first_structurally_clean_script_is_the_single_story_authority(
    act_count,
    dialogue,
):
    """A wildly off-shape script is STILL the single story authority.

    2026-08-14: the ids were "far-below-guidance" / "far-above-guidance",
    because the parameter was a word request the script missed by miles.
    There is no length guidance to miss any more, so the cases now pair an
    extreme script with an extreme act count. The assertion is unchanged and
    the point is stronger: the first structurally clean draft is accepted,
    whatever its size.
    """
    calls: list[dict[str, object]] = []
    raw = _script(dialogue)

    def creative_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return raw

    accepted_raw, parsed, telemetry = F2._pass_script(
        creative_fn,
        _pack(),
        _treatment(),
        "A source digest that supplies fiction fuel.",
        F2._build_envelope(act_count),
        ["SELA"],
    )

    actual_words = F2.canonical_word_count(dialogue)
    assert accepted_raw == raw
    assert len(calls) == telemetry["attempts"] == 1
    assert len(telemetry["attempt_trace"]) == 1
    assert parsed.character_word_count == actual_words
    assert telemetry["attempt_trace"][0].character_words == actual_words

    treatment = _treatment()
    draft = F2.build_final_draft(
        accepted_raw,
        parsed,
        treatment,
        p3_attempts=tuple(telemetry["attempt_trace"]),
    )
    F2._validate_selected_final_draft(draft, treatment=treatment)
    assert draft.parsed.character_word_count == actual_words
    with pytest.raises(FrozenInstanceError):
        draft.raw_source = "changed"


@pytest.mark.parametrize(
    ("first_response", "first_outcome"),
    [
        ("", "parse_rejected"),
        (_malformed_script(), "parse_rejected"),
    ],
    ids=("empty-transport", "malformed-markup"),
)
def test_only_structural_transport_failure_opens_another_script_call(
    first_response,
    first_outcome,
):
    responses = iter((first_response, _script("The signal is steady now.")))
    calls: list[dict[str, object]] = []

    def creative_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return next(responses)

    _raw, parsed, telemetry = F2._run_markup_ladder(
        creative_fn,
        pass_id="script",
        system="Return one complete radio play in the required markup.",
        base_user="Write the complete episode now.",
        envelope=F2._build_envelope(3),
        cast_names=["SELA"],
        initial_temperature=0.75,
    )

    traces = telemetry["attempt_trace"]
    assert len(calls) == telemetry["attempts"] == 2
    assert [trace.outcome for trace in traces] == [
        first_outcome,
        "accepted",
    ]
    assert traces[-1].character_words == parsed.character_word_count
    for call in calls:
        roles = [row["role"] for row in call["messages"]]
        assert roles == ["system", "user"]
