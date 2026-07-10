"""Meta-split chunk 1 (operator directive 2026-07-09) -- the produced-story
summary pass.

`meta` was always meant to carry a brief of the ACTUAL produced story for
downstream consumer prompts; the pre-generation interpreter digest
(`meta["news"]`) got conflated with it. K.5.5's reflection is deliberately a
mood board (content gate bans plot/names), so this NEW K.5.6 pass summarizes
the composed episode itself -- real names + plot + stated ending -- under the
distinct key `meta["produced_story"]`.

Pins: the input builder keeps real names; the content gate's accept/reject
classes; the success/failure delta shapes; and the music-prompt last-ditch
fallback preferring the produced logline over the source digest.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_story_brief as SB  # noqa: E402


def _ledger() -> dict:
    lines = []
    lines.append({"speaker_role": "announcer",
                  "speaker": "ANNOUNCER",
                  "text": "Tonight, a reel that would not stay shelved."})
    for i in range(24):
        who = ("Mara Voss", "Tom Hale")[i % 2]
        lines.append({
            "speaker_role": "character",
            "speaker": who.split()[0].upper(),
            "text": f"Line {i}: the label says 1931 but the film disagrees.",
        })
    lines.append({"speaker_role": "announcer",
                  "speaker": "ANNOUNCER",
                  "text": "The archive keeps its own hours."})
    return {
        "meta": {"episode_title": "The Brittle Reel", "style": "noir_archive"},
        "cast": [
            {"name": "Mara Voss", "character_description": "an archivist"},
            {"name": "Tom Hale", "character_description": "a projectionist"},
        ],
        "lines": lines,
    }


# ---------------------------------------------------------------------------
# Input builder -- real names intact, windowed excerpts
# ---------------------------------------------------------------------------


def test_builder_keeps_real_names_and_windows():
    text = SB._build_produced_story_input(_ledger())
    assert "Mara Voss" in text            # cast roster, unscrubbed
    assert "CAST:" in text
    assert "OPENING (" in text
    assert "CLOSING (" in text
    assert "character_a" not in text      # the reflection scrub must NOT run
    assert "source_entity" not in text


def test_builder_short_episode_has_no_middle_window():
    led = _ledger()
    led["lines"] = led["lines"][:6]
    text = SB._build_produced_story_input(led)
    assert "OPENING (" in text
    assert "MIDDLE (" not in text
    assert "CLOSING (" not in text


# ---------------------------------------------------------------------------
# Content gate
# ---------------------------------------------------------------------------


def _model(logline: str, subject: str = "a stubborn archive reel"):
    return SB.ProducedStoryModel(logline=logline, subject=subject)


GOOD_LOGLINE = (
    "Archivist Mara Voss and projectionist Tom Hale trace a mislabeled "
    "reel to its true year, and the archive closes with the record set right."
)


def test_gate_accepts_a_grounded_plot_logline():
    assert SB._validate_produced_story(_model(GOOD_LOGLINE), _ledger()) is None


def test_gate_rejects_markup_and_brackets_and_labels():
    led = _ledger()
    assert "has_markup" in SB._validate_produced_story(
        _model('Mara said "never" and walked out of the archive forever.'),
        led,
    )
    assert "has_bracket" in SB._validate_produced_story(
        _model("Mara Voss finds the reel [SFX: hum] and saves the archive."),
        led,
    )
    assert "speaker_labeled" in SB._validate_produced_story(
        _model("MARA: the label says 1931 but the film itself disagrees."),
        led,
    )


def test_gate_rejects_a_logline_that_names_nobody():
    reasons = SB._validate_produced_story(
        _model(
            "A keeper and a stranger argue over a light that blinks a "
            "name until dawn settles the question."
        ),
        _ledger(),
    )
    assert reasons is not None and "names_no_cast_member" in reasons


def test_gate_stopword_cast_name_cannot_trivially_pass():
    """Local-fanout QA 2026-07-09: a cast name like 'The Stranger' must not
    let a logline pass grounding just by containing 'the'."""
    led = _ledger()
    led["cast"] = [{"name": "The Stranger"}]
    rejected = SB._validate_produced_story(
        _model(
            "A keeper argues with the light over a name that blinks until "
            "dawn settles the question for good."
        ),
        led,
    )
    assert rejected is not None and "names_no_cast_member" in rejected
    accepted = SB._validate_produced_story(
        _model(
            "The Stranger walks out of the fog, trades one true name for "
            "the keeper's silence, and the light burns on."
        ),
        led,
    )
    assert accepted is None


def test_builder_exact_cap_boundary_keeps_closing_window():
    """Local-fanout QA 2026-07-09: with spoken lines exactly equal to
    OPEN+CLOSE caps, the closing window must still be emitted (the old
    strict-greater check silently dropped the whole second half)."""
    led = _ledger()
    total = (SB._PRODUCED_STORY_OPENING_CAP
             + SB._PRODUCED_STORY_CLOSING_CAP)
    led["lines"] = [
        {"speaker_role": "character", "speaker": "MARA",
         "text": f"Boundary line {i} about the reel."}
        for i in range(total)
    ]
    text = SB._build_produced_story_input(led)
    assert f"CLOSING ({SB._PRODUCED_STORY_CLOSING_CAP} lines):" in text
    assert "MIDDLE (" not in text
    # No overlap: the last opening row and first closing row differ.
    assert f"Boundary line {SB._PRODUCED_STORY_OPENING_CAP - 1}" in text
    assert f"Boundary line {SB._PRODUCED_STORY_OPENING_CAP}" in text


def test_gate_skips_name_grounding_when_cast_is_empty():
    led = _ledger()
    led["cast"] = []
    assert SB._validate_produced_story(
        _model(
            "A keeper and a stranger argue over a light that blinks a "
            "name until dawn settles the question."
        ),
        led,
    ) is None


# ---------------------------------------------------------------------------
# Entrypoint deltas
# ---------------------------------------------------------------------------


def test_success_delta_shape():
    reply = json.dumps({
        "logline": GOOD_LOGLINE,
        "subject": "a mislabeled archive reel",
    })

    def technical_fn(*args, **kwargs):
        return reply

    delta = SB.run_produced_story_summary(
        _ledger(), technical_fn, technical_model_id="stub/technical"
    )
    assert delta["produced_story_status"] == "ok"
    assert delta["produced_story"]["logline"] == GOOD_LOGLINE
    assert delta["produced_story"]["subject"] == "a mislabeled archive reel"
    assert delta["produced_story_model_id"] == "stub/technical"
    assert delta["produced_story_source"] == "llm_post_composition"


def test_failure_delta_never_raises_and_omits_the_field():
    def technical_fn(*args, **kwargs):
        raise RuntimeError("loader exploded")

    delta = SB.run_produced_story_summary(
        _ledger(), technical_fn, technical_model_id="stub/technical"
    )
    assert delta["produced_story_status"].startswith("failed:")
    assert "produced_story" not in delta


# ---------------------------------------------------------------------------
# Music-prompt last-ditch fallback prefers the produced logline
# ---------------------------------------------------------------------------


def test_music_last_ditch_mines_produced_logline(monkeypatch):
    import nodes._otr_music_prompt as MP

    seen: dict = {}

    def fake_suffix(text):
        seen["seed"] = text
        return ", tense"

    monkeypatch.setattr(MP, "_mood_suffix", fake_suffix)
    cue = next(iter(MP.CUE_DURATIONS))
    meta = {
        "produced_story": {"logline": "Mara Voss saves the brittle reel."},
        "news": {"script_brief": "source digest text"},
    }
    prompt, _dur = MP.compose_music_prompt(meta, cue)
    assert seen["seed"] == "Mara Voss saves the brittle reel."
    assert isinstance(prompt, str) and prompt


def test_music_last_ditch_floor_stays_news_for_old_ledgers(monkeypatch):
    import nodes._otr_music_prompt as MP

    seen: dict = {}

    def fake_suffix(text):
        seen["seed"] = text
        return ""

    monkeypatch.setattr(MP, "_mood_suffix", fake_suffix)
    cue = next(iter(MP.CUE_DURATIONS))
    meta = {"news": {"script_brief": "source digest text"}}
    prompt, _dur = MP.compose_music_prompt(meta, cue)
    assert seen["seed"] == "source digest text"
    assert prompt.startswith("atmospheric")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
