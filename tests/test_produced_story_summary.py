"""Meta-split chunk 1 (operator directive 2026-07-09) -- the produced-story
summary pass.

`meta` was always meant to carry a brief of the ACTUAL produced story for
downstream consumer prompts; the pre-generation interpreter digest
(`meta["news"]`) got conflated with it. K.5.5's reflection is deliberately a
mood board, so this NEW K.5.6 pass summarizes
the composed episode itself -- real names + plot + stated ending -- under the
distinct key `meta["produced_story"]`.

Pins: the input builder keeps real names; any nonblank provider-bounded summary
is accepted without semantic retry; success/failure delta shapes remain stable;
and the music fallback prefers the produced logline over the source digest.
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
    assert "CAST: Mara Voss, Tom Hale" in text
    assert "OPENING (" in text
    assert "CLOSING (" in text


def test_builder_short_episode_has_no_middle_window():
    led = _ledger()
    led["lines"] = led["lines"][:6]
    text = SB._build_produced_story_input(led)
    assert "OPENING (" in text
    assert "MIDDLE (" not in text
    assert "CLOSING (" not in text


# ---------------------------------------------------------------------------
# Non-gating schema
# ---------------------------------------------------------------------------


GOOD_LOGLINE = (
    "Archivist Mara Voss and projectionist Tom Hale trace a mislabeled "
    "reel to its true year, and the archive closes with the record set right."
)


def test_schema_accepts_arbitrary_nonblank_bounded_summary_text():
    model = SB.ProducedStoryModel(logline="x", subject="y")
    assert model.logline == "x"
    assert model.subject == "y"


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


def test_arbitrary_safe_summary_vocabulary_is_accepted_without_retry():
    calls = []

    def technical_fn(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps({
            "logline": "A keeper waits in the smoking room.",
            "subject": "one",
        })

    delta = SB.run_produced_story_summary(_ledger(), technical_fn)
    assert delta["produced_story"] == {
        "logline": "A keeper waits in the smoking room.",
        "subject": "one",
    }
    assert len(calls) == 1


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
