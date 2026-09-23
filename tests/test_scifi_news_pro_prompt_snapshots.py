"""Load-bearing prompt shapes for the six Fable model seams."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_routing as ROUTING


@pytest.fixture(scope="module")
def stages():
    ROUTING._REGISTRY = None
    try:
        return dict(ROUTING.resolve_story_pack("scifi_news_pro").prompt_stages)
    finally:
        ROUTING._REGISTRY = None


def test_exact_six_model_seams(stages):
    assert set(stages) == {
        "scifi_news_pro_dossier_system",
        "scifi_news_pro_pitch_system",
        "scifi_news_pro_treatment_system",
        "scifi_news_pro_news_read_system",
        "scifi_news_pro_script_system",
        "scifi_news_pro_casting_system",
    }


@pytest.mark.parametrize(
    "seam",
    [
        # scifi_news_pro_dossier_system is DELIBERATELY ABSENT as of
        # 2026-08-25 -- see test_dossier_seam_requests_labelled_sections below.
        "scifi_news_pro_pitch_system",
        "scifi_news_pro_treatment_system",
        "scifi_news_pro_news_read_system",
        "scifi_news_pro_casting_system",
    ],
)
def test_structured_seams_request_one_json_artifact(stages, seam):
    assert "return one json object only" in stages[seam].lower()


def test_dossier_seam_requests_labelled_sections_not_json(stages):
    """P0 is the ONE structured seam that does not ask for JSON.

    A local 2-4B technical model failed all three ladder rungs emitting an
    unclosed JSON object (evidence:
    docs/2026-08-25-leg1-dossier-failure-evidence.md). Nesting is what a small
    model cannot hold, so this seam asks for labelled bullet sections and
    Python assembles the object -- same schema, same validation, same ladder.
    The seam is exempted from the sibling JSON invariant ABOVE rather than
    silently failing it, so the exemption is visible and deliberate.
    """
    prompt = stages["scifi_news_pro_dossier_system"]
    lowered = prompt.lower()
    assert "return one json object only" not in lowered
    assert "labelled sections" in lowered
    for header in ("FACTS:", "NUMBERS:", "PEOPLE:",
                   "PLACES:", "THINGS:", "VECTORS:"):
        assert header in prompt, f"dossier seam lost the {header} header"


def test_script_seam_owns_complete_plain_text_grammar(stages):
    prompt = stages["scifi_news_pro_script_system"]
    for marker in (
        "TITLE: <episode title>",
        # The MUSIC placeholder stopped asking for instruments on 2026-09-13.
        # The instruments and the tempo come from the show's own palette
        # (`_otr_music_palette`), so a cue that named its own fought them -- the
        # lesson this lane's own format example already carried ("MUSIC: the
        # theme, up and under", with a comment recording that naming
        # instruments over a TR-909 palette made the engine resolve toward the
        # strings). What the seam asks for is the FEELING; this test still pins
        # that the grammar teaches a MUSIC line at all, which is its job.
        "MUSIC: <the feeling of the moment, a few words, no instruments>",
        "SCENE <n>: <concrete setting>",
        "ANNOUNCER: <spoken words>",
        "<CAST NAME>: <spoken words>",
        "CODA: <spoken coda>",
        "END.",
    ):
        assert marker in prompt
    assert "Write the whole episode once" in prompt
    assert "requested duration is loose generation guidance" in prompt


def test_pitch_seam_describes_the_direct_pitch_artifact(stages):
    prompt = stages["scifi_news_pro_pitch_system"]
    for field in (
        "frame_card", "logline", "hook", "scifi_device", "cast_size",
        "ending_shape",
    ):
        assert field in prompt
    assert "dossier facts" in prompt.lower()


def test_casting_seam_assigns_voices_without_rewriting_story(stages):
    prompt = stages["scifi_news_pro_casting_system"]
    assert "available voice-stock id" in prompt.lower()
    assert "copy names and voice-stock ids exactly" in prompt.lower()
    assert "do not rewrite story text" in prompt.lower()


def test_the_cast_ceiling_in_the_seams_is_the_real_number(stages):
    """The seams used to say "N_MAX", which nothing ever substituted.

    So the model read the characters N_MAX as its cast ceiling while
    MAX_SPEAKING_CAST was what actually bound, in pydantic. Fixed
    2026-09-23 by writing the real number into both seams; this test
    exists because a hardcoded number in a pack can drift away from the
    constant it mirrors, and nothing else would notice.
    """
    from nodes._otr_scifi_news_pro import MAX_SPEAKING_CAST

    for name, text in stages.items():
        if not isinstance(text, str):
            continue
        assert "N_MAX" not in text, (
            "%s ships an unsubstituted placeholder to the model" % name)
    pitch = stages["scifi_news_pro_pitch_system"]
    treatment = stages["scifi_news_pro_treatment_system"]
    assert "1 through %d" % MAX_SPEAKING_CAST in pitch, pitch
    assert "no more than %d cast members" % MAX_SPEAKING_CAST in treatment
