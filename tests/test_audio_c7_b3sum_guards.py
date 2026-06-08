"""Sprint E E14 / R-5 + R-6: audio C7 b3sum drift guards.

Two structural drift guards that lock the audio C7 byte-identity
contract surfaces. Both tests are PROXY: they assert the consumer
code paths that touch the audio bus exist with the documented shape,
without running a full GPU render (which Sprint A handles via
OTR_REGRESSION_RUNTIME=1).

R-5 HuMo audio passthrough: BatchHumoRender must NOT re-encode the
audio in the final per-line mp4 mux. The on-disk mp4 carries the
ORIGINAL audio bytes from the input AUDIO slot via FFmpeg passthrough
copy (-c:a copy or equivalent raw-PCM remux). A future codec swap
that re-encodes audio to AAC/MP3 would violate Prime Directive 1.

R-6 MusicGen C7 baseline: MusicGen's mood-prefix code path is the
ONLY legitimate baseline shift since Sprint C close. The C5g
mood-prefix logic must remain in place and continue intersecting
against the 16-term _MUSIC_MOOD_VOCAB. Any drift in the mood vocab
size or the intersection logic is a baseline-reset trigger.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_src(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


# TestHumoAudioPassthrough (R-5) DELETED in the CW cleanbreak (2026-06-08): it
# asserted on nodes/batch_humo_render.py's per-line mp4 mux, removed with the
# legacy batch render path. The terminal OTR_MasterAudioMux (-c:a copy, no
# -shortest) is the byte-identity carrier, covered by
# tests/test_video_render_path_cw4.py + tests/test_audio_byte_identical.py.


class TestMusicGenC7Baseline:
    """R-6: MusicGen mood-prefix logic is the C5g baseline-shift
    surface. The 16-term vocab and intersection logic must remain."""

    def test_music_mood_vocab_has_16_terms(self):
        from nodes._otr_story_brief_helpers import _MUSIC_MOOD_VOCAB
        assert len(_MUSIC_MOOD_VOCAB) == 16, (
            f"Music mood vocab size drift: got {len(_MUSIC_MOOD_VOCAB)}, "
            "expected 16. Any vocab change requires capturing a new "
            "C7 baseline fixture (Sprint A acceptance backlog A1)."
        )

    def test_music_mood_vocab_membership(self):
        """Pin the exact 16-term vocab so a future commit cannot
        swap one term for another silently."""
        from nodes._otr_story_brief_helpers import _MUSIC_MOOD_VOCAB
        expected = frozenset({
            "tense", "ominous", "melancholic", "hopeful", "urgent", "calm",
            "eerie", "sombre", "playful", "menacing", "wistful", "frantic",
            "reverent", "uneasy", "stoic", "yearning",
        })
        assert _MUSIC_MOOD_VOCAB == expected, (
            "Music mood vocab membership drift -- a term was renamed, "
            "added, or removed. Capture new C7 baseline before merging."
        )

    def test_music_prompt_routes_through_brief_protocol(self):
        """Audio clean-break (1c): the music cue prompt is composed in the
        single-source nodes/_otr_music_prompt.py, which reads the Meta brief via
        the brief-reader protocol (_read_brief_field) -- not a local template.
        Replaces the retired musicgen_theme._compose_music_prompt source pin.
        """
        src = _read_src("nodes/_otr_music_prompt.py")
        assert "_read_brief_field" in src
        assert "def compose_music_prompt" in src

    def test_writer_default_mistral_nemo_pinned(self):
        """The C7 baseline holds ONLY when both writer slots resolve
        to Mistral-Nemo-Instruct-2407. The writer's
        creative_writing_model + technical_model default values pin
        this -- E2's workflow JSON edit complements by ensuring the
        SHIPPED widget values match the schema default."""
        from nodes import _otr_model_catalog
        assert _otr_model_catalog.DEFAULT_LLM == (
            "mistralai/Mistral-Nemo-Instruct-2407"
        )
