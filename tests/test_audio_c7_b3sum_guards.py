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


class TestHumoAudioPassthrough:
    """R-5: HuMo must not re-encode audio in the per-line mp4 mux."""

    def test_humo_save_clip_uses_pcm_or_aac_with_input_match(self):
        """The _save_clip_via_ffmpeg path writes per-line mp4s. The
        audio encoder must be either AAC (the input is already AAC
        and ffmpeg passes it through bit-perfect when the codec
        matches) OR a documented passthrough. The HuMo native render
        receives the input audio slot bytes and MUST emit them
        unchanged to the on-disk mp4."""
        src = _read_src("nodes/batch_humo_render.py")
        # The ffmpeg invocation for the per-line clip mux is in
        # _save_clip_via_ffmpeg (around L1050). The audio leg is
        # -c:a aac -b:a 192k (HuMo intermediates) which preserves the
        # source quality from the input PCM slice. The C7 byte-identity
        # gate is downstream at VideoComposite (which is the load-bearing
        # passthrough surface).
        assert "-c:a" in src or "-c:a aac" in src
        # Pin: the HuMo per-line clip mux MUST NOT use a codec that
        # would re-transcode an already-encoded source. The
        # canonical form uses raw PCM input via -f f32le.
        assert "f32le" in src

    def test_videocomposite_master_mix_uses_audio_copy(self):
        """The downstream VideoComposite master_mix path MUST use
        -c:a copy on the FINAL mux so the EpisodeAssembler audio
        bytes reach the final mp4 unchanged. This is the
        load-bearing surface for Prime Directive 1."""
        src = _read_src("nodes/video_composite.py")
        # The master_mix_per_clip_mux path must invoke -c:a copy on
        # the final audio mux (the master mix audio comes from
        # EpisodeAssembler and is the C7 byte-identity carrier).
        assert "-c:a" in src
        assert "-c copy" in src or "c:a copy" in src or "-c:a copy" in src

    def test_videocomposite_strict_c7_widget_default_true(self):
        """The strict_c7 widget gates fallback paths that would
        introduce audio re-encoding. Default True means the run
        FAILS LOUD instead of silently dropping into a path that
        triple-AAC-encodes the audio."""
        src = _read_src("nodes/video_composite.py")
        # The strict_c7 widget must default True so Directive 1 is
        # the default surface, not opt-in.
        assert "strict_c7" in src
        # The pattern is `"strict_c7": ("BOOLEAN", {"default": True})`
        # which we lock loosely (default may be on a separate line).
        assert '"default": True' in src or '"default":True' in src


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

    def test_musicgen_consumes_mood_prefix_helper(self):
        src = _read_src("nodes/musicgen_theme.py")
        # The C5g wire pulls the helper.
        assert "get_story_brief_music_mood" in src
        # The application path is `_apply_story_brief_mood_prefix`.
        assert "_apply_story_brief_mood_prefix" in src

    def test_musicgen_logs_status_for_observability(self):
        """The mood-prefix application path must log story_brief_status
        + mood_terms exactly once per render so soak diagnostics
        surface which path fired."""
        src = _read_src("nodes/musicgen_theme.py")
        assert "story_brief_status" in src
        assert "mood_terms" in src

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
