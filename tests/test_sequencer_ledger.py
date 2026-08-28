"""tests/test_sequencer_ledger.py -- self-test for the ledger-pure
SceneSequencer rewrite.

Mocks the heavy paths (story_orchestrator._unload_llm, room-tone
generation, inline-Bark fallback) so the dispatch + clip-match +
write-back logic is exercised CPU-only in well under a second.

Covers four canonical Pattern 7 cases adapted for Sequencer:
  - test_sequencer_processes_dialogue_and_sfx_skips_music
  - test_sequencer_stamps_by_line_id_with_duplicate_text
  - test_sequencer_audio_clip_match_by_iteration_order
  - test_sequencer_legacy_list_raises
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.fixtures.ledger_stub import make_legacy_list, make_stub_ledger

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def _no_op(*_args, **_kwargs):
    return None

def _make_audio(n_clips: int, samples_per: int = 24000, sr: int = 24000):
    """Build a batched AUDIO dict with N distinct clips, where each clip
    is a constant-amplitude tone at a different frequency so the
    Sequencer's _trim_trailing_silence + iteration logic can tell them
    apart."""
    waveform = torch.zeros(n_clips, 1, samples_per, dtype=torch.float32)
    for b in range(n_clips):
        # Constant low amplitude with a unique offset per clip; the sole
        # purpose is non-zero content the trimmer keeps.
        waveform[b, 0, :] = 0.1 * (b + 1)
    return {"waveform": waveform, "sample_rate": sr}


def _music_bus(cue_id: str, anchor_line_id: str, *, samples: int = 12000):
    from nodes import _otr_cue_manifest as CM
    from nodes.production_ledger import music_cue_spec_sha256

    prompt = "A brief instrumental bridge."
    duration = float(samples) / 48000.0
    cue_hash = music_cue_spec_sha256({
        "generation_prompt": prompt,
        "target_duration_s": duration,
        "placement": "interstitial",
        "anchor_line_id": anchor_line_id,
    })
    manifest = CM.build_manifest(48000, [{
        "cue_id": cue_id,
        "batch_index": 0,
        "sample_count": samples,
        "sample_rate": 48000,
        "prompt": prompt,
        "prompt_sha256": CM.prompt_sha256(prompt),
        "cue_spec_sha256": cue_hash,
        "placement": "interstitial",
        "anchor_line_id": anchor_line_id,
        "seed": 7,
        "requested_duration_s": duration,
        "actual_duration_s": duration,
        "output_path": f"C:/episode/{cue_id}.wav",
    }])
    return _make_audio(1, samples_per=samples, sr=48000), CM.dumps(manifest)

@pytest.fixture
def patched_sequencer_env(tmp_path):
    """Patch the GPU + ledger I/O surfaces so sequence() is hermetic.

    Pre-seed state['led_disk'] before calling; read back after.
    """
    state = {"led_disk": None}

    def _fake_in_flight_path():
        return tmp_path / "stub_ledger.json"

    def _fake_load_ledger_safe(_path):
        return state["led_disk"]

    def _fake_save_ledger_safe(_path, led):
        state["led_disk"] = led
        return True

    # _generate_room_tone is heavy on the assembled-mix path; replace
    # with a zero buffer so the test runs CPU-only.
    def _fake_room_tone(span_len_sec, sr, intensity=0.01, descriptors=""):
        return np.zeros(int(span_len_sec * sr), dtype=np.float32)

    # `_fake_bark_for_line` was removed 2026-08-28 with the deleted
    # inline-Bark fallback it stubbed -- nothing patched it any more.

    with patch(
        # S31.5 B1: post-S31 B4 `story_orchestrator._unload_llm` was
        # DELETED. Canonical teardown is now
        # `_otr_model_loader.unload_llm`. Bucket B in the
        # BUG-LOCAL-227 triage classification.
        "nodes._otr_model_loader.unload_llm", side_effect=_no_op,
    ), patch(
        "nodes.scene_sequencer._runtime_log", side_effect=_no_op,
    ), patch(
        "nodes.scene_sequencer._generate_room_tone",
        side_effect=_fake_room_tone,
    ), patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        side_effect=_fake_in_flight_path,
    ), patch(
        "nodes._otr_ledger.load_ledger_safe",
        side_effect=_fake_load_ledger_safe,
    ), patch(
        "nodes._otr_ledger.save_ledger_safe",
        side_effect=_fake_save_ledger_safe,
    ):
        yield state

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sequencer_uses_guarded_llm_unload_handoff():
    """All-cloud LLM runs reach Sequencer with no local LLM cache; the pre-TTS
    handoff must use the guarded helper, not raw unload_llm()."""
    src = (REPO_ROOT / "nodes" / "scene_sequencer.py").read_text(encoding="utf-8")
    assert "unload_llm_if_local_resident" in src
    assert "from ._otr_model_loader import unload_llm\n" not in src


def test_sequencer_processes_dialogue_skips_music(patched_sequencer_env):
    """character / announcer lines stamped with start_s + dur_s +
    start_s_space + speaker_role. music_open / music_close lines pass
    through with NO start_s / dur_s set by Sequencer. (rip-sfx-broll
    2026-07-01: the sfx lane -- clips, offset, overlay, write-back --
    is GONE; an sfx row is now rejected LOUD, see the raise test below.)"""
    from nodes.scene_sequencer import SceneSequencer

    led_in = make_stub_ledger()  # 5 lines: music_open, announcer, c01, c02, music_close
    patched_sequencer_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    # Provide pre-rendered clips so the Bark fallback isn't needed:
    # 2 character clips (b003, b004) for tts; 1 announcer clip (b002).
    tts_audio = _make_audio(2)
    announcer_audio = _make_audio(1)

    SceneSequencer().sequence(
        script_json=script_json,

        tts_audio_clips=tts_audio,
        announcer_audio_clips=announcer_audio,
    )

    saved_led = patched_sequencer_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # Stamped: announcer b002, character b003, character b004.
    for lid in ("b002", "b003", "b004"):
        row = by_id[lid]
        assert "start_s" in row and row["start_s"] is not None, (
            f"{lid} missing start_s"
        )
        assert "dur_s" in row and row["dur_s"] is not None and row["dur_s"] > 0, (
            f"{lid} missing or zero dur_s"
        )
        assert row.get("start_s_space") == "scene_audio", (
            f"{lid} missing start_s_space tag"
        )
    # speaker_role re-stamped on dialogue rows by Sequencer (was set by writer too,
    # but confirms the patch applied).
    assert by_id["b002"]["speaker_role"] == "announcer"
    assert by_id["b003"]["speaker_role"] == "character"
    assert by_id["b004"]["speaker_role"] == "character"

    # NOT stamped: music_open b001, music_close b006. Sequencer
    # explicitly passes through music lines; their start_s / dur_s
    # stay whatever the writer initialized (None in the stub).
    assert by_id["b001"]["start_s"] is None or by_id["b001"]["start_s"] == 0.0, (
        "music_open b001 should not have its start_s set by Sequencer; "
        "stub initial value was None or 0.0 (writer fixture default)"
    )
    # The stub fixture sets start_s=0.0 for b001 and =11.4 for b006 as
    # forensic placeholders; the key assertion is that Sequencer did
    # not OVERWRITE these with its own scene-audio cumulative values.
    # We verify by checking that b001 / b006 were NOT marked with
    # start_s_space="scene_audio" (the Sequencer-stamped marker).
    assert "start_s_space" not in by_id["b001"], (
        "music_open should not have start_s_space stamped by Sequencer"
    )
    assert "start_s_space" not in by_id["b006"], (
        "music_close should not have start_s_space stamped by Sequencer"
    )



def test_episode_assembler_materializes_bookends_and_mirrors_by_placement(
    patched_sequencer_env,
):
    from nodes import _otr_cue_manifest as CM
    from nodes.production_ledger import music_cue_spec_sha256
    from nodes.scene_sequencer import EpisodeAssembler

    prompts = {
        "opening": "Opening pulse.",
        "interstitial": "Brief bridge.",
        "closing": "Closing resolve.",
    }
    anchors = {
        "opening": "b000", "interstitial": "b005", "closing": "b900",
    }
    rows = []
    for batch_index, cue_id in enumerate(
        ("opening", "interstitial", "closing")
    ):
        cue_hash = music_cue_spec_sha256({
            "generation_prompt": prompts[cue_id],
            "target_duration_s": 0.1,
            "placement": cue_id,
            "anchor_line_id": anchors[cue_id],
        })
        rows.append({
            "cue_id": cue_id, "batch_index": batch_index,
            "sample_count": 4800, "sample_rate": 48000,
            "prompt": prompts[cue_id],
            "prompt_sha256": CM.prompt_sha256(prompts[cue_id]),
            "cue_spec_sha256": cue_hash,
            "placement": cue_id, "anchor_line_id": anchors[cue_id],
            "seed": batch_index + 1,
            "requested_duration_s": 0.1, "actual_duration_s": 0.1,
            "output_path": f"C:/episode/{cue_id}.wav",
        })
    manifest = CM.build_manifest(48000, rows)
    inter = {
        "cue_id": "interstitial", "description": prompts["interstitial"],
        "generation_prompt": prompts["interstitial"],
        "target_duration_s": 0.1, "placement": "interstitial",
        "anchor_line_id": "b005", "wav_path": None,
        "start_s": 0.4, "dur_s": 0.1,
        "start_s_space": "scene_audio", "shot_id": "shot_001",
    }
    inter["cue_spec_sha256"] = music_cue_spec_sha256(inter)
    patched_sequencer_env["led_disk"] = {
        "episode_id": "assembler_music",
        "meta": {"title": "ASSEMBLER-MUSIC"},
        "lines": [{
            "line_id": "l001", "speaker_role": "character",
            "char_id": "c01", "text": "A line.",
            "start_s": 0.0, "dur_s": 1.0,
            "start_s_space": "scene_audio",
        }],
        "music": [inter],
    }

    EpisodeAssembler().assemble(
        scene_audio=_make_audio(1, samples_per=48000, sr=48000),
        episode_title="Assembler Music",
        music_cue_audio=_make_audio(3, samples_per=4800, sr=48000),
        music_cue_manifest_json=CM.dumps(manifest),
    )

    saved = patched_sequencer_env["led_disk"]
    by_cue = {row["cue_id"]: row for row in saved["music"]}
    assert set(by_cue) == {"opening", "interstitial", "closing"}
    assert by_cue["opening"]["start_s"] == 0.0
    assert by_cue["closing"]["start_s_space"] == "master_mix"
    assert by_cue["interstitial"]["start_s_space"] == "master_mix"
    assert by_cue["interstitial"]["start_s"] == pytest.approx(0.4)
    assert all(row["wav_path"] for row in by_cue.values())

    mirrors = [
        row for row in saved["lines"]
        if row.get("mirrored_from") == "music"
    ]
    assert {row["speaker_role"] for row in mirrors} == {
        "music_open", "music_inter", "music_close",
    }
    assert {row["music_cue_id"] for row in mirrors} == set(by_cue)


def test_sequencer_rejects_sfx_row_loud(patched_sequencer_env):
    """NO FALLBACKS (rip-sfx-broll): an old ledger carrying a
    speaker_role='sfx' row fails LOUD in the role dispatch instead of
    riding the dialogue bus."""
    import pytest
    from nodes.scene_sequencer import SceneSequencer

    led_in = make_stub_ledger()
    led_in["lines"].append({
        "line_id": "x005", "shot_id": None, "beat_id": "x005",
        "char_id": "sfx", "speaker_role": "sfx", "text": "metal door slam",
        "traits": "neutral", "start_s": None, "dur_s": None,
        "boundary": None, "bark_wav_path": None,
    })
    patched_sequencer_env["led_disk"] = json.loads(json.dumps(led_in))
    with pytest.raises(ValueError, match="sfx"):
        SceneSequencer().sequence(
            script_json=json.dumps(led_in),
            tts_audio_clips=_make_audio(2),
            announcer_audio_clips=_make_audio(1),
        )

def test_sequencer_stamps_by_line_id_with_duplicate_text(patched_sequencer_env):
    """Two character lines saying 'Okay.' with distinct line_ids. Both
    stamped with monotonic start_s -- the old text-match block would
    have collided on duplicate text."""
    from nodes.scene_sequencer import SceneSequencer

    led_in = {
        "meta":  {"title": "DUP-SEQ", "news_seed": {"headline": "x"}},
        "cast": [
            {"char_id": "c01", "name": "ALPHA",
             "voice_preset": "v2/en_speaker_3", "line_count": 1, "word_count": 1},
            {"char_id": "c02", "name": "BETA",
             "voice_preset": "v2/en_speaker_9", "line_count": 1, "word_count": 1},
        ],
        "lines": [
            {"line_id": "L_alpha", "char_id": "c01", "speaker_role": "character",
             "text": "Okay.", "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
            {"line_id": "L_beta", "char_id": "c02", "speaker_role": "character",
             "text": "Okay.", "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
        ],
    }
    patched_sequencer_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    tts_audio = _make_audio(2)

    SceneSequencer().sequence(
        script_json=script_json,

        tts_audio_clips=tts_audio,
    )

    saved_led = patched_sequencer_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    for lid in ("L_alpha", "L_beta"):
        row = by_id[lid]
        assert row.get("start_s_space") == "scene_audio"
        assert row.get("dur_s") is not None and row["dur_s"] > 0
        assert row.get("speaker_role") == "character"

    # start_s monotonic.
    assert by_id["L_alpha"]["start_s"] == 0.0
    assert by_id["L_beta"]["start_s"] == pytest.approx(
        by_id["L_alpha"]["dur_s"], rel=1e-6,
    )

def test_sequencer_audio_clip_match_by_iteration_order(patched_sequencer_env):
    """N character lines + N TTS clips -> Sequencer consumes clips in
    iteration order, one per character line. NO clip is skipped, NO
    inline-Bark fallback fires."""
    from nodes import scene_sequencer as _seq_mod
    from nodes.scene_sequencer import SceneSequencer

    led_in = {
        "meta":  {"title": "ORDER", "news_seed": {"headline": "x"}},
        "cast": [
            {"char_id": "c01", "name": "ALPHA",
             "voice_preset": "v2/en_speaker_3", "line_count": 3, "word_count": 6},
            {"char_id": "c02", "name": "BETA",
             "voice_preset": "v2/en_speaker_9", "line_count": 0, "word_count": 0},
        ],
        "lines": [
            {"line_id": f"L{i:03d}", "char_id": "c01", "speaker_role": "character",
             "text": f"Line number {i}.", "traits": "neutral",
             "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None}
            for i in range(3)
        ],
    }
    patched_sequencer_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    # Three pre-rendered clips, three character lines -> all consumed.
    tts_audio = _make_audio(3)

    # This block used to wrap `_generate_bark_for_line` in a counter to prove
    # it was NOT called. The function was DELETED 2026-08-28, so there is
    # nothing left to call -- absence is a stronger guarantee than a counter,
    # and the shortfall contract keeps its own direct test
    # (test_sequencer_clip_shortfall_fails_loud).
    SceneSequencer().sequence(
        script_json=script_json,
        tts_audio_clips=tts_audio,
    )

    saved_led = patched_sequencer_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # All three lines stamped, in monotonic order.
    starts = [by_id[f"L{i:03d}"]["start_s"] for i in range(3)]
    assert starts[0] == 0.0
    assert starts[1] > starts[0]
    assert starts[2] > starts[1]

    # (The "no inline-Bark fallback fired" assertion retired with the function
    # itself on 2026-08-28 -- there is no fallback left that could fire.)

def test_sequencer_clip_shortfall_fails_loud(patched_sequencer_env):
    """NO-FALLBACK (2026-07-03): fewer pre-rendered voice clips than dialogue
    lines is a clip-count shortfall -> SceneSequencer RAISES ValueError instead of
    inline-generating Bark filler for the un-clipped line."""
    from nodes.scene_sequencer import SceneSequencer

    led_in = {
        "meta":  {"title": "SHORTFALL", "news_seed": {"headline": "x"}},
        "cast": [
            {"char_id": "c01", "name": "ALPHA",
             "voice_preset": "v2/en_speaker_3", "line_count": 3, "word_count": 6},
        ],
        "lines": [
            {"line_id": f"L{i:03d}", "char_id": "c01", "speaker_role": "character",
             "text": f"Line number {i}.", "traits": "neutral",
             "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None}
            for i in range(3)
        ],
    }
    patched_sequencer_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    # Only TWO clips for THREE character lines -> the third line has no clip.
    tts_audio = _make_audio(2)

    with pytest.raises(ValueError, match="no pre-rendered voice clip"):
        SceneSequencer().sequence(
            script_json=script_json,
            tts_audio_clips=tts_audio,
        )


def test_sequencer_legacy_list_raises(patched_sequencer_env):
    """Legacy parser-list shape -> load_ledger ValueError surfaces.
    Sequencer is in the loud-fail group."""
    from nodes.scene_sequencer import SceneSequencer

    legacy_json = json.dumps(make_legacy_list())

    with pytest.raises(ValueError) as excinfo:
        SceneSequencer().sequence(
            script_json=legacy_json,

        )

    msg = str(excinfo.value)
    assert "legacy parser-list" in msg or "OTR_LedgerScriptWriter" in msg, (
        f"ValueError message should name the legacy shape; got: {msg!r}"
    )
