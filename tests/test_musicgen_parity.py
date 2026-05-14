"""S25 acceptance tests for MusicGen parity (MG-1, MG-2, MG-4, MG-5).

Each test pins one of the contract changes from the S25 sprint so the
parity with AudioGen post-S24 stays enforced as the codebase evolves.

  MG-1 (BUG-LOCAL-211): _save_wav returns explicit bool.
  MG-2 (BUG-LOCAL-212): writeback gates wav_path on save proof.
                        Indirectly verified via MG-1 + _save_ok plumbing.
  MG-4 (BUG-LOCAL-214): _silent_audio_dict honors caller duration.
  MG-5 (BUG-LOCAL-215): NODE_CLASS_MAPPINGS aligned to OTR_ prefix.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from nodes.musicgen_theme import (
    MUSICGEN_SAMPLE_RATE,
    _save_wav,
    _silent_audio_dict,
)


def test_save_wav_returns_bool_on_success(tmp_path):
    """S25/MG-1 (BUG-LOCAL-211). _save_wav -> bool. True when
    os.replace lands the .tmp -> canonical move."""
    p = str(tmp_path / "ok.wav")
    out = _save_wav(p, np.zeros(100, dtype=np.float32), MUSICGEN_SAMPLE_RATE)
    assert out is True
    assert os.path.isfile(p)


def test_save_wav_returns_false_on_failure(tmp_path):
    """S25/MG-1. False when the write path raises (e.g. parent dir
    doesn't exist). Failure path must not crash the caller."""
    # Parent doesn't exist -> soundfile.write raises -> _save_wav
    # catches, logs, returns False.
    p = str(tmp_path / "no_such_dir" / "nested" / "ok.wav")
    out = _save_wav(p, np.zeros(100, dtype=np.float32), MUSICGEN_SAMPLE_RATE)
    assert out is False
    assert not os.path.isfile(p)


def test_silent_audio_dict_honors_duration():
    """S25/MG-4 (BUG-LOCAL-214). Prior contract was hardcoded 0.1s
    regardless of cue; the 12s opening + 8s closing cues got 100 ms
    of silence on the ImportError fallback path and broke the
    EpisodeAssembler timeline. Per-cue duration now propagates."""
    d12 = _silent_audio_dict(duration_sec=12.0)
    assert d12["sample_rate"] == MUSICGEN_SAMPLE_RATE
    assert d12["waveform"].shape == (1, 1, int(MUSICGEN_SAMPLE_RATE * 12.0))

    d_short = _silent_audio_dict(duration_sec=0.1)
    assert d_short["waveform"].shape == (1, 1, int(MUSICGEN_SAMPLE_RATE * 0.1))

    d_zero = _silent_audio_dict(duration_sec=0)
    assert d_zero["waveform"].shape == (1, 1, 0)


def test_silent_audio_dict_requires_duration():
    """The signature now requires duration_sec. Older callers passing
    only sample_rate must be updated."""
    with pytest.raises(TypeError):
        _silent_audio_dict()  # type: ignore[call-arg]


def test_node_registered_under_otr_prefix():
    """S25/MG-5 (BUG-LOCAL-215). The in-module NODE_CLASS_MAPPINGS
    is keyed on the canonical 'OTR_MusicGenTheme' (matches
    __init__.py:_NODE_MODULES registration). The legacy bare
    'MusicGenTheme' key must not appear."""
    from nodes.musicgen_theme import NODE_CLASS_MAPPINGS
    assert "OTR_MusicGenTheme" in NODE_CLASS_MAPPINGS
    assert "MusicGenTheme" not in NODE_CLASS_MAPPINGS


def test_node_display_name_has_no_placeholder():
    """S25/MG-5. The display name shouldn't carry the literal
    '[EMOJI]' placeholder string -- minimal-emoji convention."""
    from nodes.musicgen_theme import NODE_DISPLAY_NAME_MAPPINGS
    for k, v in NODE_DISPLAY_NAME_MAPPINGS.items():
        assert "[EMOJI]" not in v, (
            f"{k!r} display name has literal placeholder: {v!r}"
        )
        assert "[TODO]" not in v
        assert "[PLACEHOLDER]" not in v
