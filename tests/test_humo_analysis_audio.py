"""Wave 1 / 1g -- HuMo 16 kHz mono ANALYSIS clone contract (plan S0.1).

The master episode_audio is fanned to HuMo AND SignalLost and may be stereo at
its native rate (I-5); HuMo must build an independent 16 kHz mono analysis copy
for AudioEncoderEncode WITHOUT mutating the master. These tests prove the clone
(mutating the analysis never touches the master) and the master-vs-analysis
contract.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

from nodes.batch_humo_render import HUMO_ANALYSIS_SR, _humo_analysis_audio

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_episode_audio_master_vs_humo_analysis_contract():
    # Master: stereo, native 48 kHz, distinct L/R values.
    wf = torch.zeros(1, 2, 4800)
    wf[0, 0, :] = 1.0
    wf[0, 1, :] = 3.0
    master = {"waveform": wf, "sample_rate": 48000}

    analysis = _humo_analysis_audio(master)

    # Analysis: 16 kHz mono.
    assert analysis["sample_rate"] == HUMO_ANALYSIS_SR == 16000
    assert analysis["waveform"].dim() == 3
    assert analysis["waveform"].shape[0] == 1 and analysis["waveform"].shape[1] == 1
    # Resampled length tracks the 16k/48k ratio (4800 -> ~1600).
    assert abs(int(analysis["waveform"].shape[-1]) - 1600) <= 5
    # Downmix is the channel mean (1.0, 3.0) -> 2.0.
    assert abs(float(analysis["waveform"].mean()) - 2.0) < 0.05

    # MASTER IS UNTOUCHED: still stereo, 48 kHz, original values (NOT pinned mono).
    assert master["sample_rate"] == 48000
    assert tuple(master["waveform"].shape) == (1, 2, 4800)
    assert torch.allclose(master["waveform"][0, 0], torch.ones(4800))
    assert torch.allclose(master["waveform"][0, 1], torch.full((4800,), 3.0))


def test_clone_mutation_proves_the_clone():
    # Even on the no-resample path (already 16 kHz mono), the analysis buffer is
    # a CLONE: mutating it must not reach back into the master.
    master = {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}
    analysis = _humo_analysis_audio(master)
    assert analysis["waveform"] is not master["waveform"]
    analysis["waveform"].add_(5.0)
    assert int(torch.count_nonzero(master["waveform"])) == 0, "master was aliased/mutated"


def test_already_16k_mono_passthrough_shape():
    master = {"waveform": torch.ones(1, 1, 800), "sample_rate": 16000}
    analysis = _humo_analysis_audio(master)
    assert analysis["sample_rate"] == 16000
    assert tuple(analysis["waveform"].shape) == (1, 1, 800)
    assert torch.allclose(analysis["waveform"], torch.ones(1, 1, 800))


def test_empty_waveform_is_safe():
    master = {"waveform": torch.zeros(1, 2, 0), "sample_rate": 48000}
    analysis = _humo_analysis_audio(master)
    assert analysis["sample_rate"] == 16000
    assert tuple(analysis["waveform"].shape) == (1, 1, 0)


def test_two_dim_input_is_promoted_and_monoed():
    master = {"waveform": torch.zeros(2, 320), "sample_rate": 16000}  # [C, T]
    analysis = _humo_analysis_audio(master)
    assert analysis["waveform"].dim() == 3
    assert analysis["waveform"].shape[1] == 1  # mono


def test_analysis_runs_on_cpu():
    master = {"waveform": torch.zeros(1, 1, 320), "sample_rate": 16000}
    analysis = _humo_analysis_audio(master)
    assert analysis["waveform"].device.type == "cpu"  # I-11


def test_never_humo_roles_are_skipped_upstream():
    # Document the contract: music / announcer / sfx never reach the analysis
    # path (they are skipped at the render loop via is_never_humo_role).
    from nodes.batch_humo_render import is_never_humo_role, SPEAKER_ROLE_CHARACTER

    assert is_never_humo_role("announcer") is True
    assert is_never_humo_role("music_open") is True
    assert is_never_humo_role("sfx") is True
    assert is_never_humo_role(SPEAKER_ROLE_CHARACTER) is False


def test_source_has_no_em_dash():
    # CLAUDE.md rule: no em-dash in OTR python source (cp1252 subprocess decode
    # trap). The file carries other unicode in pre-existing comments; only the
    # em-dash is forbidden. The 1g additions are ASCII by construction.
    src = (REPO_ROOT / "nodes" / "batch_humo_render.py").read_text(encoding="utf-8")
    assert "—" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
