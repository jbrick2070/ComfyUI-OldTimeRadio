"""An interstitial (act-break bridge) cue must reach ``<ep>_master.wav``.

SceneSequencer leaves a silent gap for every ``music_inter`` line and the music
node renders a cue for it. EpisodeAssembler used to mix that cue AFTER it had
written and hashed the master, so the cue reached only the returned tensor
(read by OTR_SignalLostVideo) while the WAV that OTR_MasterAudioMux publishes
and the video lanes condition on kept the silence. These tests drive the real
``assemble`` and read the file it wrote. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes import _otr_cue_manifest as CM  # noqa: E402
from nodes import _otr_ledger as oled  # noqa: E402
from nodes import scene_sequencer as sc  # noqa: E402

SR = 48000
XFADE = int(SR * 500 / 1000.0)  # assemble's default crossfade_ms=500
GAP_START_S = 3.0
GAP_S = 4.0
FOLEY_POLICY = json.dumps(
    {"effective_video_models": {"character_video": "ltx25_foley_16gb"}})


def _tone(seconds, freq, amp, channels=2):
    t = torch.arange(int(seconds * SR), dtype=torch.float32) / SR
    return (amp * torch.sin(2 * math.pi * freq * t)).unsqueeze(0).repeat(channels, 1)


def _scene_with_gap():
    """3 s of tone, the 4 s silent gap SceneSequencer inserts for a bridge,
    then 3 s of tone."""
    return torch.cat([_tone(GAP_START_S, 220.0, 0.2),
                      torch.zeros(2, int(GAP_S * SR)),
                      _tone(3.0, 330.0, 0.2)], dim=-1).unsqueeze(0)


def _row(cue_id, batch_index, samples, placement, anchor):
    prompt = "test cue %s" % cue_id
    return {
        "cue_id": cue_id, "batch_index": batch_index, "sample_count": samples,
        "sample_rate": SR, "prompt": prompt,
        "prompt_sha256": CM.prompt_sha256(prompt), "cue_spec_sha256": None,
        "placement": placement, "anchor_line_id": anchor, "seed": 1,
        "requested_duration_s": samples / SR, "actual_duration_s": samples / SR,
        "output_path": "",
    }


def _assemble(tmp_path, monkeypatch, *, themes=True, bridge=True, policy=""):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    ledger_path = audio_dir / "bridge_ep_ledger.json"
    lines = [
        {"line_id": "b001", "speaker_role": "character", "start_s": 0.0,
         "dur_s": GAP_START_S, "start_s_space": "scene_audio", "text": "a"},
        {"line_id": "b002", "speaker_role": "music_inter",
         "start_s": GAP_START_S, "dur_s": GAP_S,
         "start_s_space": "scene_audio", "text": ""},
        {"line_id": "b003", "speaker_role": "character",
         "start_s": GAP_START_S + GAP_S, "dur_s": 3.0,
         "start_s_space": "scene_audio", "text": "b"},
    ]
    ledger_path.write_text(json.dumps({"lines": lines, "meta": {}}),
                           encoding="utf-8")

    cues, rows = [], []
    if themes:
        cues += [_tone(2.0, 440.0, 0.3), _tone(2.0, 550.0, 0.3)]
        rows += [_row("open_01", 0, 2 * SR, "opening", None),
                 _row("close_01", 1, 2 * SR, "closing", None)]
    if bridge:
        cues.append(_tone(5.0, 660.0, 0.3))
        rows.append(_row("inter_01", len(rows), 5 * SR, "interstitial", "b002"))
    width = max(c.shape[-1] for c in cues)
    batch = torch.zeros(len(cues), 2, width)
    for i, c in enumerate(cues):
        batch[i, :, :c.shape[-1]] = c
    manifest = CM.build_manifest(SR, rows)

    scene = {"waveform": _scene_with_gap(), "sample_rate": SR}
    scene_before = scene["waveform"].clone()
    monkeypatch.setattr(oled, "in_flight_ledger_path", lambda: ledger_path)
    _audio, out_path, _info, _done = sc.EpisodeAssembler().assemble(
        scene_audio=scene,
        music_cue_audio={"waveform": batch, "sample_rate": SR},
        music_cue_manifest_json=CM.dumps(manifest),
        video_policy_json=policy,
    )
    data, _sr = sf.read(out_path, dtype="float32", always_2d=True)
    shift = (2 * SR - XFADE) if themes else 0
    start = shift + int(GAP_START_S * SR)
    window = data[start:start + int(GAP_S * SR)]
    return {
        "path": Path(out_path),
        "window_rms": float(np.sqrt(np.mean(np.square(window)))),
        "ledger": json.loads(ledger_path.read_text(encoding="utf-8")),
        "scene_untouched": torch.equal(scene_before, scene["waveform"]),
        "bridge_master_s": start / SR,
    }


def test_the_bridge_cue_is_in_the_published_master_wav(tmp_path, monkeypatch):
    got = _assemble(tmp_path, monkeypatch)
    assert got["window_rms"] > 0.05, (
        "the bridge window of <ep>_master.wav is silent -- the interstitial cue "
        "never reached the file the mux publishes")


def test_the_stamped_master_hash_is_the_file_with_the_cue(tmp_path, monkeypatch):
    got = _assemble(tmp_path, monkeypatch)
    file_sha = hashlib.sha256(got["path"].read_bytes()).hexdigest()
    assert got["ledger"]["audio"]["master_audio_sha256"] == file_sha


def test_without_a_bridge_cue_the_gap_stays_silent(tmp_path, monkeypatch):
    got = _assemble(tmp_path, monkeypatch, bridge=False)
    assert got["window_rms"] < 1e-6


def test_the_mix_never_writes_into_the_upstream_scene_tensor(tmp_path,
                                                              monkeypatch):
    """No opening or closing theme: the episode waveform IS the scene tensor,
    so mixing in place would rewrite AudioEnhance's cached output."""
    got = _assemble(tmp_path, monkeypatch, themes=False)
    assert got["scene_untouched"]
    assert got["window_rms"] > 0.05


def test_the_foley_route_provisional_master_carries_the_cue(tmp_path,
                                                            monkeypatch):
    got = _assemble(tmp_path, monkeypatch, policy=FOLEY_POLICY)
    assert got["ledger"]["audio"]["master_wav_flavour"] == (
        sc.MASTER_WAV_PRE_LOUDNESS)
    assert got["window_rms"] > 0.05


def test_the_stamped_bridge_row_points_where_the_cue_was_mixed(tmp_path,
                                                               monkeypatch):
    got = _assemble(tmp_path, monkeypatch)
    rows = [r for r in got["ledger"].get("music") or []
            if r.get("cue_id") == "inter_01"]
    assert len(rows) == 1
    assert rows[0]["start_s_space"] == "master_mix"
    assert rows[0]["start_s"] == pytest.approx(got["bridge_master_s"], abs=1e-3)


def test_with_no_interstitial_cue_the_helper_returns_its_input():
    wave = torch.zeros(1, 2, SR)
    manifest = CM.build_manifest(SR, [_row("open_01", 0, SR, "opening", None)])
    out = sc.EpisodeAssembler()._mix_interstitial_cues(
        wave, sample_rate=SR, shift_samples=0,
        music_cue_audio={"waveform": torch.zeros(1, 2, SR), "sample_rate": SR},
        cue_manifest=manifest, owns_waveform=False)
    assert out is wave
