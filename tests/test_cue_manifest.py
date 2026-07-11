"""720-bakeoff C3 -- music cue-manifest schema + slice/byte-parity contract.

The manifest is the ONLY authority mapping a batch row back to a cue. These
gates pin: schema conformance + fail-loud validation (dup cue_id / batch_index,
non-contiguous indices, bad sample_count / placement / version), the
sample_count DIRECT-slice recovery (MF-G: no silence trim), and byte-parity of
the legacy 3-cue synth path through the pack -> slice round-trip.

Headless (CUDA masked by tests/conftest.py). ASCII-only, no em-dash.
"""
from __future__ import annotations

import json

import pytest
import torch

from nodes import _otr_cue_manifest as CM


def _row(cue_id, batch_index, *, sample_count=10, sample_rate=32000,
         placement="interstitial", **kw):
    row = {
        "cue_id": cue_id,
        "batch_index": batch_index,
        "sample_count": sample_count,
        "sample_rate": sample_rate,
        "prompt": "p",
        "prompt_sha256": CM.prompt_sha256("p"),
        "cue_spec_sha256": None,
        "placement": placement,
        "anchor_line_id": None,
        "seed": 1,
        "requested_duration_s": 1.0,
        "actual_duration_s": 1.0,
        "output_path": "",
    }
    row.update(kw)
    return row


# --------------------------------------------------------------------------- #
# Schema conformance + fail-loud validation
# --------------------------------------------------------------------------- #
def test_build_dump_parse_roundtrip():
    m = CM.build_manifest(32000, [
        _row("opening", 0, placement="opening"),
        _row("closing", 1, placement="closing"),
    ])
    text = CM.dumps(m)
    back = CM.parse_manifest(text, batch_size=2)
    assert back == m
    assert back["manifest_version"] == CM.MANIFEST_VERSION


def test_empty_string_parses_to_none():
    assert CM.parse_manifest("") is None
    assert CM.parse_manifest("   ") is None
    assert CM.parse_manifest(None) is None


def test_dup_cue_id_raises():
    with pytest.raises(CM.CueManifestError):
        CM.build_manifest(32000, [_row("a", 0), _row("a", 1)])


def test_dup_batch_index_raises():
    with pytest.raises(CM.CueManifestError):
        CM.build_manifest(32000, [_row("a", 0), _row("b", 0)])


def test_noncontiguous_batch_index_raises():
    with pytest.raises(CM.CueManifestError):
        CM.build_manifest(32000, [_row("a", 0), _row("b", 2)])


def test_bad_sample_count_raises():
    with pytest.raises(CM.CueManifestError):
        CM.build_manifest(32000, [_row("a", 0, sample_count=0)])


def test_bad_placement_raises():
    with pytest.raises(CM.CueManifestError):
        CM.build_manifest(32000, [_row("a", 0, placement="bogus")])


def test_row_rate_must_match_manifest_rate():
    with pytest.raises(CM.CueManifestError):
        CM.build_manifest(32000, [_row("a", 0, sample_rate=48000)])


def test_wrong_version_raises():
    m = CM.build_manifest(32000, [_row("a", 0)])
    m["manifest_version"] = 999
    with pytest.raises(CM.CueManifestError):
        CM.validate_manifest(m)


def test_missing_required_field_raises():
    m = CM.build_manifest(32000, [_row("a", 0)])
    del m["cues"][0]["sample_count"]
    with pytest.raises(CM.CueManifestError):
        CM.validate_manifest(m)


def test_batch_size_mismatch_raises():
    text = CM.dumps(CM.build_manifest(32000, [_row("a", 0), _row("b", 1)]))
    with pytest.raises(CM.CueManifestError):
        CM.parse_manifest(text, batch_size=3)


def test_malformed_json_raises():
    with pytest.raises(CM.CueManifestError):
        CM.parse_manifest("{not json")


def test_index_helpers():
    m = CM.build_manifest(32000, [
        _row("opening", 0, placement="opening", anchor_line_id="L0"),
        _row("inter_01", 1, anchor_line_id="L1"),
    ])
    assert set(CM.index_by_cue_id(m)) == {"opening", "inter_01"}
    by_anchor = CM.index_by_anchor_line_id(m)
    assert by_anchor["L1"]["cue_id"] == "inter_01"
    # rows without an anchor are skipped
    m2 = CM.build_manifest(32000, [_row("x", 0)])
    assert CM.index_by_anchor_line_id(m2) == {}


# --------------------------------------------------------------------------- #
# sample_count direct-slice recovery (MF-G)
# --------------------------------------------------------------------------- #
def test_sample_count_slice_recovers_unpadded_cue():
    from nodes._otr_audio_engines import pack_audio_batch

    a = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, 5)   # len 5
    b = torch.arange(1, 11, dtype=torch.float32).reshape(1, 1, 10)  # len 10
    batch = pack_audio_batch(
        [{"waveform": a, "sample_rate": 32000},
         {"waveform": b, "sample_rate": 32000}],
        sample_rate=32000, mono=True)
    wf = batch["waveform"]
    assert int(wf.shape[0]) == 2 and int(wf.shape[-1]) == 10  # padded to longest
    # Slice row 0 by its true sample_count (5): recovers a exactly, no pad tail.
    assert torch.equal(wf[0, :, :5], a[0])
    # The padded tail of the shorter cue is zeros.
    assert torch.count_nonzero(wf[0, :, 5:]) == 0
    assert torch.equal(wf[1, :, :10], b[0])


# --------------------------------------------------------------------------- #
# byte-parity: legacy 3-cue synth through pack -> slice
# --------------------------------------------------------------------------- #
def _det_clip(prompt, duration_s, seed):
    """Deterministic per-(seed) clip: distinct length + content per cue so the
    pack/pad/slice round-trip is actually exercised."""
    n = 8 + (abs(int(seed)) % 5)                 # 8..12 distinct lengths
    scale = float((abs(int(seed)) % 97) + 1)
    wf = (torch.arange(n, dtype=torch.float32) + scale).reshape(1, 1, n)
    return {"waveform": wf, "sample_rate": 32000}


def test_legacy_three_cue_byte_parity(monkeypatch):
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_audio_engines.base import mono_safe
    from nodes.stable_audio_theme import StableAudioTheme

    calls = []

    def _gen(prompt, duration_s, seed):
        calls.append({"prompt": prompt, "duration_s": duration_s, "seed": seed})
        return _det_clip(prompt, duration_s, seed)

    monkeypatch.setattr(get_engine("musicgen"), "generate_clip", _gen)

    ledger = json.dumps({"schema_version": "l3-2026-05-14",
                         "cast": [], "lines": [], "meta": {}})
    out = StableAudioTheme().generate(script_json=ledger, engine="musicgen")
    batch = out[0]["waveform"]
    manifest = CM.parse_manifest(out[1], batch_size=3)

    # Each batch row, sliced by manifest sample_count, is byte-identical to a
    # standalone (canonicalized) render of that cue's exact prompt+seed.
    for row in manifest["cues"]:
        bi = row["batch_index"]
        sc = row["sample_count"]
        call = calls[bi]
        expected = mono_safe(_det_clip(
            call["prompt"], call["duration_s"], call["seed"]))["waveform"][0]
        assert int(expected.shape[-1]) == sc
        assert torch.equal(batch[bi, :, :sc], expected)
