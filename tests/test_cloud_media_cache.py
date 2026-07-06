"""S0 offline tests: billing cache + canonicalization contract."""
from __future__ import annotations

import json

import pytest

from nodes._otr_shared import cloud_media_cache as cmc
from nodes._otr_shared.cloud_media_backend import CloudMediaError
from nodes._otr_shared.cloud_media_canonical import (
    CANONICALIZER_VERSION,
    canonicalize_audio,
    validate_partner_result,
)


@pytest.fixture()
def cache_root(tmp_path):
    return tmp_path / "cloud_cache"


def _key(seed=7, **over):
    kwargs = dict(
        row_id="cloud_flux_pro",
        resolved_slug="flux_pro/v1",
        params={"prompt": "a 1940s radio", "size": "1472x832"},
        input_hashes={"init_image": "ab" * 32},
        seed=seed,
        output_contract_version=1,
        adapter_version=1,
        canonicalizer_version=CANONICALIZER_VERSION,
        schema_version="pin-2026-07-02",
    )
    kwargs.update(over)
    return cmc.RequestCacheKey.build(**kwargs)


def _asset(tmp_path, content=b"PNG-BYTES-PLACEHOLDER"):
    p = tmp_path / "canonical.png"
    p.write_bytes(content)
    return p


def _manifest():
    return {
        "media_type": "image",
        "canonicalizer_version": CANONICALIZER_VERSION,
        "row_id": "cloud_flux_pro",
    }


# -- key determinism ----------------------------------------------------------


def test_key_deterministic_across_param_order():
    a = _key(params={"b": 2, "a": 1})
    b = _key(params={"a": 1, "b": 2})
    assert a.hexdigest() == b.hexdigest()


def test_key_changes_with_inputs():
    assert _key().hexdigest() != _key(seed=8).hexdigest()
    assert _key().hexdigest() != _key(
        input_hashes={"init_image": "cd" * 32}).hexdigest()
    assert _key().hexdigest() != _key(canonicalizer_version=99).hexdigest()


def test_key_seed_none_supported():
    """seed=None (provider without seed support) is a valid, distinct key."""
    assert _key(seed=None).hexdigest() != _key(seed=7).hexdigest()


# -- store / lookup roundtrip -------------------------------------------------


def test_store_then_lookup_roundtrip(tmp_path, cache_root):
    key = _key()
    stored = cmc.cache_store(key, _asset(tmp_path), _manifest(),
                             cache_root=cache_root)
    hit = cmc.cache_lookup(key, cache_root=cache_root)
    assert hit is not None
    assert hit.path.read_bytes() == b"PNG-BYTES-PLACEHOLDER"
    assert hit.manifest["content_sha256"] == stored.manifest["content_sha256"]
    assert hit.manifest["cache_key"] == key.hexdigest()


def test_lookup_miss_on_empty(cache_root):
    assert cmc.cache_lookup(_key(), cache_root=cache_root) is None


def test_corrupt_asset_is_miss_and_quarantined(tmp_path, cache_root, capsys):
    key = _key()
    stored = cmc.cache_store(key, _asset(tmp_path), _manifest(),
                             cache_root=cache_root)
    stored.path.write_bytes(b"TAMPERED")
    assert cmc.cache_lookup(key, cache_root=cache_root) is None
    assert "quarantined" in capsys.readouterr().out
    # quarantine renamed the entry; a fresh store works again
    cmc.cache_store(key, _asset(tmp_path), _manifest(), cache_root=cache_root)
    assert cmc.cache_lookup(key, cache_root=cache_root) is not None


def test_missing_manifest_field_rejected(tmp_path, cache_root):
    bad = {"media_type": "image"}  # no canonicalizer_version / row_id
    with pytest.raises(CloudMediaError):
        cmc.cache_store(_key(), _asset(tmp_path), bad, cache_root=cache_root)


def test_store_missing_file_fails_closed(cache_root, tmp_path):
    with pytest.raises(CloudMediaError):
        cmc.cache_store(_key(), tmp_path / "nope.png", _manifest(),
                        cache_root=cache_root)


# -- staleness ---------------------------------------------------------------


def test_needs_recanonicalize_on_version_bump():
    m = {"canonicalizer_version": CANONICALIZER_VERSION}
    assert cmc.needs_recanonicalize(m, CANONICALIZER_VERSION) is False
    assert cmc.needs_recanonicalize(m, CANONICALIZER_VERSION + 1) is True


def test_needs_recanonicalize_on_missing_strip_proof():
    m = {"canonicalizer_version": CANONICALIZER_VERSION}
    assert cmc.needs_recanonicalize(m, CANONICALIZER_VERSION,
                                    must_strip_audio=True) is True
    m["audio_stripped_proof"] = True
    assert cmc.needs_recanonicalize(m, CANONICALIZER_VERSION,
                                    must_strip_audio=True) is False


# -- episode copy --------------------------------------------------------------


def test_copy_into_episode_verifies(tmp_path, cache_root):
    key = _key()
    cmc.cache_store(key, _asset(tmp_path), _manifest(), cache_root=cache_root)
    hit = cmc.cache_lookup(key, cache_root=cache_root)
    ep = tmp_path / "episodes" / "ep_001"
    dest = cmc.copy_into_episode(hit, ep, "beat_003.png")
    assert dest.is_file() and dest.read_bytes() == b"PNG-BYTES-PLACEHOLDER"


# -- canonicalization contract --------------------------------------------------


def test_validate_partner_result_shape(tmp_path):
    p = tmp_path / "dl.bin"
    p.write_bytes(b"x")
    ok = validate_partner_result({"path": str(p), "content_type": "image/png"})
    assert ok["path"] == str(p) and ok["raw_meta"] == {}


def test_validate_partner_result_fails_closed(tmp_path):
    with pytest.raises(CloudMediaError):
        validate_partner_result({"content_type": "image/png"})
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(CloudMediaError):
        validate_partner_result({"path": str(empty),
                                 "content_type": "image/png"})


def test_canonicalize_audio_conforms_wav_and_reuses_local_rms(tmp_path):
    """C8 (cloud-audio S0): canonicalize_audio decodes -> 44.1k WAV and matches
    loudness by REUSING the local RMS leveler (scene_sequencer, -16 dBFS target,
    NOT LUFS). A quiet tone must be BOOSTED toward the target; the output is a
    real WAV CanonicalAsset with a duration."""
    import math
    import shutil
    import subprocess
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required")

    src = tmp_path / "quiet.wav"                       # -30 dBFS tone (below -16)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=220:duration=1.0", "-af", "volume=-30dB",
                    "-ar", "24000", "-ac", "1", str(src)], check=True)
    raw = validate_partner_result({"path": str(src), "content_type": "audio/wav"})
    asset = canonicalize_audio(raw, {"sample_rate": 44100,
                                     "stereo_policy": "mono"}, session=None)
    assert asset.media_type == "audio" and asset.container == "wav"
    assert asset.path.is_file() and asset.duration_s and asset.duration_s > 0.9

    def _rms_dbfs(path):
        raw_pcm = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a",
             "-f", "f32le", "-ac", "1", "-ar", "44100", "-"],
            capture_output=True).stdout
        import numpy as np
        x = np.frombuffer(raw_pcm, dtype=np.float32)
        r = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        return 20.0 * math.log10(max(r, 1e-10))

    before, after = _rms_dbfs(src), _rms_dbfs(asset.path)
    assert after > before + 3.0                       # quiet tone was boosted
    assert after <= -12.0                              # toward -16 dBFS, peak-safe


def test_canonicalize_audio_missing_result_raises():
    with pytest.raises(CloudMediaError):
        canonicalize_audio({"path": "/nope/missing.wav",
                            "content_type": "audio/wav"}, {}, session=None)
