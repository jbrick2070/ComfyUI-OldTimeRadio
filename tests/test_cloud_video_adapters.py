"""S3-core cloud video adapters + canonicalize_video (2026-07-02).

Offline (`no-network`): the bridge is monkeypatched; the canonicalizer runs
on a locally generated ffmpeg fixture. Live provider proof stays behind the
operator-gated smokes (pass04 sec 8 labels).
"""
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from nodes._otr_shared.cloud_media_backend import (
    CloudErrorCode,
    CloudMediaError,
)
from nodes._otr_shared.cloud_media_canonical import (
    CanonicalAsset,
    canonicalize_video,
)
from nodes._otr_video_engines import eng_cloud_video as ecv
from nodes._otr_video_engines import registry as vreg

_FFMPEG = shutil.which("ffmpeg")

_CLOUD_ROWS = ("cloud_kling_avatar", "cloud_seedance_2", "cloud_wan_i2v")


# --------------------------------------------------------------------------- #
# registration + menu surface
# --------------------------------------------------------------------------- #


def test_all_cloud_rows_registered_with_capabilities():
    for name in _CLOUD_ROWS:
        assert vreg.is_registered(name), name
        row = vreg.CAPABILITIES[name]
        assert row["cpu_ok"] is True


def test_cloud_rows_never_default():
    # S3-core: selectable picks ONLY -- no cloud row declares default_roles,
    # so automatic selection can never land on one (menu ORDER is cosmetic;
    # default_engine_for_role is the automatic-selection surface).
    for eng in (ecv.KlingAvatar, ecv.Seedance2, ecv.WanI2V):
        assert tuple(eng.default_roles) == ()
    for role in ("announcer_visual", "music_visual", "character_video"):
        default = vreg.default_engine_for_role(role)
        assert not str(default).startswith("cloud_"), (role, default)


def test_reactivity_descriptors_match_pass04():
    assert ecv.KlingAvatar.reactivity == "required_audio_ref"
    assert ecv.Seedance2.reactivity == "required_audio_ref"
    assert ecv.WanI2V.reactivity == "mute_only"
    assert all(e.must_strip_audio for e in
               (ecv.KlingAvatar, ecv.Seedance2, ecv.WanI2V))


def test_schema_grounded_v3_video_rows_are_partner_invocable():
    # Partner Node video rows are credit-billed through Comfy's hidden auth
    # context. A dropdown pick is the enable; missing login/credits fail at
    # invoke time, not as a stale director-side "awaiting paid smoke" block.
    assert ecv.Seedance2.invocable is True
    assert ecv.WanI2V.invocable is True
    assert ecv.Seedance2.invocability_reason == ""
    assert ecv.WanI2V.invocability_reason == ""


def test_assert_usable_no_enable_flag(monkeypatch):
    """Operator directive 2026-07-02: the dropdown pick is the enable --
    assert_usable must NOT gate on OTR_ENABLE_COMFY_CLOUD_MEDIA (only
    ffmpeg presence + a healthy pin row)."""
    monkeypatch.delenv("OTR_ENABLE_COMFY_CLOUD_MEDIA", raising=False)
    ecv.KlingAvatar.assert_usable({}, {})            # must not raise


# --------------------------------------------------------------------------- #
# partner input builders (bridge monkeypatched -- no network)
# --------------------------------------------------------------------------- #


def _fixture_png(tmp_path):
    from PIL import Image
    p = tmp_path / "init.png"
    Image.new("RGB", (64, 36), (200, 120, 40)).save(str(p))
    return str(p)


def _fixture_wav(tmp_path):
    import numpy as np
    import soundfile as sf
    p = tmp_path / "voice.wav"
    t = np.linspace(0, 1.0, 16000, dtype="float32")
    sf.write(str(p), (0.2 * np.sin(2 * 3.14159 * 220 * t)), 16000)
    return str(p)


def _request(tmp_path, **over):
    req = {
        "shot_id": "shot_b002",
        "init_image": _fixture_png(tmp_path),
        "audio_ref": _fixture_wav(tmp_path),
        "text_prompt": "a person talking to the viewer",
        "seed_bundle": {"request_seed": 7},
        "canvas": {"w": 832, "h": 448, "fps": 25},
    }
    req.update(over)
    return req


def test_kling_avatar_partner_inputs(tmp_path, monkeypatch):
    captured = {}

    def _fake_invoke(node_key, inputs, *, timeout_s, estimated_usd=0.0):
        captured.update(node_key=node_key, inputs=inputs,
                        timeout_s=timeout_s, estimated_usd=estimated_usd)
        return {"path": _fixture_png(tmp_path), "content_type": "video/mp4",
                "duration_s": None, "provider_job_id": "j1", "raw_meta": {}}

    import nodes._otr_shared.cloud_media_invoke as cmi
    monkeypatch.setattr(cmi, "invoke_partner_node", _fake_invoke)
    raw = ecv.KlingAvatar.render_clip(_request(tmp_path), {})
    assert captured["node_key"] == "cloud_kling_avatar"
    ins = captured["inputs"]
    assert ins["mode"] and ins["seed"] == 7
    assert ins["prompt"].startswith("a person")
    assert hasattr(ins["image"], "ndim") and ins["image"].ndim == 4
    assert set(ins["sound_file"]) == {"waveform", "sample_rate"}
    assert raw["provider_job_id"] == "j1"


def test_kling_avatar_missing_audio_fails_loud(tmp_path):
    req = _request(tmp_path, audio_ref="")
    with pytest.raises(RuntimeError, match="audio_ref"):
        ecv.KlingAvatar._partner_inputs(req)


def test_wan_i2v_sends_v3_model_dict_without_audio(tmp_path, monkeypatch):
    # The pinned Wan row has NO top-level prompt input; prompt/negative/duration
    # ride inside the DYNAMICCOMBO_V3 model dict accepted by Wan2ImageToVideoApi.
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan2.7-i2v")
    monkeypatch.setenv("OTR_CLOUD_WAN_RESOLUTION", "1080p")
    monkeypatch.setenv("OTR_CLOUD_WAN_DURATION", "6")
    req = _request(tmp_path, seed_bundle={"request_seed": 2147483655})
    ins = ecv.WanI2V._partner_inputs(req)
    assert set(ins) == {"first_frame", "model", "prompt_extend",
                        "seed", "watermark"}
    assert "prompt" not in ins and "audio" not in ins
    assert set(ins["model"]) == {
        "model", "prompt", "negative_prompt", "resolution", "duration"}
    assert ins["model"]["model"] == "wan2.7-i2v"
    assert ins["model"]["prompt"].startswith("a person")
    assert ins["model"]["negative_prompt"] == ""
    assert ins["model"]["resolution"] == "1080P"
    assert ins["model"]["duration"] == 6
    assert ins["prompt_extend"] is False
    assert ins["seed"] == 7


def test_wan_i2v_rejects_v3_placeholder_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "COMFY_DYNAMICCOMBO_V3")
    with pytest.raises(CloudMediaError, match="placeholder"):
        ecv.WanI2V._partner_inputs(_request(tmp_path))


def test_seedance_reference_row_sends_v3_model_dict(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTR_CLOUD_SEEDANCE_MODEL", "dreamina-seedance-2-0-fast-260128")
    monkeypatch.setenv("OTR_CLOUD_SEEDANCE_DURATION", "5")
    req = _request(tmp_path, seed_bundle={"request_seed": 2147483656})
    ins = ecv.Seedance2._partner_inputs(req)
    assert set(ins) == {"model", "seed", "watermark"}
    assert ins["seed"] == 8
    assert ins["watermark"] is False
    model = ins["model"]
    assert model["model"] == "Seedance 2.0 Fast"
    assert model["prompt"].startswith("a person")
    assert model["resolution"] == "720p"
    assert model["ratio"] == "adaptive"
    assert model["duration"] == 5
    assert model["generate_audio"] is False
    assert set(model["reference_images"]) == {"image_1"}
    assert set(model["reference_audios"]) == {"audio_1"}
    assert hasattr(model["reference_images"]["image_1"], "ndim")
    assert set(model["reference_audios"]["audio_1"]) == {
        "waveform", "sample_rate"}


# --------------------------------------------------------------------------- #
# canonicalize_video (real ffmpeg on a generated AV fixture)
# --------------------------------------------------------------------------- #


def _make_av_fixture(tmp_path) -> Path:
    """2s 128x72 test clip WITH an audio track (the strip target)."""
    out = tmp_path / "provider.mp4"
    subprocess.run(
        [_FFMPEG, "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=128x72:rate=12:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(out)],
        check=True, capture_output=True, timeout=120)
    return out


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_canonicalize_video_strips_audio_and_conforms(tmp_path):
    src = _make_av_fixture(tmp_path)
    raw = {"path": str(src), "content_type": "video/mp4",
           "duration_s": None, "provider_job_id": "job-9",
           "raw_meta": {}}
    asset = canonicalize_video(raw, {"w": 320, "h": 192, "fps": 25})
    assert isinstance(asset, CanonicalAsset)
    assert asset.path.is_file() and asset.path.suffix == ".mp4"
    assert asset.media_type == "video"
    assert asset.width == 320 and asset.height == 192 and asset.fps == 25.0
    assert asset.duration_s and asset.duration_s > 1.0
    assert len(asset.sha256) == 64
    assert asset.provider_job_id == "job-9"
    assert any("stripped" in w for w in asset.validation_warnings)
    # strip PROOF: re-probe the canonical output -> zero audio streams
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         str(asset.path)], capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout)["streams"]
    assert all(s["codec_type"] != "audio" for s in streams)


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_canonicalize_video_missing_file_fails_closed(tmp_path):
    raw = {"path": str(tmp_path / "never.mp4"), "content_type": "video/mp4",
           "duration_s": None, "provider_job_id": None, "raw_meta": {}}
    with pytest.raises(CloudMediaError) as ei:
        canonicalize_video(raw, {"w": 320, "h": 192, "fps": 25})
    assert ei.value.code is CloudErrorCode.CORRUPT_OUTPUT


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_canonicalize_video_bad_request_fails_closed(tmp_path):
    src = _make_av_fixture(tmp_path)
    raw = {"path": str(src), "content_type": "video/mp4",
           "duration_s": None, "provider_job_id": None, "raw_meta": {}}
    with pytest.raises(CloudMediaError) as ei:
        canonicalize_video(raw, {"w": 320})           # h/fps missing
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_engine_canonicalize_returns_clip_dict(tmp_path):
    src = _make_av_fixture(tmp_path)
    raw = {"path": str(src), "content_type": "video/mp4",
           "duration_s": None, "provider_job_id": "job-5", "raw_meta": {}}
    clip = ecv.KlingAvatar.canonicalize(raw, _request(tmp_path), {})
    assert clip["has_audio"] is False
    assert clip["engine_id"] == "cloud_kling_avatar"
    assert clip["type"] == "video" and clip["container"] == "mp4"
    assert clip["fps"] == 25 and clip["frame_count"] > 0
    assert Path(clip["path"]).is_file()
    assert clip["provider_job_id"] == "job-5"
    assert len(clip["content_sha256"]) == 64
