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
from nodes._otr_video_engines import render_driver as rd

_FFMPEG = shutil.which("ffmpeg")

_CLOUD_ROWS = (
    "cloud_kling_avatar", "cloud_seedance_2", "cloud_wan_i2v",
    "cloud_wan_i2v_audio",
)
_MUSIC_OPEN_RISKY_PROMPT = (
    "Continuous shot, same console throughout. Dial whip-pans across "
    "frequencies. Tube filaments ignite from cold to white-hot. Speaker "
    "grille vibrates aggressively. Dynamic dolly push forward.")


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
    for eng in (ecv.KlingAvatar, ecv.Seedance2, ecv.WanI2V, ecv.WanI2VAudio):
        assert tuple(eng.default_roles) == ()
    for role in ("announcer_visual", "music_visual", "character_video"):
        default = vreg.default_engine_for_role(role)
        assert not str(default).startswith("cloud_"), (role, default)


def test_reactivity_descriptors_match_pass04():
    assert ecv.KlingAvatar.reactivity == "required_audio_ref"
    assert ecv.Seedance2.reactivity == "required_audio_ref"
    assert ecv.WanI2V.reactivity == "mute_only"
    assert ecv.WanI2VAudio.reactivity == "required_audio_ref"
    assert all(e.must_strip_audio for e in
               (ecv.KlingAvatar, ecv.Seedance2, ecv.WanI2V, ecv.WanI2VAudio))


def test_schema_grounded_v3_video_rows_are_partner_invocable():
    # Partner Node video rows are credit-billed through Comfy's hidden auth
    # context. A dropdown pick is the enable; missing login/credits fail at
    # invoke time, not as a stale director-side "awaiting paid smoke" block.
    assert ecv.Seedance2.invocable is True
    assert ecv.WanI2V.invocable is True
    assert ecv.WanI2VAudio.invocable is True
    assert ecv.Seedance2.invocability_reason == ""
    assert ecv.WanI2V.invocability_reason == ""
    assert ecv.WanI2VAudio.invocability_reason == ""


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


def _fixture_wav(tmp_path, dur_s=1.0):
    import numpy as np
    import soundfile as sf
    p = tmp_path / f"voice_{int(float(dur_s) * 1000)}ms.wav"
    t = np.linspace(0, float(dur_s), int(16000 * float(dur_s)),
                    endpoint=False, dtype="float32")
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


def _assert_visual_safety(prompt: str):
    lower = prompt.lower()
    assert "no blood" in lower
    assert "no guns" in lower
    assert "no knives" in lower
    assert "no smoking" in lower


def _assert_negative_safety(prompt: str):
    lower = prompt.lower()
    for term in ("blood", "guns", "knives", "smoking"):
        assert term in lower


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
    assert ecv._KLING_AVATAR_MARKER in ins["prompt"]
    _assert_visual_safety(ins["prompt"])
    assert hasattr(ins["image"], "ndim") and ins["image"].ndim == 4
    assert set(ins["sound_file"]) == {"waveform", "sample_rate"}
    assert ins["sound_file"]["waveform"].shape[-1] == 32000
    assert raw["provider_job_id"] == "j1"


def test_kling_avatar_missing_audio_fails_loud(tmp_path):
    req = _request(tmp_path, audio_ref="")
    with pytest.raises(RuntimeError, match="audio_ref"):
        ecv.KlingAvatar._partner_inputs(req)


def test_kling_avatar_mode_alias_and_seed_clamp(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_KLING_MODE", "Professional Mode")
    req = _request(tmp_path, seed_bundle={"request_seed": 2147483656})
    ins = ecv.KlingAvatar._partner_inputs(req)
    assert ins["mode"] == "pro"
    assert ins["seed"] == 8


def test_kling_avatar_rejects_unknown_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_KLING_MODE", "turbo")
    with pytest.raises(RuntimeError, match="OTR_CLOUD_KLING_MODE"):
        ecv.KlingAvatar._partner_inputs(_request(tmp_path))


def test_kling_avatar_pads_request_audio_to_provider_floor(tmp_path):
    req = _request(tmp_path, audio_ref=_fixture_wav(tmp_path, dur_s=0.5))
    ins = ecv.KlingAvatar._partner_inputs(req)
    assert ins["sound_file"]["sample_rate"] == 16000
    assert ins["sound_file"]["waveform"].shape[-1] == 32000


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
    assert ecv._WAN_SMOOTH_MARKER in ins["model"]["prompt"]
    _assert_visual_safety(ins["model"]["prompt"])
    assert "jump cuts" in ins["model"]["negative_prompt"]
    _assert_negative_safety(ins["model"]["negative_prompt"])
    assert ins["model"]["resolution"] == "1080P"
    assert ins["model"]["duration"] == 6
    assert ins["prompt_extend"] is False
    assert ins["seed"] == 7


@pytest.mark.parametrize(
    "legacy_selector",
    ("wan-2.2-i2v", "wan2.2-i2v", "wan2.2_i2v", "Wan 2.2 I2V",
     "wan-2.7-i2v"),
)
def test_wan_i2v_normalizes_legacy_cloud_model_aliases(
        tmp_path, monkeypatch, legacy_selector):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", legacy_selector)
    ins = ecv.WanI2V._partner_inputs(_request(tmp_path))
    assert ins["model"]["model"] == "wan2.7-i2v"


def test_wan_i2v_rejects_unknown_model_selector(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan3.0-i2v")
    with pytest.raises(RuntimeError, match="unsupported Wan model selector"):
        ecv.WanI2V._partner_inputs(_request(tmp_path))


def test_wan_i2v_duration_ceil_avoids_fractional_beat_underrun(
        tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan2.7-i2v")
    monkeypatch.delenv("OTR_CLOUD_WAN_DURATION", raising=False)
    req = _request(tmp_path, timing={"target_frame_count": 51})
    ins = ecv.WanI2V._partner_inputs(req)
    assert ins["model"]["duration"] == 3


def test_wan_i2v_negative_prompt_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan2.7-i2v")
    monkeypatch.setenv("OTR_CLOUD_WAN_NEGATIVE_PROMPT", "motion artifacts")
    ins = ecv.WanI2V._partner_inputs(_request(tmp_path))
    assert ins["model"]["negative_prompt"].startswith("motion artifacts")
    _assert_negative_safety(ins["model"]["negative_prompt"])


def test_wan_i2v_audio_sends_declared_optional_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan2.7-i2v")
    req = _request(tmp_path, audio_ref=_fixture_wav(tmp_path, dur_s=0.5))
    ins = ecv.WanI2VAudio._partner_inputs(req)
    assert ecv.WanI2VAudio.node_key == "cloud_wan_i2v"
    assert set(ins) == {"first_frame", "model", "prompt_extend",
                        "seed", "watermark", "audio"}
    assert set(ins["audio"]) == {"waveform", "sample_rate"}
    assert ins["audio"]["sample_rate"] == 16000
    assert ins["audio"]["waveform"].shape[-1] == 32000
    _assert_visual_safety(ins["model"]["prompt"])


def test_wan_i2v_audio_missing_audio_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan2.7-i2v")
    with pytest.raises(RuntimeError, match="audio_ref"):
        ecv.WanI2VAudio._partner_inputs(_request(tmp_path, audio_ref=""))


def test_wan_i2v_rejects_v3_placeholder_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "COMFY_DYNAMICCOMBO_V3")
    with pytest.raises(CloudMediaError, match="placeholder"):
        ecv.WanI2V._partner_inputs(_request(tmp_path))


def test_seedance_prompt_conditioner_softens_risky_music_open():
    conditioned, meta = ecv._condition_seedance_prompt(
        _MUSIC_OPEN_RISKY_PROMPT)
    lower = conditioned.lower()

    assert meta["changed"] is True
    assert meta["softeners_applied"] == [
        "dynamic_dolly_push",
        "whip_pans",
        "white_hot",
        "aggressively",
    ]
    assert "whip-pans" not in lower
    assert "white-hot" not in lower
    assert "vibrates aggressively" not in lower
    assert "dynamic dolly push" not in lower
    assert "slowly sweeps across frequencies" in lower
    assert "bright warm glow" in lower
    assert "vibrates subtly" in lower
    assert "slow controlled dolly push forward" in lower
    assert "\n\n" in conditioned
    assert ecv._SEEDANCE_SMOOTH_MARKER in conditioned


def test_seedance_prompt_conditioner_is_idempotent():
    first, first_meta = ecv._condition_seedance_prompt(
        _MUSIC_OPEN_RISKY_PROMPT)
    second, second_meta = ecv._condition_seedance_prompt(first)
    third, third_meta = ecv._condition_seedance_prompt(second)

    assert first_meta["changed"] is True
    assert second == first
    assert third == first
    assert second_meta["changed"] is False
    assert third_meta["changed"] is False
    assert second_meta["original_sha8"] == second_meta["conditioned_sha8"]
    assert third_meta["softeners_applied"] == []


def test_seedance_prompt_conditioner_rejects_empty_prompt():
    with pytest.raises(ValueError, match="non-empty prompt"):
        ecv._condition_seedance_prompt("   ")


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
    assert set(model) == {
        "model", "prompt", "resolution", "ratio", "duration",
        "generate_audio", "reference_images", "reference_audios"}
    assert model["model"] == "Seedance 2.0 Fast"
    assert model["prompt"].startswith("a person")
    assert ecv._SEEDANCE_SMOOTH_MARKER in model["prompt"]
    _assert_visual_safety(model["prompt"])
    assert model["resolution"] == "720p"
    assert model["ratio"] == "adaptive"
    assert model["duration"] == 5
    assert model["generate_audio"] is False
    assert set(model["reference_images"]) == {"image_1"}
    assert set(model["reference_audios"]) == {"audio_1"}
    assert hasattr(model["reference_images"]["image_1"], "ndim")
    assert set(model["reference_audios"]["audio_1"]) == {
        "waveform", "sample_rate"}


def test_seedance_short_beat_requests_provider_minimum(tmp_path, monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_SEEDANCE_DURATION", raising=False)
    req = _request(tmp_path, timing={"target_frame_count": 50})
    ins = ecv.Seedance2._partner_inputs(req)
    assert ins["model"]["duration"] == 4


def test_seedance_two_second_ledger_beat_requests_minimum_and_16x9(
        tmp_path, monkeypatch):
    """2s ledger beat -> request builder -> real Seedance partner payload.

    This proves the under-minimum cloud-duration policy on a ledger-shaped
    request, not only a hand-built adapter dict.
    """
    monkeypatch.setenv("OTR_CLOUD_SEEDANCE_MODEL", "Seedance 2.0 Fast")
    monkeypatch.setenv("OTR_CLOUD_SEEDANCE_RATIO", "16:9")
    monkeypatch.delenv("OTR_CLOUD_SEEDANCE_DURATION", raising=False)
    init_png = _fixture_png(tmp_path)
    audio_wav = _fixture_wav(tmp_path)
    shot = {
        "shot_id": "shot_b020",
        "source_line_ids": ["b020"],
        "role": "music_visual",
        "group_id": "grp_music",
        "engine_id": "cloud_seedance_2",
        "family": "audio_conditioned_video",
        "target_frame_count": 50,
        "render_request_hash": "hash_b020",
        "cache_keys": {"request_hash": "hash_b020"},
        "creative": {"text_prompt": _MUSIC_OPEN_RISKY_PROMPT},
    }
    ledger = {
        "audio": {
            "master_audio_sha256": rd.FROZEN_AUDIO_SHA,
            "ledger_frozen": True,
        },
        "meta": {"visual_style": "sci_fi_radio"},
        "lines": [{
            "line_id": "b020",
            "speaker_role": "music_inter",
            "char_id": "",
            "start_s": 0.0,
            "dur_s": 2.0,
            "audio_wav_path": audio_wav,
        }],
        "images": {"images": [{
            "object_id": "still_b020_music",
            "kind": "scene_open",
            "beat_id": "b020",
            "path": init_png,
        }]},
        "video": {"fps": 25, "shots": [shot]},
    }

    req = rd.build_request_from_shot(shot, ledger)
    assert req["timing"]["target_frame_count"] == 50
    assert req["timing"]["target_duration_s"] == 2.0
    assert req["asset_refs"]["init_image"] == init_png
    assert req["audio_ref"]["path"] == audio_wav
    assert req["canvas"]["w"] > req["canvas"]["h"]
    _assert_visual_safety(req["text_prompt"])

    ins = ecv.Seedance2._partner_inputs(req)
    assert set(ins) == {"model", "seed", "watermark"}
    model = ins["model"]
    assert set(model) == {
        "model", "prompt", "resolution", "ratio", "duration",
        "generate_audio", "reference_images", "reference_audios"}
    assert model["ratio"] == "16:9"
    assert model["duration"] == 4
    assert ecv._SEEDANCE_SMOOTH_MARKER in model["prompt"]
    _assert_visual_safety(model["prompt"])
    assert "whip-pans" not in model["prompt"].lower()
    assert hasattr(model["reference_images"]["image_1"], "ndim")
    assert set(model["reference_audios"]["audio_1"]) == {
        "waveform", "sample_rate"}


def test_prompt_conditioners_use_engine_specific_markers(
        tmp_path, monkeypatch):
    prompt = "handheld dolly with a rapid zoom"
    req = _request(tmp_path, text_prompt=prompt)

    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan2.7-i2v")
    wan = ecv.WanI2V._partner_inputs(req)
    assert wan["model"]["prompt"].startswith(prompt)
    assert ecv._WAN_SMOOTH_MARKER in wan["model"]["prompt"]
    assert ecv._SEEDANCE_SMOOTH_MARKER not in wan["model"]["prompt"]
    _assert_visual_safety(wan["model"]["prompt"])

    kling = ecv.KlingAvatar._partner_inputs(req)
    assert kling["prompt"].startswith(prompt)
    assert ecv._KLING_AVATAR_MARKER in kling["prompt"]
    assert ecv._SEEDANCE_SMOOTH_MARKER not in kling["prompt"]
    _assert_visual_safety(kling["prompt"])

    razzle = ecv.WordRazzle._partner_inputs(req)
    assert prompt in razzle["prompt"]
    assert ecv._WAN_SMOOTH_MARKER not in razzle["prompt"]
    assert ecv._KLING_AVATAR_MARKER not in razzle["prompt"]
    _assert_visual_safety(razzle["prompt"])
    _assert_negative_safety(razzle["negative_prompt"])


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
