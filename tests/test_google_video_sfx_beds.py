"""Google video SFX-bed engine regressions."""
from __future__ import annotations

import ast
import math
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from nodes._otr_google_api.client import GoogleAPIRequestShapeError
from nodes._otr_shared.cloud_media_backend import CloudMediaError
from nodes._otr_shared.cloud_media_canonical import extract_sfx_bed_from_provider_video
from nodes._otr_story_brief_helpers import append_sfx_audio_safety_clause
from nodes._otr_video_engines import eng_google_omni_video as OMNI
from nodes._otr_video_engines import eng_google_veo_video as VEO
from nodes._otr_video_engines import eng_google_vid_sfx as SFX
from nodes._otr_video_engines import registry as vreg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "nodes" / "_otr_video_engines" / "eng_google_vid_sfx.py"
_FFMPEG = shutil.which("ffmpeg")


def _req(**over):
    r = {
        "shot_id": "shot_sfx_001",
        "text_prompt": "a storm-lit radio tower, slow cinematic drift",
        "seed_bundle": {"request_seed": 9},
        "canvas": {"w": 832, "h": 448, "fps": 25},
        "timing": {"target_frame_count": 125},
    }
    r.update(over)
    return r


def _clear_video_env(monkeypatch):
    for name in (
        "OTR_GOOGLE_VEO_MODEL_ID",
        "OTR_GOOGLE_OMNI_MODEL_ID",
        "OTR_GOOGLE_VIDEO_MODEL_ID",
        "OTR_GOOGLE_VIDEO_MODEL",
        "OTR_GOOGLE_VEO_ASPECT",
        "OTR_GOOGLE_VIDEO_ASPECT",
        "OTR_GOOGLE_VEO_DURATION_S",
        "OTR_GOOGLE_VIDEO_DURATION_S",
        "OTR_GOOGLE_VEO_RESOLUTION",
        "OTR_GOOGLE_VIDEO_RESOLUTION",
        "OTR_GOOGLE_VEO_SFX_SUBMIT_MIN_INTERVAL_S",
    ):
        monkeypatch.delenv(name, raising=False)


def _mp4_fixture(tmp_path, *, with_audio: bool, audio_filter: str = "") -> pathlib.Path:
    out = tmp_path / ("provider_av.mp4" if with_audio else "provider_v.mp4")
    cmd = [
        _FFMPEG,
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=128x72:rate=12:duration=1",
    ]
    if with_audio:
        audio_src = "sine=frequency=330:duration=1"
        if audio_filter:
            audio_src += "," + audio_filter
        cmd += [
            "-f",
            "lavfi",
            "-i",
            audio_src,
        ]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(out))
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return out


def _audio_streams(path: pathlib.Path) -> list[dict]:
    import json

    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    doc = json.loads(p.stdout or "{}")
    return [s for s in doc.get("streams") or [] if s.get("codec_type") == "audio"]


def _audio_rms_dbfs(path: pathlib.Path) -> float:
    p = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    audio = np.frombuffer(p.stdout, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    return 20.0 * math.log10(max(rms, 1e-12))


def _clear_sfx_loudness_env(monkeypatch):
    for name in (
        "OTR_SFX_STEM_TARGET_RMS_DBFS",
        "OTR_SFX_STEM_MAX_BOOST_DB",
        "OTR_SFX_STEM_MAX_CUT_DB",
        "OTR_SFX_STEM_GATE_DBFS",
        "OTR_SFX_STEM_PEAK_CEILING",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sfx_prompt_helper_terms_and_idempotency():
    out = append_sfx_audio_safety_clause("rain on roof")
    lower = out.lower()
    for term in (
        "environmental ambience and foley only",
        "no spoken words",
        "no singing",
        "no lyrics",
        "no narration",
        "no voiceover",
        "no intelligible speech",
        "no subtitles",
        "no captions",
        "no readable text",
        "no explicit guns",
        "knives",
        "weapons",
        "nudity",
    ):
        assert term in lower
    assert append_sfx_audio_safety_clause(out) == out


def test_google_vid_sfx_engines_register_and_capabilities():
    for name, model in SFX.SFX_ENGINE_MODEL_IDS.items():
        assert vreg.is_registered(name)
        eng = vreg.get_engine(name)
        assert eng.model == model
        assert eng.family == "text_to_video"
        assert eng.required_inputs == ("text_prompt",)
        assert eng.provider_side is True
        assert eng.accepts_audio_ref is False
        row = vreg.CAPABILITIES[name]
        assert row == {
            "required_toolchain": None,
            "requires_sidecar": False,
            "device_backends": ["cuda", "cpu", "mps"],
            "requires_vendor": None,
            "needs_fp8_te": False,
            "needs_fp4_te": False,
            "practical_without_gpu": True,
            "sidecar_conditional": False,
            "model_requirements": [],
        }


def test_sfx_module_has_no_heavy_or_google_sdk_imports_at_module_scope():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    module_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.add(node.module.split(".")[0])
    assert "google" not in module_imports
    assert "PIL" not in module_imports
    assert "torch" not in module_imports
    assert "numpy" not in module_imports


def test_sfx_model_ids_ignore_hostile_env_vars(monkeypatch):
    _clear_video_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "KEY")
    monkeypatch.setenv("OTR_GOOGLE_VEO_MODEL_ID", "hostile-model")
    monkeypatch.setenv("OTR_GOOGLE_OMNI_MODEL_ID", "hostile-omni")
    monkeypatch.setenv("OTR_GOOGLE_VIDEO_MODEL_ID", "hostile-shared")
    seen = {}

    def _post(path, payload, **kwargs):
        seen["path"] = path
        seen["payload"] = payload
        return {"name": "operations/sfx-op"}

    def _get(path, **kwargs):
        return {
            "name": "operations/sfx-op",
            "done": True,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {"video": {"uri": "https://example.test/sfx.mp4"}}
                    ]
                }
            },
        }

    monkeypatch.setattr(SFX, "post_json", _post)
    monkeypatch.setattr(SFX, "get_json", _get)
    monkeypatch.setattr(SFX, "download_media", lambda uri, **kwargs: b"mp4")
    monkeypatch.setattr(
        SFX,
        "_pace_veo_sfx_submit",
        lambda *, engine_name, model: seen.setdefault("paced", (engine_name, model)),
    )
    vreg.get_engine("google_vid_sfx_veo_fast").render_clip(_req(), {})
    assert seen["paced"] == (
        "google_vid_sfx_veo_fast", "veo-3.1-fast-generate-preview")
    assert seen["path"] == (
        "/v1beta/models/veo-3.1-fast-generate-preview:predictLongRunning")
    assert seen["payload"]["parameters"]["generateAudio"] is True


def test_veo_sfx_submit_pacer_is_off_by_default_in_test_mode(monkeypatch):
    _clear_video_env(monkeypatch)
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    assert SFX._veo_sfx_submit_min_interval_s() == 0.0


def test_veo_sfx_submit_pacer_waits_between_live_submissions(monkeypatch):
    _clear_video_env(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_VEO_SFX_SUBMIT_MIN_INTERVAL_S", "65")
    monkeypatch.setattr(SFX, "_VEO_SFX_LAST_SUBMIT_AT", 0.0)
    state = {"now": 100.0}
    sleeps: list[float] = []

    monkeypatch.setattr(SFX.time, "monotonic", lambda: state["now"])

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(SFX.time, "sleep", _sleep)

    SFX._pace_veo_sfx_submit(
        engine_name="google_vid_sfx_veo_fast",
        model="veo-3.1-fast-generate-preview",
    )
    state["now"] += 10.0
    SFX._pace_veo_sfx_submit(
        engine_name="google_vid_sfx_veo_fast",
        model="veo-3.1-fast-generate-preview",
    )

    assert sleeps == pytest.approx([55.0])


def test_veo_sfx_payload_prompts_audio_and_base_veo_stays_silent(monkeypatch):
    _clear_video_env(monkeypatch)
    sfx_payload = SFX._veo_sfx_payload(
        "veo-3.1-lite-generate-preview",
        _req(),
        engine_name="google_vid_sfx_veo_lite",
    )
    base_payload = VEO._request_payload("veo-3.1-lite-generate-preview", _req())
    assert sfx_payload["parameters"]["generateAudio"] is True
    assert "generateAudio" not in base_payload["parameters"]
    assert "no spoken words" in sfx_payload["instances"][0]["prompt"].lower()
    with pytest.raises(GoogleAPIRequestShapeError, match="OTR audio"):
        SFX._veo_sfx_payload(
            "veo-3.1-lite-generate-preview",
            _req(audio_ref={"path": "voice.wav"}),
            engine_name="google_vid_sfx_veo_lite",
        )


def test_omni_sfx_payload_rejects_audio_ref():
    payload = SFX._omni_sfx_payload(
        "gemini-omni-flash-preview",
        _req(),
        engine_name="google_vid_sfx_omni",
    )
    assert payload["model"] == "gemini-omni-flash-preview"
    assert "no spoken words" in payload["input"].lower()
    with pytest.raises(GoogleAPIRequestShapeError, match="OTR audio"):
        SFX._omni_sfx_payload(
            "gemini-omni-flash-preview",
            _req(audio_ref={"path": "voice.wav"}),
            engine_name="google_vid_sfx_omni",
        )


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_extract_sfx_fails_loud_on_no_audio_provider_mp4(tmp_path):
    src = _mp4_fixture(tmp_path, with_audio=False)
    raw = {"path": str(src), "content_type": "video/mp4", "duration_s": None,
           "provider_job_id": "no-audio", "raw_meta": {}}
    with pytest.raises(CloudMediaError, match="SFX unavailable"):
        extract_sfx_bed_from_provider_video(raw)


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_extract_sfx_emits_normalized_48k_stereo_wav_from_provider_mp4(
    tmp_path, monkeypatch):
    _clear_sfx_loudness_env(monkeypatch)
    src = _mp4_fixture(tmp_path, with_audio=True)
    raw = {"path": str(src), "content_type": "video/mp4", "duration_s": None,
           "provider_job_id": "with-audio", "raw_meta": {}}
    asset = extract_sfx_bed_from_provider_video(raw)
    assert asset.path.suffix == ".wav"
    assert asset.path.name.endswith(".sfx.wav")
    assert asset.media_type == "audio"
    assert asset.provider_job_id == "with-audio"
    streams = _audio_streams(asset.path)
    assert len(streams) == 1
    assert streams[0]["codec_name"] == "pcm_s16le"
    assert streams[0]["sample_rate"] == "48000"
    assert int(streams[0]["channels"]) == 2
    assert any("sfx_rms_normalized target=-20.0dBFS" in w
               for w in asset.validation_warnings)
    assert -21.0 <= _audio_rms_dbfs(asset.path) <= -19.0


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_extract_sfx_boosts_low_google_bed_without_burying_it(
    tmp_path, monkeypatch):
    _clear_sfx_loudness_env(monkeypatch)
    src = _mp4_fixture(tmp_path, with_audio=True, audio_filter="volume=0.15")
    raw = {"path": str(src), "content_type": "video/mp4", "duration_s": None,
           "provider_job_id": "quiet-audio", "raw_meta": {}}

    asset = extract_sfx_bed_from_provider_video(raw)

    assert -21.0 <= _audio_rms_dbfs(asset.path) <= -19.0
    assert any("gain=" in w and "after=" in w
               for w in asset.validation_warnings)


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_sfx_canonicalize_returns_silent_clip_plus_sfx_stem(tmp_path):
    src = _mp4_fixture(tmp_path, with_audio=True)
    raw = {"path": str(src), "content_type": "video/mp4", "duration_s": None,
           "provider_job_id": "sfx-job", "raw_meta": {}}
    clip = vreg.get_engine("google_vid_sfx_veo_lite").canonicalize(raw, _req(), {})
    assert clip["engine_id"] == "google_vid_sfx_veo_lite"
    assert clip["has_audio"] is False
    assert pathlib.Path(clip["path"]).is_file()
    assert pathlib.Path(clip["sfx_stem_path"]).is_file()
    assert clip["sfx_duration_s"] > 0
    assert len(clip["sfx_sha256"]) == 64
    assert _audio_streams(pathlib.Path(clip["path"])) == []
    assert _audio_streams(pathlib.Path(clip["sfx_stem_path"]))


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_existing_google_video_engines_stay_silent(tmp_path):
    src = _mp4_fixture(tmp_path, with_audio=True)
    raw = {"path": str(src), "content_type": "video/mp4", "duration_s": None,
           "provider_job_id": "silent-job", "raw_meta": {}}
    for name, module in (
        ("google_veo_video", VEO),
        ("google_omni_video", OMNI),
    ):
        clip = vreg.get_engine(name).canonicalize(dict(raw), _req(), {})
        assert clip["engine_id"] == name
        assert clip["has_audio"] is False
        assert "sfx_stem_path" not in clip
        assert _audio_streams(pathlib.Path(clip["path"])) == []
