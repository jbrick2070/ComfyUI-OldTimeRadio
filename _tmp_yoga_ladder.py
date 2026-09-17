"""Yoga ladder: $0 preflight -> $2 Foley mux smoke -> $70 deluxe 1-act.

Does not touch :8188. Does not kill a busy :8000 queue.
Media cap $90. Sonnet creative, Luna tech, Luma stills, Sonilo, Foley+LTX.
My Story house playground + recur_frac.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

URL = "http://127.0.0.1:8000"
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
OUT = Path(r"C:\Users\jeffr\Documents\ComfyUI\output")
STILL = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
    r"\signal_lost_the_bird_of_dawning_20260915_200747"
    r"\stills\still_music_opening_001_c181a09bae3a.png"
)
PREFIX = "otr/obs/ltx25_foley_mux_smoke"
STATUS = ROOT / "tmp" / "yoga_ladder.json"
PY_DOCS = Path(r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe")
PACK_A = ROOT / "nodes" / "_otr_voice_node_common.py"
PACK_B = Path(
    r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI"
    r"\custom_nodes\comfyui-old-time-radio\nodes\_otr_voice_node_common.py"
)


def _hydrate_user_env(name: str) -> None:
    if os.environ.get(name):
        return
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        try:
            value, _ = winreg.QueryValueEx(key, name)
        finally:
            key.Close()
        if value:
            os.environ[name] = str(value)
    except OSError:
        return


def _queue():
    try:
        return requests.get(URL + "/queue", timeout=5).json()
    except Exception:
        return None


def _busy(q) -> bool:
    if not q:
        return False
    return bool(q.get("queue_running") or q.get("queue_pending"))


def _samefile_ok() -> bool:
    if not PACK_B.is_file():
        print("[yoga] FAIL install pack missing", PACK_B, flush=True)
        return False
    if not os.path.samefile(PACK_A, PACK_B):
        print("[yoga] FAIL pack copies are not the same file", flush=True)
        return False
    blob = PACK_A.read_text(encoding="utf-8")
    if "_floor_voice_line" not in blob or "spoken_text" not in blob:
        print("[yoga] FAIL floor helpers missing from pack", flush=True)
        return False
    print("[yoga] pack samefile + floor helpers OK", flush=True)
    return True


def _object_info_has_ltx() -> bool:
    try:
        info = requests.get(
            URL + "/object_info/LtxApi25ImageToVideo", timeout=15).json()
        return "LtxApi25ImageToVideo" in info
    except Exception as exc:
        print("[yoga] object_info LTX", type(exc).__name__, exc, flush=True)
        return False


def _object_info_has_foley_engine() -> bool:
    try:
        info = requests.get(
            URL + "/object_info/OTR_VideoDirector", timeout=20).json()
        node = info.get("OTR_VideoDirector") or {}
        inputs = ((node.get("input") or {}).get("required") or {})
        combo = inputs.get("announcer_video_model") or []
        options = combo[0] if combo and isinstance(combo[0], list) else []
        blob = " ".join(str(x) for x in options)
        return "cloud_ltx25_foley_plus" in blob
    except Exception as exc:
        print("[yoga] object_info Foley", type(exc).__name__, exc, flush=True)
        return False


def _recycle() -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["OTR_ENABLE_COMFY_CREDITS"] = "1"
    os.environ["OTR_CLOUD_FANOUT"] = "8"
    os.environ["OTR_CLOUD_VIDEO_FANOUT"] = "8"
    os.environ["OTR_CLOUD_MEDIA_BUDGET_USD"] = "90"
    os.environ["OTR_CLOUD_VIDEO_TIMEOUT_S"] = "1800"
    import _tmp_recycle_8000_fanout as recycle
    rc = int(recycle.main())
    if rc != 0:
        raise SystemExit("recycle rc=%s" % rc)
    deadline = time.time() + 120
    while time.time() < deadline:
        if _object_info_has_ltx() and _object_info_has_foley_engine():
            print("[yoga] :8000 has LTX partner + Foley engine", flush=True)
            return
        time.sleep(2)
    raise SystemExit("recycle finished but Foley/LTX object_info missing")


def _upload_image() -> str:
    with STILL.open("rb") as fh:
        resp = requests.post(
            URL + "/upload/image",
            files={"image": (STILL.name, fh, "image/png")},
            data={"overwrite": "true"},
            timeout=30,
        )
    resp.raise_for_status()
    name = str(resp.json().get("name") or STILL.name)
    print("[yoga] uploaded still=%s" % name, flush=True)
    return name


def _ltx_prompt(image_name: str) -> dict:
    return {
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "5": {
            "class_type": "LtxApi25ImageToVideo",
            "inputs": {
                "image": ["4", 0],
                "model": "LTX-2.5 (Fast)",
                "model.duration": "5",
                "model.resolution": "1920x1080",
                "model.fps": "25",
                "model.generate_audio": True,
                "prompt": (
                    "Use the start image as the first frame and keep this "
                    "radio, this wet stone, and this dusk. The camera dollies "
                    "in from a three-quarter angle as the three vacuum tubes "
                    "flare from amber to white-hot. The speaker cloth "
                    "vibrates. Orange dial light pulses. Wind pushes mist "
                    "across the battlement. Continuous uncut take, hard "
                    "sidelight, no text, no black frames, no voices."
                ),
                "seed": 42,
            },
        },
        "2": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["5", 0],
                "filename_prefix": PREFIX,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def _wait_history(prompt_id: str, timeout_s: float = 900) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = requests.get(URL + "/history/" + prompt_id, timeout=20).json()
        row = hist.get(prompt_id) if isinstance(hist, dict) else None
        if row:
            status = ((row.get("status") or {}).get("status_str") or "")
            if status in ("success", "error"):
                return row
        print("[yoga] waiting partner clip %s" % prompt_id, flush=True)
        time.sleep(8)
    raise SystemExit("partner clip timeout %s" % prompt_id)


def _find_mp4(row: dict) -> Path:
    outputs = row.get("outputs") or {}
    names = []
    for node in outputs.values():
        for key in ("gifs", "videos", "images"):
            for item in node.get(key) or []:
                name = item.get("filename") or item.get("name")
                sub = item.get("subfolder") or ""
                if name:
                    names.append((sub, name))
    for sub, name in names:
        cand = OUT / sub / name if sub else OUT / name
        if cand.is_file() and cand.suffix.lower() in {".mp4", ".webm", ".mkv"}:
            return cand
        obs = OBS / name
        if obs.is_file():
            return obs
    # Prefix search
    hits = sorted(OBS.glob("ltx25_foley_mux_smoke*"), key=lambda p: p.stat().st_mtime)
    if hits:
        return hits[-1]
    raise SystemExit("partner clip succeeded but no video file in %s" % names)


def _mux_smoke(video_path: Path) -> None:
    from nodes._otr_video_engines import foley_stems as fs
    from nodes._otr_shared import ffprobe as _fp

    work = ROOT / "tmp" / "yoga_foley_mux"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    harvest = work / "smoke.src.wav"
    stem = work / "smoke_foley.wav"
    assembled = work / "smoke_beat_foley.wav"
    master_path = work / "smoke_master.wav"
    fs.extract_pcm16_wav_from_video(str(video_path), str(harvest))
    arr, rate = fs.read_pcm16_wav(str(harvest))
    if arr.size == 0 or float(abs(arr).max()) == 0.0:
        raise SystemExit("FAIL harvested Foley is silent")
    probe = _fp.probe_json(str(video_path), entries="stream=nb_frames,avg_frame_rate,codec_type")
    frames = 0
    fps = 25
    for stream in probe.get("streams") or []:
        if str(stream.get("codec_type") or "") != "video":
            continue
        try:
            frames = int(stream.get("nb_frames") or 0)
        except (TypeError, ValueError):
            frames = 0
        raw = str(stream.get("avg_frame_rate") or "25/1")
        if "/" in raw:
            num, den = raw.split("/", 1)
            if float(den):
                fps = max(1, int(round(float(num) / float(den))))
    if frames < 1:
        # 5s * 25fps is the asked length
        frames = 125
    matched = fs.conform_stem_to_frame_count(arr, rate, frames, fps)
    n_samples, n_ch = fs.write_pcm16_wav(str(stem), matched, rate)
    receipts = fs.assemble_beat_foley_segments(
        [(str(stem), 0, frames)],
        str(assembled),
        expect_frames=frames,
        fps=fps,
    )
    muxed = fs.mux_native_audio_into_beat_clip(
        str(video_path), receipts["foley_path"], fps=fps)
    # Short master the length of the beat, then mix at 0.50.
    import numpy as np
    master = np.full((n_ch, n_samples), 0.05, dtype=np.float32)
    fs.write_pcm16_wav(str(master_path), master, rate)
    mixed, stats = fs.mix_foley_under_master(
        master, rate,
        [{
            "foley_path": receipts["foley_path"],
            "start_s": 0.0,
            "start_s_space": "master_mix",
            "frame_count": frames,
            "engine_id": "cloud_ltx25_foley_plus",
        }],
        fps=fps,
        lane_ids=fs.route_lane_ids(json.dumps({
            "effective_video_models": {
                "character": "cloud_ltx25_foley_plus",
            }
        })),
    )
    if int(stats.get("placed") or 0) < 1:
        raise SystemExit("FAIL mix placed 0 Foley windows: %s" % stats)
    if float(abs(mixed).max()) == 0.0:
        raise SystemExit("FAIL mixed master is silence")
    print(
        "[yoga] MUX SMOKE OK frames=%d fps=%d harvest=%dHz x%d placed=%s "
        "gain=%s av=%.3f/%.3f"
        % (frames, fps, rate, n_ch, stats.get("placed"),
           stats.get("global_master_gain"),
           muxed.get("audio_duration_s"), muxed.get("video_duration_s")),
        flush=True,
    )


def _partner_and_mux() -> str:
    key = (os.environ.get("OTR_COMFY_API_KEY") or "").strip()
    if not key:
        raise SystemExit("OTR_COMFY_API_KEY missing")
    if not STILL.is_file():
        raise SystemExit("missing Luma still %s" % STILL)
    image_name = _upload_image()
    client_id = str(uuid.uuid4())
    resp = requests.post(
        URL + "/prompt",
        json={
            "prompt": _ltx_prompt(image_name),
            "client_id": client_id,
            "extra_data": {"api_key_comfy_org": key},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit("POST /prompt HTTP %s %s" % (
            resp.status_code, resp.text[:500]))
    body = resp.json()
    if body.get("error") or body.get("node_errors"):
        raise SystemExit("submit error %s" % body)
    prompt_id = body.get("prompt_id")
    print("[yoga] partner QUEUED %s (~$2, 5s Foley)" % prompt_id, flush=True)
    row = _wait_history(prompt_id)
    status = ((row.get("status") or {}).get("status_str") or "")
    if status != "success":
        print(json.dumps(row.get("status") or {}, indent=2)[:2000], flush=True)
        raise SystemExit("partner clip RESULT %s" % status)
    video = _find_mp4(row)
    print("[yoga] partner clip on disk %s (%d bytes)" % (
        video, video.stat().st_size), flush=True)
    _mux_smoke(video)
    return prompt_id


def _submit_deluxe() -> str:
    import _tmp_submit_cloud as cloud

    os.environ.setdefault("OTR_ENABLE_COMFY_CREDITS", "1")
    os.environ.setdefault("OTR_CLOUD_FANOUT", "8")
    os.environ.setdefault("OTR_CLOUD_VIDEO_FANOUT", "8")
    os.environ.setdefault("OTR_CLOUD_MEDIA_BUDGET_USD", "90")
    sys.argv = [
        "_tmp_submit_cloud.py",
        "--workflow",
        str(ROOT / "workflows" / "variants" / "otr_cloud_deluxe_7act.json"),
        "--act-count", "1",
        "--comfyui-url", URL,
        "--source-bank", "my_story",
        "--visual-style", "recur_frac",
        "--source-ref", "",
        "--no-wait",
    ]
    # Patch slot A to Sonnet after argparse by wrapping main -- do it here.
    orig_args = cloud._Args

    class _ArgsSonnet(orig_args):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.set.append(
                "OTR_LedgerScriptWriter.comfy_slot_a_model="
                "anthropic/claude-sonnet-5")

    cloud._Args = _ArgsSonnet
    try:
        rc = int(cloud.main())
    finally:
        cloud._Args = orig_args
    if rc != 0:
        raise SystemExit("deluxe submit rc=%s" % rc)
    return "see-stdout"


def main() -> int:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["OTR_ENABLE_COMFY_CREDITS"] = "1"
    os.environ["OTR_CLOUD_FANOUT"] = "8"
    os.environ["OTR_CLOUD_VIDEO_FANOUT"] = "8"
    os.environ["OTR_CLOUD_MEDIA_BUDGET_USD"] = "90"
    os.environ["OTR_CLOUD_VIDEO_TIMEOUT_S"] = "1800"
    os.environ.setdefault(
        "OTR_OBS_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
    os.environ.setdefault(
        "OTR_OUTPUT_DIR", r"C:\Users\jeffr\Documents\ComfyUI\output")
    _hydrate_user_env("OTR_COMFY_API_KEY")
    if not (os.environ.get("OTR_COMFY_API_KEY") or "").strip():
        print("[yoga] FAIL no OTR_COMFY_API_KEY", flush=True)
        return 3
    if not _samefile_ok():
        return 3
    q = _queue()
    if q is not None and _busy(q):
        print("[yoga] FAIL :8000 busy -- not recycling, not spending", flush=True)
        return 3
    print("[yoga] step 0 recycle :8000 onto floor commit", flush=True)
    _recycle()
    print("[yoga] step 1 partner Foley + episode-mux functions", flush=True)
    smoke_id = _partner_and_mux()
    print("[yoga] smokes passed -- queueing $70 deluxe 1-act", flush=True)
    # Re-read queue; smoke should be finished.
    q = _queue()
    if q is not None and _busy(q):
        print("[yoga] FAIL queue busy after smoke", flush=True)
        return 3
    sys.argv = [
        "_tmp_submit_cloud.py",
        "--workflow",
        str(ROOT / "workflows" / "variants" / "otr_cloud_deluxe_7act.json"),
        "--act-count", "1",
        "--comfyui-url", URL,
        "--source-bank", "my_story",
        "--visual-style", "recur_frac",
        "--source-ref", "",
        "--no-wait",
    ]
    import _tmp_submit_cloud as cloud
    orig = cloud._Args

    class _ArgsSonnet(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.set.append(
                "OTR_LedgerScriptWriter.comfy_slot_a_model="
                "anthropic/claude-sonnet-5")

    cloud._Args = _ArgsSonnet
    try:
        rc = int(cloud.main())
    finally:
        cloud._Args = orig
    payload = {
        "smoke_prompt_id": smoke_id,
        "smoke": "pass",
        "deluxe_workflow": "otr_cloud_deluxe_7act",
        "act_count": "1",
        "source_bank": "my_story",
        "visual_style": "recur_frac",
        "writer": "anthropic/claude-sonnet-5 + luna",
        "stills": "cloud_luma_photon_flash",
        "music": "sonilo",
        "video": "cloud_ltx25_foley_plus",
        "budget_usd": 90,
        "submit_rc": rc,
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[yoga] wrote %s" % STATUS, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
