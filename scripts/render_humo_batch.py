#!/usr/bin/env python3
"""HuMo batch orchestrator (Pattern B).

Reads a Production Ledger JSON + master episode WAV, slices per-line audio,
copies portraits into ComfyUI's input dir, and POSTs HuMo prompts to a
running ComfyUI server one at a time. ComfyUI's model cache stays warm
between prompts, so cold-load cost (~50s) amortizes across N clips and
each subsequent clip pays only sampling + decode cost.

Usage (production):
  python scripts/render_humo_batch.py \\
      --ledger output/old_time_radio/signal_lost_<title>_ledger.json \\
      --master-wav output/old_time_radio/signal_lost_<title>.wav \\
      --portraits-dir output \\
      --out-dir output/old_time_radio/<title>_humo \\
      --comfyui-url http://localhost:8000 \\
      --scope all

Usage (test with auto-slice — when ledger has no per-line timing):
  python scripts/render_humo_batch.py \\
      --ledger <ledger.json> \\
      --master-wav <master.wav> \\
      --portraits-dir output \\
      --out-dir output/<title>_humo \\
      --auto-slice 3.88 \\
      --max-clips 5

Scope options:
  all                  every line in ledger.lines[]
  first-per-scene      first line of each scene
  cold-open            just the first line
  custom:l001,l005,l012  comma-separated line_ids

Architecture: pure Python stdlib + ffmpeg via subprocess. No new ComfyUI
custom nodes. Uses TEST_humo workflow JSON as the prompt template, swaps
LoadImage / LoadAudio / CLIPTextEncode / SaveVideo per clip.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HuMo batch orchestrator (Pattern B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ledger",
        required=True,
        type=Path,
        help="Path to signal_lost_*_ledger.json",
    )
    p.add_argument(
        "--master-wav",
        type=Path,
        default=None,
        help="Path to master episode WAV. If omitted, the ledger's "
        "final_audio_path is used. If that's empty, the master MP4's "
        "audio track is extracted.",
    )
    p.add_argument(
        "--portraits-dir",
        type=Path,
        default=Path(r"C:\Users\jeffr\Documents\ComfyUI\output"),
        help="Directory containing PASS1 character portraits "
        "(otr_humo_pass1_portrait_*.png).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where finished HuMo MP4s land.",
    )
    p.add_argument(
        "--comfyui-url",
        default="http://localhost:8000",
        help="Running ComfyUI server URL.",
    )
    p.add_argument(
        "--comfyui-input-dir",
        type=Path,
        default=Path(r"C:\Users\jeffr\Documents\ComfyUI\input"),
        help="ComfyUI's input directory (where LoadImage/LoadAudio look).",
    )
    p.add_argument(
        "--scope",
        default="all",
        help="all / first-per-scene / cold-open / custom:l001,l005,...",
    )
    p.add_argument(
        "--auto-slice",
        type=float,
        default=None,
        help="If ledger lines lack timing, auto-slice master WAV into "
        "windows of this length (seconds). Recommended: 3.88 for HuMo "
        "native length=97.",
    )
    p.add_argument(
        "--clip-length",
        type=float,
        default=7.0,
        help="HuMo clip duration in seconds. Default 7.0 lands at 175 "
        "frames -> 177 (clamped by Wan 2.1 VAE 4n+1 rule = 7.08s "
        "actual). HuMo's empirical sweet spot on RTX 5080 16 GB at "
        "640x640 fp8. Older runs used 3.88s (length=97); 7.0s is "
        "twice the line density per clip and matches Jeffrey's "
        "FULL-workflow spec. NOTE: when --auto-slice is set to a "
        "different value, the audio stride uses --auto-slice but the "
        "audio window per clip is hardcoded to --clip-length below "
        "(line ~749), which means --auto-slice and --clip-length "
        "must match to avoid overlap or gaps. The wrapper "
        "test_humo_batch_concat.ps1 passes -AutoSlice 7.0 to keep "
        "them aligned by default.",
    )
    p.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Cap on how many clips to render (for test runs).",
    )
    p.add_argument(
        "--workflow-template",
        type=Path,
        default=Path("workflows/otr_videoplan_TEST_humo.json"),
        help="GUI-format workflow JSON to clone the HuMo subgraph from.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan, don't submit any prompts.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between /history polls while a prompt runs.",
    )
    p.add_argument(
        "--prompt-timeout",
        type=float,
        default=900.0,
        help="Hard cap on a single prompt's wall-clock (seconds). At "
        "6:15/clip baseline we're well under this.",
    )
    # Goal 3 daisy-chain mechanic. Honours per-clip `boundary` field on
    # ledger lines (shot_start / beat_start / beat_resume / continue):
    #   shot_start  : ref_image = clean speaker portrait (full reset)
    #   beat_start  : ref_image = clean speaker portrait (clean reset, new
    #                 speaker in this shot, no chain)
    #   beat_resume : ref_image = this speaker's last frame from earlier in
    #                 this shot (cinematic continuity on speaker return)
    #   continue    : ref_image = previous clip's last frame
    # MVP: naive chain (no alpha blend). Hybrid alpha blend is layered on
    # later once Stage 0/1/2 metrics from the continuity brief land.
    # `--no-chain` falls back to fresh-portrait-every-clip (the pre-Goal-3
    # behaviour) for A/B comparison.
    p.add_argument(
        "--chain",
        dest="chain",
        action="store_true",
        default=True,
        help="(default) Honour ledger boundary field for daisy-chain anchoring.",
    )
    p.add_argument(
        "--no-chain",
        dest="chain",
        action="store_false",
        help="Disable daisy-chain; every clip uses the clean speaker portrait.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------

def load_ledger(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_lines(lines: list[dict], scope: str, scenes: list[dict]) -> list[dict]:
    """Apply --scope filter to ledger.lines."""
    if not lines:
        return []
    if scope == "all":
        return lines
    if scope == "cold-open":
        return [lines[0]]
    if scope == "first-per-scene":
        # First line in each scene (by scene_id appearance)
        seen: set[str] = set()
        out: list[dict] = []
        for ln in lines:
            scene_id = ln.get("scene_id") or ""
            if scene_id and scene_id not in seen:
                seen.add(scene_id)
                out.append(ln)
        return out
    if scope.startswith("custom:"):
        wanted = {s.strip() for s in scope.split(":", 1)[1].split(",") if s.strip()}
        return [ln for ln in lines if str(ln.get("line_id")) in wanted]
    raise ValueError(f"unknown --scope value: {scope!r}")


def find_portrait_for_speaker(
    speaker: str | None,
    cast: list[dict],
    portraits_dir: Path,
) -> Path | None:
    """Find the PASS1 portrait PNG that best matches `speaker`.

    Strategy:
      1. cast[].portrait_path if populated
      2. otr_humo_pass1_portrait_*.png by index in cast list
      3. otr_videoplan_pass3_*.png as fallback (composite shots)
    """
    if not speaker:
        speaker = ""
    speaker_norm = speaker.upper().strip()

    # 1. Direct path from ledger cast entry
    for c in cast:
        if (c.get("name") or "").upper().strip() == speaker_norm:
            p = c.get("portrait_path")
            if p and Path(p).exists():
                return Path(p)

    # 2. PASS1 humo portraits indexed by cast position
    cast_names = [(c.get("name") or "").upper().strip() for c in cast]
    if speaker_norm in cast_names:
        idx = cast_names.index(speaker_norm)
        candidates = sorted(
            list(portraits_dir.glob("otr/stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_humo_pass1_portrait_*.png"))
        )
        # Use modulo so even if we have fewer portraits than cast, we
        # always return SOMETHING.
        if candidates:
            return candidates[idx % len(candidates)]

    # 3. Any HuMo portrait, first one
    candidates = sorted(portraits_dir.glob("otr_humo_pass1_portrait_*.png"))
    if candidates:
        return candidates[0]

    # 4. Last resort: any PNG in input dir
    return None


# Slug rule must stay in lockstep with render_flux_batch.slugify so the
# composite filenames written by FLUX get found by HuMo. Both functions
# lowercase, replace runs of non-[a-z0-9] with `_`, strip leading and
# trailing underscores.
_SLUG_RE_CONSUME = re.compile(r"[^a-z0-9]+")


def _composite_slug(s: str, limit: int = 40) -> str:
    s = (s or "").strip().lower()
    s = _SLUG_RE_CONSUME.sub("_", s).strip("_")
    return (s[:limit] or "unnamed")


def find_composite_for_shot_speaker(
    shot_id: str | None,
    speaker: str | None,
    portraits_dir: Path,
) -> Path | None:
    """Find the per-(shot, speaker) FLUX composite written by render_flux_batch.

    Filename convention from render_flux_batch.build_target_list:
        otr_humo_pass3_<shot_slug>_<speaker_slug>_*.png
    where shot_slug truncates at 24 chars and speaker_slug at 40 chars.

    Returns the newest matching PNG, or None if no composite is on disk
    yet -- caller falls back to the pass1 portrait so the run still
    proceeds against legacy data.
    """
    if not shot_id or not speaker:
        return None
    shot_slug = _composite_slug(shot_id, limit=24)
    speaker_slug = _composite_slug(speaker, limit=40)
    # 2026-04-26: pass3 outputs nested under output/otr/stills/. Look
    # there first, then mid-day's flat output/otr_stills/, then the
    # original root-output location for backwards compatibility with
    # all generations of renders.
    new_pat = f"otr/stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    mid_pat = f"otr_stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    legacy_pat = f"otr_humo_pass3_{shot_slug}_{speaker_slug}_*.png"
    candidates = sorted(
        list(portraits_dir.glob(new_pat))
        + list(portraits_dir.glob(mid_pat))
        + list(portraits_dir.glob(legacy_pat)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# ffmpeg slicing
# ---------------------------------------------------------------------------

def slice_audio(
    src_wav_or_mp4: Path,
    start_s: float,
    duration_s: float,
    out_wav: Path,
) -> None:
    """Slice a mono 16 kHz PCM WAV from src starting at start_s for duration_s."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_s}",
        "-i", str(src_wav_or_mp4),
        "-t", f"{duration_s}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_master_audio(mp4_path: Path, out_wav: Path) -> Path:
    """Extract a full mono-16k WAV from an episode MP4."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp4_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


def get_audio_duration(path: Path) -> float:
    """ffprobe duration of an audio/video file."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def extract_last_frame(mp4_path: Path, out_png: Path) -> Path:
    """Extract the last frame of an MP4 as a PNG.

    Uses ffmpeg with `-sseof -0.04` (seek to 40 ms before EOF) plus
    `-frames:v 1 -update 1` to grab a single image. Output is a PNG
    next to the MP4 (or wherever ``out_png`` points). Used by the
    Goal 3 daisy-chain mechanic to feed clip N's terminal frame as
    the next clip's reference image.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof", "-0.04",
        "-i", str(mp4_path),
        "-frames:v", "1",
        "-update", "1",
        str(out_png),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_png


# ---------------------------------------------------------------------------
# ComfyUI API
# ---------------------------------------------------------------------------

class ComfyClient:
    """Minimal ComfyUI HTTP client (stdlib only)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def submit_prompt(self, api_prompt: dict) -> str:
        """Submit an API-format prompt graph. Returns prompt_id."""
        body = {"prompt": api_prompt, "client_id": self.client_id}
        out = self._post_json("/prompt", body)
        if "prompt_id" not in out:
            raise RuntimeError(f"ComfyUI rejected prompt: {out}")
        return out["prompt_id"]

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout_s: float,
        poll_interval_s: float,
    ) -> dict:
        """Poll /history/<id> until it completes or we time out.
        Returns the history entry."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                hist = self._get_json(f"/history/{prompt_id}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    time.sleep(poll_interval_s)
                    continue
                raise
            entry = hist.get(prompt_id)
            if entry and entry.get("status", {}).get("completed"):
                return entry
            time.sleep(poll_interval_s)
        raise TimeoutError(f"prompt {prompt_id} did not complete in {timeout_s}s")

    def get_queue_state(self) -> tuple[int, int]:
        """Returns (running_count, pending_count)."""
        q = self._get_json("/queue")
        return len(q.get("queue_running", [])), len(q.get("queue_pending", []))


# ---------------------------------------------------------------------------
# HuMo prompt builder
# ---------------------------------------------------------------------------

# Wan 2.1 VAE compresses 4 frames into 1 latent, so HuMo's `length` widget
# must satisfy (length - 1) % 4 == 0. Common valid values:
#   length=97  -> 3.88s @ 25fps  (yesterday's "stable floor" baseline)
#   length=113 -> 4.52s
#   length=125 -> 5.00s
#   length=153 -> 6.12s
#   length=177 -> 7.08s          (verified working at 640x640 fp8 2026-04-25)
HUMO_FPS = 25
HUMO_MIN_FRAMES = 33   # smaller frame counts have hung this hardware; treat as floor
HUMO_MAX_FRAMES = 177  # last empirically verified value on RTX 5080 Laptop 16GB


def humo_length_for_dur(dur_s: float, *, fps: int = HUMO_FPS) -> int:
    """Pick the smallest valid HuMo `length` >= round(dur_s * fps).

    Wan 2.1 VAE temporal compression requires length = 4n + 1.
    Floored at HUMO_MIN_FRAMES (33) -- below that has hung this hardware.
    Capped at HUMO_MAX_FRAMES (177) -- last empirically verified ceiling.
    """
    target = max(1, round(float(dur_s) * fps))
    # ceil((target - 1) / 4) gives the smallest n such that 4n + 1 >= target
    n = (target - 1 + 3) // 4
    frames = 4 * n + 1
    if frames < HUMO_MIN_FRAMES:
        frames = HUMO_MIN_FRAMES
    if frames > HUMO_MAX_FRAMES:
        frames = HUMO_MAX_FRAMES
    return frames


# API-format prompt for the HuMo subgraph (no FLUX). Built fresh per clip
# with the per-clip widget overrides.
def build_humo_prompt(
    image_filename: str,
    audio_filename: str,
    positive_prompt: str,
    save_prefix: str,
    seed: int,
    width: int = 480,
    height: int = 832,
    length: int = 97,
    steps: int = 6,
    cfg: float = 1.0,
) -> dict:
    """Construct ComfyUI API-format prompt for one HuMo clip.

    Mirrors the HuMo subgraph wiring from
    workflows/otr_videoplan_TEST_humo.json (nodes 7..21):

      UNETLoader(HuMo 14B fp8 e4m3fn scaled, Kijai)
        -> LoraLoaderModelOnly(lightx2v)
          -> ModelSamplingSD3(shift=8)
            -> KSampler

      CLIPLoader(umt5_xxl)
        -> CLIPTextEncode pos -+
        -> CLIPTextEncode neg -+-> WanHuMoImageToVideo

      VAELoader(wan_2.1_vae) -+-> WanHuMoImageToVideo
                              +-> VAEDecode

      LoadImage -> WanHuMoImageToVideo.ref_image
      LoadAudio -> AudioEncoderEncode + CreateVideo
      AudioEncoderLoader(whisper) -> AudioEncoderEncode -> WanHuMoImageToVideo

      KSampler -> VAEDecode -> CreateVideo -> SaveVideo

    Negative prompt is the ByteDance Chinese default — empirically the
    best HuMo negative on the official template.
    """
    chinese_negative = (
        "色调艳丽，过曝，静态，"
        "细节模糊不清，字幕，风格"
        "，作品，画作，画面，静止"
        "，整体发灰，最差质量，低"
        "质量，JPEG压缩残留，丑陋的"
        "，残缺的，多余的手指，画"
        "得不好的手部，画得不好的"
        "脸部，畸形的，毁容的，形"
        "态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背"
        "景，三条腿，背景人很多，"
        "倒着走"
    )

    return {
        # 7: UNETLoader (HuMo 14B fp8 e4m3fn scaled — Kijai mirror)
        # Production HuMo per ROADMAP Hardware Floor scoring (37.5/45 vs
        # 17B fp8's 36/45). Validated in the RunningHub HuMo+InfiniteTalk
        # A3 stack and tuned by Kijai for 16 GB cards.
        # Fallbacks (per ROADMAP):
        #   - "humo_17B_fp8_e4m3fn.safetensors" (UNETLoader) — 36/45, slower
        #   - "Wan2_1-HuMo-17B_Q5_K_M.gguf" (UnetLoaderGGUF) — 30/45, smoke runs
        "7": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                "weight_dtype": "default",
            },
            "_meta": {"title": "Load HuMo 14B fp8 e4m3fn scaled (Kijai)"},
        },
        # 8: LoraLoaderModelOnly (lightx2v)
        "8": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
                "strength_model": 1.0,
                "model": ["7", 0],
            },
        },
        # 9: ModelSamplingSD3 (shift=8)
        "9": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": 8.0, "model": ["8", 0]},
        },
        # 10: CLIPLoader (umt5_xxl)
        "10": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        # 11: CLIPTextEncode positive
        "11": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["10", 0]},
        },
        # 12: CLIPTextEncode negative
        "12": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": chinese_negative, "clip": ["10", 0]},
        },
        # 13: VAELoader (wan_2.1_vae)
        "13": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        },
        # 14: LoadAudio
        "14": {
            "class_type": "LoadAudio",
            "inputs": {"audio": audio_filename},
        },
        # 15: AudioEncoderLoader (whisper)
        "15": {
            "class_type": "AudioEncoderLoader",
            "inputs": {"audio_encoder_name": "whisper_large_v3_fp16.safetensors"},
        },
        # 16: AudioEncoderEncode
        "16": {
            "class_type": "AudioEncoderEncode",
            "inputs": {"audio_encoder": ["15", 0], "audio": ["14", 0]},
        },
        # 17: LoadImage
        "17": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # 18: WanHuMoImageToVideo
        "18": {
            "class_type": "WanHuMoImageToVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "positive": ["11", 0],
                "negative": ["12", 0],
                "vae": ["13", 0],
                "audio_encoder_output": ["16", 0],
                "ref_image": ["17", 0],
            },
        },
        # 19: KSampler
        "19": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["9", 0],
                "positive": ["18", 0],
                "negative": ["18", 1],
                "latent_image": ["18", 2],
            },
        },
        # 20: VAEDecode
        "20": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["19", 0], "vae": ["13", 0]},
        },
        # 21: CreateVideo (25 fps)
        "21": {
            "class_type": "CreateVideo",
            "inputs": {
                "fps": 25.0,
                "images": ["20", 0],
                "audio": ["14", 0],
            },
        },
        # 22: SaveVideo
        "22": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": save_prefix,
                "format": "auto",
                "codec": "auto",
                "video": ["21", 0],
            },
        },
    }


# ---------------------------------------------------------------------------
# Per-line plan
# ---------------------------------------------------------------------------

def build_clip_plan(
    ledger: dict,
    args: argparse.Namespace,
    master_wav: Path,
) -> list[dict]:
    """Decide which lines to render and what audio/portrait/prompt each gets.

    Returns a list of dicts, one per clip:
      { line_id, speaker, audio_src_path, audio_filename_in_input,
        portrait_src_path, portrait_filename_in_input, positive_prompt,
        save_prefix, seed, start_s, dur_s }
    """
    cast = ledger.get("cast") or []
    scenes = ledger.get("scenes") or []
    lines = ledger.get("lines") or []

    # BUG-LOCAL-074 (Bug A): ledger lines carry `char_id`, not `speaker`.
    # production_ledger.set_lines (production_ledger.py:321-353) and
    # story_orchestrator.py:4719-4725 both write rows with `char_id` and
    # rely on cast[*].name for the human-readable name. Build a lookup so
    # plan entries can carry a real speaker name into prompt building,
    # portrait selection, and the daisy-chain map. Older legacy ledgers
    # that populated `speaker` directly still resolve via the ln.speaker
    # branch first (back-compat).
    cid_to_name: dict[str, str] = {
        (c.get("char_id") or ""): (c.get("name") or "")
        for c in cast
        if c.get("char_id")
    }

    selected = filter_lines(lines, args.scope, scenes)

    # If timing is missing AND --auto-slice is set, distribute clips
    # evenly across the master audio.
    use_auto_slice = bool(args.auto_slice)
    if use_auto_slice:
        if master_wav.exists():
            master_dur = get_audio_duration(master_wav)
        else:
            # Dry-run: master may not exist yet. Fall back to ledger
            # total or a generous default.
            master_dur = float(ledger.get("total_episode_dur_s") or 600.0)
    else:
        master_dur = 0.0

    plan: list[dict] = []
    for idx, ln in enumerate(selected):
        line_id = str(ln.get("line_id") or f"l{idx + 1:03d}")
        # Speaker resolution priority:
        #   1. ln.speaker (legacy/explicit override)
        #   2. cast[char_id == ln.char_id].name (current schema)
        #   3. "" (graceful fallback -- portrait lookup uses first PNG)
        speaker = (
            ln.get("speaker")
            or cid_to_name.get(ln.get("char_id") or "", "")
            or ""
        ).strip()

        # --- timing ---
        start_s = ln.get("start_s")
        dur_s = ln.get("dur_s")
        if (start_s is None or start_s == "") and use_auto_slice:
            stride = float(args.auto_slice)
            start_s = idx * stride
            dur_s = stride
        if start_s is None or start_s == "":
            print(
                f"[skip] line {line_id} has no timing and --auto-slice not set",
                file=sys.stderr,
            )
            continue
        # Always cap to clip-length so audio matches HuMo native window
        dur_s = float(args.clip_length)
        start_s = float(start_s)
        # Don't overrun master
        if use_auto_slice and start_s + dur_s > master_dur:
            print(
                f"[skip] line {line_id} starts past master end "
                f"({start_s:.2f} + {dur_s:.2f} > {master_dur:.2f})",
                file=sys.stderr,
            )
            continue

        # --- portrait ---
        portrait_src = find_portrait_for_speaker(speaker, cast, args.portraits_dir)
        if not portrait_src:
            print(
                f"[skip] line {line_id} speaker={speaker!r} has no portrait",
                file=sys.stderr,
            )
            continue

        # Per-(shot, speaker) composite from render_flux_batch.py — used as
        # the fresh-anchor reference for shot_start / beat_start clips when
        # one is on disk. Falls back to the cast portrait if no composite
        # has been rendered yet (back-compat with pre-FLUX-batch runs).
        composite_src = find_composite_for_shot_speaker(
            ln.get("shot_id"), speaker, args.portraits_dir,
        )

        # --- positive prompt ---
        # Try cast.description, fall back to a generic OTR scene line.
        speaker_desc = ""
        for c in cast:
            if (c.get("name") or "").upper().strip() == speaker.upper():
                speaker_desc = (c.get("description") or "").strip()
                break
        if not speaker_desc:
            speaker_desc = (
                f"A {speaker.lower()} character speaks calmly with subtle facial "
                f"expressions"
            ) if speaker else "A character speaks calmly with subtle facial expressions"
        positive_prompt = (
            f"{speaker_desc}, dimly lit interior, ambient cinematic lighting, "
            f"35mm film grain, shallow depth of field"
        )

        # --- staged input filenames ---
        portrait_filename = f"humo_clip_{line_id}_portrait.png"
        audio_filename = f"humo_clip_{line_id}_speech.wav"

        # 2026-04-26: outputs nested under output/otr/videos/<episode_id>/.
        # Single OTR parent dir keeps ComfyUI's output/ root tidy.
        # Per-episode subdir keyed by the silent_test ledger's
        # episode_id so multi-episode runs don't collide.
        _ep_id = ledger.get("episode_id", "episode")
        save_prefix = f"otr/videos/{_ep_id}/humo_{line_id}"
        seed = (idx * 1009 + 7) & 0x7FFFFFFFFFFFFFFF  # deterministic per-line

        plan.append({
            "line_id": line_id,
            "shot_id": ln.get("shot_id"),
            "beat_id": ln.get("beat_id"),
            "boundary": ln.get("boundary"),
            "speaker": speaker,
            "audio_src_path": str(master_wav),
            "audio_filename_in_input": audio_filename,
            "portrait_src_path": str(portrait_src),
            "composite_src_path": str(composite_src) if composite_src else None,
            "portrait_filename_in_input": portrait_filename,
            "positive_prompt": positive_prompt,
            "save_prefix": save_prefix,
            "seed": seed,
            "start_s": start_s,
            "dur_s": dur_s,
        })
    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # 1. Validate ledger + master WAV
    ledger = load_ledger(args.ledger)
    master_wav: Path
    if args.master_wav:
        master_wav = args.master_wav
    elif ledger.get("final_audio_path"):
        master_wav = Path(ledger["final_audio_path"])
    elif ledger.get("final_video_path"):
        # Extract audio from the master MP4
        mp4 = Path(ledger["final_video_path"])
        if not mp4.exists():
            print(f"FATAL: ledger final_video_path missing: {mp4}", file=sys.stderr)
            return 2
        extracted = args.out_dir / f"{ledger.get('episode_id', 'episode')}_master.wav"
        print(f"[info] extracting master audio: {mp4} -> {extracted}")
        if not args.dry_run:
            extract_master_audio(mp4, extracted)
        master_wav = extracted
    else:
        print("FATAL: no master WAV available (no --master-wav, no ledger.final_audio_path, no final_video_path)", file=sys.stderr)
        return 2

    if not args.dry_run and not master_wav.exists():
        print(f"FATAL: master WAV missing: {master_wav}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.comfyui_input_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build plan
    plan = build_clip_plan(ledger, args, master_wav)
    if args.max_clips:
        plan = plan[: args.max_clips]
    if not plan:
        print("FATAL: empty clip plan (no lines passed filter / had timing)", file=sys.stderr)
        return 3

    print(f"[plan] {len(plan)} clip(s) queued, target: {args.out_dir}")
    for c in plan:
        print(
            f"  {c['line_id']}  speaker={c['speaker']:<12}  "
            f"start={c['start_s']:6.2f}s  portrait={Path(c['portrait_src_path']).name}"
        )
    if args.dry_run:
        print("[dry-run] not submitting any prompts.")
        return 0

    # 3. Connect to ComfyUI
    client = ComfyClient(args.comfyui_url)
    try:
        running, pending = client.get_queue_state()
    except Exception as e:
        print(f"FATAL: cannot reach ComfyUI at {args.comfyui_url}: {e}", file=sys.stderr)
        return 4
    print(f"[comfy] queue state running={running} pending={pending}")

    # 4. Render each clip
    # Goal 3 daisy-chain state. Reset on shot boundary so per-speaker
    # last-frame mapping never leaks across shots (a returning speaker
    # in shot N+1 should NOT see their last frame from shot N).
    prev_last_frame_path: Path | None = None
    prev_shot_id: str | None = None
    speaker_last_frame_in_shot: dict[str, Path] = {}

    started_at = time.time()
    completed = 0
    for idx, clip in enumerate(plan, 1):
        line_id = clip["line_id"]
        clip_started = time.time()
        speaker = clip["speaker"]
        boundary = clip.get("boundary") or "shot_start"
        clip_shot_id = clip.get("shot_id")
        print(
            f"\n[{idx}/{len(plan)}] {line_id}  ({speaker})  boundary={boundary}"
        )

        # If we crossed into a new shot, drop the per-speaker map so a
        # returning speaker in the new shot won't chain from their old
        # shot's last frame.
        if prev_shot_id is not None and clip_shot_id != prev_shot_id:
            speaker_last_frame_in_shot.clear()

        # 4a. Stage reference image into ComfyUI input/
        # Anchor-mode lookup, in priority order per boundary type:
        #   shot_start / beat_start -> per-(shot, speaker) FLUX composite
        #     when present; otherwise fall back to clean cast portrait.
        #     The composite (otr_humo_pass3_<shot>_<speaker>) carries
        #     scene context so HuMo's first frame opens with the right
        #     framing/lighting. Falls back to the cast portrait
        #     (otr_humo_pass1_portrait_<speaker>) if FLUX hasn't rendered
        #     a composite yet, so legacy runs still work.
        #   beat_resume -> this speaker's own last frame from earlier in
        #     the same shot. Falls back to composite, then portrait.
        #   continue -> previous clip's last frame.
        composite_src = (
            Path(clip["composite_src_path"]) if clip.get("composite_src_path")
            else None
        )
        portrait_src = Path(clip["portrait_src_path"])

        def _fresh_anchor() -> tuple[Path, str]:
            if composite_src and composite_src.exists():
                return composite_src, f"composite ({composite_src.name})"
            return portrait_src, "clean portrait"

        chain_source: str = "clean portrait"  # for log only
        if args.chain:
            if boundary == "continue" and prev_last_frame_path \
                    and prev_last_frame_path.exists():
                ref_src = prev_last_frame_path
                chain_source = "prev clip last frame"
            elif boundary == "beat_resume":
                speaker_prev = speaker_last_frame_in_shot.get(speaker)
                if speaker_prev and speaker_prev.exists():
                    ref_src = speaker_prev
                    chain_source = (f"speaker last frame in shot "
                                    f"({speaker_prev.name})")
                else:
                    # Speaker is supposedly returning but we don't have a
                    # prior frame on hand (first run after restart, etc.).
                    # Fall back to composite-or-portrait so the run keeps
                    # going.
                    ref_src, fallback_label = _fresh_anchor()
                    chain_source = f"{fallback_label} (beat_resume fallback)"
            else:
                # shot_start, beat_start, or unknown boundary -- fresh anchor
                ref_src, chain_source = _fresh_anchor()
        else:
            ref_src, chain_source = _fresh_anchor()
            chain_source += " (--no-chain)"

        portrait_dst = args.comfyui_input_dir / clip["portrait_filename_in_input"]
        shutil.copy2(ref_src, portrait_dst)
        print(f"    ref      -> {portrait_dst.name}  [{chain_source}]")

        # 4b. Slice audio into ComfyUI input/
        audio_dst = args.comfyui_input_dir / clip["audio_filename_in_input"]
        slice_audio(
            Path(clip["audio_src_path"]),
            clip["start_s"],
            clip["dur_s"],
            audio_dst,
        )
        print(
            f"    audio    -> {audio_dst.name} "
            f"({clip['start_s']:.2f}s + {clip['dur_s']:.2f}s)"
        )

        # 4c. Build + submit prompt.
        # Per-line HuMo length: snap line.dur_s up to nearest 4n+1 frame count.
        # Lets a single ledger mix [7s, 4.5s, 5s, ...] logical clips in one run
        # without forcing a global clip_length. CLI --clip-length is still
        # honoured as a default when a line lacks dur_s.
        humo_length = (
            humo_length_for_dur(clip["dur_s"])
            if clip.get("dur_s")
            else int(args.clip_length * 25)
        )
        api_prompt = build_humo_prompt(
            image_filename=clip["portrait_filename_in_input"],
            audio_filename=clip["audio_filename_in_input"],
            positive_prompt=clip["positive_prompt"],
            save_prefix=clip["save_prefix"],
            seed=clip["seed"],
            length=humo_length,
        )
        print(
            f"    humo     -> length={humo_length} "
            f"({humo_length / HUMO_FPS:.2f}s, target dur_s={clip['dur_s']:.2f}s)"
        )
        try:
            prompt_id = client.submit_prompt(api_prompt)
        except Exception as e:
            print(f"    SUBMIT FAILED: {e}", file=sys.stderr)
            continue
        print(f"    prompt_id={prompt_id}")

        # 4d. Wait for completion
        try:
            entry = client.wait_for_completion(
                prompt_id,
                timeout_s=args.prompt_timeout,
                poll_interval_s=args.poll_interval,
            )
        except TimeoutError as e:
            print(f"    TIMEOUT: {e}", file=sys.stderr)
            continue
        clip_dur = time.time() - clip_started

        # 4e. Move output MP4 from ComfyUI's output dir to --out-dir.
        # SaveVideo writes to ComfyUI's output/<save_prefix>_<NNNNN>_.mp4
        # where save_prefix is built in build_clip_plan as
        # "otr/videos/<episode_id>/humo_<line_id>" (see line 781). The
        # earlier search hit otr/audio/humo_episode/ + old_time_radio/
        # humo_episode/, neither of which match the prefix; candidates
        # came back empty, the else-branch crashed on undefined
        # `comfy_output_root`, and the orchestrator died after clip 1.
        # BUG-LOCAL-074 (Bug B): point the search at the actual save
        # root and keep the legacy paths as a back-compat fallback.
        _comfy_output = Path(r"C:\Users\jeffr\Documents\ComfyUI\output")
        _ep_id = ledger.get("episode_id", "episode")
        save_root = _comfy_output / "otr" / "videos" / _ep_id
        search_roots = [
            save_root,
            _comfy_output / "otr" / "audio" / "humo_episode",
            _comfy_output / "old_time_radio" / "humo_episode",
        ]
        candidates = []
        for _root in search_roots:
            if _root.exists():
                # SaveVideo appends "_<NNNNN>_" before ".mp4". Match both
                # the current "humo_<line_id>_*.mp4" prefix and legacy
                # "<line_id>_*.mp4" so older episodes still resolve.
                candidates.extend(_root.glob(f"humo_{line_id}_*.mp4"))
                candidates.extend(_root.glob(f"{line_id}_*.mp4"))
        candidates = sorted(
            candidates, key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            src_mp4 = candidates[0]
            dst_mp4 = args.out_dir / f"{line_id}.mp4"
            shutil.move(str(src_mp4), str(dst_mp4))
            print(f"    DONE in {clip_dur:.1f}s -> {dst_mp4.name}")
            completed += 1

            # Goal 3 chain state update. Extract this clip's last frame
            # and stash it for the next clip's anchor decision. Always
            # update prev_last_frame_path (used by `continue` boundary)
            # and the per-speaker map for this shot (used by `beat_resume`).
            if args.chain:
                try:
                    last_frame_dst = (args.out_dir
                                      / f"{line_id}_last.png")
                    extract_last_frame(dst_mp4, last_frame_dst)
                    prev_last_frame_path = last_frame_dst
                    if speaker:
                        speaker_last_frame_in_shot[speaker] = last_frame_dst
                    print(f"    last_frame -> {last_frame_dst.name}")
                except subprocess.CalledProcessError as exc:
                    # Don't tank the run if ffmpeg blows up extracting the
                    # last frame -- next clip will fall back to portrait.
                    print(
                        f"    WARN: last-frame extract failed: {exc}",
                        file=sys.stderr,
                    )
                    prev_last_frame_path = None
            prev_shot_id = clip_shot_id
        else:
            print(
                f"    WARN: completed but no MP4 found in {save_root}",
                file=sys.stderr,
            )
            # Skip chain update so a missing MP4 doesn't poison the next clip.
            prev_shot_id = clip_shot_id

    elapsed = time.time() - started_at
    print(
        f"\n[done] {completed}/{len(plan)} clips in {elapsed:.1f}s "
        f"({elapsed / max(1, completed):.1f}s avg/clip)"
    )
    return 0 if completed == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
