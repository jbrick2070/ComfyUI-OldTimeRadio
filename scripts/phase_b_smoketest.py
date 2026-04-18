"""
phase_b_smoketest.py -- Minimal end-to-end test of Phase B anchor_gen wiring

Fires a tiny Bridge+Poll workflow against a running ComfyUI on port 8000,
then watches io/visual_out/<job_id>/ for:
  - STATUS.json evolution (SPAWNED -> PROCESSING -> READY / ERROR / WORKER_DEAD)
  - per-shot meta.json showing anchor_used/cache_hit/seed
  - render.png files actually landing on disk

Assumes the invoking shell has OTR_VISUAL_ANCHOR=sd15 set, which ComfyUI
inherited when launched. The sidecar worker reads the env var at module
import time (inside worker.py), so no per-job toggle is needed.

Usage (PowerShell):
    $env:OTR_VISUAL_ANCHOR = "sd15"
    & C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe `
        C:\\Users\\jeffr\\Documents\\ComfyUI\\custom_nodes\\ComfyUI-OldTimeRadio\\scripts\\phase_b_smoketest.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


COMFYUI = "http://127.0.0.1:8000"
OTR_ROOT = Path(__file__).resolve().parent.parent
IO_OUT = OTR_ROOT / "io" / "visual_out"
IO_IN = OTR_ROOT / "io" / "visual_in"
CLIENT_ID = f"phase_b_smoketest_{int(time.time())}"


# Minimal canonical script_json: 3 scenes, each with 1 dialogue line.
# Bridge validates that at least one canonical token type appears; this
# payload has 5 of them (title/scene_break/dialogue/environment).
SCRIPT_LINES = [
    {"type": "title", "text": "Phase B Smoketest"},
    {"type": "environment", "description": "Transmitter room. Static."},
    {"type": "dialogue", "character": "ANNOUNCER",
     "text": "Signal lost. Reception is unclear."},
    {"type": "scene_break"},
    {"type": "dialogue", "character": "ANNOUNCER",
     "text": "Stand by for emergency broadcast."},
]


def http_get_json(path: str):
    with urllib.request.urlopen(f"{COMFYUI}{path}", timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post_json(path: str, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def build_api_prompt() -> dict:
    """Minimal API-format prompt: Bridge + Poll + Renderer-as-terminator.

    Bridge spawns the sidecar worker; Poll blocks until STATUS.json
    writes a terminal status. We include Renderer with an empty shotlist
    and a non-existent audio path SOLELY to satisfy ComfyUI's
    prompt_no_outputs gate (the OUTPUT_NODE flag lives only on Renderer).
    With shotlist_json="{}" the Renderer short-circuits at the
    "no shot assets found" branch and never invokes ffmpeg, so this
    is a safe no-op terminator for the smoketest.
    """
    return {
        "1": {
            "class_type": "OTR_VisualBridge",
            "inputs": {
                "script_json": json.dumps(SCRIPT_LINES),
                # Suffix a timestamp so ComfyUI's execution_cached gate
                # does not short-circuit a repeat smoketest run. Without
                # this, identical inputs would return a cached job_id and
                # skip spawning the sidecar entirely.
                "episode_title": f"Phase B Smoketest {int(time.time())}",
                "production_plan_json": "{}",
                "scene_manifest_json": "{}",
                "lane": "faithful",
                "chaos_ops": "",
                "chaos_seed": int(time.time()) & 0xFFFF,
                "sidecar_enabled": True,
            },
        },
        "2": {
            "class_type": "OTR_VisualPoll",
            "inputs": {
                "visual_job_id": ["1", 0],
            },
        },
        "3": {
            "class_type": "OTR_VisualRenderer",
            "inputs": {
                "visual_assets_path": ["2", 0],
                "final_audio_path": "C:/nonexistent_smoketest_audio.wav",
                "shotlist_json": "{}",
                "episode_title": "Phase B Smoketest",
                "crt_postfx": False,
                "output_resolution": "960x540",
            },
        },
    }


def recent_jobs(since: float) -> list[Path]:
    """Return io/visual_out subdirs modified after `since`."""
    if not IO_OUT.exists():
        return []
    return sorted(
        [p for p in IO_OUT.iterdir() if p.is_dir() and p.stat().st_mtime >= since],
        key=lambda p: p.stat().st_mtime,
    )


def read_status(job_dir: Path) -> dict | None:
    f = job_dir / "STATUS.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def dump_shot_meta(job_dir: Path) -> None:
    """Print any per-shot meta.json files that mention anchor_*."""
    shots = sorted(
        [p for p in job_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()]
    )
    if not shots:
        print("  (no shot_dirs with meta.json yet)")
        return
    for shot in shots:
        meta_f = shot / "meta.json"
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  {shot.name}/meta.json UNREADABLE: {e}")
            continue
        anchor_used = meta.get("anchor_used", "(field missing)")
        cache_hit = meta.get("anchor_cache_hit", "(field missing)")
        seed = meta.get("anchor_seed", "(field missing)")
        render_png = shot / "render.png"
        png_size = render_png.stat().st_size if render_png.exists() else 0
        print(
            f"  {shot.name}: anchor_used={anchor_used} "
            f"cache_hit={cache_hit} seed={seed} "
            f"render.png={'MISSING' if png_size == 0 else f'{png_size}b'}"
        )


def main() -> int:
    print(f"=== Phase B anchor_gen smoketest ===")
    # NOTE: this prints the DRIVER's env, not ComfyUI's. The flag lives on
    # the ComfyUI process that spawned the sidecar worker; we cannot inspect
    # it over HTTP. Check ComfyUI's own launch shell if you need to confirm.
    print(f"  driver OTR_VISUAL_ANCHOR={os.environ.get('OTR_VISUAL_ANCHOR', '(unset in driver — must be set in ComfyUI shell)')}")
    print(f"  ComfyUI: {COMFYUI}")
    print(f"  Client:  {CLIENT_ID}")
    print(f"  Shots:   {len(SCRIPT_LINES)} script_lines")

    # Liveness check
    try:
        stats = http_get_json("/system_stats")
        free_mb = stats["devices"][0].get("vram_free", 0) // 1_000_000
        print(f"  VRAM free at start: {free_mb} MB")
    except Exception as e:
        print(f"  ComfyUI not reachable: {e}")
        return 2

    t0 = time.time()
    prompt = build_api_prompt()
    try:
        resp = http_post_json(
            "/prompt",
            {"prompt": prompt, "client_id": CLIENT_ID},
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"POST /prompt HTTP {e.code}: {body[:500]}")
        return 3
    except Exception as e:
        print(f"POST /prompt failed: {e}")
        return 3

    prompt_id = resp.get("prompt_id", "?")
    print(f"  prompt_id: {prompt_id}")
    print(f"  queued in {time.time() - t0:.2f}s")
    print()

    # Poll for job output directory to appear, then watch STATUS.json
    job_dir: Path | None = None
    deadline = time.time() + 60 * 15  # 15 minute overall ceiling
    last_status = None
    last_shot_dump = 0.0

    while time.time() < deadline:
        # Locate job dir (Bridge creates it after parse; so ~1-2s after POST)
        if job_dir is None:
            candidates = recent_jobs(t0 - 5)
            # Exclude dirs that obviously belong to other clients
            if candidates:
                job_dir = candidates[-1]
                print(f"  Job dir:  {job_dir.relative_to(OTR_ROOT)}")
        if job_dir is not None:
            s = read_status(job_dir)
            if s and s != last_status:
                status = s.get("status", "?")
                detail = s.get("detail", "")
                print(
                    f"  [{int(time.time() - t0):4d}s] STATUS={status:14s} "
                    f"detail={detail[:120]}"
                )
                last_status = s
                if status in {
                    "READY", "ERROR", "OOM", "TIMEOUT",
                    "ENV_NOT_FOUND", "WORKER_MISSING",
                    "SPAWN_FAILED", "DRY_RUN", "SIDECAR_UNAVAILABLE",
                    "WORKER_DEAD",
                }:
                    break
            # Every 30s, dump shot meta so user sees progress
            if time.time() - last_shot_dump > 30:
                print("  --- shot meta so far ---")
                dump_shot_meta(job_dir)
                last_shot_dump = time.time()

        time.sleep(1.0)

    # Final summary
    print()
    print(f"=== Summary (elapsed {time.time() - t0:.1f}s) ===")
    if job_dir is None:
        print("  NO JOB DIR created -- Bridge probably rejected script_json")
        return 4

    print(f"  Job: {job_dir.name}")
    final_status = read_status(job_dir) or {}
    print(f"  Final STATUS: {final_status.get('status', 'UNKNOWN')}")
    print(f"  Final detail: {final_status.get('detail', '')}")
    print()
    print("  Per-shot anchor meta:")
    dump_shot_meta(job_dir)
    print()
    print(f"  Full job dir: {job_dir}")

    return 0 if final_status.get("status") == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
