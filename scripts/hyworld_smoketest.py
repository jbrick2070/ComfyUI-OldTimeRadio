"""
hyworld_smoketest.py  --  HyWorld node smoke test (supersoaker pattern)
========================================================================
End-to-end smoke test for the v2.0 HyWorld integration nodes.
Exercises the full pipeline: Bridge -> Poll -> Renderer using
synthetic script_lines (no ComfyUI needed, no GPU needed).

This runs in the main Python env (not hyworld2).  It tests:
    1. shotlist.py deterministic mapping (pure Python)
    2. OTR_HyworldBridge contract file generation (dry run mode)
    3. worker.py stub execution (placeholder PNGs)
    4. OTR_HyworldPoll status detection
    5. OTR_HyworldRenderer ffmpeg-free path (asset collection + log)
    6. Round-trip JSON contract validity
    7. Fallback path (FALLBACK routing works without crash)

Strategic gates (halt + report):
    * Any exception in the pipeline
    * shotlist produces 0 shots from valid script_lines
    * Bridge writes 0-byte contract files
    * Worker stub produces 0 assets
    * Poll returns unexpected status
    * Renderer crashes on FALLBACK input

Usage:
    python scripts/hyworld_smoketest.py
    python scripts/hyworld_smoketest.py --verbose
    python scripts/hyworld_smoketest.py --cleanup    # remove io/ artifacts after

Automation:
    Desktop Commander can run this via:
        cd /d C:\\Users\\jeffr\\Documents\\ComfyUI\\custom_nodes\\ComfyUI-OldTimeRadio
        C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe scripts/hyworld_smoketest.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup — add repo root to path so we can import otr_v2
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_OTR_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_OTR_ROOT))

# ---------------------------------------------------------------------------
# Synthetic test data — realistic OTR script_lines
# ---------------------------------------------------------------------------
SYNTHETIC_SCRIPT_LINES = [
    {"type": "title", "value": "The Last Signal from Vault 7"},
    {"type": "scene_break", "scene": "1"},
    {"type": "environment", "description": "fluorescent hum, distant traffic, rain on concrete"},
    {
        "type": "dialogue",
        "character_name": "COMMANDER",
        "voice_traits": "male, 50s, weary",
        "line": "Hold the line. I repeat, hold the line.",
    },
    {
        "type": "dialogue",
        "character_name": "LIEUTENANT",
        "voice_traits": "female, 30s, frantic",
        "line": "Sir, the outer hull is compromised. We have maybe four minutes.",
    },
    {"type": "sfx", "description": "metal stress groaning, deep bass shudder"},
    {"type": "pause", "kind": "beat", "duration_ms": 200},
    {
        "type": "dialogue",
        "character_name": "COMMANDER",
        "voice_traits": "male, 50s, weary",
        "line": "Four minutes is a long time when you have nowhere left to go.",
    },
    {"type": "scene_break", "scene": "2"},
    {"type": "environment", "description": "vacuum silence, faint radio static, creaking bulkhead"},
    {
        "type": "dialogue",
        "character_name": "DR_VALE",
        "voice_traits": "female, 40s, calm",
        "line": "The readings are stable. Whatever it is, it is not hostile.",
    },
    {"type": "sfx", "description": "electronic chirp sequence, ascending pitch"},
    {
        "type": "dialogue",
        "character_name": "COMMANDER",
        "voice_traits": "male, 50s, weary",
        "line": "Define not hostile.",
    },
    {"type": "pause", "kind": "beat", "duration_ms": 200},
    {
        "type": "dialogue",
        "character_name": "DR_VALE",
        "voice_traits": "female, 40s, calm",
        "line": "It hasn't killed us yet. That is the entirety of what I know.",
    },
    {"type": "scene_break", "scene": "3"},
    {"type": "environment", "description": "deep engine thrum through bulkhead, red emergency lighting"},
    {"type": "sfx", "description": "hatch seal disengaging with pneumatic hiss"},
    {
        "type": "dialogue",
        "character_name": "ANNOUNCER",
        "voice_traits": "formal, narrator",
        "line": "And so it ends where it began. In the dark, listening.",
    },
]

EPISODE_TITLE = "The Last Signal from Vault 7"


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

class SmokeResults:
    """Accumulates pass/fail for each probe."""

    def __init__(self):
        self.probes: list[dict] = []
        self.start_time = time.monotonic()

    def record(self, name: str, passed: bool, detail: str = ""):
        self.probes.append({"name": name, "passed": passed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def all_passed(self) -> bool:
        return all(p["passed"] for p in self.probes)

    @property
    def summary(self) -> str:
        passed = sum(1 for p in self.probes if p["passed"])
        total = len(self.probes)
        elapsed = time.monotonic() - self.start_time
        return f"{passed}/{total} probes passed in {elapsed:.1f}s"


def test_shotlist(results: SmokeResults, verbose: bool = False) -> dict | None:
    """Test 1: shotlist.py deterministic mapping."""
    try:
        from otr_v2.hyworld.shotlist import generate_shotlist

        shotlist = generate_shotlist(SYNTHETIC_SCRIPT_LINES, EPISODE_TITLE)

        results.record("shotlist_returns_dict", isinstance(shotlist, dict))
        results.record("shotlist_has_shots", len(shotlist.get("shots", [])) > 0,
                        f"{len(shotlist.get('shots', []))} shots")
        results.record("shotlist_has_anchor_hash", len(shotlist.get("style_anchor_hash", "")) == 12,
                        shotlist.get("style_anchor_hash", ""))
        results.record("shotlist_scene_count", shotlist.get("scene_count", 0) == 3,
                        f"expected 3, got {shotlist.get('scene_count', 0)}")

        # Verify each shot has required fields
        required_fields = {"shot_id", "scene_ref", "duration_sec", "camera", "env_prompt", "mood"}
        for shot in shotlist.get("shots", []):
            missing = required_fields - set(shot.keys())
            if missing:
                results.record("shot_fields_complete", False, f"{shot.get('shot_id','?')} missing {missing}")
                break
        else:
            results.record("shot_fields_complete", True, "all shots have required fields")

        # Determinism check: same input -> same output
        shotlist2 = generate_shotlist(SYNTHETIC_SCRIPT_LINES, EPISODE_TITLE)
        results.record("shotlist_deterministic",
                        json.dumps(shotlist, sort_keys=True) == json.dumps(shotlist2, sort_keys=True))

        # Camera mapping check: COMMANDER (weary) should get "slow handheld, close"
        first_shot = shotlist["shots"][0] if shotlist["shots"] else {}
        results.record("camera_trait_mapping",
                        "handheld" in first_shot.get("camera", ""),
                        f"got: {first_shot.get('camera', 'NONE')}")

        if verbose:
            print(json.dumps(shotlist, indent=2))

        return shotlist

    except Exception as e:
        results.record("shotlist_import", False, str(e))
        traceback.print_exc()
        return None


def test_bridge_dry_run(results: SmokeResults) -> str | None:
    """Test 2: Bridge node in dry-run mode (no sidecar spawn)."""
    try:
        from otr_v2.hyworld.bridge import HyworldBridge

        node = HyworldBridge()
        job_id, shotlist_json = node.execute(
            script_json=json.dumps(SYNTHETIC_SCRIPT_LINES),
            episode_title=EPISODE_TITLE,
            sidecar_enabled=False,
        )

        results.record("bridge_returns_job_id", bool(job_id) and not job_id.startswith("PARSE_ERROR"),
                        job_id)
        results.record("bridge_returns_shotlist", len(shotlist_json) > 10)

        # Check contract files exist
        job_dir = _OTR_ROOT / "io" / "hyworld_in" / job_id
        results.record("bridge_writes_contract_dir", job_dir.is_dir(), str(job_dir))

        expected_files = ["script_lines.json", "shotlist.json", "production_plan.json",
                          "scene_manifest.json", "meta.json"]
        for fname in expected_files:
            fpath = job_dir / fname
            exists = fpath.exists() and fpath.stat().st_size > 0
            results.record(f"contract_{fname}", exists,
                            f"{fpath.stat().st_size}B" if fpath.exists() else "MISSING")

        # Verify meta.json content
        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        results.record("meta_has_job_id", meta.get("job_id") == job_id)
        results.record("meta_lane_faithful", meta.get("lane") == "faithful")

        return job_id

    except Exception as e:
        results.record("bridge_import", False, str(e))
        traceback.print_exc()
        return None


def test_worker_stub(results: SmokeResults, job_id: str) -> None:
    """Test 3: Worker stub generates placeholder assets."""
    try:
        from otr_v2.hyworld.worker import run_stub

        job_dir = _OTR_ROOT / "io" / "hyworld_in" / job_id
        run_stub(job_dir)

        out_dir = _OTR_ROOT / "io" / "hyworld_out" / job_id

        # Check STATUS.json
        status_file = out_dir / "STATUS.json"
        results.record("worker_status_exists", status_file.exists())

        if status_file.exists():
            status = json.loads(status_file.read_text(encoding="utf-8"))
            results.record("worker_status_ready", status.get("status") == "READY",
                            status.get("status", "NONE"))

        # Check per-shot assets
        shotlist = json.loads((job_dir / "shotlist.json").read_text(encoding="utf-8"))
        shots = shotlist.get("shots", [])
        assets_found = 0
        for shot in shots:
            shot_dir = out_dir / shot["shot_id"]
            if (shot_dir / "render.png").exists():
                size = (shot_dir / "render.png").stat().st_size
                if size > 100:  # valid PNG is at least a few hundred bytes
                    assets_found += 1

        results.record("worker_assets_generated", assets_found == len(shots),
                        f"{assets_found}/{len(shots)} shots have render.png")

    except Exception as e:
        results.record("worker_stub", False, str(e))
        traceback.print_exc()


def test_poll(results: SmokeResults, job_id: str) -> str | None:
    """Test 4: Poll node reads status correctly."""
    try:
        from otr_v2.hyworld.poll import HyworldPoll

        node = HyworldPoll()
        assets_path, status, detail = node.execute(
            hyworld_job_id=job_id,
            timeout_sec=5,
            poll_interval_sec=0.5,
        )

        results.record("poll_status_ready", status == "READY", status)
        results.record("poll_assets_path_valid", assets_path != "FALLBACK" and Path(assets_path).is_dir(),
                        assets_path[:80])

        return assets_path

    except Exception as e:
        results.record("poll_import", False, str(e))
        traceback.print_exc()
        return None


def test_poll_fallback(results: SmokeResults) -> None:
    """Test 5: Poll node handles PARSE_ERROR gracefully."""
    try:
        from otr_v2.hyworld.poll import HyworldPoll

        node = HyworldPoll()
        assets_path, status, detail = node.execute(
            hyworld_job_id="PARSE_ERROR_test123",
            timeout_sec=2,
        )

        results.record("poll_fallback_path", assets_path == "FALLBACK")
        results.record("poll_fallback_status", status == "PARSE_ERROR")

    except Exception as e:
        results.record("poll_fallback", False, str(e))


def test_renderer_fallback(results: SmokeResults) -> None:
    """Test 6: Renderer handles FALLBACK without crashing."""
    try:
        from otr_v2.hyworld.renderer import HyworldRenderer

        node = HyworldRenderer()
        mp4_path, render_log = node.execute(
            hyworld_assets_path="FALLBACK",
            final_audio_path="/nonexistent/audio.wav",
        )

        results.record("renderer_fallback_empty_path", mp4_path == "")
        results.record("renderer_fallback_has_log", "FALLBACK" in render_log)

    except Exception as e:
        results.record("renderer_fallback", False, str(e))


def test_renderer_with_assets(results: SmokeResults, assets_path: str) -> None:
    """Test 7: Renderer processes real stub assets (no ffmpeg required for this check)."""
    try:
        from otr_v2.hyworld.renderer import HyworldRenderer

        node = HyworldRenderer()
        # Use a nonexistent audio file — renderer should log and return empty
        # (we're testing asset collection, not ffmpeg muxing)
        mp4_path, render_log = node.execute(
            hyworld_assets_path=assets_path,
            final_audio_path="/nonexistent/test_audio.wav",
            episode_title=EPISODE_TITLE,
        )

        # Even without ffmpeg, the render log should show it found the shots
        results.record("renderer_found_assets", "render.png" in render_log or "shot" in render_log.lower(),
                        f"log length: {len(render_log)} chars")
        # Audio not found is expected — renderer should have logged it
        results.record("renderer_handles_missing_audio", "not found" in render_log.lower() or mp4_path == "")

    except Exception as e:
        results.record("renderer_with_assets", False, str(e))


def test_bridge_parse_error(results: SmokeResults) -> None:
    """Test 8: Bridge handles garbage script_json gracefully."""
    try:
        from otr_v2.hyworld.bridge import HyworldBridge

        node = HyworldBridge()
        job_id, shotlist_json = node.execute(
            script_json="NOT VALID JSON {{{{",
            episode_title="Parse Error Test",
            sidecar_enabled=False,
        )

        results.record("bridge_parse_error_id", job_id.startswith("PARSE_ERROR_"))
        results.record("bridge_parse_error_empty_shots", shotlist_json == "[]")

    except Exception as e:
        results.record("bridge_parse_error", False, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HyWorld node smoke test")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Remove io/ artifacts after test")
    args = parser.parse_args()

    print("=" * 60)
    print("HyWorld Smoke Test (supersoaker pattern)")
    print(f"OTR root: {_OTR_ROOT}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = SmokeResults()

    # ---- Test 1: Shotlist ----
    print("\n--- Test 1: Shotlist (deterministic mapping) ---")
    shotlist = test_shotlist(results, args.verbose)

    # ---- Test 2: Bridge dry run ----
    print("\n--- Test 2: Bridge (dry run, no sidecar) ---")
    job_id = test_bridge_dry_run(results)

    # ---- Test 3: Worker stub ----
    if job_id:
        print("\n--- Test 3: Worker stub (placeholder assets) ---")
        test_worker_stub(results, job_id)

    # ---- Test 4: Poll ----
    assets_path = None
    if job_id:
        print("\n--- Test 4: Poll (status detection) ---")
        assets_path = test_poll(results, job_id)

    # ---- Test 5: Poll fallback ----
    print("\n--- Test 5: Poll fallback (PARSE_ERROR routing) ---")
    test_poll_fallback(results)

    # ---- Test 6: Renderer fallback ----
    print("\n--- Test 6: Renderer fallback (FALLBACK routing) ---")
    test_renderer_fallback(results)

    # ---- Test 7: Renderer with assets ----
    if assets_path:
        print("\n--- Test 7: Renderer with stub assets ---")
        test_renderer_with_assets(results, assets_path)

    # ---- Test 8: Bridge parse error ----
    print("\n--- Test 8: Bridge parse error handling ---")
    test_bridge_parse_error(results)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"SMOKE TEST RESULT: {results.summary}")

    if results.all_passed:
        print("STATUS: ALL PROBES PASSED")
    else:
        print("STATUS: FAILURES DETECTED")
        for p in results.probes:
            if not p["passed"]:
                print(f"  FAILED: {p['name']}" + (f" — {p['detail']}" if p["detail"] else ""))

    print("=" * 60)

    # ---- Write report ----
    report_path = _OTR_ROOT / "logs" / "hyworld_smoketest_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# HyWorld Smoke Test Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Result:** {results.summary}\n")
        f.write(f"**Status:** {'ALL PASSED' if results.all_passed else 'FAILURES DETECTED'}\n\n")
        f.write("| Probe | Status | Detail |\n")
        f.write("|---|---|---|\n")
        for p in results.probes:
            status = "PASS" if p["passed"] else "FAIL"
            f.write(f"| {p['name']} | {status} | {p['detail']} |\n")

    print(f"\nReport: {report_path}")

    # ---- Cleanup ----
    if args.cleanup:
        io_dir = _OTR_ROOT / "io"
        if io_dir.exists():
            shutil.rmtree(io_dir)
            print(f"Cleaned up: {io_dir}")

    return 0 if results.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
