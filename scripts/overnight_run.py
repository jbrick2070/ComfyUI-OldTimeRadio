"""
overnight_run.py — Full episode orchestration for OldTimeRadio v2.0.

Usage:
    python scripts/overnight_run.py --mode=safe --episode=<path> [--single-scene=<id>] [--dry-run]

Order:
    1. preflight.py
    2. Parse script -> Director LLM -> Segmenter -> Reconciler -> PromptBuilder -> fully-materialized matrix
    3. If --dry-run: dump dry_run_plan.json, exit 0
    4. Audio pass (Bark/Kokoro/SFX/Music) -> SceneSequencer -> EpisodeAssembler
    5. Visual pass per scene: anchor -> boundary -> ProductionBus
    6. Final concat
    7. Write logs/run_<ts>.json + append logs/ledger.jsonl

Exit nonzero if completed: false.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

log = logging.getLogger("OTR")


def _git_sha():
    """Get current git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=_REPO_ROOT
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _config_sha(config_path):
    """SHA-256 of the config file for ledger reproducibility."""
    try:
        h = hashlib.sha256()
        with open(config_path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:12]
    except Exception:
        return "unknown"


def _load_config(config_path=None):
    """Load rtx5080.yaml config."""
    import yaml

    if config_path is None:
        config_path = os.path.join(_REPO_ROOT, "config", "rtx5080.yaml")

    if not os.path.isfile(config_path):
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), config_path


def _write_run_log(run_id, events):
    """Write the full run log as logs/run_<ts>.json."""
    log_dir = os.path.join(_REPO_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_log_path = os.path.join(log_dir, f"run_{run_id}.json")

    with open(run_log_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    log.info("[OvernightRun] Run log: %s", run_log_path)
    return run_log_path


def _append_ledger(entry):
    """Append a single-line JSON entry to logs/ledger.jsonl."""
    log_dir = os.path.join(_REPO_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ledger_path = os.path.join(log_dir, "ledger.jsonl")

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    log.info("[OvernightRun] Ledger entry appended: %s", ledger_path)


def main():
    parser = argparse.ArgumentParser(
        description="OldTimeRadio v2.0 Overnight Run Orchestrator")
    parser.add_argument("--mode", choices=["safe", "experiment"],
                        default="safe",
                        help="Pipeline mode: safe (keyframes) or experiment (animated)")
    parser.add_argument("--episode", required=True,
                        help="Path to episode script JSON or text file")
    parser.add_argument("--single-scene", dest="single_scene", default=None,
                        help="Render only this scene ID (e.g., s03)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Dump plan JSON and exit without rendering")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (default: config/rtx5080.yaml)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(_REPO_ROOT, "otr_runtime.log"),
                encoding="utf-8"
            ),
        ]
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    events = []  # Accumulate run events for run log
    t_run_start = time.time()

    log.info("[OvernightRun] === Run %s starting ===", run_id)
    log.info("[OvernightRun] Mode: %s, Episode: %s, Seed: %d",
             args.mode, args.episode, args.seed)

    # ── 1. Load config and preflight ──────────────────────────────
    config, config_path = _load_config(args.config)

    # Override mode from CLI
    config["mode"] = args.mode

    from scripts.preflight import run as preflight_run
    preflight_result = preflight_run(config)

    events.append({
        "phase": "preflight",
        "ts": datetime.now().isoformat(),
        "passed": preflight_result["passed"],
        "hard_fails": preflight_result["hard_fails"],
        "warnings": preflight_result["warnings"],
    })

    if not preflight_result["passed"]:
        log.error("[OvernightRun] Preflight FAILED. Aborting.")
        _write_run_log(run_id, events)
        sys.exit(1)

    # ── 2. Parse script ───────────────────────────────────────────
    if not os.path.isfile(args.episode):
        log.error("[OvernightRun] Episode file not found: %s", args.episode)
        sys.exit(1)

    with open(args.episode, "r", encoding="utf-8") as f:
        episode_content = f.read()

    # Determine format: JSON script or raw text
    try:
        script_lines = json.loads(episode_content)
    except (json.JSONDecodeError, TypeError):
        # Raw text: split into lines, treat each as dialogue
        script_lines = [
            {"type": "dialogue", "line": line.strip()}
            for line in episode_content.strip().split("\n")
            if line.strip()
        ]

    events.append({
        "phase": "parse",
        "ts": datetime.now().isoformat(),
        "total_lines": len(script_lines),
    })

    # ── 2b. Segmenter -> Reconciler -> PromptBuilder ──────────────
    from nodes.scene_segmenter import segment
    from nodes.director_reconciler import reconcile
    from nodes.prompt_builder import build

    seg_scenes = segment(script_lines)

    # Director plan: for now use empty if not available from LLM
    # In full pipeline, Director LLM produces this; for overnight_run
    # we accept it as a companion file (episode.director.json) or empty
    director_path = args.episode.rsplit(".", 1)[0] + ".director.json"
    director_json = {}
    if os.path.isfile(director_path):
        with open(director_path, "r", encoding="utf-8") as f:
            director_json = json.load(f)
        log.info("[OvernightRun] Loaded director plan: %s", director_path)

    director_scenes = director_json.get("visual_plan", {}).get("scenes", [])
    reconciled = reconcile(director_scenes, seg_scenes)
    scene_prompts = build(reconciled, director_json)

    # Apply single-scene filter if requested
    if args.single_scene:
        scene_prompts = [
            sp for sp in scene_prompts
            if sp.scene_id == args.single_scene
        ]
        if not scene_prompts:
            log.error("[OvernightRun] Scene '%s' not found in %d scenes",
                      args.single_scene, len(reconciled))
            sys.exit(1)

    events.append({
        "phase": "prompt_build",
        "ts": datetime.now().isoformat(),
        "seg_scene_count": len(seg_scenes),
        "director_scene_count": len(director_scenes),
        "final_scene_count": len(scene_prompts),
        "single_scene": args.single_scene,
    })

    log.info("[OvernightRun] %d scenes ready for rendering",
             len(scene_prompts))

    # ── 3. Dry-run exit ───────────────────────────────────────────
    if args.dry_run:
        plan = {
            "run_id": run_id,
            "mode": args.mode,
            "seed": args.seed,
            "episode": args.episode,
            "scene_count": len(scene_prompts),
            "scenes": [
                {
                    "scene_id": sp.scene_id,
                    "anchor_prompt": sp.anchor_prompt[:100] + "...",
                    "motion_prompt": sp.motion_prompt[:100] + "...",
                    "motion": sp.motion,
                    "duration_s": sp.duration_s,
                }
                for sp in scene_prompts
            ],
        }

        dry_run_path = os.path.join(_REPO_ROOT, "dry_run_plan.json")
        with open(dry_run_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

        log.info("[OvernightRun] Dry-run plan written: %s", dry_run_path)
        print(f"Dry-run plan: {dry_run_path}")
        print(f"  Scenes: {len(scene_prompts)}")
        print(f"  Mode: {args.mode}")
        sys.exit(0)

    # ── 4. Audio pass ─────────────────────────────────────────────
    # Audio generation is handled by the existing v1.x pipeline
    # (BatchBark/BatchKokoro -> SceneSequencer -> EpisodeAssembler).
    # overnight_run delegates to ComfyUI queue for audio, or expects
    # pre-rendered audio from a previous run.
    #
    # For v2.0 RC1, audio is assumed to exist. The visual pass is
    # the focus of this orchestrator.
    log.info("[OvernightRun] Audio pass: delegated to ComfyUI pipeline "
             "(pre-existing or queued separately)")

    events.append({
        "phase": "audio",
        "ts": datetime.now().isoformat(),
        "status": "delegated_to_comfyui",
    })

    # ── 5. Visual pass ────────────────────────────────────────────
    from nodes.production_bus_v2 import ProductionBusV2

    bus_mode = "keyframes" if args.mode == "safe" else "animated"
    bus = ProductionBusV2(config, mode=bus_mode)

    output_dir = os.path.join(_REPO_ROOT, "output", "v2_runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    clip_paths = []
    scene_results = []
    fallback_count = 0
    peak_vram_gb = 0.0

    for s_idx, sp in enumerate(scene_prompts):
        t_scene = time.time()
        log.info("[OvernightRun] Rendering scene %d/%d [%s] mode=%s",
                 s_idx + 1, len(scene_prompts), sp.scene_id, bus_mode)

        try:
            # For keyframes mode, we need an anchor image path.
            # In the full pipeline, this comes from SD3.5 via CharacterForge
            # + ScenePainter + VisualCompositor. For overnight_run standalone,
            # we generate a placeholder or use pre-rendered anchors.
            anchor_dir = os.path.join(output_dir, "anchors")
            os.makedirs(anchor_dir, exist_ok=True)
            anchor_path = os.path.join(anchor_dir,
                                       f"anchor_{sp.scene_id}.png")

            # If no anchor exists, create a black placeholder
            if not os.path.isfile(anchor_path):
                _create_placeholder_anchor(anchor_path, 1024, 1024)

            clip_path = bus.render_scene(
                sp, anchor_path, sp.duration_s)
            clip_paths.append(clip_path)

            scene_time = time.time() - t_scene
            scene_results.append({
                "scene_id": sp.scene_id,
                "status": "ok",
                "clip_path": clip_path,
                "render_time_s": round(scene_time, 1),
            })

        except Exception as e:
            scene_time = time.time() - t_scene
            log.error("[OvernightRun] Scene %s failed: %s",
                      sp.scene_id, e, exc_info=True)
            fallback_count += 1
            scene_results.append({
                "scene_id": sp.scene_id,
                "status": "failed",
                "error": str(e)[:200],
                "render_time_s": round(scene_time, 1),
            })

    # Track peak VRAM if torch available
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    except ImportError:
        pass

    events.append({
        "phase": "visual",
        "ts": datetime.now().isoformat(),
        "mode": bus_mode,
        "scene_count": len(scene_prompts),
        "clips_rendered": len(clip_paths),
        "fallback_count": fallback_count,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "scene_results": scene_results,
    })

    # ── 6. Final concat ───────────────────────────────────────────
    final_path = os.path.join(output_dir, f"episode_{run_id}.mp4")

    if clip_paths:
        try:
            bus.concat(clip_paths, final_path)
            log.info("[OvernightRun] Final video: %s", final_path)
        except Exception as e:
            log.error("[OvernightRun] Concat failed: %s", e)
            events.append({
                "phase": "concat",
                "ts": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)[:200],
            })
    else:
        log.warning("[OvernightRun] No clips to concatenate")

    # ── 7. Write logs ─────────────────────────────────────────────
    total_s = time.time() - t_run_start
    completed = (len(clip_paths) == len(scene_prompts) and
                 fallback_count == 0)

    events.append({
        "phase": "summary",
        "ts": datetime.now().isoformat(),
        "total_s": round(total_s, 1),
        "completed": completed,
    })

    _write_run_log(run_id, events)

    ledger_entry = {
        "run_id": run_id,
        "ts": datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "config_sha": _config_sha(config_path),
        "seed": args.seed,
        "mode": args.mode,
        "episode_id": os.path.basename(args.episode),
        "scene_count": len(scene_prompts),
        "total_s": round(total_s, 1),
        "fallback_count": fallback_count,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "completed": completed,
    }
    _append_ledger(ledger_entry)

    # Final report
    print(f"\n{'=' * 60}")
    print(f"OvernightRun {'COMPLETED' if completed else 'FINISHED WITH ISSUES'}")
    print(f"  Run ID:     {run_id}")
    print(f"  Mode:       {args.mode}")
    print(f"  Scenes:     {len(clip_paths)}/{len(scene_prompts)}")
    print(f"  Fallbacks:  {fallback_count}")
    print(f"  Peak VRAM:  {peak_vram_gb:.1f} GB")
    print(f"  Duration:   {total_s:.0f}s")
    if os.path.isfile(final_path):
        print(f"  Output:     {final_path}")
    print(f"{'=' * 60}")

    sys.exit(0 if completed else 1)


def _create_placeholder_anchor(path, width, height):
    """Create a black placeholder PNG for scenes without real anchors.

    Uses FFmpeg to avoid PIL dependency.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "lavfi",
             "-i", f"color=c=black:s={width}x{height}:d=1",
             "-frames:v", "1",
             path],
            capture_output=True, timeout=10
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
