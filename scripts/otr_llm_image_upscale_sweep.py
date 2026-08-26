"""One-act pressure test: all 7 curated local LLMs, a variety of local image
models, a variety of still engines, and both upscale engines -- NO video.

WHY THIS SHAPE (operator, 2026-08-25). *"Let's do a one act test of all seven
[LLMs] onboard. Use varying image models ... and test a variety of upscalers.
So that make them pressure test the whole items. But not video because I
wanted to go quickly."* Then, correcting the first cut of this driver: *"just
change the dropdown values for each leg just as a human would."*

THE SANCTIONED LEVER, unchanged from ``otr_gpu_soak_matrix.py``: the video and
image engine widgets are MANAGED and ``patch_creative`` refuses them outright
(BUG-08.06 stranded-COMBO class), so a profile's ``role_overrides`` /
``upscale_stage`` -- exactly what a human clicking the dropdowns and saving the
graph would produce -- is the only legitimate way to set them from a headless
driver. The 7 profiles this script drives
(``config/profiles/otr_soak_llmsweep_0{1..7}.json``) are that: nothing here
edits ``workflows/otr_canonical.json`` or hand-pokes ``widgets_values``.

THE LLM is set differently, on purpose: ``creative_writing_model`` /
``technical_model`` ARE on the CREATIVE_WHITELIST, so they go through the
runner's own ``--creative-model`` / ``--technical-model`` flags -- the exact
channel ``otr_canonical_api_run.py`` offers for this, applied AFTER the
profile (verified: ``apply_profile_to_workflow`` runs before
``_apply_writer_shortcuts`` in ``build_api_prompt``), so the CLI value wins
over each profile's own placeholder ``llm.creative_model`` /
``technical_model``.

THE MATRIX (7 legs, one episode each, act_count=1 -- exactly 3 voiced beats,
one per role, which is what lets a SINGLE leg exercise 3 different stills and
3 different image engines at once):

* LLM pairing is CYCLIC: leg N's creative model is row N, its technical model
  is row N+1 (mod 7). Every one of the 7 curated local rows plays creative
  exactly once and technical exactly once across the whole sweep -- the
  operator's separate ruling that a row untested in both roles is a rip
  candidate needs exactly this evidence.
* Each leg's 3 roles (announcer/music/character) get 3 DISTINCT still engines
  from {still_flat, still_motion, still_pan, still_word} and 3 DISTINCT local
  image engines from {z_image_turbo, flux_gen1, flux2_klein, lumina_image,
  ideogram4_local} -- rotated leg-to-leg so every engine in both pools is
  exercised by more than one role over the sweep.
* Upscale alternates the only two real engines (``spandrel_esrgan`` on odd
  legs, ``off`` on even) -- there is no third option to rotate; recorded so
  nobody goes looking for one.
* Bank is PINNED to ``scifi_news_pro`` for every leg. It is the one lane
  verified to drive BOTH the creative and technical writer slots (the
  2026-08-25 sweep-design fan-out); an unpinned ``roll`` bank could land on a
  lane that never reaches technical_model, proving nothing about that slot.
* NOT covered, by request: no video engine is ever selected (``still``-lane
  profiles only, matching ``otr_gpu_soak_matrix.py --lanes still``'s
  precedent) -- the operator wants this fast, and a still is the cheap signal.

SEQUENTIAL, ONE LEG AT A TIME (CLAUDE.md scope rule -- one GPU, no async CUDA
streams). A failed leg is LOGGED and the sweep continues; a receipt JSON is
rewritten after every leg so killing the run mid-flight still leaves a
complete record.

Usage:
    python scripts/otr_llm_image_upscale_sweep.py
    python scripts/otr_llm_image_upscale_sweep.py --legs 1,3,5
    python scripts/otr_llm_image_upscale_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "otr_canonical_api_run.py"
RECEIPT = REPO / "docs" / "2026-08-25-llm-image-upscale-sweep-receipt.json"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BANK = "scifi_news_pro"
VISUAL_STYLE = "sci_fi_radio"

#: SECONDS. Raised 2400 -> 7200 on 2026-08-25 after two legs were reported
#: FAIL purely because this budget expired, one of which had not started.
#:
#: TWO THINGS THIS NUMBER HAS TO ABSORB, and the first one is not obvious:
#:
#: 1. ComfyUI SERIALIZES prompts. Each leg POSTs its prompt and immediately
#:    starts its own clock, but the server runs one prompt at a time -- so a
#:    leg queued behind a slow predecessor burns its entire budget sitting in
#:    `queue_pending` having rendered nothing. Leg 4 of the first run "failed"
#:    in exactly this way: 40 minutes of pure queue wait, zero work done, and
#:    the receipt called it a failure. A per-leg timeout is really a
#:    QUEUE-WAIT + RENDER budget unless the driver waits for the queue to
#:    drain first, which is what `_wait_for_idle_queue` below now does.
#: 2. `spandrel_esrgan` costs 3-4 minutes PER SEGMENT (18+ segments on a
#:    one-act episode). See docs/SOAK_LEG_GUIDE.md section 8A.
DEFAULT_TIMEOUT = 7200

#: Poll the server's queue before submitting, so a leg's timeout measures its
#: OWN render rather than its predecessor's. Cheap: one HTTP GET every 15s.
COMFY_URL = "http://127.0.0.1:8000"
QUEUE_POLL_S = 15
QUEUE_DRAIN_MAX_S = 10800


def _wait_for_idle_queue(*, max_wait_s: int = QUEUE_DRAIN_MAX_S) -> bool:
    """Block until the ComfyUI queue is empty, so this leg's clock is its own.

    Returns True when the queue drained, False on timeout/unreachable. Never
    raises: a driver that dies because a status probe failed is worse than one
    that submits slightly early.
    """
    import urllib.request

    deadline = time.monotonic() + max_wait_s
    announced = False
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/queue", timeout=10) as r:
                q = json.load(r)
            busy = len(q.get("queue_running") or []) + len(q.get("queue_pending") or [])
            if busy == 0:
                return True
            if not announced:
                print(f"[llmsweep]   waiting for queue to drain ({busy} ahead)",
                      flush=True)
                announced = True
        except Exception:
            return False  # unreachable: let the leg try and fail honestly
        time.sleep(QUEUE_POLL_S)
    return False


#: The 7 curated LOCAL LLM rows, in catalog order. Full dropdown labels
#: (size suffix included) -- validate_model_id resolves them.
LLM_ROWS = (
    "mistralai/Mistral-Nemo-Instruct-2407 (12.0 GB)",
    "google/gemma-4-E2B-it (3.0 GB)",
    "google/gemma-4-E4B-it (4.5 GB)",
    "google/gemma-4-12b-it (11.9 GB)",
    "unsloth/gemma-4-12b-it-GGUF (17.4 GB)",
    "unsloth/Qwen3-8B-GGUF (10.3 GB)",
    "google/gemma-2-2b-it (2.6 GB)",
)
STILL_ENGINES = ("still_flat", "still_motion", "still_pan", "still_word")
IMAGE_ENGINES = ("z_image_turbo", "flux_gen1", "flux2_klein",
                 "lumina_image", "ideogram4_local")
ROLES = ("announcer", "music", "character")

#: Which leg carries the upscaler. Exactly ONE, deliberately -- see
#: DEFAULT_TIMEOUT note 2 and docs/SOAK_LEG_GUIDE.md section 8A.
UPSCALE_LEG = 7


def load_matrix() -> list[dict]:
    """Derive the leg matrix. COMPUTED, not read from a sidecar file.

    It used to live in `scripts/_llm_sweep_matrix.json`, which `.gitignore`
    excludes as a scratch file (`scripts/_*.json`) -- so the committed script
    could not actually run from a fresh clone. The matrix is fully
    deterministic, so it belongs in the code that uses it.

    Leg N: creative = row N, technical = row N+1 (mod 7), so every row plays
    BOTH slots exactly once across the sweep. Stills and images rotate so no
    leg repeats an engine across its three roles.
    """
    legs = []
    for i in range(len(LLM_ROWS)):
        legs.append({
            "index": i + 1,
            "creative": LLM_ROWS[i],
            "technical": LLM_ROWS[(i + 1) % len(LLM_ROWS)],
            "stills": {ROLES[r]: STILL_ENGINES[(i * 3 + r) % len(STILL_ENGINES)]
                       for r in range(3)},
            "images": {ROLES[r]: IMAGE_ENGINES[(i * 3 + r) % len(IMAGE_ENGINES)]
                       for r in range(3)},
            "upscale_engine": ("spandrel_esrgan" if i + 1 == UPSCALE_LEG
                               else "off"),
        })
    return legs


def leg(entry: dict, *, timeout: int, dry_run: bool) -> dict:
    idx = entry["index"]
    profile = f"otr_soak_llmsweep_{idx:02d}"
    creative = entry["creative"]
    technical = entry["technical"]
    stamp = datetime.datetime.now().strftime("%H%M%S")
    leg_label = (
        f"LLMSWEEP{idx:02d} "
        f"c={creative.split(' (')[0]} t={technical.split(' (')[0]} "
        f"upscale={entry['upscale_engine']}"
    )
    cmd = [
        sys.executable, str(RUNNER),
        "--act-count", "1",
        "--source-bank", BANK,
        "--visual-style", VISUAL_STYLE,
        "--profile", profile,
        "--creative-model", creative,
        "--technical-model", technical,
        "--timeout", str(timeout),
    ]
    if dry_run:
        # --offline-schemas is REQUIRED with --dry-run: without it the
        # runner still calls the live /object_info endpoint to resolve
        # widget choices, so a dry-run with no server booted fails on a
        # ConnectionError rather than actually validating anything.
        # (Caught 2026-08-25: the first cut of this driver omitted the
        # flag and blanket-passed every dry-run leg regardless of outcome
        # -- a false green in the exact class this project's own admission
        # rule exists to catch. Never trust dry_run alone as a verdict.)
        cmd += ["--dry-run", "--offline-schemas"]
    print(f"[llmsweep] leg {idx} START {stamp} {leg_label}", flush=True)
    if not dry_run:
        # Make this leg's timeout measure ITS OWN render, not the queue
        # wait behind a slower predecessor (see DEFAULT_TIMEOUT note 1).
        _wait_for_idle_queue()
    print(f"[llmsweep]   cmd: {' '.join(cmd)}", flush=True)
    started = datetime.datetime.now()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True,
            timeout=timeout + 600,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if dry_run:
            ok = proc.returncode == 0 and "DRY_RUN complete" in out
        else:
            ok = "RESULT SUCCESS" in out
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        out, ok, rc = "harness timeout", False, -1
    finished = datetime.datetime.now()
    elapsed = (finished - started).total_seconds() / 60.0
    tail = [ln for ln in out.splitlines()
            if "RESULT" in ln or "Exception" in ln or "Error" in ln]
    print(f"[llmsweep] leg {idx} {'PASS' if ok else 'FAIL'} "
          f"{elapsed:.1f} min rc={rc} {tail[-1][:160] if tail else ''}",
          flush=True)
    return {
        "leg": idx, "leg_label": leg_label, "profile": profile,
        "creative_model": creative, "technical_model": technical,
        "stills": entry["stills"], "images": entry["images"],
        "upscale_engine": entry["upscale_engine"],
        "ok": ok, "rc": rc, "minutes": round(elapsed, 1),
        "stdout_tail": out[-4000:] if not ok else "",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs", default=None,
                     help="comma-separated 1-based leg numbers to run "
                          "(default: all 7)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--dry-run", action="store_true",
                     help="build and dump each prompt without POSTing")
    args = ap.parse_args(argv)

    matrix = load_matrix()
    if args.legs:
        wanted = {int(x) for x in args.legs.split(",")}
        matrix = [m for m in matrix if m["index"] in wanted]

    results = []
    for entry in matrix:
        r = leg(entry, timeout=args.timeout, dry_run=args.dry_run)
        results.append(r)
        RECEIPT.write_text(json.dumps({
            "generated_at": datetime.datetime.now().isoformat(),
            "bank": BANK, "visual_style": VISUAL_STYLE,
            "legs_run": len(results),
            "legs_total": len(matrix),
            "pass": sum(1 for x in results if x["ok"]),
            "fail": sum(1 for x in results if not x["ok"]),
            "results": results,
        }, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["ok"])
    print(f"[llmsweep] DONE {passed}/{len(results)} passed. "
          f"Receipt: {RECEIPT}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
