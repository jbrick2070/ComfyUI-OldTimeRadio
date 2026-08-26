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

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "otr_canonical_api_run.py"
MATRIX_FILE = HERE / "_llm_sweep_matrix.json"
RECEIPT = REPO / "docs" / "2026-08-25-llm-image-upscale-sweep-receipt.json"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BANK = "scifi_news_pro"
VISUAL_STYLE = "sci_fi_radio"
DEFAULT_TIMEOUT = 2400  # seconds; one-act stills-only legs are cheap


def load_matrix() -> list[dict]:
    return json.loads(MATRIX_FILE.read_text(encoding="utf-8"))


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
