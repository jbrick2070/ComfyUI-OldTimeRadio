"""Overnight pressure test: the SMALLEST curated local LLM in BOTH writer slots,
across every runnable source bank and every local image/still engine.

WHY THIS SHAPE (operator, 2026-08-26). After the labelled-sections dossier fix
qualified live on `scifi_news_pro`, the operator asked for the harder proof:
*"gemma-4-E2B-it technical AND creative ... that really proves this is a robust
fix"*, then *"if it passes scifi_pro move onto the other banks to see we have
good parser jsons for all media source banks even when using lil ole
gemma-4-E2B-it as both creative and technical -- that's the true test."*

So `google/gemma-4-E2B-it` (3.0 GB, the smallest row in the catalog) is pinned
in BOTH the creative and technical slots on EVERY leg. A 2B model is the
worst case for structured extraction -- it is the model whose unclosed JSON
produced the original failure -- so a bank that survives it survives anything
above it.

THE SANCTIONED LEVER, unchanged from `otr_llm_image_upscale_sweep.py`: the
image and video engine widgets are MANAGED and `patch_creative` refuses them,
so a profile's `role_overrides` -- exactly what a human clicking the
announcer / music / character dropdowns and saving the graph would produce --
is the only legitimate way to set them from a headless driver. Operator's own
framing: *"you just change the dropdowns for music announcer and char beats
and image model as a human would."* Nothing here edits
`workflows/otr_canonical.json` or pokes `widgets_values`.

The LLM is set differently and on purpose: `creative_writing_model` /
`technical_model` ARE on the CREATIVE_WHITELIST, so they ride the runner's own
`--creative-model` / `--technical-model` flags, applied AFTER the profile --
the CLI value wins over each profile's placeholder.

THE MATRIX (8 legs, one one-act episode each):

* 4 banks x 2 engine profiles. `scifi_news_pro` is deliberately ABSENT: it
  qualified live on 2026-08-26 (RESULT SUCCESS + obs_publish OK + mp4 on
  disk) and re-proving it would spend 25 minutes of GPU on a settled result.
* The two profiles together exercise ALL FIVE local image engines
  (z_image_turbo, flux_gen1, flux2_klein, lumina_image, ideogram4_local) and
  ALL FOUR still lanes (still_flat, still_motion, still_pan, still_word),
  three of each per leg -- which is what `--act-count 1` buys: exactly three
  voiced beats, one per role.
* Every one of these four banks genuinely drives the TECHNICAL slot, so the
  test is honest for both models. media_archive / public_domain / shakespeare
  reach it through `_run_source_interpreter(technical_model_id=...)`; original
  reaches it through its own concept pass. A bank that never touched the
  technical slot would prove nothing about it.
* Visual style is pinned per bank to the style authored for that lane, so the
  published episodes are watchable rather than merely valid -- obs is the
  success signal the operator actually reads.

SEQUENTIAL, ONE LEG AT A TIME (CLAUDE.md scope rule -- one GPU, no async CUDA
streams). ComfyUI serializes prompts anyway, so each leg waits for the queue
to drain before submitting; otherwise a leg burns its whole timeout sitting in
`queue_pending` having rendered nothing and gets reported as a failure it
never had. A failed leg is LOGGED and the sweep CONTINUES -- one bad bank must
not cost the other seven legs -- and the receipt is rewritten after every leg
so killing the run mid-flight still leaves a complete record.

Usage:
    python scripts/otr_bank_engine_sweep.py
    python scripts/otr_bank_engine_sweep.py --legs 1,3
    python scripts/otr_bank_engine_sweep.py --dry-run
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
RECEIPT = REPO / "docs" / "2026-08-26-bank-engine-e2b-sweep-receipt.json"

#: The model under test, in BOTH slots, on every leg. Full dropdown label --
#: `validate_model_id` resolves the size suffix.
E2B = "google/gemma-4-E2B-it (3.0 GB)"

#: Bank -> the visual style authored for that lane. scifi_news_pro is absent
#: on purpose: it qualified live on 2026-08-26 and does not need re-proving.
BANKS = (
    ("media_archive", "archival_documentary"),
    ("original", "sci_fi_radio"),
    ("public_domain", "storybook_engraving"),
    ("shakespeare", "shakespeare_stage_realism"),
)

#: Two profiles that BETWEEN THEM cover every local image engine and every
#: still lane. Leg 01: z_image_turbo / flux_gen1 / flux2_klein over
#: still_flat / still_motion / still_pan. Leg 02: lumina_image /
#: ideogram4_local / z_image_turbo over still_word / still_flat / still_motion.
PROFILES = ("otr_soak_llmsweep_01", "otr_soak_llmsweep_02")

#: THE VIDEO LANES THAT RUN ON THE DEFAULT BOOT (`--lanes video`), smallest
#: model first so a night that runs short still covers the most engines.
#:
#: WHY ONLY THESE EIGHT: a profile's `launch.env` is a BOOT contract, not a
#: per-leg switch, so a lane whose shipping profile asks for a different boot
#: cannot share this server. Deliberately EXCLUDED and owed a separate boot:
#:   * humo / humo_14B_169 / humo_1.7B / humo_1.7B_169 -- want
#:     OTR_HEADLESS_RESERVE_VRAM_GB=2.921 + OTR_HEADLESS_DISABLE_PINNED=1
#:   * ltx_audio_in -- wants OTR_HEADLESS_DISABLE_PINNED=1 (no reserve); it
#:     cleared the 14.5 GiB gate by ~35 MB on a stock boot, which is why it
#:     keeps the diet
#: wan_i2v was RETIRED 2026-08-26 (19.82 GiB of weights vs a 14.5 GiB
#: target); the lane type survives on-card as wan_ti2v.
#:   * minimax_h3_video / minimax_h3_audio_in -- the only lanes the ENGINE
#:     itself refuses to run on the stock boot; it accepts the measured `h3`
#:     streaming boot or the physical-8-GB `h3_8gb_lab` launch shape
#: Cloud lanes (word_razzle, cloud_*, google_*) are excluded outright: the
#: render happens provider-side, so they prove nothing about a local model.
VIDEO_LANES = (
    ("otr_w45_ltx_8gb", "ltx_8gb"),
    ("otr_w45_wan_ti2v", "wan_ti2v"),
    ("otr_w45_fastwan", "fastwan_8gb"),
    ("otr_w45_mesh_stage", "mesh_stage"),
    ("otr_ghost_signal_v3_haunted", "animatediff15_v3_haunted_video"),
    ("otr_w45_ltx_video", "ltx_video"),
    ("otr_ltx25_high_video", "ltx25_video"),
)

#: SECONDS. Inherited from otr_llm_image_upscale_sweep.py, which raised it to
#: 7200 after legs were reported FAIL purely because the budget expired. This
#: number absorbs a full one-act render; the queue wait is handled separately
#: by _wait_for_idle_queue so a leg's clock measures its OWN work.
DEFAULT_TIMEOUT = 7200

COMFY_URL = "http://127.0.0.1:8000"
QUEUE_POLL_S = 15
QUEUE_DRAIN_MAX_S = 14400


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
            busy = (len(q.get("queue_running") or [])
                    + len(q.get("queue_pending") or []))
            if busy == 0:
                return True
            if not announced:
                print(f"[banksweep]   waiting for queue to drain ({busy} ahead)",
                      flush=True)
                announced = True
        except Exception:
            return False  # unreachable: let the leg try and fail honestly
        time.sleep(QUEUE_POLL_S)
    return False


def load_matrix(lanes: str = "image") -> "list[dict]":
    """Derive the leg matrix. COMPUTED, not read from a sidecar file.

    `.gitignore` excludes `scripts/_*.json` as scratch, so a matrix stored
    there could not run from a fresh clone -- the reason the sibling sweep
    driver moved its matrix into code. This one is deterministic too, so it
    belongs beside the code that uses it.

    `image` walks every bank against both engine profiles. `video` walks the
    default-boot video lanes, ROTATING the bank so the video sweep broadens
    bank coverage rather than re-proving one lane eight times.
    """
    legs = []
    index = 0
    if lanes == "video":
        for profile, engine in VIDEO_LANES:
            bank, style = BANKS[index % len(BANKS)]
            index += 1
            legs.append({
                "index": index,
                "bank": bank,
                "visual_style": style,
                "profile": profile,
                "video_engine": engine,
            })
        return legs
    for bank, style in BANKS:
        for profile in PROFILES:
            index += 1
            legs.append({
                "index": index,
                "bank": bank,
                "visual_style": style,
                "profile": profile,
                "video_engine": "",
            })
    return legs


def leg(entry: dict, *, timeout: int, dry_run: bool) -> dict:
    idx = entry["index"]
    bank = entry["bank"]
    profile = entry["profile"]
    stamp = datetime.datetime.now().strftime("%H%M%S")
    engine = entry.get("video_engine") or ""
    leg_label = (f"BANKSWEEP{idx:02d} bank={bank} profile={profile} c=t=E2B"
                 + (f" video={engine}" if engine else ""))
    cmd = [
        sys.executable, str(RUNNER),
        "--act-count", "1",
        "--source-bank", bank,
        "--visual-style", entry["visual_style"],
        "--profile", profile,
        "--creative-model", E2B,
        "--technical-model", E2B,
        "--timeout", str(timeout),
    ]
    if dry_run:
        # --offline-schemas is REQUIRED with --dry-run: without it the runner
        # still calls the live /object_info endpoint to resolve widget
        # choices, so a dry-run with no server booted dies on a
        # ConnectionError rather than validating anything. The sibling
        # driver's first cut omitted it and blanket-passed every leg -- a
        # false green. Never read dry_run alone as a verdict.
        cmd += ["--dry-run", "--offline-schemas"]
    print(f"[banksweep] leg {idx} START {stamp} {leg_label}", flush=True)
    if not dry_run:
        _wait_for_idle_queue()
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
    print(f"[banksweep] leg {idx} {'PASS' if ok else 'FAIL'} "
          f"{elapsed:.1f} min rc={rc} {tail[-1][:160] if tail else ''}",
          flush=True)
    return {
        "leg": idx, "leg_label": leg_label,
        "bank": bank, "visual_style": entry["visual_style"],
        "profile": profile, "video_engine": engine,
        "creative_model": E2B, "technical_model": E2B,
        "ok": ok, "rc": rc, "minutes": round(elapsed, 1),
        "stdout_tail": out[-4000:] if not ok else "",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs", default=None,
                    help="comma-separated 1-based leg numbers (default: all)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and dump each prompt without POSTing")
    ap.add_argument("--lanes", choices=("image", "video"), default="image",
                    help="image: every bank x both engine profiles (stills). "
                         "video: the default-boot video lanes, bank rotated. "
                         "humo / ltx_audio_in / minimax_h3 are NOT here -- "
                         "their profiles request a different server boot.")
    args = ap.parse_args(argv)

    matrix = load_matrix(args.lanes)
    if args.legs:
        wanted = {int(x) for x in args.legs.split(",")}
        matrix = [m for m in matrix if m["index"] in wanted]

    # One receipt per LANE MODE. Sharing a path would let the video sweep
    # clobber the image sweep's record, which is the same way the sibling
    # driver's receipt was destroyed by a later run of itself -- the reason
    # docs/2026-08-25-leg1-dossier-failure-evidence.md had to be recovered
    # from a server log at all.
    receipt = (RECEIPT if args.lanes == "image"
               else RECEIPT.with_name(RECEIPT.name.replace(
                   "-sweep-receipt.json", "-video-sweep-receipt.json")))
    results = []
    for entry in matrix:
        results.append(leg(entry, timeout=args.timeout, dry_run=args.dry_run))
        receipt.write_text(json.dumps({
            "generated_at": datetime.datetime.now().isoformat(),
            "model_under_test": E2B,
            "slots": "creative AND technical (both)",
            "lanes": args.lanes,
            "legs_run": len(results),
            "legs_total": len(matrix),
            "pass": sum(1 for x in results if x["ok"]),
            "fail": sum(1 for x in results if not x["ok"]),
            "results": results,
        }, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["ok"])
    print(f"[banksweep] DONE {passed}/{len(results)} passed. "
          f"Receipt: {receipt}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
