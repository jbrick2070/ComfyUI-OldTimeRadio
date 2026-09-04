"""One 1-act episode per VIDEO LANE, on the pod, with the operator's settings.

WHY NOT `otr_bank_engine_sweep.py --lanes video`: that driver exists to prove the
SMALLEST writer survives every bank, so it hard-pins `gemma-4-E2B-it` in both
slots and pins one visual style per bank. The operator asked for the opposite on
both axes -- *"NO USE 12B ... on both creative and technical"* and *"leave the
video style randomizer on"* -- so this drives the same per-leg runner
(`otr_canonical_api_run.py`, which loads the REAL workflows/otr_canonical.json
and asserts that path) with those two settings instead. Nothing here edits a
runner, a profile, or the canonical JSON.

NO `--title` IS PASSED, DELIBERATELY. The harness label becomes the published
title card; the operator's words were "it bled into my title". Legs are told
apart by their log file and the ledger's own timestamp.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

# Paths are env-overridable so this is a tool rather than one pod's script.
# The default is this file's own repo, which is right when it runs from
# scripts/ and WRONG when a copy has been dropped somewhere convenient -- on
# the pod it lived at /workspace/, where the naive parent-of-parent resolves to
# "/" and every leg dies instantly with
#   can't open file '/scripts/otr_canonical_api_run.py'
# So the location is VERIFIED rather than assumed, and an unusable root is a
# named refusal instead of sixteen mystery failures.
REPO = os.environ.get(
    "OTR_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(REPO, "scripts", "otr_canonical_api_run.py")
if not os.path.isfile(RUNNER):
    raise SystemExit(
        "cannot find the canonical runner at %s\n"
        "Set OTR_REPO_ROOT to the ComfyUI-OldTimeRadio checkout, e.g.\n"
        "  export OTR_REPO_ROOT=/workspace/runpod-slim/ComfyUI/custom_nodes/"
        "ComfyUI-OldTimeRadio" % RUNNER)
COMFY = os.environ.get("OTR_COMFYUI_URL", "http://127.0.0.1:8188")

WRITER = "google/gemma-4-12b-it (11.9 GB)"      # both slots, operator 2026-09-03
STYLE = "roll (any style)"                       # the randomizer, left ON
ACTS = "1"

#: (profile, engine, source_bank). Banks rotate so the sweep broadens bank
#: coverage instead of re-proving one bank seven times -- same reasoning as the
#: sweep driver this replaces.
BANKS = ("media_archive", "original", "public_domain", "shakespeare")

#: THE FULL LOCAL VIDEO MATRIX, ordered by value so a run that is cut short
#: still covers what the operator asked about first (foley, then the joint-AV
#: lanes, then wan/AnimateDiff, then the cheap visual families).
#:
#: THE IMAGE ENGINE IS CHOSEN BY THE PROFILE -- it is not a runner flag, and the
#: sanctioned lever is `role_overrides`, exactly what a human clicking the
#: dropdowns would save. So the profile for each lane is picked to SPREAD the
#: five local image engines instead of letting every leg default to
#: z_image_turbo: ideogram4_local, flux_gen1, lumina_image, flux2_klein and
#: `ideo` each get at least one leg. Cloud image engines (google_image,
#: cloud_nano_banana_2) are excluded -- they render provider-side and prove
#: nothing about this pod.
#: EVERY LANE RUNS ON z_image_turbo, AND THAT IS A CONSTRAINT, NOT A CHOICE.
#: Measured on this pod 2026-09-03: `z_image_turbo_bf16.safetensors` is the ONLY
#: image model on disk. flux2_klein, flux_gen1, lumina_image and ideogram4_local
#: are all absent and none is offered by `otr_fetch_lane_weights.py` -- they are
#: the "manual public files" tiers of RUNPOD_INSTALL section 4. Pairing a lane
#: with a profile that names one of them fails ADAPTER-level usability before a
#: pixel is rendered:
#:     still_music_opening_001: image engine 'ideogram4_local' failed ...
#:     missing_model -- requires all four artifacts; missing: ideogram4_nvfp4_mix
#: So image-model spread is a SEPARATE errand that needs those downloads first.
#:
#: ALSO LEARNED THE HARD WAY: do not pair a big writer with a small-card profile.
#: `otr_nvidia_8gb_haunted` is an 8 GB row, and pinning the 12B writer against it
#: raises VRAMFitFailedError in 0.1 min -- the profile's own VRAM policy refuses
#: the model, regardless of the 24 GB actually available. The `otr_w45_*` rows
#: carry no such ceiling.
#:
#: Ordered cheap-first so a run that is cut short has banked the most lanes.
LANES = [
    ("otr_w45_viz_camera", "viz_camera"),
    ("otr_w45_viz_green", "viz_green"),
    ("otr_w45_still_flat", "still_flat"),
    ("otr_w45_still_motion", "still_motion"),
    ("otr_w45_still_pan", "still_pan"),
    ("otr_w45_still_word", "still_word"),
    ("otr_w45_word_razzle", "word_razzle"),
    ("otr_w45_mesh_stage", "mesh_stage"),
    ("otr_w45_animatediff15_v3_haunted_video", "animatediff15_v3_haunted_video"),
    ("otr_w45_wan_ti2v", "wan_ti2v"),
    ("otr_w45_fastwan", "fastwan_8gb"),
    ("otr_w45_ltx_8gb", "ltx_8gb"),
    ("otr_w45_ltx_video", "ltx_video"),
    # The LTX 2.5 family is LAST on purpose: re-confirmed 2026-09-03 to OOM at
    # `decode` on a 24 GB Ada card, matching the evidence ledger's existing
    # 4090/LTX-2.5 negative. Kept in the list so a bigger card runs them.
    ("otr_w45_ltx25_foley_plus", "ltx25_foley_plus"),
    ("otr_w45_ltx25_mime", "ltx25_mime"),
    ("otr_ltx25_high_video", "ltx25_video"),
]

#: Lanes that need their OWN BOOT (different launch env), so they cannot share
#: the server the seven above run on. Run with --group h3 / --group humo AFTER
#: rebooting ComfyUI with that group's env.
GROUPS = {
    # Every lane that shares the STANDARD boot -- joint-AV foley/mime, wan,
    # AnimateDiff, the LTX family, and the cheap still/viz families. Folded into
    # one group on purpose: they need no reboot between them, so splitting them
    # would only cost model reloads.
    "video": LANES,
    # MiniMax H3 -- the engine itself refuses the stock boot; run these only
    # after rebooting ComfyUI with the h3 launch env.
    "h3": [
        ("otr_rot_h3_lumina", "h3_low_video"),          # lumina_image
        ("otr_h3_low_audio_in", "h3_low_audio_in"),     # z_image_turbo
        ("otr_nvidia_8gb_h3", "minimax_h3_video"),      # flux2_klein
    ],
    # HuMo -- wants OTR_HEADLESS_RESERVE_VRAM_GB + DISABLE_PINNED, so its own boot.
    "humo": [
        ("otr_rot_humo_klein", "humo"),                 # flux2_klein
        ("otr_w45_humo_1_7b", "humo_1.7B"),
        ("otr_w45_humo_1_7b_169", "humo_1.7B_169"),
        ("otr_w45_humo_14b_169", "humo_14B_169"),
    ],
    # ltx_audio_in wants DISABLE_PINNED with no reserve -- its own boot again.
    "audio_in": [
        ("otr_w45_ltx_audio_in", "ltx_audio_in"),
    ],
}


def _stamp():
    return datetime.datetime.now().strftime("%H:%M:%S")


def run_leg(idx, profile, engine, bank, timeout_s, logdir):
    log_path = os.path.join(logdir, "leg%02d_%s.log" % (idx, engine))
    cmd = [
        sys.executable, RUNNER,
        "--profile", profile,
        "--source-bank", bank,
        "--visual-style", STYLE,
        "--creative-model", WRITER,
        "--technical-model", WRITER,
        "--act-count", ACTS,
        "--comfyui-url", COMFY,
        "--timeout", str(timeout_s),
    ]
    print("[%s] leg %d START engine=%-32s bank=%-14s profile=%s"
          % (_stamp(), idx, engine, bank, profile), flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("### engine=%s bank=%s profile=%s writer=%s style=%s\n"
                 % (engine, bank, profile, WRITER, STYLE))
        fh.flush()
        try:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=REPO, timeout=timeout_s + 300)
        except subprocess.TimeoutExpired:
            rc = 124
    mins = (time.time() - t0) / 60.0
    verdict = "PASS" if rc == 0 else "FAIL"
    print("[%s] leg %d %s  %.1f min rc=%d  engine=%s  log=%s"
          % (_stamp(), idx, verdict, mins, rc, engine, os.path.basename(log_path)),
          flush=True)
    return {"leg": idx, "engine": engine, "bank": bank, "profile": profile,
            "rc": rc, "verdict": verdict, "minutes": round(mins, 1),
            "log": log_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="video", choices=sorted(GROUPS))
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--logdir", default="/workspace/lane_legs")
    ap.add_argument("--only", default=None,
                    help="comma-separated engine ids to run from the group")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    lanes = GROUPS[args.group]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        lanes = [x for x in lanes if x[1] in want]

    print("=== group=%s  %d lane(s)  writer=%s  style=%s  acts=%s ==="
          % (args.group, len(lanes), WRITER, STYLE, ACTS), flush=True)
    results = []
    for i, (profile, engine) in enumerate(lanes, start=1):
        bank = BANKS[(i - 1) % len(BANKS)]
        results.append(run_leg(i, profile, engine, bank, args.timeout, args.logdir))
        receipt = os.path.join(args.logdir, "receipt_%s.json" % args.group)
        with open(receipt, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    passed = sum(1 for r in results if r["rc"] == 0)
    print("=== DONE %d/%d passed ===" % (passed, len(results)), flush=True)
    for r in results:
        print("  %-32s %-6s %5.1f min" % (r["engine"], r["verdict"], r["minutes"]),
              flush=True)


if __name__ == "__main__":
    main()
