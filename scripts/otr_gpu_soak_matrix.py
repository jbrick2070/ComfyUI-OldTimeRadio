"""Keep the GPU busy: 1-act episodes across the live banks and visual styles.

WHY THIS SHAPE. The operator asked for continuous test episodes while he is
remote -- "1 act across random local banks, styles and local video/still
models". Banks and styles ARE rotated here. Engines are NOT, and that is a
design constraint rather than an omission: the video/image engine widgets are
MANAGED, so `patch_creative` refuses them (the BUG-08.06 stranded-COMBO
class) and the sanctioned surface is a capability profile's `role_overrides`.
Authoring soak profiles is its own change with its own emit/ratify gate. Every
leg therefore runs the canonical engines -- still_flat + z_image_turbo.

SEQUENTIAL BY DESIGN. One GPU, one render at a time (CLAUDE.md scope rule:
no async CUDA streams, no queue refactor). Each leg runs to a terminal state
before the next is submitted, and a failed leg is LOGGED and skipped rather
than stopping the campaign -- an overnight soak that dies on leg 3 is worth
less than one that reports 3 failures out of 20.

A receipt JSON is rewritten after EVERY leg, so killing the run mid-flight
still leaves a complete record of what has finished.

Usage:
    python scripts/otr_gpu_soak_matrix.py --legs 12
    python scripts/otr_gpu_soak_matrix.py --legs 0      # until stopped
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import random
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "otr_canonical_api_run.py"

#: The five runnable banks after the 2026-08-16 scifi_news rip. Rotated
#: EXPLICITLY rather than left to the canonical roll, so a finite soak gets
#: even coverage instead of luck -- a roll can hand you four shakespeares.
BANKS = [
    "media_archive",
    "original",
    "scifi_news_pro",
    "public_domain",
    "shakespeare",
]

#: Every non-sentinel visual style the live registry offers.
STYLES = [
    "anime", "archival_documentary", "cartoon", "paper_origami",
    "recur_frac", "sci_fi_radio", "shakespeare_stage_realism",
    "storybook_engraving", "video_art", "visual_storybased",
]

#: Engine rotation goes through CAPABILITY PROFILES, which is the sanctioned
#: surface: the video/image widgets are MANAGED and `patch_creative` refuses
#: them outright so a run cannot strand a COMBO value (BUG-08.06 class). Each
#: profile below differs from `16gb_full` in role_overrides ONLY, so a leg
#: differs from a normal render in exactly the engine under test. They are
#: listed in build_variants.LANE_PRESETS so they never emit a shipping
#: variant -- soak instruments, not platform targets.
PROFILES = [
    "otr_soak_still_flat_z_image_turbo",
    "otr_soak_still_flat_flux_gen1",
    "otr_soak_still_motion_lumina_image",
    "otr_soak_still_motion_flux2_klein",
    "otr_soak_still_pan_flux_gen1",
    "otr_soak_still_pan_ideo",
    "otr_soak_still_word_flux2_klein",
    "otr_soak_still_word_z_image_turbo",
    "otr_soak_word_razzle_ideo",
    "otr_soak_word_razzle_lumina_image",
]


def leg(index, bank, style, profile, timeout) -> dict:
    stamp = datetime.datetime.now().strftime("%H%M%S")
    short = profile.replace("otr_soak_", "")
    title = f"SOAK{index:02d} {bank} {style} {short}"
    cmd = [
        sys.executable, str(RUNNER),
        "--act-count", "1",
        "--title", title,
        "--source-bank", bank,
        "--visual-style", style,
        "--profile", profile,
        "--timeout", str(timeout),
    ]
    started = datetime.datetime.now()
    print(f"[soak] leg {index} START {stamp} bank={bank} style={style} "
          f"engines={short}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, timeout=timeout + 600)
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = "RESULT SUCCESS" in out
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        out, ok, rc = "harness timeout", False, -1
    elapsed = (datetime.datetime.now() - started).total_seconds() / 60.0
    tail = [ln for ln in out.splitlines() if "RESULT" in ln or "Exception" in ln]
    print(f"[soak] leg {index} {'PASS' if ok else 'FAIL'} "
          f"{elapsed:.1f} min rc={rc} {tail[-1][:110] if tail else ''}",
          flush=True)
    return {"leg": index, "bank": bank, "style": style,
            "profile": profile, "ok": ok,
            "rc": rc, "minutes": round(elapsed, 1), "title": title}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs", type=int, default=12,
                    help="0 = run until stopped")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    print(f"[soak] rotating {len(BANKS)} banks x {len(STYLES)} styles x "
          f"{len(PROFILES)} engine profiles, 1 act per leg", flush=True)

    results, index = [], 0
    out_dir = REPO / "otr_soak_receipts"
    out_dir.mkdir(exist_ok=True)
    receipt = out_dir / (
        "soak_%s.json" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        while args.legs == 0 or index < args.legs:
            index += 1
            row = leg(index, rng.choice(BANKS), rng.choice(STYLES),
                      rng.choice(PROFILES), args.timeout)
            results.append(row)
            receipt.write_text(json.dumps(results, indent=1), encoding="utf-8")
    except KeyboardInterrupt:
        print("[soak] interrupted -- receipts kept", flush=True)

    passed = sum(1 for r in results if r["ok"])
    print(f"\n[soak] {passed}/{len(results)} passed. receipt: {receipt}")
    for r in results:
        if not r["ok"]:
            print(f"        FAIL leg {r['leg']}: {r['bank']} + {r['style']} "
                  f"+ {r['profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
