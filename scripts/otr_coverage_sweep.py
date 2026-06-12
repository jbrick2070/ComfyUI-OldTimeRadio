"""otr_coverage_sweep.py -- the GATE A dropdown coverage sweep (3D plan sec. 0
item 4), powered by the GATE B S2 applier.

Every engine option in the VIDEO dropdowns (announcer_visual / music_visual /
other_beats) renders a 30-word FULL episode on the live server, one leg per
(slot, engine), via an EPHEMERAL capability profile applied by the ONE applier
-- never a hand-coded patch list. Options are ENUMERATED FROM THE REGISTRY
(intersected with enabled(16gb_full)); engines outside the enable-set
(cu128-toolkit 3D darks) are recorded SKIPPED_DISABLED, never silently
omitted. Each leg runs the capstone gates (playable obs final, byte-identical
master audio, output hygiene, VRAM ceiling) with expect_engine="" (the
dropdown-rotation mode -- completion gates hold, the strict humo histogram
does not).

Usage (ComfyUI venv python; live server on :8000):
    python scripts\\otr_coverage_sweep.py [--only slot_or_engine_substr]
Results: scripts/_otr_soak_capstone_results/sweep_*.json (per leg, via
run_leg) + scripts/coverage_sweep_summary.json + LOUD stdout.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_HERE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

# Resolve the live server's output tree DYNAMICALLY (the 2026-06-12 Desktop-v2
# install move made any hardcoded pin go stale: the server now writes under
# ComfyUI-Installs, not Documents). Honor an explicit env override; otherwise
# pick the candidate whose otr/episodes tree was written most recently, and
# say so LOUDLY so a wrong pick is visible in every leg log.
_OUTPUT_CANDIDATES = (
    r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\output",
    r"C:\Users\jeffr\Documents\ComfyUI\output",
)


def _resolve_server_output() -> str:
    explicit = os.environ.get("OTR_SOAK_SERVER_OUTPUT")
    if explicit:
        print(f"[sweep] server output (env override): {explicit}", flush=True)
        return explicit

    def _episodes_mtime(root):
        try:
            return os.path.getmtime(os.path.join(root, "otr", "episodes"))
        except OSError:
            return -1.0

    scored = [(_episodes_mtime(c), c) for c in _OUTPUT_CANDIDATES]
    best_mtime, best = max(scored)
    if best_mtime < 0:
        raise SystemExit(
            "[sweep] FATAL: no candidate server output tree has otr/episodes; "
            "set OTR_SOAK_SERVER_OUTPUT explicitly. Candidates tried: "
            + "; ".join(_OUTPUT_CANDIDATES))
    print(f"[sweep] server output (auto, newest otr/episodes): {best}",
          flush=True)
    return best


os.environ["OTR_SOAK_SERVER_OUTPUT"] = _resolve_server_output()

from nodes._otr_shared.capability_profiles import (  # noqa: E402
    availability, cross_validate_profile, load_profile, load_widget_mapping,
)
from nodes._otr_video_engines import registry as vreg  # noqa: E402
from nodes._otr_video_engines import (  # noqa: E402,F401  (register adapters)
    cheap_families, eng_character_3d, eng_humo, eng_latentsync,
    eng_ltx_video, eng_wan_i2v,
)

import _otr_soak_capstone as soak  # noqa: E402

SUMMARY_PATH = os.path.join(_HERE, "coverage_sweep_summary.json")

#: OTR_VideoDirector slot -> (profile role_overrides key, registry role token).
SLOTS = (
    ("announcer_visual", "announcer_visual"),
    ("music_visual", "music_visual"),
    ("other_beats_visual", "character_video"),
)


def enumerate_options():
    """(slot_key, engine, reason) per dropdown option, FROM THE REGISTRY.
    reason: "run" when in enabled(16gb_full); else the availability code."""
    profile = load_profile("16gb_full")
    avail = availability(profile, vreg.CAPABILITIES)
    out = []
    for slot_key, role in SLOTS:
        for engine in vreg.engines_for_role(role):
            reason = avail.get(engine, "undeclared")
            out.append((slot_key, engine, "run" if reason == "ok" else reason))
    return out


def profile_for(slot_key: str, engine: str) -> dict:
    profile = copy.deepcopy(load_profile("16gb_full"))
    profile["role_overrides"][slot_key] = engine
    if slot_key == "other_beats_visual":
        # node 92's render default tracks the character engine so the live
        # render path and the OTR_VideoDirector policy agree.
        profile["slot_overrides"]["video_render_engine"] = engine
    return profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="substring filter on '<slot>_<engine>' leg names")
    args = ap.parse_args()

    options = enumerate_options()
    runnable = [(s, e) for s, e, r in options if r == "run"]
    skipped = [(s, e, r) for s, e, r in options if r != "run"]
    print("[sweep] %d dropdown options enumerated from the registry; "
          "%d runnable in enabled(16gb_full); %d skipped (disabled):"
          % (len(options), len(runnable), len(skipped)), flush=True)
    for s, e, r in skipped:
        print("[sweep]   SKIPPED_DISABLED %s=%s (%s)" % (s, e, r), flush=True)

    mapping = load_widget_mapping()
    results = []
    t0 = time.time()
    for slot_key, engine in runnable:
        leg = "sweep_%s_%s" % (slot_key, engine.replace(".", "_"))
        if args.only and args.only not in leg:
            continue
        profile = profile_for(slot_key, engine)
        try:
            cross_validate_profile(profile, mapping, {
                "video": vreg.CAPABILITIES,
                "audio": __import__(
                    "nodes._otr_audio_engines.registry",
                    fromlist=["CAPABILITIES"]).CAPABILITIES,
                "image": __import__(
                    "nodes._otr_image_engines.registry",
                    fromlist=["CAPABILITIES"]).CAPABILITIES,
            })
        except Exception as exc:  # noqa: BLE001 -- record, keep sweeping
            results.append({"leg": leg, "verdict": "PROFILE_INVALID",
                            "error": str(exc)[:400]})
            print("[sweep] PROFILE_INVALID %s: %s" % (leg, exc), flush=True)
            continue
        print("[sweep] === LEG %s (%s=%s) ===" % (leg, slot_key, engine),
              flush=True)
        t1 = time.time()
        try:
            rc = soak.run_leg(leg, expect_floor=False, expect_engine="",
                              profile=profile)
            verdict = "PASS" if rc == 0 else "RC_%d" % rc
        except soak.SoakFail as exc:
            verdict = "SOAK_FAIL"
            print("[sweep] SOAK_FAIL %s: %s" % (leg, exc), flush=True)
            results.append({"leg": leg, "verdict": verdict,
                            "error": str(exc)[:600],
                            "elapsed_s": round(time.time() - t1, 1)})
            continue
        except Exception as exc:  # noqa: BLE001 -- record, keep sweeping
            verdict = "ERROR"
            print("[sweep] ERROR %s: %s" % (leg, exc), flush=True)
            traceback.print_exc()
            results.append({"leg": leg, "verdict": verdict,
                            "error": repr(exc)[:600],
                            "elapsed_s": round(time.time() - t1, 1)})
            continue
        results.append({"leg": leg, "verdict": verdict,
                        "elapsed_s": round(time.time() - t1, 1)})
        print("[sweep] %s -> %s (%.0fs)" % (leg, verdict, time.time() - t1),
              flush=True)
        _write_summary(options, results, t0, done=False)

    _write_summary(options, results, t0, done=True)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    print("[sweep] COMPLETE: %d/%d legs PASS (%d skipped-disabled) in %.0f min"
          % (passed, len(results), len(skipped), (time.time() - t0) / 60.0),
          flush=True)
    return 0 if passed == len(results) else 1


def _write_summary(options, results, t0, done):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "done": done,
            "elapsed_min": round((time.time() - t0) / 60.0, 1),
            "options": [{"slot": s, "engine": e, "reason": r}
                        for s, e, r in options],
            "results": results,
        }, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
