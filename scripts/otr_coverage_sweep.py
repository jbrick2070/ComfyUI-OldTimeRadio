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

#: The CORE/BLOCKING Wan engines the GATE-A acceptance sweep must exercise --
#: never --exclude-able in acceptance, each gated behind its enable flag. The
#: sweep is NOT green until BOTH pass (GO_FORWARD section 4A M2 + M3).
CORE_WAN_ENGINES = ("wan_i2v", "wan_ti2v")
WAN_ENABLE_FLAGS = {"wan_i2v": "OTR_ENABLE_WAN_I2V",
                    "wan_ti2v": "OTR_ENABLE_WAN_TI2V"}


# --------------------------------------------------------------------------- #
# acceptance gates (pure -- CPU-testable; GO_FORWARD section 4A M2 + M3 + M5)
# --------------------------------------------------------------------------- #
def acceptance_preflight(env, registered_engines, exclude_args):
    """GATE-A acceptance preflight. Returns a list of BLOCKING problem strings
    (empty == clear). Pure: ``env`` is a mapping and ``registered_engines`` is
    the enumerated registry set, so it is CPU-testable with no server.

    M5: ``OTR_TEST_MODE`` MUST be unset, else eng_wan_i2v skips its render-phase
    VRAM<=14.5GB assert. M3: every REGISTERED core Wan engine must have its
    enable flag set to "1" (a gated-off Wan leg enumerates "run", fails
    assert_usable closed, falls back, and -- pre-M1 -- false-passes; the same
    gated_by_flag mechanism that floored HuMo-1.7B), and ``--exclude`` must not
    filter a core Wan engine (the non-Wan sweep's escape hatch is forbidden)."""
    problems = []
    test_mode = str(env.get("OTR_TEST_MODE", "")).strip()
    if test_mode not in ("", "0"):
        problems.append(
            "OTR_TEST_MODE=%r is set -- it MUST be unset for acceptance so the "
            "Wan render-phase VRAM<=14.5GB assert actually runs (M5)" % test_mode)
    for engine in CORE_WAN_ENGINES:
        if engine not in registered_engines:
            # wan_ti2v may not be registered yet (8GB tier not built). A
            # not-yet-registered core engine is caught by the RESULTS gate (M2),
            # not the enable-flag preflight.
            continue
        flag = WAN_ENABLE_FLAGS[engine]
        if str(env.get(flag, "0")).strip() != "1":
            problems.append(
                "core Wan engine %s is registered but its enable flag %s != 1 "
                "(M3: acceptance must enable it, else it falls back closed)"
                % (engine, flag))
    for ex in exclude_args:
        hit = next((e for e in CORE_WAN_ENGINES if ex and ex in e), None)
        if hit is not None:
            problems.append(
                "--exclude %r filters core Wan engine %s -- forbidden in "
                "acceptance (M3)" % (ex, hit))
    return problems


def sweep_exit_code(results, acceptance, required_engines=CORE_WAN_ENGINES):
    """The sweep process exit code (0 == GREEN). GO_FORWARD section 4A M2.

    The old ``return 0 if passed == len(results)`` made ``0 == 0`` read GREEN on
    an EMPTY result set (``--only``/``--exclude`` filtered everything out, or
    ``wan_ti2v`` is unregistered). Here an empty result set is always RED, and a
    GREEN run needs every leg to PASS. In ``acceptance`` mode it is additionally
    RED unless every required core engine (wan_i2v AND wan_ti2v) is present in
    the results with a PASS verdict."""
    if not results:
        return 1
    if not all(r.get("verdict") == "PASS" for r in results):
        return 1
    if acceptance:
        passed_legs = [r.get("leg", "") for r in results
                       if r.get("verdict") == "PASS"]
        for engine in required_engines:
            token = engine.replace(".", "_")
            if not any(token in leg for leg in passed_legs):
                return 1
    return 0


def should_forbid_fallback(acceptance, strict_fallback):
    """The M1 no-runtime-fallback (CS-1) gate is ON in ``acceptance`` mode OR in
    ``strict_fallback`` mode. ``--strict-fallback`` DECOUPLES the CS-1 gate from
    the full GATE-A Wan-engine requirement (M2), so the NON-Wan permutation soak
    -- the set that originally false-greened -- can re-run with M1 active and a
    clean GREEN/RED exit, WITHOUT wan_ti2v built. Pure."""
    return bool(acceptance or strict_fallback)


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
    ap.add_argument("--exclude", action="append", default=[],
                    help="engine/leg SUBSTRING(s) to SKIP (repeatable); e.g. "
                         "--exclude wan. availability() is pure profile-fit and "
                         "never reads OTR_ENABLE_WAN_I2V, so Wan enumerates as "
                         "runnable unless filtered here (the non-Wan sweep). "
                         "NOTE: substring -- '--exclude humo' ALSO drops "
                         "humo_1.7B (humo is a prefix). Use --exclude-engine for "
                         "an exact engine-id match.")
    ap.add_argument("--exclude-engine", action="append", default=[],
                    help="EXACT engine-id(s) to SKIP (repeatable). Unlike "
                         "--exclude (substring), this matches the engine id "
                         "exactly -- so '--exclude-engine humo' drops ONLY the "
                         "14B humo leg and KEEPS humo_1.7B. Guarded like "
                         "--exclude: forbidden against core Wan in acceptance.")
    ap.add_argument("--acceptance", action="store_true",
                    help="GATE-A acceptance mode (GO_FORWARD 4A): assert ZERO "
                         "runtime fallbacks per leg (M1), preflight the Wan "
                         "enable flags + OTR_TEST_MODE-unset + forbid --exclude "
                         "of core Wan (M3/M5), and require wan_i2v AND wan_ti2v "
                         "to PASS (M2). Without it the sweep is the informational "
                         "dropdown-rotation coverage pass.")
    ap.add_argument("--strict-fallback", action="store_true",
                    help="Turn ON the M1 no-runtime-fallback (CS-1) gate WITHOUT "
                         "the full GATE-A Wan-engine requirement -- so the NON-Wan "
                         "permutation soak (the set that originally false-greened) "
                         "re-runs with M1 active and a clean GREEN/RED exit, no "
                         "wan_ti2v needed. Use with --exclude wan. Implied by "
                         "--acceptance.")
    args = ap.parse_args()
    forbid_fallback = should_forbid_fallback(args.acceptance, args.strict_fallback)

    options = enumerate_options()
    runnable = [(s, e) for s, e, r in options if r == "run"]
    skipped = [(s, e, r) for s, e, r in options if r != "run"]
    print("[sweep] %d dropdown options enumerated from the registry; "
          "%d runnable in enabled(16gb_full); %d skipped (disabled):"
          % (len(options), len(runnable), len(skipped)), flush=True)
    for s, e, r in skipped:
        print("[sweep]   SKIPPED_DISABLED %s=%s (%s)" % (s, e, r), flush=True)

    if args.acceptance:
        enumerated = {e for _s, e, _r in options}
        # Guard BOTH exclude paths against core Wan (M5): an exact engine
        # exclude must not be a back door around the substring guard.
        problems = acceptance_preflight(os.environ, enumerated,
                                        args.exclude + args.exclude_engine)
        if problems:
            print("[sweep] ACCEPTANCE PREFLIGHT FAILED (%d blocker(s)):"
                  % len(problems), flush=True)
            for p in problems:
                print("[sweep]   BLOCK: %s" % p, flush=True)
            return 2
        print("[sweep] acceptance preflight OK: OTR_TEST_MODE unset, core Wan "
              "enable flags set, no --exclude of core Wan", flush=True)
    if forbid_fallback:
        print("[sweep] M1 no-runtime-fallback gate: ON (%s)"
              % ("acceptance" if args.acceptance else "strict-fallback"),
              flush=True)

    mapping = load_widget_mapping()
    results = []
    t0 = time.time()
    for slot_key, engine in runnable:
        leg = "sweep_%s_%s" % (slot_key, engine.replace(".", "_"))
        if args.only and args.only not in leg:
            continue
        _excl = next((x for x in args.exclude if x and (x in leg or x in engine)),
                     None)
        if _excl is not None:
            print("[sweep]   SKIPPED_EXCLUDED %s (matched --exclude %r)"
                  % (leg, _excl), flush=True)
            continue
        # Exact engine-id exclude (e.g. drop 14B 'humo' but keep 'humo_1.7B').
        if engine in args.exclude_engine:
            print("[sweep]   SKIPPED_EXCLUDED %s (matched --exclude-engine %r)"
                  % (leg, engine), flush=True)
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
                              profile=profile, forbid_fallback=forbid_fallback)
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
    rc = sweep_exit_code(results, acceptance=args.acceptance)
    print("[sweep] COMPLETE: %d/%d legs PASS (%d skipped-disabled) in %.0f min "
          "-> exit %d%s"
          % (passed, len(results), len(skipped), (time.time() - t0) / 60.0, rc,
             " [ACCEPTANCE]" if args.acceptance else ""), flush=True)
    if rc != 0 and not results:
        print("[sweep] RED: zero legs ran (filters excluded everything, or no "
              "engine registered) -- an empty sweep is NOT a pass (M2)",
              flush=True)
    elif rc != 0 and args.acceptance:
        missing = [e for e in CORE_WAN_ENGINES
                   if not any(e.replace(".", "_") in r.get("leg", "")
                              for r in results if r.get("verdict") == "PASS")]
        if missing:
            print("[sweep] RED: acceptance requires PASS for core Wan engine(s) "
                  "%r -- absent from results (M2)" % missing, flush=True)
    return rc


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
