"""Operator GPU-smoke probe for the parked video engines (humo / latentsync /
ltx_video / wan_i2v) -- READY-to-run, never a fake pass.

The heavy engines ship default-OFF / dark: their CPU adapters (assert_usable,
the pure request build, canonicalize, the fallback wiring) are done, but the
in-process / sidecar FORWARD is an intentional ``NotImplementedError`` GPU-smoke
slice the operator finishes on the 5080. This probe automates everything that
CAN be checked without faking the render: it runs the real dependency pilot,
reads the opt-in flag, calls the real ``assert_usable`` (surfacing
GATED_BY_FLAG / MISSING_MODEL / INCOMPATIBLE_PROFILE with the ckpt path), checks
the SageAttention state (BUG-070), builds the real inference request, resolves
the real fallback chain, probes machine-wide NVML VRAM, and prints a READY /
NOT-READY verdict with the exact next steps. ``--run-render`` additionally drives
the adapter lifecycle and reports the real outcome (the GPU slice raises
NotImplementedError until the operator wires it -- reported honestly, never a
pass). For humo it also demonstrates the LOUD humo -> latentsync ->
still_kenburns fallback restamp on a simulated OOM.

Run on the 5080 AFTER: install the wrapper + ckpts, set the engine's ckpt env,
``setx OTR_ENABLE_<ENGINE> 1``, restart ComfyUI. Cold-import clean; UTF-8, no
BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import importlib.util
import json as _json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nodes._otr_shared import retry_taxonomy as _rt          # noqa: E402
from nodes._otr_shared.fallback import resolve_fallback_chain  # noqa: E402
from nodes._otr_video_engines import motion_common as _mc     # noqa: E402
from nodes._otr_video_engines import registry as _vreg        # noqa: E402
# Register every engine + the cheap-family radio floor.
from nodes._otr_video_engines import eng_humo                 # noqa: E402,F401
from nodes._otr_video_engines import eng_latentsync           # noqa: E402,F401
from nodes._otr_video_engines import eng_ltx_video            # noqa: E402,F401
from nodes._otr_video_engines import eng_wan_i2v              # noqa: E402,F401
from nodes._otr_video_engines import cheap_families           # noqa: E402,F401

#: Per-engine operator metadata (install hint + whether Sage matters).
ENGINES = {
    "humo": {
        "flag": "OTR_ENABLE_HUMO", "kind": "in-process",
        "install": "HuMo wrapper + ckpts (umt5_xxl CLIP, wan_2.1 VAE, "
                   "whisper_large_v3 audio encoder, HuMo diffusion/LoRA); "
                   "set OTR_HUMO_CKPT",
        "sage": "tolerated (verify on GPU)", "has_fallback": True},
    "latentsync": {
        "flag": "OTR_ENABLE_LATENTSYNC", "kind": "cu128 py3.10 sidecar",
        "install": "cu128 sidecar venv + bytedance/LatentSync repo + worker; "
                   "set OTR_LATENTSYNC_VENV / _WORKER / _REPO",
        "sage": "isolated in sidecar", "has_fallback": True},
    "ltx_video": {
        "flag": "OTR_ENABLE_LTX_VIDEO", "kind": "in-process",
        "install": "LTX-Video wrapper + ckpt; set OTR_LTX_VIDEO_CKPT",
        "sage": "MUST be clear (BUG-070 int8-PV abort)", "has_fallback": False},
    "wan_i2v": {
        "flag": "OTR_ENABLE_WAN_I2V", "kind": "in-process (sidecar under Sage)",
        "install": "Wan 2.2 I2V wrapper + ckpt; set OTR_WAN_I2V_CKPT",
        "sage": "escalates to a cu128 sidecar", "has_fallback": False},
}


def _load_dep_pilot():
    """Import scripts/otr_video_dep_pilot.py (shares the dep-pilot helpers)."""
    src = os.path.join(_REPO_ROOT, "scripts", "otr_video_dep_pilot.py")
    spec = importlib.util.spec_from_file_location("otr_video_dep_pilot", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_request(init_image="", audio_ref="", text_prompt="", frames=13):
    """A minimal VideoRequest-shaped dict the adapters' request builders read."""
    return {
        "shot_id": "smoke_shot", "request_id": "smoke",
        "text_prompt": text_prompt or "a 1940s radio studio, warm tungsten light",
        "asset_refs": {"init_image": init_image} if init_image else {},
        "audio_ref": {"path": audio_ref} if audio_ref else None,
        "base_clip_ref": None,
        "timing": {"target_frame_count": int(frames)},
        "canvas": {"w": 832, "h": 480, "aspect_policy": "pad"},
        "seed_bundle": {"request_seed": 12345},
        "init_w": 480, "init_h": 832,
    }


def _registry_fallback_of(name):
    if _vreg.is_registered(name):
        return getattr(_vreg.get_engine(name), "fallback_engine", None)
    return None


def demonstrate_humo_fallback():
    """Walk + LOUD-restamp the real humo -> latentsync -> still_kenburns chain on
    a simulated OOM (CPU-real; proves the wiring the GPU OOM will trigger)."""
    chain = resolve_fallback_chain("humo", _registry_fallback_of)
    section = {"video_revision": 1, "shots": [], "runtime_fallback_decisions": []}
    logs = []
    for frm, to in zip(chain, chain[1:]):
        rec = _rt.build_fallback_decision(
            shot_id="smoke_shot", beat_id="b_smoke", from_engine=frm,
            to_engine=to, kind=_rt.FailureKind.OOM, video_revision=1)
        section = _rt.append_runtime_fallback_decision(section, rec)
        logs.append(_rt.format_swap_log(rec))
    return {"chain": chain, "logs": logs,
            "converges_to_floor": chain[-1] == "still_kenburns",
            "decisions": section["runtime_fallback_decisions"]}


def attempt_render(eng, req, *, vram_ceiling):
    """Drive prepare -> render_clip -> canonicalize -> teardown; report honestly.

    The forward is the operator's GPU-smoke slice (NotImplementedError until
    wired) -- reported as such, NEVER a fake pass. On a real render it also
    asserts the NVML VRAM ceiling and returns the canonical clip.
    """
    prepared = None
    try:
        prepared = eng.prepare(host_caps={}, profile={}, session_ctx={})
    except NotImplementedError as e:
        return {"status": "gpu_slice_not_implemented", "stage": "prepare",
                "detail": str(e).splitlines()[0]}
    except Exception as e:                       # noqa: BLE001 - report, never fake
        return {"status": "not_ready", "stage": "prepare",
                "detail": "%s: %s" % (type(e).__name__, str(e).splitlines()[0])}
    try:
        raw = eng.render_clip(req, prepared)
        clip = eng.canonicalize(raw, req, {})
        used = _mc.vram_used_mb()
        within = used is None or used <= int(vram_ceiling)
        return {"status": "rendered", "clip": clip, "vram_used_mb": used,
                "vram_within_ceiling": within}
    except NotImplementedError as e:
        return {"status": "gpu_slice_not_implemented", "stage": "render_clip",
                "detail": str(e).splitlines()[0]}
    except Exception as e:                       # noqa: BLE001
        return {"status": "render_failed", "stage": "render_clip",
                "detail": "%s: %s" % (type(e).__name__, str(e).splitlines()[0])}
    finally:
        try:
            eng.teardown(prepared)
        except Exception:                        # noqa: BLE001 - teardown best-effort
            pass


def run_smoke(engine, *, init_image="", audio_ref="", text_prompt="",
              frames=13, run_render=False, vram_ceiling=14500):
    """Run every CPU-checkable readiness probe for ``engine`` + (optionally) the
    live render attempt. Returns a structured report dict."""
    meta = ENGINES[engine]
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("registered", _vreg.is_registered(engine),
        "in the static video registry (V-6)")

    try:
        pilot = _load_dep_pilot()
        v = pilot.probe_one(engine, do_import=True)
        add("dep_pilot_import_clean", v.get("import_clean_ready", False),
            "status=%s" % v.get("status"))
    except Exception as e:                        # noqa: BLE001
        add("dep_pilot_import_clean", False, "pilot error: %s" % e)

    flag = meta["flag"]
    add("flag_set", os.getenv(flag, "0") == "1", "%s=%s" % (flag, os.getenv(flag, "0")))

    eng = _vreg.get_engine(engine)
    try:
        eng.assert_usable(host_caps={}, profile={})
        add("assert_usable", True, "usable")
    except Exception as e:                        # noqa: BLE001 - EngineUnusable etc.
        reason = getattr(e, "reason", None)
        add("assert_usable", False, "%s -- %s"
            % (getattr(reason, "value", type(e).__name__), str(e).splitlines()[0]))

    sage = _mc.sageattention_patched()
    add("sage_state", True, "patched=%s (%s)" % (sage, meta["sage"]))

    if meta["has_fallback"]:
        chain = resolve_fallback_chain(engine, _registry_fallback_of)
        add("fallback_chain", chain[-1] == "still_kenburns", " -> ".join(chain))

    try:
        if hasattr(eng, "_build_render_request"):
            built = eng._build_render_request(
                _minimal_request(init_image, audio_ref, text_prompt, frames))
        elif hasattr(eng, "_build_worker_request"):
            built = eng._build_worker_request(
                _minimal_request(init_image, audio_ref, text_prompt, frames),
                "smoke_out.mp4")
        else:
            built = {}
        add("request_build", True, "keys=%s" % sorted(built))
    except Exception as e:                        # noqa: BLE001
        add("request_build", False, "%s: %s" % (type(e).__name__, e))

    used = _mc.vram_used_mb()
    add("nvml_vram", True,
        "used=%s MB (ceiling %d)" % (used, vram_ceiling) if used is not None
        else "NVML unavailable (run on the 5080)")

    report = {"engine": engine, "kind": meta["kind"], "checks": checks}
    report["ready"] = all(c["ok"] for c in checks
                          if c["name"] in ("registered", "dep_pilot_import_clean",
                                           "flag_set", "assert_usable"))
    if engine == "humo":
        report["fallback_demo"] = demonstrate_humo_fallback()
    if run_render:
        report["render_attempt"] = attempt_render(
            eng, _minimal_request(init_image, audio_ref, text_prompt, frames),
            vram_ceiling=vram_ceiling)
    return report


def _next_steps(engine):
    meta = ENGINES[engine]
    return [
        "1. Install: %s" % meta["install"],
        "2. setx %s 1   (then restart ComfyUI so the env + module cache reload)"
        % meta["flag"],
        "3. Dep pilot: python scripts\\otr_video_dep_pilot.py --python "
        "<venv py> --json   (and --scan-custom-nodes)",
        "4. Implement + run the in-process forward (the NotImplementedError "
        "GPU slice): load + render_clip on the 5080.",
        "5. Re-run this probe with --run-render: assert VRAM <= 14.5 GB, render "
        "twice for determinism%s."
        % (" + exercise the humo -> latentsync -> still_kenburns fallback"
           if engine == "humo" else ""),
    ]


def print_report(report):
    print("=" * 70)
    print("ENGINE: %s  (%s)" % (report["engine"], report["kind"]))
    for c in report["checks"]:
        print("  [%s] %-24s %s" % ("PASS" if c["ok"] else "FAIL",
                                    c["name"], c["detail"]))
    if "fallback_demo" in report:
        fd = report["fallback_demo"]
        print("  fallback chain: %s (converges=%s)"
              % (" -> ".join(fd["chain"]), fd["converges_to_floor"]))
        for line in fd["logs"]:
            print("    " + line)
    if "render_attempt" in report:
        ra = report["render_attempt"]
        print("  render attempt: status=%s stage=%s %s"
              % (ra.get("status"), ra.get("stage", "-"), ra.get("detail", "")))
    print("  VERDICT: %s" % ("READY for the GPU render slice"
                             if report["ready"] else "NOT READY"))
    if not report["ready"]:
        print("  next steps:")
        for s in _next_steps(report["engine"]):
            print("    " + s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Operator GPU-smoke readiness probe for the parked video "
                    "engines (never a fake pass).")
    ap.add_argument("--engine", choices=list(ENGINES) + ["all"], default="all")
    ap.add_argument("--init-image", default="")
    ap.add_argument("--audio-ref", default="")
    ap.add_argument("--text-prompt", default="")
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--run-render", action="store_true",
                    help="drive the adapter lifecycle (GPU slice reported "
                         "honestly; never faked)")
    ap.add_argument("--vram-ceiling", type=int, default=14500)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    engines = list(ENGINES) if args.engine == "all" else [args.engine]
    reports = [run_smoke(e, init_image=args.init_image, audio_ref=args.audio_ref,
                         text_prompt=args.text_prompt, frames=args.frames,
                         run_render=args.run_render,
                         vram_ceiling=args.vram_ceiling)
               for e in engines]
    if args.json:
        print(_json.dumps(reports, indent=2, default=str))
    else:
        for r in reports:
            print_report(r)
    # exit 0 only if every selected engine is READY (honest on the CPU box).
    return 0 if all(r["ready"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
