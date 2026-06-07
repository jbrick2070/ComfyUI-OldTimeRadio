#!/usr/bin/env python
"""OTR video dependency-pilot harness (A-S4 -- the lipsync-sidecar unblock).

The opt-in video engines that run in an ISOLATED cu128 sidecar venv (latentsync
now; LTX / Wan / HuMo as they land) need Blackwell / cu128 dependency validation
on the operator's GPU box BEFORE a live render is trusted: importing the engine
library must NOT swap the resident torch or drag in a determinism-breaking,
sm_120-brittle dependency (xformers / flash_attn / sageattention -- BUG-070).
This is the video twin of ``scripts/otr_audio_dep_pilot.py`` (one pattern, three
namespaces).

It is deliberately HEADLESS-SAFE: with the engine library absent (the box-fresh
state, and the pytest sandbox) every probe reports ``lib_absent`` without
crashing and without pulling any banned dependency. With the library installed
in its isolated venv (the operator GPU run -- point ``--python`` at it) it
snapshots ``torch.__version__`` + the presence of every BANNED_DEPS entry BEFORE
and AFTER importing the engine library and FAILS the engine if the import changed
torch or dragged in a banned dep, and records the ASSUMED render call the live
wiring must confirm.

Isolation: each engine is probed in its OWN subprocess (``--probe-one``), so a
polluted import never contaminates the parent. This is a diagnostic tool, never
an import-time side effect: run it, read the verdict. Promotion (install the
venv, flip OTR_ENABLE_LATENTSYNC, run the live GPU smoke) stays a human step.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys

# Dependencies a video-engine import must NOT pull onto the Blackwell / cu128
# sidecar stack: they break render-twice determinism and are brittle on sm_120
# (BUG-070). An engine whose import makes any appear is disqualified until fixed.
BANNED_DEPS = ("xformers", "flash_attn", "sageattention")

TOOL_VERSION = "1"

# The opt-in / sidecar-isolated video engines this pilot verifies. ``assumed_call``
# is the render interface the live GPU wiring must confirm AFTER this pilot proves
# the import is clean -- the single source of truth the worker + adapter point at.
OPT_IN_ENGINES = {
    "latentsync": {
        "lib_module": "latentsync",
        "adapter_class": "LatentSyncEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_LATENTSYNC",
        "assumed_call": (
            "python -m scripts.inference --unet_config_path "
            "configs/unet/stage2_512.yaml --inference_ckpt_path "
            "checkpoints/latentsync_unet.pt --video_path <base_clip> "
            "--audio_path <audio> --video_out_path <out> --seed <int>  "
            "# TODO-for-GPU-smoke: confirm the config/ckpt paths run headless on "
            "sm_120; optimize to a preloaded LipsyncPipeline reused per request"
        ),
    },
}


# --------------------------------------------------------------------------- #
# Pure helpers (no engine import; headless-testable)
# --------------------------------------------------------------------------- #
def _ensure_repo_on_path():
    """Put the repo root on sys.path so ``from nodes import ...`` resolves when
    this file runs as a standalone script (e.g. an isolated --probe-one)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _module_present(name):
    """True iff ``name`` is importable, WITHOUT importing it (find_spec only)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def dep_snapshot():
    """Snapshot torch version + presence of every BANNED_DEPS entry.

    ``torch`` is imported lazily (usually already resident); the banned deps are
    probed with ``find_spec`` so the snapshot never imports them.
    """
    try:
        import torch
        torch_version = str(torch.__version__)
    except Exception:  # noqa: BLE001 -- torch absent is a valid snapshot state
        torch_version = None
    snap = {"torch": torch_version}
    for dep in BANNED_DEPS:
        snap[dep] = _module_present(dep)
    return snap


def snapshot_violations(before, after):
    """Plan violations between two ``dep_snapshot`` results.

    A violation is: torch's version string changed (an engine swapped torch), or
    a banned dep was absent before and present after (the import dragged it in).
    """
    out = []
    b_torch = before.get("torch")
    a_torch = after.get("torch")
    if b_torch and a_torch and b_torch != a_torch:
        out.append(f"torch version changed: {b_torch} -> {a_torch}")
    for dep in BANNED_DEPS:
        if (not before.get(dep)) and after.get(dep):
            out.append(f"banned dependency pulled in by import: {dep}")
    return out


def _adapter_forward(engine_name):
    """Bound adapter forward for ``engine_name`` from the video registry, or
    ``None``. Importing the registry pulls NO engine library (adapters
    lazy-import inside the worker venv), so this is safe headless."""
    spec = OPT_IN_ENGINES[engine_name]
    _ensure_repo_on_path()
    try:
        from nodes._otr_video_engines.registry import get_engine
        adapter = get_engine(engine_name)
    except Exception:  # noqa: BLE001 -- a missing registry is a real finding
        return None
    return getattr(adapter, spec["forward"], None)


# --------------------------------------------------------------------------- #
# Probing (per-engine; subprocess-isolated by default)
# --------------------------------------------------------------------------- #
def probe_one(engine_name, *, do_import=True):
    """Probe ONE engine in THIS process. Headless-safe: a missing library is a
    clean ``lib_absent`` verdict, never an exception. Returns a verdict dict."""
    if engine_name not in OPT_IN_ENGINES:
        return {"engine": engine_name, "status": "unknown_engine"}
    spec = OPT_IN_ENGINES[engine_name]
    fwd = _adapter_forward(engine_name)
    verdict = {
        "engine": engine_name,
        "lib_module": spec["lib_module"],
        "adapter_class": spec["adapter_class"],
        "forward": spec["forward"],
        "flag": spec["flag"],
        "assumed_call": spec["assumed_call"],
        "adapter_registered": fwd is not None,
        "forward_present": fwd is not None,
        "banned_deps_clean": True,
        "violations": [],
        "import_clean_ready": False,
    }
    if not do_import:
        verdict["status"] = "not_imported"
        return verdict

    before = dep_snapshot()
    try:
        importlib.import_module(spec["lib_module"])
    except ImportError:
        verdict["status"] = "lib_absent"
        verdict["note"] = (
            f"{spec['lib_module']} not installed -- run "
            f"scripts\\_otr_latentsync_install.ps1 in this engine's isolated "
            f"venv and re-run on the GPU box to verify the import + render"
        )
        return verdict
    except Exception as exc:  # noqa: BLE001 -- a broken import is a real finding
        verdict["status"] = "import_error"
        verdict["note"] = f"{type(exc).__name__}: {exc}"
        return verdict

    after = dep_snapshot()
    violations = snapshot_violations(before, after)
    verdict["violations"] = violations
    verdict["banned_deps_clean"] = not violations
    verdict["status"] = "lib_present"
    # Structural precondition only -- the real render verification against
    # ``assumed_call`` is still a human step on the GPU box before promotion.
    verdict["import_clean_ready"] = bool(fwd is not None and not violations)
    return verdict


def probe_isolated(engine_name, *, python=None, timeout=600):
    """Probe ONE engine in a FRESH subprocess so a polluted import cannot
    contaminate the parent. ``python`` may point at a per-engine venv."""
    py = python or sys.executable
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["OTR_VIDEO_DEP_PILOT_CHILD"] = "1"
    try:
        proc = subprocess.run(
            [py, os.path.abspath(__file__), "--probe-one", engine_name, "--json"],
            capture_output=True, text=True, env=env, timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return {"engine": engine_name, "status": "probe_subprocess_error",
                "note": f"{type(exc).__name__}: {exc}"}
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    return {"engine": engine_name, "status": "probe_subprocess_error",
            "note": f"no JSON verdict on stdout (rc={proc.returncode}); "
                    f"stderr tail: {(proc.stderr or '')[-200:]}"}


def run_pilot(engines=None, *, isolated=True, python=None):
    """Probe each engine; return the aggregate report. Never raises."""
    names = list(engines) if engines else list(OPT_IN_ENGINES)
    verdicts = []
    for name in names:
        if isolated:
            verdicts.append(probe_isolated(name, python=python))
        else:
            verdicts.append(probe_one(name, do_import=True))
    ready = [v for v in verdicts if v.get("import_clean_ready")]
    return {
        "tool_version": TOOL_VERSION,
        "banned_deps": list(BANNED_DEPS),
        "engines": verdicts,
        "ready_count": len(ready),
        "engine_count": len(verdicts),
        "all_imports_clean": bool(verdicts) and len(ready) == len(verdicts),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_human(report):
    print(f"OTR video dependency pilot (tool v{report['tool_version']})")
    print(f"banned deps watched: {', '.join(report['banned_deps'])}")
    print("")
    for v in report["engines"]:
        print(f"[{v.get('status', '?')}] {v['engine']}  (flag {v.get('flag', '?')})")
        if v.get("note"):
            print(f"    note: {v['note']}")
        print(f"    adapter registered: {v.get('adapter_registered')}  "
              f"forward present: {v.get('forward_present')}")
        for viol in v.get("violations") or []:
            print(f"    VIOLATION: {viol}")
        print(f"    assumed call (verify-at-build): {v.get('assumed_call')}")
        print("")
    print(f"import-clean engines: {report['ready_count']}/{report['engine_count']}")
    print("next: install each lib in its isolated venv on the GPU box, verify the "
          "assumed_call by hand, then run the live OTR_ENABLE_<engine>=1 smoke.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OTR video engine dependency pilot (A-S4): the sidecar unblock."
    )
    parser.add_argument("--probe-one", metavar="ENGINE", default=None,
                        help="probe a single engine in THIS process and print its JSON verdict")
    parser.add_argument("--engines", default=None,
                        help="comma-separated subset of engines to probe")
    parser.add_argument("--no-isolation", action="store_true",
                        help="probe in-process instead of per-engine subprocesses")
    parser.add_argument("--python", default=None,
                        help="python executable for isolated probes (point at a per-engine venv)")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of a human report")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 unless every engine import is clean")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.probe_one:
        print(json.dumps(probe_one(args.probe_one, do_import=True), sort_keys=True))
        return 0

    engines = None
    if args.engines:
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    report = run_pilot(engines, isolated=not args.no_isolation, python=args.python)
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        _print_human(report)
    if args.strict and not report["all_imports_clean"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
