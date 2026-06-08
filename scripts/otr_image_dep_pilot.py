#!/usr/bin/env python
"""OTR image dependency-pilot harness (Subproject C -- the image-sidecar unblock).

The third namespace twin of ``scripts/otr_audio_dep_pilot.py`` +
``scripts/otr_video_dep_pilot.py`` (one pattern, three namespaces). It validates
the dependency posture of the pluggable IMAGE engines before a live render is
trusted, and -- unlike video, whose engines are ALL cu128-sidecar-isolated -- it
recognises that image engines come in TWO integration modes:

* **Sidecar-isolated** (exposes ``SIDECAR_ENV``): runs in a pinned cu128 venv
  subprocess (the protected cu130 main venv must not be contaminated). Z-Image
  (C2) is one. For these the real risk is the same as video: importing the engine
  pipeline must NOT swap the resident torch or drag in a determinism-breaking,
  sm_120-brittle dep (xformers / flash_attn / sageattention -- BUG-070). This
  pilot carries the BEFORE/AFTER snapshot probe for exactly that, run in the
  sidecar venv (point ``--python`` at it and ``--lib`` at the confirmed package).
* **In-stack** (exposes ``MODEL_ENV``): runs on the main venv via an
  already-installed loader (the ComfyUI-GGUF loader for the GGUF peers; ComfyUI's
  native loaders for the native peers). A GGUF/native checkpoint is dtype-isolated
  and pulls no contaminating dep, so these need no sidecar probe -- their
  dependency posture is already proven by the per-adapter cold-import tests
  (``test_image_engine_*`` / ``test_image_platform_c1``). This pilot reports them
  as in-stack with that note, never inventing a sidecar probe they do not need.

It is deliberately HEADLESS-SAFE: it enumerates engines from the (cold-import
clean) image registry and reports each one's integration mode + the exact
verify-at-build contract WITHOUT importing any model framework. The snapshot probe
only runs when the operator explicitly asks for it on the GPU box. A diagnostic
tool, never an import-time side effect: run it, read the verdict. Promotion
(install the sidecar venv, flip ``OTR_ENABLE_<ENGINE>``, run the live GPU smoke)
stays a human step. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

# Deps an engine import must NOT pull onto the Blackwell / cu128 sidecar stack:
# they break render-twice determinism and are brittle on sm_120 (BUG-070). Same
# watch-list as the video pilot (one taxonomy, three namespaces).
BANNED_DEPS = ("xformers", "flash_attn", "sageattention")

TOOL_VERSION = "1"


# --------------------------------------------------------------------------- #
# Pure helpers (no engine import; headless-testable)
# --------------------------------------------------------------------------- #
def _ensure_repo_on_path():
    """Put the repo root on sys.path so ``from nodes import ...`` resolves when
    this file runs as a standalone script."""
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


def _engine_module(engine):
    """The adapter's defining module object (to read its SIDECAR_ENV / MODEL_ENV
    contract). Returns None if it cannot be resolved."""
    mod_name = getattr(type(engine), "__module__", None)
    if not mod_name:
        return None
    return sys.modules.get(mod_name)


def classify_engine(engine):
    """Classify ONE image engine's integration mode + verify-at-build contract.

    ``isolation`` is ``sidecar`` if the adapter module exposes ``SIDECAR_ENV``
    (runs in a cu128 venv subprocess), ``in_stack`` if it exposes ``MODEL_ENV``
    (runs on the main venv via an installed loader), else ``default`` (the gen-1
    in-stack engine with no opt-in flag, e.g. Flux). The dependency posture follows
    from the mode -- a sidecar engine needs the BEFORE/AFTER import probe on the
    GPU box; an in-stack engine is covered by the cold-import tests.
    """
    mod = _engine_module(engine)
    sidecar_env = getattr(mod, "SIDECAR_ENV", None) if mod else None
    model_env = getattr(mod, "MODEL_ENV", None) if mod else None
    flag = getattr(engine, "requires_flag", None)
    if sidecar_env:
        isolation = "sidecar"
        gate_env = sidecar_env
        posture = ("cu128 sidecar subprocess; run the BEFORE/AFTER import probe in "
                   "the sidecar venv (--python <venv> --lib <pkg>) to prove the "
                   "import does not swap torch or pull xformers/flash_attn/"
                   "sageattention (BUG-070), then the live GPU smoke")
    elif model_env:
        isolation = "in_stack"
        gate_env = model_env
        posture = ("in-stack via an installed loader (GGUF: ComfyUI-GGUF; native: "
                   "ComfyUI loaders); dtype-isolated, no sidecar -- dependency "
                   "posture proven by the per-adapter cold-import tests")
    else:
        isolation = "default"
        gate_env = None
        posture = ("gen-1 in-stack default; no opt-in flag, no sidecar -- the "
                   "shipped loader path owns residency")
    return {
        "engine": getattr(engine, "name", "?"),
        "isolation": isolation,
        "enable_flag": flag,
        "gate_env": gate_env,
        "commercial_clean": bool(getattr(engine, "commercial_clean", False)),
        "verify_at_build": posture,
    }


def enumerate_engines():
    """Classify every registered image engine. Headless-safe: importing the image
    registry is cold-import clean (pulls no model framework). Returns a list of
    classification dicts sorted by engine name; an unimportable registry yields an
    empty list, never an exception."""
    _ensure_repo_on_path()
    try:
        from nodes._otr_image_engines import registry as ireg
        names = ireg.all_engine_names()
        return [classify_engine(ireg.get_engine(n)) for n in names]
    except Exception:  # noqa: BLE001 -- a missing/broken registry is a real finding
        return []


# --------------------------------------------------------------------------- #
# Sidecar import probe (only on the GPU box; --python at the sidecar venv)
# --------------------------------------------------------------------------- #
def probe_sidecar_import(lib_module):
    """Import ``lib_module`` in THIS process, snapshotting torch + banned deps
    BEFORE and AFTER. Headless-safe: a missing library is a clean ``lib_absent``
    verdict, never an exception. Meant to be run inside the cu128 sidecar venv
    (the caller points ``--python`` there)."""
    verdict = {"lib_module": lib_module, "violations": [],
               "banned_deps_clean": True, "import_clean_ready": False}
    before = dep_snapshot()
    try:
        importlib.import_module(lib_module)
    except ImportError:
        verdict["status"] = "lib_absent"
        verdict["note"] = (f"{lib_module} not installed -- install it in the cu128 "
                           f"sidecar venv and re-run with --python at that venv")
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
    verdict["import_clean_ready"] = not violations
    return verdict


def run_pilot(lib_module=None):
    """Enumerate + classify every image engine; optionally run the sidecar import
    probe for ``lib_module``. Never raises."""
    engines = enumerate_engines()
    sidecar = [e for e in engines if e["isolation"] == "sidecar"]
    report = {
        "tool_version": TOOL_VERSION,
        "banned_deps": list(BANNED_DEPS),
        "engines": engines,
        "engine_count": len(engines),
        "sidecar_count": len(sidecar),
        "in_stack_count": sum(1 for e in engines if e["isolation"] == "in_stack"),
    }
    if lib_module:
        report["sidecar_probe"] = probe_sidecar_import(lib_module)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_human(report):
    print(f"OTR image dependency pilot (tool v{report['tool_version']})")
    print(f"banned deps watched: {', '.join(report['banned_deps'])}")
    print("")
    for e in report["engines"]:
        clean = "commercial-clean" if e["commercial_clean"] else "license-restricted"
        print(f"[{e['isolation']}] {e['engine']}  (flag {e['enable_flag']}; {clean})")
        if e["gate_env"]:
            print(f"    fail-closed gate env: {e['gate_env']}")
        print(f"    verify-at-build: {e['verify_at_build']}")
        print("")
    probe = report.get("sidecar_probe")
    if probe is not None:
        print(f"sidecar import probe [{probe.get('status', '?')}] "
              f"{probe['lib_module']}")
        if probe.get("note"):
            print(f"    note: {probe['note']}")
        for v in probe.get("violations") or []:
            print(f"    VIOLATION: {v}")
        print("")
    print(f"engines: {report['engine_count']} "
          f"({report['sidecar_count']} sidecar, {report['in_stack_count']} in-stack)")
    print("next: for each sidecar engine install its lib in the cu128 venv on the "
          "GPU box, run this with --python <venv> --lib <pkg>, then the live "
          "OTR_ENABLE_<engine>=1 smoke. In-stack engines are cold-import-tested.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OTR image engine dependency pilot: the image-sidecar unblock."
    )
    parser.add_argument("--lib", default=None,
                        help="import this module in-process and snapshot torch + "
                             "banned deps before/after (run inside the cu128 sidecar "
                             "venv via --python)")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of a human report")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if a requested sidecar import probe is not clean")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    report = run_pilot(lib_module=args.lib)
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        _print_human(report)
    if args.strict and "sidecar_probe" in report:
        if not report["sidecar_probe"].get("import_clean_ready"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
