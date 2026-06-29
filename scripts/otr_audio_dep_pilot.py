#!/usr/bin/env python
"""OTR audio dependency-pilot harness (plan Wave 3 / F -- the G1 unblock).

The dep-pilot audio engines (chatterbox / indextts2 / stable_audio_music) need
Blackwell / cu130 dependency validation before G1 wires their real library
forward calls: chatterbox + stable_audio_music are flag-gated opt-ins; indextts2
was PROMOTED 2026-06-04 to the shipped char_voice default but runs in an isolated
oop_venv Path B worker, so it stays pilot-gated until this harness verifies, on
the operator's GPU box, that importing each library is SAFE and that its forward
can bind an external ``torch.Generator`` (the bit_exact precondition -- base.py
``supports_external_generator``).

This module is the harness the operator runs to do that verification. It is
deliberately HEADLESS-SAFE: with the engine libraries absent (the box-fresh
state) every probe reports ``lib_absent`` without crashing and without pulling
any banned dependency. With a library installed (the operator GPU run) it:

  * snapshots ``torch.__version__`` + the presence of every BANNED_DEPS entry
    BEFORE and AFTER importing the engine library, and FAILS the engine if the
    import changed torch or dragged in xformers / flash_attn (plan F: "hard-fail
    on xformers/flash_attn/torch-change");
  * records the ASSUMED library call signature (``assumed_call``) that G1 must
    wire -- the single source of truth the adapter TODO-for-F comments point at;
  * introspects whether the adapter forward can bind an external generator.

Isolation: each engine is probed in its OWN subprocess (``--probe-one``), so a
polluted import in one never contaminates the parent or another engine. Point
``--python`` at a per-engine venv to get the plan's "isolated venv import per
opt-in lib". Network is forced OFFLINE (no weight fetch during a probe).

This is a diagnostic tool, never an import-time side effect. Run it; read the
verdict. It writes blind nothing -- promotion (flip supports_external_generator,
wire G1) stays a human step after the verdict is read. UTF-8, no BOM, ASCII.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys

# Dependencies an engine import must NOT pull onto the Blackwell / cu130 stack
# (plan F): they break byte-identical determinism and are brittle on sm_120.
# An engine whose import makes either appear is disqualified from bit_exact.
BANNED_DEPS = ("xformers", "flash_attn")

TOOL_VERSION = "1"

# The opt-in engines this pilot verifies. ``assumed_call`` is the library call
# G1 must wire AFTER this pilot confirms the signature on GPU -- the single
# source of truth the adapter TODO-for-F comments point back to. Keeping it here
# (data, not prose) lets the harness diff the assumption against reality.
PROBE_ENGINES = {
    "chatterbox": {
        "lib_module": "chatterbox",
        "adapter_class": "ChatterboxEngine",
        "forward": "generate_voice",
        "assumed_call": (
            "ChatterboxTTS.generate(text, audio_prompt_path=<ref>, "
            "exaggeration=<float>, cfg=<float>, temperature=<float>, "
            "generator=<torch.Generator>)  # TODO-for-F: confirm generate() "
            "binds an external generator; else disqualified from bit_exact"
        ),
    },
    "indextts2": {
        "lib_module": "indextts",
        "adapter_class": "IndexTTS2Engine",
        "forward": "generate_voice",
        # PROMOTED 2026-06-04: indextts2 is the shipped char_voice default (no
        # flag); it stays dep-pilot-gated via its isolated oop_venv Path B worker
        # until F validates the Blackwell deps.
        "assumed_call": (
            "IndexTTS2(cfg, model_dir).infer(spk_audio_prompt=<ref>, "
            "text=<str>, emo_vector=<list>, seed=<int>, "
            "generator=<torch.Generator>)  # TODO-for-F: confirm the "
            "constructor + infer kwargs + generator support"
        ),
    },
    "dia": {
        "lib_module": "dia",
        "adapter_class": "DiaEngine",
        "forward": "generate_voice",
        "assumed_call": (
            "Dia.from_pretrained(<id>, compute_dtype=<str>).generate("
            "'[S1] <text>', audio_prompt=<ref>)  # TODO-for-F: confirm the "
            "generate() signature + that audio_prompt-only clone quality is "
            "acceptable; no external generator (seed via global RNG)"
        ),
    },
    "stable_audio_music": {
        "lib_module": "stable_audio_tools",
        "adapter_class": "StableAudioMusicEngine",
        "forward": "generate_clip",
        "assumed_call": (
            "ComfyUI-native SA3 sampler / generate_diffusion_cond(model, "
            "steps=<int>, conditioning=[{prompt, seconds_total}], seed=<int>, "
            "generator=<torch.Generator>)  # TODO-for-F: confirm the native "
            "SA3 path binds a torch.Generator for byte-identical render-twice"
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

    ``torch`` is imported lazily (it is usually already resident); the banned
    deps are probed with ``find_spec`` so the snapshot never imports them.
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
    """Plan-F violations between two ``dep_snapshot`` results.

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


def forward_accepts_generator(fn):
    """Tri-state: does ``fn`` accept an external ``generator``?

    ``True`` if it has a parameter named ``generator`` or a ``**kwargs`` catch;
    ``False`` if introspectable but neither is present; ``None`` if the
    signature cannot be read (a C/builtin callable or a non-callable).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    for p in sig.parameters.values():
        if p.name == "generator":
            return True
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _adapter_forward(engine_name):
    """Bound adapter forward for ``engine_name`` from the registry, or ``None``.

    Importing the registry pulls NO engine library (adapters lazy-import inside
    load/generate), so this is safe headless.
    """
    spec = PROBE_ENGINES[engine_name]
    _ensure_repo_on_path()
    try:
        from nodes import _otr_audio_engines as AE
        adapter = AE.get_engine(engine_name)
    except Exception:  # noqa: BLE001 -- a missing registry is a real finding
        return None
    return getattr(adapter, spec["forward"], None)


# --------------------------------------------------------------------------- #
# Probing (per-engine; subprocess-isolated by default)
# --------------------------------------------------------------------------- #
def probe_one(engine_name, *, do_import=True):
    """Probe ONE engine in THIS process. Headless-safe: a missing library is a
    clean ``lib_absent`` verdict, never an exception. Returns a verdict dict."""
    if engine_name not in PROBE_ENGINES:
        return {"engine": engine_name, "status": "unknown_engine"}
    spec = PROBE_ENGINES[engine_name]
    fwd = _adapter_forward(engine_name)
    binds_gen = forward_accepts_generator(fwd) if fwd is not None else None
    verdict = {
        "engine": engine_name,
        "lib_module": spec["lib_module"],
        "adapter_class": spec["adapter_class"],
        "forward": spec["forward"],        "assumed_call": spec["assumed_call"],
        "adapter_registered": fwd is not None,
        "adapter_forward_binds_generator": binds_gen,
        "banned_deps_clean": True,
        "violations": [],
        "supports_external_generator_ready": False,
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
            f"{spec['lib_module']} not installed -- install it in this engine's "
            f"isolated venv and re-run on the GPU box to verify the forward"
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
    # Structural precondition only -- the real library-call verification against
    # ``assumed_call`` is still a human step on the GPU box before promotion.
    verdict["supports_external_generator_ready"] = bool(binds_gen and not violations)
    return verdict


def probe_isolated(engine_name, *, python=None, timeout=600):
    """Probe ONE engine in a FRESH subprocess so a polluted import cannot
    contaminate the parent. ``python`` may point at a per-engine venv."""
    py = python or sys.executable
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["OTR_AUDIO_DEP_PILOT_CHILD"] = "1"
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
    names = list(engines) if engines else list(PROBE_ENGINES)
    verdicts = []
    for name in names:
        if isolated:
            verdicts.append(probe_isolated(name, python=python))
        else:
            verdicts.append(probe_one(name, do_import=True))
    ready = [v for v in verdicts if v.get("supports_external_generator_ready")]
    return {
        "tool_version": TOOL_VERSION,
        "banned_deps": list(BANNED_DEPS),
        "engines": verdicts,
        "ready_count": len(ready),
        "engine_count": len(verdicts),
        "all_structural_preconditions_met": bool(verdicts) and len(ready) == len(verdicts),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_human(report):
    print(f"OTR audio dependency pilot (tool v{report['tool_version']})")
    print(f"banned deps watched: {', '.join(report['banned_deps'])}")
    print("")
    for v in report["engines"]:
        print(f"[{v.get('status', '?')}] {v['engine']}  (flag {v.get('flag', '?')})")
        if v.get("note"):
            print(f"    note: {v['note']}")
        print(
            f"    adapter registered: {v.get('adapter_registered')}  "
            f"forward binds generator: {v.get('adapter_forward_binds_generator')}"
        )
        for viol in v.get("violations") or []:
            print(f"    VIOLATION: {viol}")
        print(f"    assumed call (TODO-for-F): {v.get('assumed_call')}")
        print("")
    print(
        f"structural-ready engines: {report['ready_count']}/{report['engine_count']}"
    )
    print(
        "next: install each lib in its isolated venv on the GPU box, verify the "
        "assumed_call by hand, flip supports_external_generator=True, wire G1."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OTR audio engine dependency pilot (plan F): the G1 unblock."
    )
    parser.add_argument(
        "--probe-one", metavar="ENGINE", default=None,
        help="probe a single engine in THIS process and print its JSON verdict",
    )
    parser.add_argument(
        "--engines", default=None,
        help="comma-separated subset of engines to probe",
    )
    parser.add_argument(
        "--no-isolation", action="store_true",
        help="probe in-process instead of per-engine subprocesses",
    )
    parser.add_argument(
        "--python", default=None,
        help="python executable for isolated probes (point at a per-engine venv)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a human report",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 unless every engine is structurally ready",
    )
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
    if args.strict and not report["all_structural_preconditions_met"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
