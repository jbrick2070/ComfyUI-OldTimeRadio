#!/usr/bin/env python
"""OTR video dependency-pilot harness (A-S4 -- the opt-in-engine dep gate).

The opt-in video engines that run in an ISOLATED cu128 sidecar venv (the
character_3d lanes; LTX / Wan / HuMo as they land) need Blackwell / cu128
dependency validation
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
venv, flip the engine's OTR_ENABLE_* flag, run the live GPU smoke) stays a human
step.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import ast
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

# A custom node that imports any of these at MODULE SCOPE contaminates the
# protected stack at ComfyUI STARTUP -- before the per-engine gate ever runs
# (ComfyUI-KJNodes is the known offender: it can import sageattention at import
# time, globally swapping attention and breaking render-twice determinism,
# BUG-070). The startup scan greps every installed custom node __init__.py for
# these module-scope imports + any ``torch.hub.load`` (a nondeterministic network
# download banned on the determinism path).
STARTUP_CONTAMINANTS = ("sageattention", "xformers", "flash_attn")

TOOL_VERSION = "2"

# The opt-in / sidecar-isolated video engines this pilot verifies. ``assumed_call``
# is the render interface the live GPU wiring must confirm AFTER this pilot proves
# the import is clean -- the single source of truth the worker + adapter point at.
OPT_IN_ENGINES = {
    # LTX-AV (audio-input) lane -- in-process, flag-gated behind OTR_ENABLE_LTX_AV.
    # Rides the same LTX/GGUF wrapper stack as ltx_video (no distinct pip lib);
    # the real probe is "the ComfyUI-GGUF + ComfyUI-LTXVideo node classes resolve"
    # which assert_usable enforces at render. Listed so the flag-gated invariant
    # (every requires_flag engine is covered) holds.
    "ltx_audio_in": {
        "lib_module": "ltx_video",
        "adapter_class": "LtxAudioInEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_LTX_AV",
        "assumed_call": (
            "in-process ComfyUI graph (I2V on the minted still + audio-conditioned "
            "concat -> LTXVSeparateAVLatent -> VAEDecodeTiled); GGUF Q3_K_M, Gemma-3 "
            "on CPU. The UNIFIED agnostic audio-in lane (music OR voice). "
            "# TODO-for-GPU-smoke: confirm the I2V audio-conditioned forward holds "
            "<=14.5GB at the floor across music + announcer + character (M4)"
        ),
    },
    "ltx_video": {
        "lib_module": "ltx_video",
        "adapter_class": "LtxVideoEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_LTX_VIDEO",
        "assumed_call": (
            "in-process: drive the installed LTX-Video ComfyUI wrapper node "
            "classes (MODEL+CLIP+VAE: ltx-video-2b ckpt + gemma text encoder + "
            "LTX 2.3 distilled LoRA @0.7); ltx_video.assert_usable MUST assert "
            "SageAttention is NOT patched (BUG-070) before the first forward.  "
            "# TODO-for-GPU-smoke: confirm the installed wrapper INPUT_TYPES, that "
            "the import does not swap torch or pull xformers/flash_attn/"
            "sageattention, and render-twice determinism on sm_120"
        ),
    },
    # 0-E easy on-ramp: 2.5D depth parallax over existing stills.
    # DepthAnythingV2-SMALL pinned (Apache-2.0; the bigger DA-V2 ckpts are
    # CC-BY-NC and banned). CPU-degradable; in-process transformers.
    "still_parallax": {
        "lib_module": "transformers",
        "lib_in_stack": True,           # spine dep; gate = local DA-V2-S snapshot
        "adapter_class": "StillParallaxEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_STILL_PARALLAX",
        "assumed_call": (
            "in-process: AutoModelForDepthEstimation (DA-V2-SMALL, "
            "local_files_only) -> one depth map per still -> pure-numpy "
            "depth-weighted sway -> encode_frames_to_silent_mp4; CUDA when "
            "present, CPU otherwise (same contract).  "
            "# TODO-for-GPU-smoke: pre-fetch depth-anything/"
            "Depth-Anything-V2-Small-hf into the HF cache, one clip on the "
            "box confirming the parallax reads + render-twice determinism"
        ),
    },
    # 0-E easy on-ramp: the traditional 3D chain. The mesher drives ComfyUI
    # CORE hy3d nodes in-process (the "lib" is the comfy host itself --
    # absent in the pytest sandbox, present in production); Blender is a
    # spawned exe, not an importable lib (OTR_BLENDER_EXE assert_usable
    # gate). Tencent community license -- E-7 record before default-on.
    "mesh_stage": {
        "lib_module": "comfy",
        "adapter_class": "MeshStageEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_MESH_STAGE",
        "assumed_call": (
            "in-process core nodes: CheckpointLoaderSimple(hy3d-dit-v2-mv) + "
            "CLIPVisionLoader/Encode -> Hunyuan3Dv2Conditioning -> "
            "KSampler(EmptyLatentHunyuan3Dv2) -> VAEDecodeHunyuan3D -> "
            "VoxelToMesh -> SaveGLB; then BUG-291 reclaim barrier; then "
            "pinned portable Blender --background --factory-startup "
            "--python-exit-code 1 (WORKBENCH matcap turntable, straight-"
            "alpha PNG dir, atomic publish).  "
            "# TODO-for-GPU-smoke: confirm the installed core hy3d node "
            "INPUT_TYPES + SaveGLB filename pattern, the E-1 VRAM probe on "
            "the 5080 (DRAFT 8000 MB), the cube self-test, and one E-6 "
            "acceptance render per slot at 1472x832"
        ),
    },
    # triposr REMOVED from the probe manifest 2026-06-29 (C3): the dark MIT 3D
    # mesher is unregistered (NotImplementedError render), so there is no adapter
    # to dep-verify. Restore the row WITH the @register when a real forward ships.
    # visualizer (2026-06-18): the low-VRAM ffmpeg-only procedural CRT scope engine.
    # No heavy/contaminating lib -- ffmpeg is a spawned exe, soundfile is the only
    # importable external dep (the floor audio path already uses it), PIL/numpy are
    # core. Flag-gated (OTR_ENABLE_VISUALIZER) so the pilot covers it, but there is
    # nothing GPU/sidecar to probe; it runs in-process on CPU.
    "visualizer": {
        "lib_module": "soundfile",
        "lib_in_stack": True,        # soundfile is a spine dep (present); the real
                                     # gate is ffmpeg + assert_usable, not a lib probe
        "adapter_class": "VisualizerEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_VISUALIZER",
        "assumed_call": (
            "in-process CPU: soundfile-decode the per-beat audio -> mono -> "
            "_analyze_audio_np + _dual_ema -> paint the full-16:9 CRT scope look "
            "(ring / orbiting particles / grid / mirrored waveform / freq bars / "
            "CRT post, COPIED torch-free into _otr_shared/scope_draw.py) -> one "
            "silent mp4 via ffmpeg stdin. Near-zero GPU; no weights, no sidecar.  "
            "# TODO-for-GPU-smoke: a full-episode run with visualizer on all three "
            "roles, audio byte-identical, render time + the near-zero VRAM floor"
        ),
    },
    "wan_i2v": {
        "lib_module": "wan",
        "adapter_class": "WanI2VEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_WAN_I2V",
        "assumed_call": (
            "in-process by default (sidecar_optional -> sidecar when SageAttention "
            "is resident): drive the installed Wan 2.2 i2v wrapper (two-expert "
            "HIGH/LOW MoE + lightx2v/SVI_v2_PRO/distill LoRAs + ModelSamplingSD3 "
            "sigma-split); the init_image is padded/cropped into the canvas, never "
            "stretched.  # TODO-for-GPU-smoke: clear the KJNodes pin audit (gate F: "
            "numpy<2 / transformers<=4.51.3), confirm the wrapper expert-pair / "
            "LoRA-order / sigma-split via assert_usable, and render-twice "
            "determinism on sm_120"
        ),
    },
    # GO_FORWARD 4A (2026-06-14): the 8GB-tier Wan2.2 TI2V-5B sibling. Same
    # in-process / sidecar_optional model, but the 5B graph (Wan22ImageToVideoLatent
    # + the Wan2.2 VAE, M8) and a GGUF UNET loader. Its 5B core node class was
    # captured from a live /object_info before the engine was coded.
    "wan_ti2v": {
        "lib_module": "wan",
        "adapter_class": "WanTi2vEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_WAN_TI2V",
        "assumed_call": (
            "in-process by default (sidecar_optional -> sidecar when SageAttention "
            "is resident): drive the installed Wan 2.2 TI2V-5B graph "
            "(UnetLoaderGGUF Q5_K_M + ModelSamplingSD3 shift 5.0 + "
            "Wan22ImageToVideoLatent + the Wan2.2 VAE); the init_image is "
            "padded/cropped into the canvas, never stretched.  "
            "# TODO-for-GPU-smoke: confirm the 5B latent-node INPUT_TYPES, that the "
            "Wan2.2 VAE decode fits the medium VRAM tier, and render-twice "
            "determinism on sm_120"
        ),
    },
    "humo": {
        "lib_module": "humo",
        "adapter_class": "HuMoEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_HUMO",
        "assumed_call": (
            "in-process: drive the installed HuMo ComfyUI wrapper node classes "
            "(loads MODEL+CLIP+VAE+AUDIO_ENCODER internally via "
            "comfy.model_management: umt5_xxl CLIP + wan_2.1 VAE + "
            "whisper_large_v3 audio encoder + the HuMo diffusion model / LoRA "
            "tier); audio-conditioned image-to-video, native 480x832 @25fps; the "
            "portrait init is padded/cropped into the canvas, never stretched.  "
            "# TODO-for-GPU-smoke: confirm the installed wrapper INPUT_TYPES + the "
            "MODEL/CLIP/VAE/AUDIO_ENCODER handles, that the import does not swap "
            "torch or pull xformers/flash_attn/sageattention, the VRAM<=14.5 GB "
            "peak, and render-twice determinism on sm_120"
        ),
    },
    # The 1.7B HuMo downgrade tier (OOM/VRAM hard auto-downgrade target before
    # the still floor). Same in-process HuMo wrapper + flag as the 14B keystone;
    # its own UNET, NO LoRA, more steps -- a lighter real talking face.
    "humo_1.7B": {
        "lib_module": "humo",
        "adapter_class": "HuMo17BEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_HUMO",
        "assumed_call": (
            "in-process: same HuMo ComfyUI wrapper node classes as the 14B tier "
            "but the 1.7B UNET, LoRA-free, ~20 steps / cfg 1.0 (de-blued from 5.0 "
            "which cast blue, 2026-06-17; own "
            "OTR_HUMO_17B_* knobs). The OOM/VRAM hard auto-downgrade lands here "
            "before the still floor so a heavy episode keeps a real face.  "
            "# TODO-for-GPU-smoke: confirm the 1.7B weight loads, fits a tighter "
            "VRAM budget than the 14B, does not swap torch or pull "
            "xformers/flash_attn/sageattention, and render-twice determinism"
        ),
    },
    "humo_1.7B_169": {
        "lib_module": "humo",
        "adapter_class": "HuMo17BLandscapeEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_HUMO",
        "assumed_call": (
            "in-process: the humo_1.7B tier rendered at 16:9 LANDSCAPE 832x480 "
            "(render_aspect='wide') instead of the 480x832 portrait, cfg 2.5 (the "
            "16:9 sweet spot measured 2026-06-16; own OTR_HUMO_17B_169_CFG knob). "
            "A deliberate operator dropdown pick alongside portrait humo_1.7B, NOT "
            "in the auto-downgrade chain.  # TODO-for-GPU-smoke: confirm the 16:9 "
            "forward holds the same VRAM budget as portrait 1.7B and render-twice "
            "determinism"
        ),
    },
    "humo_14B_169": {
        "lib_module": "humo",
        "adapter_class": "HuMo14BLandscapeEngine",
        "forward": "render_clip",
        "flag": "OTR_ENABLE_HUMO",
        "assumed_call": (
            "in-process: the 14B keystone rendered at 16:9 LANDSCAPE 832x480 "
            "(render_aspect='wide') instead of the 480x832 portrait, same 14B ckpt "
            "+ lightx2v distill + 6 steps / cfg 1.0. The 14B latents match "
            "wan_2.1_vae so it is colour-correct (no 1.7B blue cast) -- the 06-09 "
            "quality in the 16:9 look. A deliberate operator dropdown pick, NOT in "
            "the auto-downgrade chain.  # TODO-for-GPU-smoke: confirm the 16:9 "
            "forward fits the 14.5 GB ceiling and render-twice determinism"
        ),
    },
    # triposg_talk / hunyuan3d_talk / trellis_talk REMOVED from the probe manifest
    # 2026-06-29 (C3): the dark character_3d scaffolds are unregistered
    # (NotImplementedError render), so there is no adapter to dep-verify. Restore
    # the rows WITH the @register + package import when a real forward ships.
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


# --------------------------------------------------------------------------- #
# Custom-node startup-contamination scan (KJNodes gate F + torch.hub ban)
# --------------------------------------------------------------------------- #
def custom_nodes_dir():
    """The ComfyUI ``custom_nodes`` directory (this repo's parent). The scan is
    headless-safe when it is absent."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _iter_node_init_files(nodes_dir):
    """Yield ``(node_name, init_py_path)`` for each installed custom node with a
    top-level ``__init__.py``. Headless-safe: a missing dir yields nothing."""
    try:
        entries = sorted(os.listdir(nodes_dir))
    except OSError:
        return
    for name in entries:
        node_dir = os.path.join(nodes_dir, name)
        if not os.path.isdir(node_dir):
            continue
        init_py = os.path.join(node_dir, "__init__.py")
        if os.path.isfile(init_py):
            yield name, init_py


def _is_torch_hub_load(func):
    """True iff an AST call target is the attribute chain ``torch.hub.load``."""
    return (
        isinstance(func, ast.Attribute) and func.attr == "load"
        and isinstance(func.value, ast.Attribute) and func.value.attr == "hub"
        and isinstance(func.value.value, ast.Name) and func.value.value.id == "torch"
    )


def scan_source_for_contaminants(src):
    """Pure: parse one module source; return a finding dict.

    ``module_scope_imports`` = sorted ``STARTUP_CONTAMINANTS`` imported at TOP
    LEVEL (module scope -- these fire at ComfyUI startup, before any gate);
    ``torch_hub_load`` = True if a ``torch.hub.load(...)`` call appears anywhere.
    A source that will not parse is recorded (``parse_error``), never raised.
    """
    finding = {"module_scope_imports": [], "torch_hub_load": False,
               "parse_error": False}
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        finding["parse_error"] = True
        return finding
    banned = set()
    for node in tree.body:                       # module scope == top-level body
        if isinstance(node, ast.Import):
            for imp in node.names:                   # ast.alias node; name 'imp'
                root = (imp.name or "").split(".")[0]
                if root in STARTUP_CONTAMINANTS:
                    banned.add(root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in STARTUP_CONTAMINANTS:
                banned.add(root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_torch_hub_load(node.func):
            finding["torch_hub_load"] = True
            break
    finding["module_scope_imports"] = sorted(banned)
    return finding


def scan_custom_nodes(nodes_dir=None):
    """Scan every installed custom node ``__init__.py`` for startup contaminants
    + ``torch.hub.load``. Headless-safe: a missing dir -> an empty, clean report;
    an unreadable / unparseable file is skipped, never raised. Returns a report
    dict with the FLAGGED nodes + a ``startup_audit_clean`` bool."""
    d = nodes_dir if nodes_dir is not None else custom_nodes_dir()
    flagged = []
    for name, init_py in _iter_node_init_files(d):
        try:
            with open(init_py, "r", encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError:
            continue
        f = scan_source_for_contaminants(src)
        if f["module_scope_imports"] or f["torch_hub_load"]:
            flagged.append({
                "node": name, "init": init_py,
                "module_scope_imports": f["module_scope_imports"],
                "torch_hub_load": f["torch_hub_load"],
            })
    return {
        "custom_nodes_dir": d,
        "scanned_dir_present": bool(d) and os.path.isdir(d),
        "startup_contaminants": list(STARTUP_CONTAMINANTS),
        "flagged_nodes": flagged,
        "startup_audit_clean": len(flagged) == 0,
    }


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

    if spec.get("lib_in_stack"):
        # The engine's libraries are SPINE dependencies (already resident in
        # the main venv -- e.g. transformers for still_parallax). Re-importing
        # the spine inside the pilot proves nothing and would smear the dep
        # snapshot for the later probes; the engine's REAL gate is its local
        # model snapshot, enforced fail-closed by assert_usable, never a lib
        # import. import_clean_ready stays False (readiness is a GPU/asset
        # call, same as every other row).
        verdict["status"] = "lib_in_stack"
        verdict["note"] = (
            f"{spec['lib_module']} is a spine dependency (already in the "
            f"main venv); the gating asset is the LOCAL model snapshot "
            f"checked by assert_usable -- nothing to import-probe here"
        )
        return verdict

    before = dep_snapshot()
    try:
        importlib.import_module(spec["lib_module"])
    except ImportError:
        verdict["status"] = "lib_absent"
        verdict["note"] = (
            f"{spec['lib_module']} not installed -- install it in this engine's "
            f"isolated venv and re-run on the GPU box to verify the import + render"
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


def run_pilot(engines=None, *, isolated=True, python=None, scan_nodes=True):
    """Probe each engine; return the aggregate report. Never raises.

    When ``scan_nodes`` (default), also runs the custom-node startup-contamination
    audit (KJNodes gate F + torch.hub ban) and reports it under
    ``custom_node_scan`` / ``startup_audit_clean``.
    """
    names = list(engines) if engines else list(OPT_IN_ENGINES)
    verdicts = []
    for name in names:
        if isolated:
            verdicts.append(probe_isolated(name, python=python))
        else:
            verdicts.append(probe_one(name, do_import=True))
    ready = [v for v in verdicts if v.get("import_clean_ready")]
    scan = scan_custom_nodes() if scan_nodes else None
    return {
        "tool_version": TOOL_VERSION,
        "banned_deps": list(BANNED_DEPS),
        "engines": verdicts,
        "ready_count": len(ready),
        "engine_count": len(verdicts),
        "all_imports_clean": bool(verdicts) and len(ready) == len(verdicts),
        "custom_node_scan": scan,
        "startup_audit_clean": (scan["startup_audit_clean"] if scan else None),
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
    scan = report.get("custom_node_scan")
    if scan is not None:
        print("custom-node startup audit (KJNodes gate F + torch.hub ban):")
        if not scan["scanned_dir_present"]:
            print(f"    custom_nodes dir not found: {scan['custom_nodes_dir']}")
        elif scan["startup_audit_clean"]:
            print("    clean -- no module-scope "
                  f"{'/'.join(scan['startup_contaminants'])} or torch.hub.load")
        else:
            for n in scan["flagged_nodes"]:
                bits = []
                if n["module_scope_imports"]:
                    bits.append("module-scope import: "
                                + ", ".join(n["module_scope_imports"]))
                if n["torch_hub_load"]:
                    bits.append("torch.hub.load")
                print(f"    FLAGGED {n['node']} -- {'; '.join(bits)}")
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
    parser.add_argument("--scan-custom-nodes", action="store_true",
                        help="scan every installed custom node __init__.py for "
                             "module-scope sage/xformers/flash_attn + torch.hub.load "
                             "(the KJNodes startup-contamination gate) and print it")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.probe_one:
        print(json.dumps(probe_one(args.probe_one, do_import=True), sort_keys=True))
        return 0

    if args.scan_custom_nodes:
        scan = scan_custom_nodes()
        print(json.dumps(scan, sort_keys=True, indent=2))
        if args.strict and not scan["startup_audit_clean"]:
            return 1
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
