"""
_gsplat_spike.py  --  2-hour gsplat install spike on Blackwell/torch 2.10/py3.12
=================================================================================

Purpose
-------
Answer ONE question: does ``pip install gsplat`` succeed on Jeffrey's
Windows + RTX 5080 Laptop + Blackwell (sm_120) + Python 3.12 + torch 2.10
+ CUDA 13.0 platform, without corrupting the main ComfyUI venv?

Branch: ``r-and-d/gsplat-spike`` ONLY.  Never merge to v2.0-alpha or main.

Operating modes (select via ``--phase``)
----------------------------------------
  --phase probe         (default, SAFE)
      Read-only environment audit.  No pip, no network writes.
      Prints torch version, CUDA build, sm_arch list, python version.

  --phase dryrun
      ``pip install --dry-run gsplat``.  Queries PyPI metadata, builds
      the dep graph, but does NOT touch site-packages.  Reveals whether
      a pre-built wheel exists for the current (python, platform, CUDA)
      triple or whether pip would fall back to a source build (which is
      the real CUDA-compile risk).

  --phase install
      DESTRUCTIVE.  Runs ``pip install gsplat`` for real.  Writes a
      pip freeze snapshot to the consultation folder BEFORE installing
      so the user can reconstruct the venv if the install breaks it.
      This mode is only invoked after the user green-lights it in chat.

  --phase smoketest
      Imports gsplat, constructs 1 000 random Gaussians, renders a
      single 256x256 frame on the default CUDA device, and reports
      (a) did it import, (b) did the CUDA rasterizer build work,
      (c) did the render return a non-NaN tensor.  Requires ``--phase
      install`` to have succeeded first.

  --phase rollback
      ``pip uninstall -y gsplat``.  Best-effort cleanup if the spike
      leaves the venv unhappy.

Output
------
All phases write their stdout/stderr and a structured JSON summary to
``docs/superpowers/consultations/2026-04-16-phase-c-splat-stack/``:

  04_gsplat_spike_probe.json
  04_gsplat_spike_dryrun.txt
  04_gsplat_spike_install.txt
  04_gsplat_spike_smoketest.json
  04_gsplat_spike_rollback.txt

The write-up (``04_gsplat_spike.md``) is authored from these artifacts
after the spike completes.

Safety guarantees
-----------------
* ``--phase probe`` and ``--phase dryrun`` are 100% read-only wrt the
  main ComfyUI venv.  Safe to run unattended.
* ``--phase install`` refuses to run without the env var
  ``OTR_GSPLAT_SPIKE_GO=1`` -- the caller must explicitly opt in.
  This matches the ``OTR_HYWORLD_ANCHOR`` gate already used in worker.py.
* A pip freeze snapshot is always captured before ``install`` so the
  venv state can be reconstructed.
* Hard 2-hour wall-clock cap (``_TIME_BUDGET_SEC``) across all phases
  from the first call.  Tracked via a stamp file in the consult dir.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 stdout so ffmpeg / pip output doesn't crash on Windows
# cp1252 when it hits a non-ASCII byte.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
_CONSULT = _REPO / "docs" / "superpowers" / "consultations" / "2026-04-16-phase-c-splat-stack"
_STAMP = _CONSULT / ".spike_started_at"

_TIME_BUDGET_SEC = 2 * 60 * 60  # 2 hours


# ---------------------------------------------------------------------------
# Time budget enforcement
# ---------------------------------------------------------------------------

def _ensure_time_budget() -> float:
    """Create the stamp file on first call; return elapsed seconds so far.

    Aborts with non-zero exit code if the 2-hour budget is exhausted.
    Jeffrey gets auditable confirmation that we respected the cap.
    """
    _CONSULT.mkdir(parents=True, exist_ok=True)
    if _STAMP.exists():
        started = float(_STAMP.read_text(encoding="utf-8").strip())
    else:
        started = time.time()
        _STAMP.write_text(f"{started}", encoding="utf-8")
    elapsed = time.time() - started
    if elapsed > _TIME_BUDGET_SEC:
        print(
            f"[budget] 2-hour cap exceeded ({elapsed:.0f}s). Aborting. "
            f"Delete {_STAMP} to restart the cap if Jeffrey chooses to continue."
        )
        sys.exit(2)
    return elapsed


# ---------------------------------------------------------------------------
# Phase: probe (read-only env audit)
# ---------------------------------------------------------------------------

def phase_probe() -> int:
    out: dict = {
        "phase": "probe",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
    }

    # torch + CUDA detection (runs even without torch installed).
    try:
        import torch  # type: ignore
        out["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            devs = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devs.append({
                    "index": i,
                    "name": props.name,
                    "capability": f"sm_{props.major}{props.minor}",
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                })
            out["torch"]["devices"] = devs
        # gsplat notes: as of 2026, gsplat ships pre-built wheels keyed
        # off torch version + CUDA version + python version.  Mismatches
        # force a source build.  Source builds use nvcc to compile
        # CUDA kernels, which MUST target the device's sm_ architecture.
        # Blackwell = sm_120 = very new; wheel support varies.
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        out["torch"]["TORCH_CUDA_ARCH_LIST"] = arch_list
    except ImportError as exc:
        out["torch"] = {"import_error": str(exc)}

    # Check if gsplat is already installed
    try:
        import gsplat  # type: ignore
        out["gsplat"] = {
            "already_installed": True,
            "version": getattr(gsplat, "__version__", "unknown"),
            "file": gsplat.__file__,
        }
    except ImportError:
        out["gsplat"] = {"already_installed": False}

    # nvcc availability (required for source build)
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        out["nvcc"] = {
            "rc": result.returncode,
            "stdout_tail": (result.stdout or "")[-300:],
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        out["nvcc"] = {"error": f"{type(exc).__name__}: {exc}"}

    # pip version
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        out["pip"] = {
            "rc": result.returncode,
            "stdout": (result.stdout or "").strip(),
        }
    except Exception as exc:  # noqa: BLE001
        out["pip"] = {"error": f"{type(exc).__name__}: {exc}"}

    out_path = _CONSULT / "04_gsplat_spike_probe.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\n[probe] wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Phase: dryrun (pip install --dry-run, no site-packages mutation)
# ---------------------------------------------------------------------------

def phase_dryrun() -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "-v", "gsplat"]
    print(f"[dryrun] $ {' '.join(cmd)}")
    out_path = _CONSULT / "04_gsplat_spike_dryrun.txt"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        out_path.write_text("TIMEOUT after 900s", encoding="utf-8")
        print("[dryrun] TIMEOUT")
        return 1

    body = (
        f"$ {' '.join(cmd)}\n"
        f"RC: {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    print(body[-2000:])
    print(f"\n[dryrun] wrote {out_path}")
    return 0 if result.returncode == 0 else 1


# ---------------------------------------------------------------------------
# Phase: install (DESTRUCTIVE -- gated by OTR_GSPLAT_SPIKE_GO)
# ---------------------------------------------------------------------------

def phase_install() -> int:
    if os.environ.get("OTR_GSPLAT_SPIKE_GO") != "1":
        print(
            "[install] BLOCKED: set OTR_GSPLAT_SPIKE_GO=1 to enable real install.\n"
            "         (PowerShell: $env:OTR_GSPLAT_SPIKE_GO='1')\n"
            "         This is a destructive operation; read 04_gsplat_spike_dryrun.txt first."
        )
        return 3

    # Freeze snapshot BEFORE we touch anything.
    freeze_path = _CONSULT / "04_gsplat_spike_freeze_before.txt"
    try:
        fr = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60,
        )
        freeze_path.write_text(fr.stdout, encoding="utf-8")
        print(f"[install] pip freeze snapshot -> {freeze_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[install] freeze snapshot failed: {exc}")
        # Do not continue if we can't make the safety snapshot.
        return 4

    cmd = [sys.executable, "-m", "pip", "install", "-v", "gsplat"]
    print(f"[install] $ {' '.join(cmd)}")
    out_path = _CONSULT / "04_gsplat_spike_install.txt"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        out_path.write_text("TIMEOUT after 3600s", encoding="utf-8")
        print("[install] TIMEOUT")
        return 5

    body = (
        f"$ {' '.join(cmd)}\n"
        f"RC: {result.returncode}\n"
        f"--- STDOUT (tail 4k) ---\n{result.stdout[-4000:]}\n"
        f"--- STDERR (tail 4k) ---\n{result.stderr[-4000:]}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    print(body[-2000:])
    print(f"\n[install] wrote {out_path}")
    return 0 if result.returncode == 0 else 1


# ---------------------------------------------------------------------------
# Phase: smoketest (import + tiny render)
# ---------------------------------------------------------------------------

def phase_smoketest() -> int:
    out: dict = {"phase": "smoketest", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out_path = _CONSULT / "04_gsplat_spike_smoketest.json"

    # Import
    try:
        t0 = time.time()
        import gsplat  # type: ignore
        out["import"] = {
            "ok": True,
            "elapsed_sec": round(time.time() - t0, 3),
            "version": getattr(gsplat, "__version__", "unknown"),
        }
    except Exception as exc:  # noqa: BLE001
        out["import"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 1

    # Tiny render.  This exercises the CUDA rasterizer that gsplat
    # builds at install time.  If this returns a sane tensor, the
    # Blackwell compile path works end-to-end.
    try:
        import torch  # type: ignore
        from gsplat import rasterization  # type: ignore

        if not torch.cuda.is_available():
            out["render"] = {"ok": False, "reason": "CUDA unavailable at runtime"}
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(json.dumps(out, indent=2))
            return 2

        device = "cuda"
        N = 1000
        torch.manual_seed(0)
        means = torch.randn(N, 3, device=device)
        quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
        scales = torch.ones(N, 3, device=device) * 0.05
        opacities = torch.ones(N, device=device) * 0.5
        colors = torch.rand(N, 3, device=device)

        W, H = 256, 256
        # Minimal view: identity rot, camera at z=5 looking at origin.
        K = torch.tensor(
            [[200.0, 0.0, W / 2], [0.0, 200.0, H / 2], [0.0, 0.0, 1.0]],
            device=device,
        ).unsqueeze(0)
        viewmat = torch.eye(4, device=device).unsqueeze(0)
        viewmat[0, 2, 3] = 5.0

        t0 = time.time()
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
        )
        elapsed = time.time() - t0
        out["render"] = {
            "ok": True,
            "elapsed_sec": round(elapsed, 3),
            "shape": list(renders.shape),
            "dtype": str(renders.dtype),
            "has_nan": bool(torch.isnan(renders).any().item()),
            "min": round(float(renders.min()), 4),
            "max": round(float(renders.max()), 4),
        }
    except Exception as exc:  # noqa: BLE001
        out["render"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\n[smoketest] wrote {out_path}")
    return 0 if out.get("render", {}).get("ok") else 1


# ---------------------------------------------------------------------------
# Phase: rollback (pip uninstall -y gsplat)
# ---------------------------------------------------------------------------

def phase_rollback() -> int:
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "gsplat"]
    print(f"[rollback] $ {' '.join(cmd)}")
    out_path = _CONSULT / "04_gsplat_spike_rollback.txt"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        out_path.write_text("TIMEOUT after 120s", encoding="utf-8")
        return 1
    body = (
        f"$ {' '.join(cmd)}\n"
        f"RC: {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    print(body)
    return 0 if result.returncode == 0 else 1


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="gsplat install spike")
    parser.add_argument(
        "--phase",
        choices=["probe", "dryrun", "install", "smoketest", "rollback"],
        default="probe",
        help="Spike phase (see module docstring).",
    )
    args = parser.parse_args()

    elapsed = _ensure_time_budget()
    print(f"[budget] elapsed {elapsed:.0f}s / {_TIME_BUDGET_SEC}s")

    dispatch = {
        "probe": phase_probe,
        "dryrun": phase_dryrun,
        "install": phase_install,
        "smoketest": phase_smoketest,
        "rollback": phase_rollback,
    }
    return dispatch[args.phase]()


if __name__ == "__main__":
    sys.exit(main())
