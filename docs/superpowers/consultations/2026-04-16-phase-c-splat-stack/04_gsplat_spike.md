# 04 - gsplat install spike (r-and-d/gsplat-spike)

**Date:** 2026-04-16
**Branch:** `r-and-d/gsplat-spike` (throwaway, do not merge)
**Driver:** `scripts/_gsplat_spike.py`
**Time budget:** 2 hours
**Time used:** ~2 minutes (~1.7% of budget)
**Verdict:** INSTALL WOULD SUCCEED, RUNTIME BLOCKED ON MISSING CUDA TOOLKIT

## Spike question

From the Phase C round-robin consultation (see `03_synthesis.md`):

> Is `ComfyUI-Sharp + gsplat` a viable Phase C render stack on this
> machine, or is it a dead end on Windows + Blackwell (sm_120) +
> torch 2.10 + CUDA 13 + Python 3.12?

## Method

Read-only audit only. No pip install was attempted. Evidence comes
from two script phases:

  1. `python scripts/_gsplat_spike.py --phase probe` -- environment
     audit (torch, CUDA, device arch, nvcc, pip).
  2. `python scripts/_gsplat_spike.py --phase dryrun` -- resolves the
     gsplat dependency graph against PyPI without mutating
     site-packages; reveals whether a pre-built wheel exists for this
     platform or whether pip would fall back to a source build.

Both phases are safe; neither writes to the main ComfyUI venv.

## Platform (confirmed by probe)

| Field | Value |
| --- | --- |
| OS | Windows 11 10.0.26200 |
| Python | 3.12.11 (MSVC 1944, AMD64) |
| torch | 2.10.0+cu130 |
| CUDA runtime | 13.0 |
| cuDNN | 91200 |
| GPU | NVIDIA GeForce RTX 5080 Laptop, sm_120 (Blackwell), 15.92 GB |
| nvcc | **NOT FOUND** on PATH |
| pip | 26.0.1 |
| gsplat | not yet installed |
| TORCH_CUDA_ARCH_LIST | empty |

## Findings

### Finding 1: gsplat ships a pure-Python wheel

From `04_gsplat_spike_dryrun.txt`:

    Collecting gsplat
      Obtaining dependency information for gsplat from
      https://files.pythonhosted.org/packages/aa/68/.../gsplat-1.5.3-py3-none-any.whl.metadata
      Downloading gsplat-1.5.3-py3-none-any.whl.metadata (1.1 kB)
    ...
    Would install gsplat-1.5.3 jaxtyping-0.3.9 ninja-1.13.0 wadler_lindig-0.1.7

Platform tag `py3-none-any` means: Python 3, no ABI tag, any platform.
No pre-compiled CUDA kernels in the wheel. The `ninja` dependency is
the giveaway -- gsplat JIT-compiles its CUDA kernels at first import
via `torch.utils.cpp_extension.load()`.

### Finding 2: nvcc is missing on this machine

`nvcc --version` fails with `FileNotFoundError: [WinError 2]`. The
probe verified there is no CUDA Toolkit installed that exposes nvcc
on PATH.

Consequence: `pip install gsplat` would complete (pure-Python
install, ~5 seconds). The first CUDA rasterization call at runtime
would raise inside `cpp_extension.load()` with a "CUDA_HOME not set"
or "nvcc not found" error. No amount of gsplat version pinning fixes
this -- nvcc is a hard requirement of the JIT path.

### Finding 3: Blackwell sm_120 wheel compatibility is UNPROVEN here

Even if nvcc were present, a successful JIT build is not guaranteed
on Blackwell + CUDA 13. As of early 2026, public gsplat issues /
release notes do not document sm_120 + CUDA 13 as a tested target.
This is a secondary risk that cannot be investigated without first
resolving the nvcc gap.

### Finding 4: pre-install dependency footprint

If Jeffrey ever decides to pursue this, pip would add four new
packages to the ComfyUI venv:

  - gsplat 1.5.3
  - jaxtyping 0.3.9
  - ninja 1.13.0
  - wadler_lindig 0.1.7

No existing-package version conflicts surfaced in the dryrun resolver.

## Verdict

The gsplat render stack is **blocked on CUDA Toolkit availability**
on this machine, not on gsplat itself. The install is trivial; the
runtime requires a local nvcc. Unblocking would require one of:

  1. Install the NVIDIA CUDA Toolkit 13.0 SDK (adds nvcc), set
     `CUDA_HOME` and put `nvcc.exe` on PATH, then re-run
     `--phase install --phase smoketest`.
  2. Wait for upstream gsplat to ship pre-compiled sm_120 wheels
     (not currently available).
  3. Choose a different render stack.

**This spike does NOT recommend option 1 right now.** Installing the
CUDA Toolkit adds ~5 GB of dev tooling to the machine and, per the
round-robin consult, the long-term maintenance cost of riding a
bleeding-edge splat stack on Blackwell was already flagged as a risk.

## Recommendation for v2.0-alpha

Unchanged from the 2026-04-16 consultation vote: **ship Stack #4 (SDXL
anchor + ffmpeg zoompan) with the camera_path trajectory engine**
that landed in commit 4ab1512 on v2.0-alpha. The camera_path module
is stack-agnostic -- if gsplat becomes viable later (Toolkit
installed, or pre-compiled wheels ship), its `sample()` output plugs
into a splat camera path with a new emit backend, no rewrite
required.

## When to re-run this spike

Re-open the throwaway branch and re-run `--phase install --phase
smoketest` if ANY of:

  * NVIDIA CUDA Toolkit 13.0 (or 13.x) is installed locally and
    `nvcc` resolves on PATH.
  * gsplat publishes Windows + Blackwell pre-compiled wheels (track
    the PyPI release feed).
  * A new Phase C consultation decides the splat investment is worth
    the Toolkit install cost.

The spike driver (`scripts/_gsplat_spike.py`) is idempotent and
safe to re-run; just delete the stamp file
(`docs/superpowers/consultations/2026-04-16-phase-c-splat-stack/.spike_started_at`)
to reset the 2-hour budget cap.

## Artifacts

  * `04_gsplat_spike_probe.json` -- environment audit (machine-readable)
  * `04_gsplat_spike_dryrun.txt` -- pip dryrun stdout/stderr
  * `04_gsplat_spike.md` -- this file

No `04_gsplat_spike_install.txt`, `_smoketest.json`, or `_rollback.txt`
were created because Phase 3 (install) was not executed.
