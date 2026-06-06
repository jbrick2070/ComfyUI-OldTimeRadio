"""A-S2 probe (e) -- PASS-PM 5th probe: a REAL subprocess sidecar's VRAM is fully
reclaimed AFTER it exits. This is the ACTUAL hardest C->A boundary -- an
in-process flux_still load/teardown does NOT test cross-process reclamation.

Parent: record machine-wide NVML baseline -> spawn a child that allocates
~OTR_SIDECAR_ALLOC_MB on the GPU and ``sys.exit(0)`` -> wait + re-probe NVML ->
PASS iff used has fallen back to <= baseline + OTR_SIDECAR_FLOOR_MB.

Set OTR_SIDECAR_PYTHON to the cu128 sidecar venv python for the true V-12 sidecar
variant; defaults to this interpreter as a cross-process proxy.

OPERATOR-RUN on the 5080. The agent does NOT run this (needs the real GPU).
"""
from __future__ import annotations

import os
import subprocess
import sys
import pathlib
import time

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes._otr_shared import gpu_residency as gr

ALLOC_MB = int(os.getenv("OTR_SIDECAR_ALLOC_MB", "10000"))
FLOOR_MB = int(os.getenv("OTR_SIDECAR_FLOOR_MB", "768"))


def _child_allocate():
    """Child: allocate ~ALLOC_MB of GPU memory, hold briefly, then exit clean."""
    import torch
    if not torch.cuda.is_available():
        print("CHILD: CUDA unavailable", flush=True)
        sys.exit(3)
    n = (ALLOC_MB * 1024 * 1024) // 4  # float32 elements
    buf = torch.empty(n, dtype=torch.float32, device="cuda")
    buf.fill_(1.0)
    torch.cuda.synchronize()
    print(f"CHILD: allocated ~{ALLOC_MB} MB", flush=True)
    time.sleep(2.0)
    sys.exit(0)


def main() -> int:
    if "--child-alloc" in sys.argv:
        _child_allocate()
        return 0
    if not gr.nvml_available():
        print("PROBE e: NO-GO: NVML unavailable (run on the 5080 with drivers)")
        return 1
    baseline = gr.probe_used_mb()
    print(f"PROBE e: baseline machine-wide NVML used = {baseline} MB")
    child_py = os.getenv("OTR_SIDECAR_PYTHON", sys.executable)
    proc = subprocess.run([child_py, __file__, "--child-alloc"],
                          capture_output=True, text=True)
    print(f"PROBE e: child rc={proc.returncode} :: {proc.stdout.strip()} {proc.stderr.strip()[:200]}")
    if proc.returncode == 3:
        print("PROBE e: NO-GO: child reported CUDA unavailable")
        return 1
    # Bounded wait for the driver to release the child's allocation.
    ok = gr.wait_until_below_mb(baseline + FLOOR_MB, attempts=5, sleep_s=2.0)
    after = gr.probe_used_mb()
    print(f"PROBE e: post-exit machine-wide NVML used = {after} MB "
          f"(baseline {baseline} + floor {FLOOR_MB})")
    if ok and after <= baseline + FLOOR_MB:
        print("PROBE e: PASS -- sidecar VRAM fully reclaimed across the process boundary")
        return 0
    print(f"PROBE e: NO-GO: {after - baseline} MB still resident after the sidecar exited "
          f"(> {FLOOR_MB} MB floor) -- the C->A handoff would OOM")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
