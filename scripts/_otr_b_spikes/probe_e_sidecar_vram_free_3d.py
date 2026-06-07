"""B-dep/B-S1 probe (e) -- the 3D render-spawn fits the 14.0 GB sub-ceiling AND
is fully reclaimed after the sidecar exits.

Unlike mesh-gen (cheap), the RENDER spawn co-loads the A2F-3D TRT engine + the
cached mesh + the rasterizer -- its PEAK is the real budget risk (passPM
MUST-FIX #3). This probe spawns a REAL subprocess that allocates the render-spawn
footprint, samples machine-wide NVML to capture the PEAK while it is resident,
asserts peak <= 14000 MB (the 3D sub-ceiling, NOT the 14.5 GB machine ceiling),
then asserts the VRAM is reclaimed below the residual floor after the child
exits. It takes the shipped AS-3 lease so the spike also smoke-tests the lease.

Tunables: OTR_B_RENDER_ALLOC_MB (child footprint, default 12000),
OTR_B_HOLD_S (hold window, default 5), OTR_SIDECAR_FLOOR_MB (post-exit floor,
default 768), OTR_B_SIDECAR_PYTHON (cu128 sidecar python; defaults to this one).

OPERATOR-RUN on the 5080. The agent does NOT run this (needs the real GPU).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _b_harness as H  # noqa: E402
from nodes._otr_shared import gpu_residency as gr  # noqa: E402

ALLOC_MB = int(os.getenv("OTR_B_RENDER_ALLOC_MB", "12000"))
HOLD_S = float(os.getenv("OTR_B_HOLD_S", "5"))
FLOOR_MB = int(os.getenv("OTR_SIDECAR_FLOOR_MB", "768"))
BUDGET_MB = H.SIDECAR_3D_BUDGET_MB  # 14000


def _child_allocate() -> None:
    import torch
    if not torch.cuda.is_available():
        print("CHILD: CUDA unavailable", flush=True)
        sys.exit(3)
    n = (ALLOC_MB * 1024 * 1024) // 4
    buf = torch.empty(n, dtype=torch.float32, device="cuda")
    buf.fill_(1.0)
    torch.cuda.synchronize()
    print(f"CHILD: allocated ~{ALLOC_MB} MB", flush=True)
    time.sleep(HOLD_S)
    sys.exit(0)


def main() -> int:
    if "--child-alloc" in sys.argv:
        _child_allocate()
        return 0
    if not gr.nvml_available():
        print("PROBE e: NO-GO: NVML unavailable (run on the 5080 with drivers)")
        return 1
    baseline = gr.probe_used_mb()
    print(f"PROBE e: baseline machine-wide NVML used = {baseline} MB; "
          f"sub-ceiling = {BUDGET_MB} MB")
    lease = None
    try:
        lease = gr.acquire(timeout_s=30)
    except Exception as e:  # noqa: BLE001
        print(f"PROBE e: NO-GO: could not take the AS-3 lease: {e}")
        return 1
    try:
        child_py = os.getenv("OTR_B_SIDECAR_PYTHON", sys.executable)
        proc = subprocess.Popen([child_py, __file__, "--child-alloc"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        peak = baseline
        deadline = time.time() + HOLD_S + 30.0
        while proc.poll() is None and time.time() < deadline:
            peak = max(peak, gr.probe_used_mb())
            time.sleep(0.25)
        out, err = proc.communicate(timeout=30)
        print(f"PROBE e: child rc={proc.returncode} :: {out.strip()} {err.strip()[:200]}")
        if proc.returncode == 3:
            print("PROBE e: NO-GO: child reported CUDA unavailable")
            return 1
        print(f"PROBE e: PEAK machine-wide NVML used = {peak} MB during the render spawn")
        reclaimed = gr.wait_until_below_mb(baseline + FLOOR_MB, attempts=5, sleep_s=2.0)
        after = gr.probe_used_mb()
        print(f"PROBE e: post-exit NVML used = {after} MB (baseline {baseline} + floor {FLOOR_MB})")
    finally:
        if lease is not None:
            gr.release(lease)
    fits = H.sidecar_vram_ok(peak, BUDGET_MB)
    freed = reclaimed and after <= baseline + FLOOR_MB
    if fits and freed:
        print(f"PROBE e: PASS -- render spawn peak {peak} MB <= {BUDGET_MB} MB sub-ceiling "
              f"and fully reclaimed across the process boundary")
        return 0
    if not fits:
        print(f"PROBE e: NO-GO: render-spawn peak {peak} MB exceeds the {BUDGET_MB} MB "
              f"3D sub-ceiling -- the sidecar would OOM at 14.0 GB")
    if not freed:
        print(f"PROBE e: NO-GO: {after - baseline} MB still resident after exit "
              f"(> {FLOOR_MB} MB floor) -- the next spawn would OOM")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
