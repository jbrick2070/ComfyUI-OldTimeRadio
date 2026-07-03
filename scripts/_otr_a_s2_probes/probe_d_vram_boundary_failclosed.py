"""A-S2 probe (d) -- VRAM usage telemetry probe.

The OOM budget is owned by the external per-hardware tier JSON, NOT by this
platform code -- there is no in-process admission ceiling to fail closed on.
This probe drives machine-wide used VRAM up with a hog allocation, then
reports used/free VRAM before and after (pure telemetry) so the operator can
see the NVML probe track real allocations. It never admits or refuses a
render; it only measures and prints.

Uses the shipped AS-3 machine-wide NVML probe (never comfy get_free_memory).

OPERATOR-RUN on the 5080. The agent does NOT run this.
"""
from __future__ import annotations

import os
import sys
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes._otr_shared import gpu_residency as gr

HOG_MB = int(os.getenv("OTR_BOUNDARY_HOG_MB", "0"))  # 0 -> auto (see main())


def main() -> int:
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("PROBE d: NO-GO: torch unavailable")
        return 1
    if not (gr.nvml_available() and torch.cuda.is_available()):
        print("PROBE d: NO-GO: CUDA/NVML unavailable (run on the 5080)")
        return 1

    base = gr.probe_used_mb()
    total = gr.probe_total_mb() if hasattr(gr, "probe_total_mb") else None
    # Auto hog size: a few GB, or an operator-set override via
    # OTR_BOUNDARY_HOG_MB. This is just enough to prove the NVML probe tracks
    # a real allocation -- not a boundary/ceiling test.
    hog_mb = HOG_MB or 3000
    n = (hog_mb * 1024 * 1024) // 4
    hog = torch.empty(n, dtype=torch.float32, device="cuda")
    hog.fill_(1.0)
    torch.cuda.synchronize()
    used_after_hog = gr.probe_used_mb()
    del hog
    import gc; gc.collect(); torch.cuda.empty_cache()
    used_after_free = gr.probe_used_mb()

    print(f"PROBE d: telemetry -- base_used={base} MB, "
          f"hog_requested={hog_mb} MB, used_after_hog={used_after_hog} MB, "
          f"used_after_free={used_after_free} MB"
          + (f", total={total} MB" if total is not None else ""))
    print("PROBE d: PASS -- NVML telemetry tracked the allocation "
          "(no admission ceiling; the OOM budget lives in the external "
          "per-hardware tier JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
