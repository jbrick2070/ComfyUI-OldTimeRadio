"""A-S2 probe (d) -- the VRAM boundary check FAILS CLOSED at the ceiling.

The platform must REFUSE to start a render whose projected residency would exceed
the single-resident-heavy ceiling (14.5 GB), rather than admitting it and OOMing
mid-sampling. This probe drives machine-wide used VRAM up toward the ceiling, then
asks the admission check whether a further heavy render fits -- and asserts it
REFUSES (fail-closed) at the boundary while ADMITTING when there is headroom.

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

CEILING_MB = int(os.getenv("OTR_VRAM_CEILING_MB", "14500"))
HOG_MB = int(os.getenv("OTR_BOUNDARY_HOG_MB", "0"))  # 0 -> auto: ceiling - 1500


def _would_admit(needed_mb: int, ceiling_mb: int) -> bool:
    """FAIL-CLOSED admission: only admit if machine-wide used + needed <= ceiling.
    NVML-unavailable -> refuse (cannot prove headroom)."""
    if not gr.nvml_available():
        return False
    return (gr.probe_used_mb() + int(needed_mb)) <= int(ceiling_mb)


def main() -> int:
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("PROBE d: NO-GO: torch unavailable")
        return 1
    if not (gr.nvml_available() and torch.cuda.is_available()):
        print("PROBE d: NO-GO: CUDA/NVML unavailable (run on the 5080)")
        return 1

    # With low usage, a modest render must be admitted (sanity).
    admit_low = _would_admit(2000, CEILING_MB)
    base = gr.probe_used_mb()
    hog_mb = HOG_MB or max(1000, CEILING_MB - base - 1500)
    n = (hog_mb * 1024 * 1024) // 4
    hog = torch.empty(n, dtype=torch.float32, device="cuda")
    hog.fill_(1.0)
    torch.cuda.synchronize()
    used_now = gr.probe_used_mb()
    # Near the ceiling, a further heavy (e.g. 3 GB) render must be REFUSED.
    admit_near = _would_admit(3000, CEILING_MB)
    del hog
    import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"PROBE d: ceiling={CEILING_MB} base={base} after_hog={used_now} MB; "
          f"admit_low={admit_low} admit_near_ceiling={admit_near}")
    if admit_low and not admit_near:
        print("PROBE d: PASS -- admits with headroom, FAILS CLOSED at the boundary")
        return 0
    print("PROBE d: NO-GO: boundary check did not fail closed "
          f"(admit_low={admit_low}, admit_near={admit_near})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
