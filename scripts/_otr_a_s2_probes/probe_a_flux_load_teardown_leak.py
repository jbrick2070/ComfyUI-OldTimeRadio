"""A-S2 probe (a) -- a heavy load -> teardown -> reload leaves NO resident VRAM.

Measures machine-wide NVML (the cross-process truth, AS-3) around: load a heavy
residency -> teardown (del + empty_cache) -> reload. PASS iff post-teardown used
returns to within OTR_LEAK_TOL_MB of baseline AND the reload shows no growth vs
the first load (a leak would compound across episodes).

If OTR_FLUX_CKPT points at a Flux checkpoint the probe loads it via ComfyUI's
loader; otherwise it uses a raw ~10 GB CUDA tensor as the heavy-residency proxy
(and says so) so the leak-accounting still exercises load/teardown/reload.

OPERATOR-RUN on the 5080. The agent does NOT run this.
"""
from __future__ import annotations

import gc
import os
import sys
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes._otr_shared import gpu_residency as gr

TOL_MB = int(os.getenv("OTR_LEAK_TOL_MB", "768"))
CKPT = os.getenv("OTR_FLUX_CKPT", "").strip()
PROXY_MB = int(os.getenv("OTR_SIDECAR_ALLOC_MB", "10000"))


def _load_heavy():
    """Return an opaque handle holding heavy GPU residency."""
    import torch
    if CKPT:
        try:
            import comfy.sd  # type: ignore
            out = comfy.sd.load_checkpoint_guess_config(
                CKPT, output_vae=True, output_clip=True)
            return ("flux", out)
        except Exception as exc:  # noqa: BLE001
            print(f"PROBE a: flux load failed ({exc}); falling back to tensor proxy")
    n = (PROXY_MB * 1024 * 1024) // 4
    t = torch.empty(n, dtype=torch.float32, device="cuda")
    t.fill_(1.0)
    torch.cuda.synchronize()
    return ("proxy", t)


def _free_cuda():
    """Release CUDA caching-allocator memory. The CALLER must drop its OWN
    reference to the heavy handle FIRST (``= None`` / ``del``) -- deleting a
    function parameter cannot free an object the caller still holds (the BUG-291
    lesson: clear the holder's ref BEFORE empty_cache, or nothing is freed)."""
    import torch
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("PROBE a: NO-GO: torch unavailable")
        return 1
    if not (gr.nvml_available() and torch.cuda.is_available()):
        print("PROBE a: NO-GO: CUDA/NVML unavailable (run on the 5080)")
        return 1

    base = gr.probe_used_mb()
    h1 = _load_heavy()
    load1 = gr.probe_used_mb()
    h1 = None                       # drop the holder's ref BEFORE empty_cache
    _free_cuda()
    gr.wait_until_below_mb(base + TOL_MB, attempts=5, sleep_s=1.0)
    after = gr.probe_used_mb()
    h2 = _load_heavy()
    load2 = gr.probe_used_mb()
    h2 = None
    _free_cuda()

    print(f"PROBE a: baseline={base} load1={load1} post_teardown={after} reload2={load2} MB")
    leaked = after - base
    grew = load2 - load1
    if leaked <= TOL_MB and grew <= TOL_MB:
        print(f"PROBE a: PASS -- teardown reclaimed (leak {leaked} MB <= {TOL_MB}; "
              f"reload growth {grew} MB <= {TOL_MB})")
        return 0
    print(f"PROBE a: NO-GO: leak {leaked} MB / reload-growth {grew} MB exceed {TOL_MB} MB")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
