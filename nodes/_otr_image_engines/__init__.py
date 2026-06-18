"""OTR image-engine namespace (the model-agnostic image-gen adapter registry).

Subproject C, one level UP from the video platform: every image generator
(Flux gen-1, and later Z-Image / Klein / Qwen / any "+ Add Custom Model") is a
pluggable adapter the user selects PER ROLE -- "Flux" is just gen 1, never a
hardcoded default. Swap the image-gen and nothing downstream changes (A's video
init-image / portrait + B's portrait->mesh resolve by ledger, not by engine).

A THIRD parallel namespace beside ``nodes/_otr_audio_engines`` (frozen) and
``nodes/_otr_video_engines`` -- same proven pattern (AS-4) via the dep-free
:mod:`nodes._otr_shared.engine_registry_base`, its OWN registry dict, zero
cross-pollution. The image protocol is a REDUCED ``prompt -> image`` set (no
``canonicalize``), mirroring the SHIPPED ``AudioEngine(Protocol)`` core.

Import-time is cold-import-clean (invariant V-12): importing this package +
:mod:`registry` + :mod:`schemas` + any adapter descriptor pulls in NO heavy lib
(torch / transformers / diffusers). Adapters lazy-import their frameworks inside
``load`` / ``render_image``, never at module scope. The Flux gen-1 adapter is
imported here (guarded) so it self-registers on package import while staying
cold-import clean.
"""
from __future__ import annotations

# Register the gen-1 Flux adapter on package import so the image registry is
# non-empty (Flux selectable per role). The adapter is cold-import clean (it
# lazy-imports torch/comfy only inside load/render), so this import pulls in
# nothing heavy. Guarded so a future packaging quirk never breaks the namespace.
try:  # pragma: no cover - trivial guard
    from . import flux_gen1 as _flux_gen1  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# C2 (build-or-NO-GO, default-OFF): the Z-Image-Turbo adapter -- a 2nd image
# engine proving the model-agnostic layer holds >=2 engines. Cold-import clean;
# greyed until OTR_ENABLE_ZIMAGE=1 + its cu128 sidecar exists. Separate guard so
# a quirk in one adapter never blocks the other from registering.
try:  # pragma: no cover - trivial guard
    from . import z_image_turbo as _z_image_turbo  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# C3 (generic peer, default-OFF): the Qwen-Image (GGUF) adapter -- a 3rd image
# engine carrying the layer to an OPEN set, via a DIFFERENT integration mode (the
# in-stack ComfyUI-GGUF loader, no cu128 sidecar) behind the same protocol.
# Cold-import clean; greyed until OTR_ENABLE_QWEN_IMAGE=1 + its GGUF weights exist.
# Its own guard so a quirk in one adapter never blocks the others from registering.
try:  # pragma: no cover - trivial guard
    from . import qwen_image as _qwen_image  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# C4-C6 (generic peers, default-OFF): more model-agnostic image engines from the
# C2 dep/license matrix, each registering identically and greyed until its own
# OTR_ENABLE_<X>=1 + weights exist. HiDream-I1 (GGUF/MIT, in-stack like Qwen);
# Lumina-Image 2.0 (native lightweight/Apache-2.0). All commercial-clean
# (MIT/Apache). Each has its OWN guard so a quirk in one never blocks the others
# -- and proves the "+ Add Custom Model" open-set story: the registry grows by
# dropping in an adapter, no other edits.
# (Chroma1-HD was DROPPED 2026-06-18 by operator decision: it is a de-restricted
# /uncensored FLUX finetune and OTR will not ship a path to that content.)
try:  # pragma: no cover - trivial guard
    from . import hidream_i1 as _hidream_i1  # noqa: F401
except Exception:  # noqa: BLE001
    pass
try:  # pragma: no cover - trivial guard
    from . import lumina_image as _lumina_image  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# C7-C8 (generic peers, default-OFF, CONDITIONAL license -> commercial_clean=False):
# matrix candidates whose license is conditional/unconfirmed, registered HONESTLY
# so the layer holds mixed-license peers and the license gate surfaces each one's
# real terms. SD 3.5 Large (Stability Community, free only below a revenue tier);
# FLUX.2 Klein (FLUX-family, verify -- plus BUG-070 SageAttention at render time).
# Each greyed until its own OTR_ENABLE_<X>=1 + weights exist; own guard each.
try:  # pragma: no cover - trivial guard
    from . import sd35_large as _sd35_large  # noqa: F401
except Exception:  # noqa: BLE001
    pass
try:  # pragma: no cover - trivial guard
    from . import flux2_klein as _flux2_klein  # noqa: F401
except Exception:  # noqa: BLE001
    pass
