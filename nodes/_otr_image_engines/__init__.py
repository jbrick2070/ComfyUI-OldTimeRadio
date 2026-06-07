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
