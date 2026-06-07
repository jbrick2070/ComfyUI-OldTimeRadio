"""OTR video-engine namespace (the model-agnostic video adapter registry).

A parallel namespace to ``nodes/_otr_audio_engines`` -- same proven pattern
(AS-4), its OWN registry dict, zero cross-pollution with audio. Every video
model (HuMo / LTX / Wan / lipsync / the cheap radio-floor families / the 3D
Model Renderer / any "+ Add Custom Model") is a pluggable adapter the user
selects PER ROLE; no model is "primary".

Import-time is cold-import-clean (invariant V-12): importing this package +
:mod:`registry` + any adapter descriptor pulls in NO heavy lib (torch /
transformers / diffusers) -- a ``test_cold_import_no_heavy_libs`` asserts it.
Adapters lazy-import their frameworks inside ``load`` / ``render_clip``, never at
module scope, so registering them here stays cold-import clean.
"""

# CW-4: register the cheap radio-floor families on package import so the platform
# is non-empty (selectable per role) before any heavy engine exists. They are
# cold-import clean (lazy ffmpeg/PIL/torch inside render_clip), so this import
# pulls in nothing heavy (invariant V-12, the cold-import test). Guarded so a
# packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import cheap_families as _cheap_families  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# M2 / A-S4: register the latentsync lipsync-overlay sidecar adapter. It is
# DEFAULT-OFF / dark -- registered so it shows in the static per-role dropdown
# (V-6) but not a default for any role and gated behind OTR_ENABLE_LATENTSYNC;
# it fails closed until the Path-B cu128 venv + worker are installed. Cold-import
# clean (lazy diffusers/torch/ffmpeg inside the worker venv, never here), so this
# import pulls in nothing heavy (invariant V-12, the cold-import test). Guarded so
# a packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_latentsync as _eng_latentsync  # noqa: F401
except Exception:  # noqa: BLE001
    pass
