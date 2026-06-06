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
module scope. Keep this ``__init__`` free of eager adapter imports.
"""
