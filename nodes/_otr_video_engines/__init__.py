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


# M2 / A-S5: register the in-process motion engines -- ltx_video (text->video)
# and wan_i2v (image->video). Both are DEFAULT-OFF / dark (empty default_roles +
# gated behind OTR_ENABLE_LTX_VIDEO / OTR_ENABLE_WAN_I2V) so they show in the
# static per-role dropdown (V-6) but are never a default and fail closed until
# the operator enables them AND the wrapper + checkpoints are installed/verified
# on the GPU box. Cold-import clean (lazy LTX/Wan wrapper + torch inside
# load/render_clip, never here), so this import pulls in nothing heavy (invariant
# V-12, the cold-import test). Guarded so a packaging quirk never breaks the
# namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_ltx_video as _eng_ltx_video  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:  # pragma: no cover - trivial guard
    from . import eng_wan_i2v as _eng_wan_i2v  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# GO_FORWARD 4A (2026-06-14): register the Wan2.2 TI2V-5B 8GB-tier engine. Like
# wan_i2v it is DEFAULT-OFF / dark (empty default_roles + gated behind
# OTR_ENABLE_WAN_TI2V) and fails closed until the GGUF + the Wan2.2 VAE are on
# disk. Its 5B core node class (Wan22ImageToVideoLatent) was captured from a live
# /object_info before coding. Cold-import clean (V-12); guarded so a packaging
# quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_wan_ti2v as _eng_wan_ti2v  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# 2026-06-15: register the LTX-2.3 AUDIO-INPUT (A2V) lane -- ltx_av_talk
# (audio_driven_face) + ltx_av_music (audio_conditioned_video). ADDITIVE + DARK:
# both are DEFAULT-OFF (empty default_roles + gated behind OTR_ENABLE_LTX_AV) so
# they show in the static per-role dropdown (V-6) but never default and fail
# closed until the GGUF unet + Gemma-3 encoder + LTX VAEs are on disk AND NVML is
# available. The golden prompt-only eng_ltx_video is NEVER imported/touched here.
# Cold-import clean (V-12: lazy LTX/GGUF wrapper + torch inside load/render_clip).
# Guarded so a packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_ltx_av as _eng_ltx_av  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# M2 / A-S6: register the in-process HuMo audio-driven-face engine -- the
# heaviest motion engine (loads MODEL+CLIP+VAE+AUDIO_ENCODER internally via
# comfy.model_management). Like LTX/Wan it is DEFAULT-OFF / dark (empty
# default_roles + gated behind OTR_ENABLE_HUMO) so it shows in the static
# per-role dropdown (V-6) but is never a default and fails closed until the
# operator enables it AND the wrapper + checkpoints are installed/verified on the
# GPU box. Cold-import clean (lazy HuMo wrapper + torch inside load/render_clip,
# never here), so this import pulls in nothing heavy (invariant V-12, the
# cold-import test). Guarded so a packaging quirk never breaks the namespace
# import.
try:  # pragma: no cover - trivial guard
    from . import eng_humo as _eng_humo  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# 0-E easy on-ramp (2026-06-11): still_parallax -- 2.5D depth parallax over
# existing stills (DepthAnythingV2-SMALL, Apache, pinned). DEFAULT-OFF /
# selectable (empty default_roles + OTR_ENABLE_STILL_PARALLAX) until operator
# look-QA; fails closed without the local model. Cold-import clean (lazy
# torch/transformers/PIL/numpy inside load/render_clip). Guarded so a
# packaging quirk never breaks the namespace import. (ltx_orbit, the first
# 0-E engine, registers inside eng_ltx_video above.)
try:  # pragma: no cover - trivial guard
    from . import eng_still_parallax as _eng_still_parallax  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# 0-E easy on-ramp (2026-06-11): mesh_stage -- the traditional local 3D
# chain (portrait -> hy3d-2mv core-node mesh -> cached GLB -> pinned
# portable Blender turntable stage -> straight-alpha frame directory).
# DEFAULT-OFF / selectable (empty default_roles + OTR_ENABLE_MESH_STAGE,
# Tencent license record gates any default-on, E-7); fails closed without
# Blender + the hy3d checkpoint. Cold-import clean (lazy torch/comfy/PIL
# inside load/render_clip). Guarded so a packaging quirk never breaks the
# namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_mesh_stage as _eng_mesh_stage  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# character_3d dark scaffold adapters (W7-pre slice): triposg_talk (the v1
# no-compile lane, S-3D-0-gated) + hunyuan3d_talk/trellis_talk (the deferred
# cu128 toolkit lane). All three are DEFAULT-OFF / dark and FAIL CLOSED until
# their gates clear. Imported UNCONDITIONALLY here so all three appear in the
# static per-role dropdown (V-6: the COMBO always shows the full registry; the
# usability gate is assert_usable, not the import).
# Cold-import clean (V-12): eng_character_3d imports only stdlib + the dep-free
# registry, no torch/diffusers/comfy at module scope. Guarded so a packaging
# quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_character_3d as _eng_character_3d  # noqa: F401
except Exception:  # noqa: BLE001
    pass
