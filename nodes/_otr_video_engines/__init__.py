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

---------------------------------------------------------------------------
HOW TO ADD YOUR OWN VIDEO ENGINE (same shape as the other three namespaces):
---------------------------------------------------------------------------

The full checklist is docs/VIDEO_LANE_PREFLIGHT.md (enforced by
tests/test_lane_preflight_matrix.py); EXTENDING_OTR.md gives the adapter
walkthrough. One declaration deserves calling out here because silence about
it is invisible until an episode quietly stops obeying the operator:

  accepts_still -- DECLARE IT EXPLICITLY, True or False (preflight G3.6).
  Every video lane is expected to render the still minted by whichever IMAGE
  engine the operator selected for that role (the image-gen dropdowns beside
  the video dropdowns on OTR_VideoDirector). Motion lanes inherit True from
  MotionEngineBase; the procedural viz_* family declares False out loud
  (no image gen exists for a visualizer). An engine that declares NEITHER
  and lists no ``init_image`` resolves to False through a getattr fallback:
  it mints no still, the operator's chosen image model is never invoked for
  that role, and the episode renders anyway -- nothing reports it.
  tests/test_still_spine_engine_coverage.py sweeps the live registry and
  fails any engine that stays silent.

And one more, if you build your lane as a SIBLING of an existing one:

  EVERY PER-ARTIFACT CONSTANT MUST TRAVEL WITH THE LANE (preflight G1.3).
  Model filename, byte floor, recipe receipt, quant token -- make each a CLASS
  attribute your sibling can override, never a module-level constant read from
  inside a method. A method reading the module constant means your lane loads
  the PARENT's weights while stamping its own receipt, which is wrong pixels
  under a confident label. This has now bitten twice here: once on the WAN
  recipe accessors (see eng_fastwan_8gb) and once on Ghost, where the module
  name had been made overridable and the byte floor beside it had not -- so a
  byte-perfect 1.67 GB module was refused as "truncated" against a floor sized
  for a 1.82 GB one. When you subclass, ask what ELSE was sized for the PARENT.
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

# video-tiers (2026-07-20): register the 8GB-tier LTX-Video 0.9.8 distilled 2B I2V
# engine (ltx_8gb). Unlike the older dark motion engines it is a NORMAL selectable
# row -- requires_flag=None, empty default_roles (selectable, not a default); no
# enable flag (registry IS the menu). It fails CLOSED (ordinary asset preflight) until
# the 0.9.8 checkpoint + the shared t5xxl_fp16 are on disk. Its 0.9.8 core node graph
# was captured from a live /object_info + a functional in-process smoke before coding.
# Cold-import clean (V-12: lazy LTX core nodes + torch inside load/render_clip).
# Guarded so a packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_ltx_8gb as _eng_ltx_8gb  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# fastwan_8gb (2026-08-01): the FastWan 2.2 TI2V-5B 3-step DMD distillation --
# a SUBCLASS of eng_wan_ti2v sharing its whole 5B substrate (beat hoist, teardown,
# frame ladder, tiled decode) and overriding only the recipe seam, the LoRA route
# and the sampler chain. ADDITIVE: wan_ti2v keeps its menu row untouched. It is a
# THROUGHPUT tier -- identical VRAM and identical motion to the incumbent, ~2.7x
# sooner -- not a quality or longer-clip upgrade. Fails CLOSED (ordinary asset
# preflight, including the LoRA) until the weights are on disk. Cold-import clean.
# Guarded so a packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_fastwan_8gb as _eng_fastwan_8gb  # noqa: F401
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


# minimax_h3_video (lane 19, 2026-08-12): the MiniMax H3 FL2VA first-frame ->
# silent video lane. A NORMAL selectable row (requires_flag=None, empty
# default_roles); it fails CLOSED until the DiT, the Qwen3-VL encoder and the
# video VAE are on disk AND the server was booted on the named `h3` contract.
# ONE registration out of eng_minimax_h3, deliberately: `minimax_h3_audio_in`
# shares this implementation module but is lane 20's to register, and two public
# ids on one internal id trips public_engines' bijection assert at import time.
# Cold-import clean (V-12: torch / numpy / the ComfyUI node registry are lazy
# inside load/render_clip). Guarded so a packaging quirk never breaks the
# namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_minimax_h3 as _eng_minimax_h3  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# ltx25_video (LTX 2.5 Chunk A, 2026-08-19): the LTX 2.5 Distilled I2V lane,
# rendered SILENT -- the model's audio latent is computed and dropped at
# LTXVSeparateAVLatent, exactly as the three sibling LTX/H3 lanes already do.
# A NORMAL selectable row (requires_flag=None, empty default_roles); it fails
# CLOSED until the Q3 DiT, the Gemma-4 encoder and BOTH VAEs are on disk.
#
# ONE registration out of eng_ltx25, and unlike lane 19's case that is not a
# sequencing choice -- the other two lanes DO NOT EXIST YET. `ltx25_mime` and
# `ltx25_foley_plus` need the model's own audio to enter the master BEFORE the
# master freezes, and video renders four topological stages after that freeze
# (OTR_EpisodeAssembler order 12 vs OTR_VideoRenderBatch order 16). That is
# Chunk B: an execution-order change with its own arc. Their ids are RESERVED
# in eng_ltx25.LTX25_RESERVED_SIBLING_IDS so nobody spends them, and they are
# deliberately absent from the menu, per the operator (2026-08-19): a dropdown
# row that cannot make an episode is worse than a missing one.
#
# Cold-import clean (V-12: torch and every LTX node class are lazy inside
# load/render_clip). Guarded so a packaging quirk never breaks the namespace
# import.
try:  # pragma: no cover - trivial guard
    from . import eng_ltx25 as _eng_ltx25  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# UNREGISTERED 2026-06-30 (still_parallax rip-out, item 2 -- "registry IS the
# menu"): the 0-E easy on-ramp 2.5D depth-parallax engine (DepthAnythingV2-SMALL)
# is no longer imported here and is NOT selectable. The SOURCE file stays on
# disk untouched (dark scaffold, same pattern as triposr/character_3d); re-add
# this import + the @register decorator + a CAPABILITIES row in the SAME
# change if it returns. mesh_stage's fallback chain now degrades directly to
# still_motion (no longer via still_parallax).

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


# UNREGISTERED 2026-06-29 (C3 -- "registry IS the menu"): the dark 3D scaffolds
# triposg_talk / hunyuan3d_talk / trellis_talk (eng_character_3d) and triposr
# (eng_triposr) render NotImplementedError, so they are NO LONGER imported here
# and are NOT selectable. The source files stay on disk; re-add this import +
# the @register decorator(s) + a CAPABILITIES row in the SAME change when a real
# forward ships (then they return to the dropdown).


# viz_green (renamed from "visualizer" 2026-06-30, item 2): the LOW-VRAM
# ffmpeg-only procedural CRT scope engine -- audio-reactive scopes rendered AS
# the per-beat picture (the resurrected full-colour video_engine look, via the
# COPIED torch-free routines in _otr_shared/scope_draw.py; zero coupling to the
# floor node / the SceneAwareScopes overlay). Selectable per role (registry IS
# the menu; the historical OTR_ENABLE_VISUALIZER flag is vestigial). Old saved
# graphs carrying "visualizer" resolve via otr_video_director's
# _LEGACY_ENGINE_ALIASES. Cold-import clean (V-12: soundfile/PIL lazy in
# render_clip). Guarded so a packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_visualizer as _eng_visualizer  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# viz_mxc_cpu (2026-06-30): the OTR multi-colored ("mxc") rainbow visualizer -- the
# creative replacement for the retired abstract floor. Pure numpy/PIL/ffmpeg (no GPU,
# no shaders; runs on any box). AUDIO-OPTIONAL (required_inputs=()) so it fits every
# role AND idles a procedural rainbow on silence (the no-image floor). Cold-import
# clean (V-12: soundfile/PIL/scope_draw lazy in render_clip). Guarded so a packaging
# quirk never breaks the namespace import. (viz_mxc_gpu shader tier is DEFERRED.)
try:  # pragma: no cover - trivial guard
    from . import eng_viz_rainbow as _eng_viz_rainbow  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# viz_mxc_mandala (2026-06-30): the Cosmic Radio Mandala -- pycairo vector CPU
# painter (tuning-eye + radio-dial rings/spokes + spectrum band), a SEPARATE
# selectable engine alongside viz_mxc_cpu (the zero-dep PIL rainbow stays the
# zero-dep alternate). Opt-in selectable, NOT a saved-widget default. Cold-import
# clean (V-12: cairo/soundfile/PIL/scope_draw lazy in load/assert_usable/
# render_clip -- cairo is NEVER at module scope). Guarded so a packaging quirk
# never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_viz_mandala as _eng_viz_mandala  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# viz_camera (2026-07-05): the OTR-native Golden Flicker camera visualizer --
# warm gold projector-camera centerpiece, spectrum spokes, and CRT post, copied
# into this repo as first-party PIL/numpy code with NO runtime dependency on the
# separate golden-flicker repo. Audio-optional, accepts_still=False, selectable
# per role like the other abstract visualizers.
try:  # pragma: no cover - trivial guard
    from . import eng_viz_camera as _eng_viz_camera  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# CLOUD partner video rows (S3 core, 2026-07-02): kling_avatar / seedance_2 /
# wan_i2v via the S0 invoke_partner_node bridge, conformed by
# canonicalize_video (provider audio ALWAYS stripped, strip-proven). Register
# unconditionally (registry IS the menu) with EMPTY default_roles -- selectable
# picks only; NO enable flag (operator directive 2026-07-02: the dropdown pick
# IS the enable; missing credentials fail LOUD at invoke-time auth).
# Cold-import clean (torch/PIL/soundfile/bridge lazy).
# Guarded so a packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_cloud_video as _eng_cloud_video  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# Direct Google Gemini Omni BYO API video adapter. Not a Partner node and not
# local GPU; selectable only, no default role.
try:  # pragma: no cover - trivial guard
    from . import eng_google_omni_video as _eng_google_omni_video  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# Direct Google Veo 3.1 BYO API video adapter. Not a Partner node and not local
# GPU; selectable only, no default role.
try:  # pragma: no cover - trivial guard
    from . import eng_google_veo_video as _eng_google_veo_video  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# (rip-sfx 2026-08-06: the eng_google_vid_sfx import block died with its
# module -- the SFX-bed lane is retired, ids guarded via RETIRED_ENGINE_IDS.)


# animatediff15_video (Ghost Signal, 2026-08-22): SD1.5 + the mm-p_0.5 v2 motion
# module through AnimateDiff-Evolved's NON-LOOPED Standard Static context, at a
# fixed 512x288 delivered by clean Lanczos to 1920x1080.
#
# THE PROMPT-ONLY LANE, and the first one that declares it. `accepts_still` is
# False and `still_plan` is EMPTY, so the image phase mints nothing for it --
# not a portrait, not a scene still, nothing -- and G3.7's lane-derived role set
# reads that declaration rather than an engine name. It is a NORMAL selectable
# row (requires_flag=None, empty default_roles) and it fails CLOSED until both
# pinned artifacts are on disk and the ADE pack is installed.
#
# Cold-import clean (V-12: torch, numpy and every ComfyUI class are lazy inside
# load/render_clip; ghost_signal_prompt imports only hashlib/re). Guarded so a
# packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_ghost_signal as _eng_ghost_signal  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# The two OFFICIAL-module Ghost peers (2026-08-22). ADDITIVE: they sit beside
# animatediff15_video, which is unchanged and remains the lane that rendered the
# published episode. They exist because the default module ships with NO LICENCE
# GRANT -- the one blocker to submitting this lane anywhere -- and because the
# spec's own Phase-0 inventory of v2/v3 was never carried out.
#
# Imported AFTER the golden lane because they subclass it. Guarded like every
# sibling so a packaging quirk never breaks the namespace import.
try:  # pragma: no cover - trivial guard
    from . import eng_ghost_signal_official as _eng_ghost_signal_official  # noqa: F401
    from . import eng_ghost_signal_cadence as _eng_ghost_signal_cadence  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# ROSTER AUDIT -- runs LAST, after every guarded adapter import above.
# ---------------------------------------------------------------------------
# Multi-clip coverage chunk 2 (2026-07-25). Position matters and is the whole
# point: this must sit at the BOTTOM of this module, not inside registry.py.
# Run from inside the registry it would report every not-yet-imported adapter
# as missing, because the imports above are what populate it.
#
# Every adapter import in this file is deliberately wrapped in a bare
# ``except Exception: pass`` so a packaging quirk can never break the namespace
# import. The price is that a BROKEN adapter disappears silently -- it never
# registers, it vanishes from the per-role dropdown, and nothing anywhere says
# so. registry.CAPABILITIES is the independent expected roster that survives
# such a failure, so comparing the two is the only way to see the hole.
#
# LOG, never raise: a box with one broken adapter must still render with the
# other thirty. CI enforcement is a test (tests/test_frame_contract.py), which
# is the right place for a hard gate.
try:  # pragma: no cover - trivial guard
    from . import registry as _registry_for_audit

    _roster = _registry_for_audit.audit_engine_roster()
    if _roster["missing"]:
        import logging as _logging

        _logging.getLogger("OTR").error(
            "[OTR video] ROSTER AUDIT: %d declared engine(s) FAILED TO "
            "REGISTER: %s -- their adapter import raised and was swallowed by "
            "the guards above, so they are silently absent from every per-role "
            "dropdown. This is a real break, not a warning.",
            len(_roster["missing"]), ", ".join(_roster["missing"]))
    if _roster["unexpected"]:
        import logging as _logging

        _logging.getLogger("OTR").warning(
            "[OTR video] ROSTER AUDIT: %d engine(s) registered without a "
            "CAPABILITIES row: %s -- 'registry IS the menu' requires a row per "
            "registered engine.",
            len(_roster["unexpected"]), ", ".join(_roster["unexpected"]))
except Exception:  # noqa: BLE001 -- the audit must never break the import
    pass
