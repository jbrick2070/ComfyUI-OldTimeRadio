"""Pluggable video-engine registry -- the model-agnostic video superstructure.

Each video *role* -- announcer visual (A), music visual (B), other-beats visual
(C: character / scene b-roll / background) -- picks its engine from this shared
registry instead of being hardcoded to one model. Adapters self-register on
import; the ``OTR_VideoDirector`` builds its per-role dropdown from the FULL
static registry (V-6) and filters by role compatibility at execute time
(``nodes/_otr_shared/role_compat.py``); ``OTR_ShotLock`` calls
:func:`assert_usable` to fail closed on an incompatible pick.

This mirrors the SHIPPED audio registry pattern (AS-4) via the dependency-free
:mod:`nodes._otr_shared.engine_registry_base` -- it does NOT import the audio
package (which hard-imports ``torch``; that would break the video cold-import
invariant V-12). One pattern, three parallel namespaces (audio frozen; video
here; image in C1).

Cold-import-clean: this module imports only ``typing`` + the dep-free shared
base. No torch / transformers / diffusers at module scope.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .._otr_shared.engine_registry_base import (
    EngineCore,
    EngineRegistry,
    EngineUnusable,
    EngineUsabilityReason,
)

__all__ = [
    "VideoEngine",
    "EngineUnusable",
    "EngineUsabilityReason",
    "register",
    "get_engine",
    "is_registered",
    "all_engine_names",
    "engines_for_role",
    "default_engine_for_role",
    "assert_usable",
]


@runtime_checkable
class VideoEngine(Protocol):
    """Contract every video-engine adapter implements.

    A structural SUPERSET of the shipped ``AudioEngine(Protocol)`` core (see
    :class:`~nodes._otr_shared.engine_registry_base.EngineCore`) -- anchored on
    that proven shape (AS-4), NOT a fresh divergent protocol. The registry only
    reads the CORE members (``name`` / ``roles`` / ``default_roles`` /
    ``commercial_clean`` / ``requires_flag`` / ``load`` / ``unload``); the render
    lifecycle below is the per-clip contract that ``OTR_VideoRenderBatch`` walks
    (built out by the CW-4+ adapters). Adapters duck-type -- inheritance is never
    required.

    ``family`` is one of: ``audio_driven_face`` | ``lipsync_overlay`` |
    ``image_to_video`` | ``text_to_video`` | ``static_image_gen`` |
    ``static_motion`` | ``abstract`` (the 3D ``character_3d`` family ships with
    Subproject B). ``required_inputs`` declares the request-level inputs the
    engine needs (tokens shared with ``role_compat``: ``text_prompt`` /
    ``init_image`` / ``audio_ref`` / ``base_clip_ref``).

    Reduced-capability engines may set ``canonicalize = None`` (e.g. an
    image-gen adapter reused via the same pattern in C1): the registry never
    calls render-lifecycle methods, so optional ones may be absent or ``None``.
    """

    # --- registry-facing core (mirrors AudioEngine(Protocol)) ---
    name: str
    roles: tuple
    default_roles: tuple
    commercial_clean: bool
    requires_flag: Optional[str]

    def load(self) -> None: ...
    def unload(self) -> None: ...

    # --- video-specific identity ---
    family: str
    required_inputs: tuple

    # --- render lifecycle (CW-4+ adapters implement; not called by registry) ---
    def assert_usable(self, host_caps, profile, request_template=None): ...
    def prepare(self, host_caps, profile, session_ctx): ...
    def render_clip(self, request, prepared): ...
    def canonicalize(self, raw, request, profile): ...
    def teardown(self, prepared) -> None: ...


# One registry instance for the video namespace (its own dict; no audio
# cross-pollution). Module-level functions bind to it so the public API matches
# the shipped audio registry's function surface 1:1 (AS-4 "one pattern").
_VIDEO_REGISTRY = EngineRegistry("video")

register = _VIDEO_REGISTRY.register
get_engine = _VIDEO_REGISTRY.get_engine
is_registered = _VIDEO_REGISTRY.is_registered
all_engine_names = _VIDEO_REGISTRY.all_engine_names
engines_for_role = _VIDEO_REGISTRY.engines_for_role
default_engine_for_role = _VIDEO_REGISTRY.default_engine_for_role
assert_usable = _VIDEO_REGISTRY.assert_usable

# Re-export the protocol core under the video namespace for adapters that prefer
# an explicit base reference (duck-typing still works without it).
__all__.append("EngineCore")


# ---------------------------------------------------------------------------
# GATE B S1 -- per-engine capability DECLARATIONS (the registry TABLE, not the
# adapters). Consumed by nodes/_otr_shared/capability_profiles.py to DERIVE the
# per-profile enable-set -- never hand-listed per profile. A new engine ships
# its own row here; zero profile edits.
#
# Keys per row (validated by capability_profiles.validate_declaration):
#   vram_class          cpu | light | medium | heavy (GPU residency class)
#   vram_estimate_mb    DRAFT estimates pending operator probe runs (Lever-1
#                       register) -- policy-grade, not benchmark-grade.
#   required_toolchain  None, or "cu128_toolkit" (source builds; operator-
#                       blocked per the 3D plan -- keeps hunyuan/trellis dark).
#   requires_sidecar    True when the engine runs in an isolated sidecar venv.
#   cpu_ok              True when the engine can run with no GPU at all
#                       (procgen/CPU lanes; the cpu_floor tier filter).
#   model_requirements  informational model-asset ids for the S5 wizard.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    "abstract": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                 "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "still_kenburns": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                       "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "station_card": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                     "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "visualizer": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "flat_still": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "flux_still": {"vram_class": "heavy", "vram_estimate_mb": 12000, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": False,
                   "model_requirements": ["flux.1-dev"]},
    "humo": {"vram_class": "heavy", "vram_estimate_mb": 14000, "required_toolchain": None,
             "requires_sidecar": False, "cpu_ok": False,
             "model_requirements": ["HuMo-17B"]},
    "humo_1.7B": {"vram_class": "medium", "vram_estimate_mb": 7000, "required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["HuMo-1.7B"]},
    # Same 1.7B checkpoint as humo_1.7B, just rendered 16:9 832x480 (~ same pixel
    # budget as the 480x832 portrait) -> identical VRAM class / estimate.
    "humo_1.7B_169": {"vram_class": "medium", "vram_estimate_mb": 7000, "required_toolchain": None,
                      "requires_sidecar": False, "cpu_ok": False,
                      "model_requirements": ["HuMo-1.7B"]},
    # Same 14B checkpoint as humo (the 2026-06-09 keystone), rendered 16:9 832x480
    # (~ same pixel budget as the 480x832 portrait) -> identical heavy VRAM class.
    # The 14B's latents match wan_2.1_vae so it is colour-correct -- the 1.7B blue
    # cast does NOT apply -- giving the operator the 06-09 quality in the 16:9 look.
    "humo_14B_169": {"vram_class": "heavy", "vram_estimate_mb": 14000, "required_toolchain": None,
                     "requires_sidecar": False, "cpu_ok": False,
                     "model_requirements": ["HuMo-17B"]},
    # GGUF splice (2026-06-15): the production LTX video recipe is the frozen
    # mini -- 22B GGUF unet + distilled LoRA @0.70 + Gemma-3 encoder + LTX video
    # VAE + projection ckpt (the 5-artifact tuple). Heavy 22B class. The 2026-06-16
    # battle adopted Q3_K_M as the default quant: measured per-clip peak ~14.8 GB
    # (at the 14.5 GB ceiling, 2.2x faster, no decode offload); Q4_K_S was ~15.8 GB
    # = over. commercial_clean (Apache GGUF + LTX-2 Community model) set True.
    "ltx_video": {"vram_class": "heavy", "vram_estimate_mb": 14000, "required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["ltx-2.3-22b-dev-gguf",
                                         "ltx-2.3-distilled-lora", "gemma-3-12b",
                                         "ltx-2.3-video-vae", "ltx-2.3-22b-dev"]},
    # still_parallax (0-E easy on-ramp): DepthAnythingV2-SMALL (~25M params,
    # Apache-2.0 -- the bigger DA-V2 ckpts are CC-BY-NC and banned) + a pure
    # numpy warp. cpu_ok: CPU-degradable by design (slower, same contract).
    "still_parallax": {"vram_class": "light", "vram_estimate_mb": 500,
                       "required_toolchain": None, "requires_sidecar": False,
                       "cpu_ok": True,
                       "model_requirements": ["depth-anything-v2-small-hf"]},
    # mesh_stage (0-E easy on-ramp): hy3d-2mv core-node mesher (in-process,
    # compile-free) + headless portable Blender stage. vram_estimate is a
    # DRAFT pending the E-1 probe on the 5080; Blender renders AFTER the
    # BUG-291 reclaim barrier so the classes never co-reside. Tencent
    # community license (E-7 record gates default-on).
    # A4 audit 2026-06-11: the all-in-one hy3d checkpoint EMBEDS the DINO
    # image encoder + ShapeVAE -- no separate clip-vision requirement.
    "mesh_stage": {"vram_class": "medium", "vram_estimate_mb": 8000,
                   "required_toolchain": None, "requires_sidecar": False,
                   "cpu_ok": False,
                   "model_requirements": ["hunyuan3d-dit-v2-mv",
                                          "blender-portable"]},
    # S1: vram_estimate raised 14000 -> 14500. The 14499 MB bare-/prompt smoke
    # was WITHOUT free_after_use, which is LOAD-BEARING -- eng_wan_i2v.render_clip
    # passes free_after_use=True so umt5-fp8 + the 14B fp8 UNET do not co-reside
    # through the sampler on the 16 GB card; that mitigation is MANDATORY, not
    # optional. S5: model_requirements is the real Wan 2.2 I2V asset id (was the
    # stale wan2.1 label; the engine ckpt default is wan2.2-i2v.safetensors).
    "wan_i2v": {"vram_class": "heavy", "vram_estimate_mb": 14500, "required_toolchain": None,
                "requires_sidecar": False, "cpu_ok": False,
                "model_requirements": ["wan2.2-i2v"]},
    # S2 (GO_FORWARD 4A): the 8GB-tier Wan2.2 TI2V-5B sibling. medium class /
    # ~8000 MB DRAFT -- the 5B GGUF UNET is ~3.6 GB but the umt5 text-encode + the
    # Wan2.2 VAE decode push the render-phase peak higher; verify on the 8GB probe
    # / the Phase-2 measured peak and tighten. model_requirements is the real 5B
    # asset id. Apache-2.0 (commercial-clean); built 2026-06-14 after the live
    # /object_info node-class capture (the registry-consistency invariant forbids a
    # row without a registered engine, so this lands WITH eng_wan_ti2v).
    "wan_ti2v": {"vram_class": "medium", "vram_estimate_mb": 8000, "required_toolchain": None,
                 "requires_sidecar": False, "cpu_ok": False,
                 "model_requirements": ["wan2.2-ti2v-5b"]},
    # triposg_talk: the v1 NO-COMPILE character_3d lane -- prebuilt cu128
    # wheels only (NO cu128_toolkit requirement; that distinction is the whole
    # point of the lane, 3D plan section 4). Still flag-gated dark at the
    # adapter (OTR_ENABLE_TRIPOSG_TALK + S-3D-0), so an enable-set "fit" never
    # means "renders today" -- assert_usable stays the usability authority.
    "triposg_talk": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                     "required_toolchain": None, "requires_sidecar": True,
                     "cpu_ok": False, "model_requirements": ["triposg"]},
    # triposr (2026-06-18): the LOWER-TIER MIT 3D mesher -- the license-clean,
    # 8GB-tier sibling of mesh_stage (TripoSR single-image->mesh, ~6-8 GB,
    # sub-second). Static mesher (image_to_video family; turntable motion only,
    # NEVER lip-sync). MIT -> commercial_clean. No cu128 toolkit (prebuilt
    # wheels / transformers), runs in-process, no sidecar. medium / ~7000 MB
    # DRAFT -- tighten on the GPU probe. Registered DARK (OTR_ENABLE_TRIPOSR);
    # kept OUT of VALIDATED_ENGINES until the forward is GPU-validated.
    "triposr": {"vram_class": "medium", "vram_estimate_mb": 7000,
                "required_toolchain": None, "requires_sidecar": False,
                "cpu_ok": False, "model_requirements": ["triposr"]},
    "hunyuan3d_talk": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                       "required_toolchain": "cu128_toolkit", "requires_sidecar": True,
                       "cpu_ok": False, "model_requirements": ["hunyuan3d-2"]},
    "trellis_talk": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                     "required_toolchain": "cu128_toolkit", "requires_sidecar": True,
                     "cpu_ok": False, "model_requirements": ["trellis"]},
    # LTX-AV (audio-input) lane -- the LTX-2.3 22B audio-conditioned engines.
    # vram_estimate is the M0-MEASURED Q3_K_M peak (13688 MB at 512x288x97, 8
    # steps, Gemma-3 encoder offloaded to CPU via device=cpu) -- under the 14500
    # ceiling; the audio VAE adds ~340 MB at the floor. Q4_K_S (15594 MB) is OVER
    # and is a quality step-up only if the ceiling is relaxed / run solo. Both
    # engines run in-process and are DEFAULT-OFF / dark (OTR_ENABLE_LTX_AV).
    # SHARP build-out (2026-06-17): the default recipe adds the distilled LoRA +
    # the projection ckpt (LTXAVTextEncoderLoader reads it) -> listed here + gated
    # in eng_ltx_av._weight_paths. Estimate stays 14000 (the free_after_use engine
    # path frees the Gemma encoder before the unet+decode peak; verify on the soak).
    "ltx_av_talk": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                    "required_toolchain": None, "requires_sidecar": False,
                    "cpu_ok": False,
                    "model_requirements": ["ltx-2.3-22b-dev-gguf", "gemma-3-12b",
                                           "ltx-2.3-audio-vae", "ltx-2.3-video-vae",
                                           "ltx-2.3-distilled-lora", "ltx-2.3-22b-dev"]},
    "ltx_av_music": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                     "required_toolchain": None, "requires_sidecar": False,
                     "cpu_ok": False,
                     "model_requirements": ["ltx-2.3-22b-dev-gguf", "gemma-3-12b",
                                            "ltx-2.3-audio-vae", "ltx-2.3-video-vae",
                                            "ltx-2.3-distilled-lora", "ltx-2.3-22b-dev"]},
    # ltx_audio_in (2026-06-26): the UNIFIED, AGNOSTIC audio-in lane -- one engine
    # for music + announcer + character (I2V on whatever still + the shot audio,
    # music or voice). Same LTX-2.3 22B audio weights / heavy class / OTR_ENABLE_LTX_AV
    # gate as the talk/music pair; accepts_still=True so the bookend still is minted.
    "ltx_audio_in": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                     "required_toolchain": None, "requires_sidecar": False,
                     "cpu_ok": False,
                     "model_requirements": ["ltx-2.3-22b-dev-gguf", "gemma-3-12b",
                                            "ltx-2.3-audio-vae", "ltx-2.3-video-vae",
                                            "ltx-2.3-distilled-lora", "ltx-2.3-22b-dev"]},
}
__all__.append("CAPABILITIES")


# ---------------------------------------------------------------------------
# TESTED-ONLY DROPDOWN GATE (2026-06-17 operator directive). The per-role video
# COMBO in OTR_VideoDirector lists ONLY engines that have been VALIDATED on the
# GPU end-to-end -- untested engines (3D talkers, wan_i2v/wan_ti2v until tested,
# the abstract / CPU-floor families, etc.) are HIDDEN from the dropdown so the
# operator cannot accidentally pick a non-working model. This is a DISPLAY gate
# only: every engine stays REGISTERED (V-6 -- all_engine_names() is unchanged, so
# role_compat / assert_usable / the force-map experiment knob still see the full
# set), and the "+ Add Custom Model" sentinel remains the escape hatch for an
# explicitly-declared custom engine.
#
# To promote an engine: add its name here AFTER a green GPU validation. This is a
# SEPARATE frozenset (not a CAPABILITIES row key) on purpose -- capability_profiles
# .validate_declaration rejects unknown CAPABILITIES keys, and "validated" is a
# build-status fact, not a hardware-capability fact.
#
# Validated VIDEO engines as of 2026-06-17 (GPU-proven end-to-end, audio
# byte-identical): the LTX lanes + the HuMo family. humo_1.7B_169 is functional
# (still cfg 2.5 / mildly cool -- see GO_FORWARD_PLAN); it is listed because it
# renders correctly, only its colour grade is unfinished.
# ---------------------------------------------------------------------------
VALIDATED_ENGINES = frozenset({
    "ltx_video",
    "ltx_av_music",
    "ltx_av_talk",
    "humo",
    "humo_1.7B",
    "humo_1.7B_169",
    "humo_14B_169",
    # wan_ti2v PROMOTED 2026-06-18: forced-lane GPU smoke PASSED on the 5080 via
    # the real adapter (OTR_VideoRenderBatch mode=single, executor thread) -- i2v
    # rendered 33 frames from a 16:9 still, engine vram_used ~8.1 GB, independent
    # NVML peak 12,945 MiB (< the 14.5 GB cap). The 8GB-tier 5B motion engine
    # (b-roll / scene / background; NO audio -> never the lip-sync path).
    "wan_ti2v",
    # visualizer PROMOTED 2026-06-18: a full visualizer-all-roles 120w episode
    # rendered END-TO-END on the 5080 (random seed, via _otr_combo_soak -> the real
    # OTR_VideoRenderBatch + mux) -> status=success after 4 soak-found robustness
    # fixes (assert_usable / b000 master-slice / idle-on-silence / 0-frame default).
    # The CPU/ffmpeg-only procedural CRT scope floor (near-zero GPU; default-ON).
    "visualizer",
    # flat_still 2026-06-18: a DEAD-FLAT static still (the selected image held with
    # NO pan/zoom, fit+pad so a face is never cropped) -- the "I want stills, not
    # video" option (operator). CPU/ffmpeg-only, no weights, no VRAM, always renders
    # -> commercial-clean + trivially valid; listed so end users can pick it.
    "flat_still",
    # mesh_stage PROMOTED 2026-06-21 (operator: "at least one 3D model ready
    # for all 3 slots"): the textured-hero PoC GPU smoke PASSED on the 5080
    # 2026-06-20 (portrait -> Hunyuan3D-2mv mesh -> Blender WORKBENCH turntable
    # -> RGBA frame directory -> directory-clip composite). family
    # image_to_video, default_roles=() so it is SELECTABLE-NOT-DEFAULT for
    # every role; gated behind OTR_ENABLE_MESH_STAGE + OTR_BLENDER_EXE.
    # NOTE: commercial_clean=False (Tencent Hunyuan community license -- the
    # E-7 license-record gate still applies before any default-on / ship).
    "mesh_stage",
})
__all__.append("VALIDATED_ENGINES")


def validated_engine_names() -> list:
    """Registered engines that are VALIDATED (GPU-proven), sorted.

    The intersection of the live registry with :data:`VALIDATED_ENGINES` -- the
    tested-only source for the OTR_VideoDirector per-role dropdowns. Intersecting
    with the registry (rather than returning the frozenset directly) keeps the
    list honest if an engine is ever renamed or unregistered.
    """
    return sorted(set(_VIDEO_REGISTRY.all_engine_names()) & VALIDATED_ENGINES)


__all__.append("validated_engine_names")
