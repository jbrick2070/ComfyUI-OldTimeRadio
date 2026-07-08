"""Pluggable video-engine registry -- the model-agnostic video superstructure.

Each video *role* -- announcer visual (A), music visual (B), character visual
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

from .._otr_shared import role_compat as _role_compat
from .._otr_shared.engine_registry_base import (
    EngineCore,
    EngineRegistry,
    EngineUnusable,
    EngineUsabilityReason,
    EngineNotRunnableError,
)

__all__ = [
    "VideoEngine",
    "VideoEngineRegistry",
    "EngineUnusable",
    "EngineUsabilityReason",
    "EngineNotRunnableError",
    "register",
    "get_engine",
    "is_registered",
    "all_engine_names",
    "engines_for_role",
    "default_engine_for_role",
    "assert_usable",
    "descriptor_for_engine",
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
    invocable: bool
    invocability_reason: str

    # --- render lifecycle (CW-4+ adapters implement; not called by registry) ---
    def assert_usable(self, host_caps, profile, request_template=None): ...
    def prepare(self, host_caps, profile, session_ctx): ...
    def render_clip(self, request, prepared): ...
    def canonicalize(self, raw, request, profile): ...
    def teardown(self, prepared) -> None: ...


class VideoEngineRegistry(EngineRegistry):
    """The VIDEO registry: eligibility is CAPABILITY, not the legacy `roles` list.

    C2 (D1 drift fix, 2026-06-30). Production already validates per-role picks by
    CAPABILITY (``role_compat.engine_fits_role``: an engine fits a role iff the role
    can supply every ``required_inputs`` token). The shared base, however, still
    gated ``engines_for_role`` / ``assert_usable`` on the per-engine ``roles``
    whitelist -- so a capability-fit engine (e.g. ltx_video for character_video) was
    rejected by the registry while production accepted it, and the soak filled a
    still instead of the video. These two methods are overridden HERE -- not in
    :class:`EngineRegistry`, which also serves the IMAGE + AUDIO registries -- to
    delegate to ``role_compat``. ``roles`` is now UI-SORT metadata only
    (``default_roles`` still sorts the dropdown).

    FAIL-SOFT: fall back to the legacy ``roles`` whitelist ONLY when an engine
    declares no ``required_inputs`` (``None`` / missing) OR the role is unknown to
    role_compat -- NEVER for ``required_inputs == ()`` (a valid capability that fits
    EVERY role; e.g. a pure-procedural engine). ``RoleCompatError`` (unknown role) is
    wrapped so the public contract is preserved: ``engines_for_role`` filters it out
    via the legacy path; ``assert_usable`` raises :class:`EngineUnusable`.
    """

    def _descriptor(self, name: str) -> dict:
        """The role_compat EngineDescriptor for a REGISTERED engine. ``required_inputs``
        is read straight off the adapter -- ``None`` (missing) is the fail-soft trigger
        and is preserved distinct from ``()`` (declared, fits all roles)."""
        eng = self._registry[name]
        ri = getattr(eng, "required_inputs", None)
        return {
            "engine_id": name,
            "roles": tuple(getattr(eng, "roles", ()) or ()),
            "required_inputs": tuple(ri) if ri is not None else None,
        }

    def _capability_decision(self, name: str, role: str):
        """``(decided, fits)``. ``decided is False`` -> the caller must fall back to
        the legacy ``roles`` whitelist (no capability declared, or unknown role)."""
        desc = self._descriptor(name)
        if desc["required_inputs"] is None:
            return (False, False)                 # no capability declared -> legacy
        try:
            return (True, _role_compat.engine_fits_role(desc, role))
        except _role_compat.RoleCompatError:
            return (False, False)                 # unknown role -> legacy

    def engines_for_role(self, role: str) -> list:
        """Names serving ``role`` BY CAPABILITY (default engine[s] first).

        Fail-soft: an engine with no declared ``required_inputs``, or any engine when
        ``role`` is unknown to role_compat, is filtered by the legacy ``roles`` list.
        The default-for-role engine still sorts first (``default_roles``)."""
        names = []
        for name, eng in self._registry.items():
            decided, fits = self._capability_decision(name, role)
            if decided:
                if fits:
                    names.append(name)
            elif role in tuple(getattr(eng, "roles", ()) or ()):
                names.append(name)                # legacy fail-soft
        names.sort(
            key=lambda n: (
                role not in tuple(getattr(self._registry[n], "default_roles", ()) or ()),
                n,
            )
        )
        return names

    def assert_usable(self, name: str, role: str) -> str:
        """Validate ``name`` may run for ``role`` BY CAPABILITY; FAIL CLOSED.

        ``MALFORMED_CONFIG`` -- not registered. ``INCOMPATIBLE_PROFILE`` -- the role
        cannot supply the engine's ``required_inputs`` (capability), or (fail-soft,
        for an engine with no declared inputs / an unknown role) the legacy ``roles``
        list does not list ``role``. Never silently resolves to another engine."""
        if not self.is_registered(name):
            raise EngineUnusable(
                name, role, EngineUsabilityReason.MALFORMED_CONFIG,
                f"no {self.kind} engine named '{name}' is registered",
                kind=self.kind,
            )
        decided, fits = self._capability_decision(name, role)
        if decided:
            if not fits:
                desc = self._descriptor(name)
                raise EngineUnusable(
                    name, role, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                    f"engine '{name}' requires inputs {tuple(desc['required_inputs'])} "
                    f"which role '{role}' does not supply",
                    kind=self.kind,
                )
            return name
        # Fail-soft: no capability declared (or unknown role) -> legacy `roles` gate.
        eng = self._registry[name]
        if role not in tuple(getattr(eng, "roles", ()) or ()):
            raise EngineUnusable(
                name, role, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                f"engine '{name}' serves roles "
                f"{tuple(getattr(eng, 'roles', ()))}, not '{role}'",
                kind=self.kind,
            )
        return name


# One registry instance for the video namespace (its own dict; no audio
# cross-pollution). Module-level functions bind to it so the public API matches
# the shipped audio registry's function surface 1:1 (AS-4 "one pattern").
_VIDEO_REGISTRY = VideoEngineRegistry("video")

register = _VIDEO_REGISTRY.register
get_engine = _VIDEO_REGISTRY.get_engine
is_registered = _VIDEO_REGISTRY.is_registered
all_engine_names = _VIDEO_REGISTRY.all_engine_names
engines_for_role = _VIDEO_REGISTRY.engines_for_role
default_engine_for_role = _VIDEO_REGISTRY.default_engine_for_role
assert_usable = _VIDEO_REGISTRY.assert_usable


def descriptor_for_engine(name: str) -> dict:
    """The role_compat EngineDescriptor (``engine_id`` / ``roles`` /
    ``required_inputs``) for a registered video engine. The SHARED builder the C2
    registry override + the C4 capability matrix test both read, so the eligibility
    rule has ONE source. Raises ``KeyError`` for an unregistered name."""
    return _VIDEO_REGISTRY._descriptor(name)

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
#   required_toolchain  None, or "cu128_toolkit" (source builds; operator-
#                       blocked per the 3D plan -- keeps hunyuan/trellis dark).
#   requires_sidecar    True when the engine runs in an isolated sidecar venv.
#   cpu_ok              True when the engine can run with no GPU at all
#                       (procgen/CPU lanes; the cpu_floor tier filter).
#   model_requirements  informational model-asset ids for the S5 wizard.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    # "abstract" + "station_card" rows REMOVED 2026-06-30 (C0 -- "registry IS the
    # menu"): both engines were UNREGISTERED (abstract redundant with visualizer;
    # station_card the broken black card), and the registry-consistency invariant
    # forbids a CAPABILITIES row without a registered engine.
    "still_motion": {"required_toolchain": None,
                       "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    # viz_green (renamed from "visualizer" 2026-06-30, item 2; old saved values
    # resolve via otr_video_director._LEGACY_ENGINE_ALIASES).
    "viz_green": {"required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    # viz_mxc_cpu (2026-06-30): the OTR rainbow visualizer -- pure numpy/PIL/ffmpeg,
    # no GPU/shaders. required_toolchain None (a GL/torch toolchain would disable it on
    # every shipped profile). GPU shader tier (viz_mxc_gpu) is a deferred separate row.
    "viz_mxc_cpu": {"required_toolchain": None,
                    "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    # viz_mxc_mandala (2026-06-30): Cosmic Radio Mandala -- pycairo vector CPU
    # painter, no GPU/shaders. required_toolchain None: pycairo is NOT in
    # requirements.video.txt (lazy-imported + probed by assert_usable so a box
    # without system libcairo never breaks any OTHER engine's install).
    "viz_mxc_mandala": {"required_toolchain": None,
                        "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    # viz_camera (2026-07-05): OTR-native Golden Flicker camera visualizer --
    # pure numpy/PIL/ffmpeg, audio-optional, no external golden-flicker import.
    "viz_camera": {"required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "still_flat": {"required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "still_pan": {"required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    # still_word (Sprint B, 2026-07-03): a still_flat sibling -- same CPU/ffmpeg
    # flat-hold render (zero VRAM), the delta is the WORD/TITLE-driven prompt its
    # base still is minted from (compose_still_word_prompt). Model-agnostic: any
    # image engine mints the still. cpu class, cpu_ok True, no model assets.
    "still_word": {"required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "humo": {"required_toolchain": None,
             "requires_sidecar": False, "cpu_ok": False,
             "model_requirements": ["HuMo-17B"]},
    "humo_1.7B": {"required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["HuMo-1.7B"]},
    # Same 1.7B checkpoint as humo_1.7B, just rendered 16:9 832x480 (~ same pixel
    # budget as the 480x832 portrait) -> identical VRAM class / estimate.
    "humo_1.7B_169": {"required_toolchain": None,
                      "requires_sidecar": False, "cpu_ok": False,
                      "model_requirements": ["HuMo-1.7B"]},
    # Same 14B checkpoint as humo (the 2026-06-09 keystone), rendered 16:9 832x480
    # (~ same pixel budget as the 480x832 portrait) -> identical heavy VRAM class.
    # The 14B's latents match wan_2.1_vae so it is colour-correct -- the 1.7B blue
    # cast does NOT apply -- giving the operator the 06-09 quality in the 16:9 look.
    "humo_14B_169": {"required_toolchain": None,
                     "requires_sidecar": False, "cpu_ok": False,
                     "model_requirements": ["HuMo-17B"]},
    # GGUF splice (2026-06-15): the production LTX video recipe is the frozen
    # mini -- 22B GGUF unet + distilled LoRA @0.70 + Gemma-3 encoder + LTX video
    # VAE + projection ckpt (the 5-artifact tuple). Heavy 22B class. The 2026-06-16
    # battle adopted Q3_K_M as the default quant: measured per-clip peak ~14.8 GB
    # (at the 14.5 GB ceiling, 2.2x faster, no decode offload); Q4_K_S was ~15.8 GB
    # = over. commercial_clean (Apache GGUF + LTX-2 Community model) set True.
    "ltx_video": {"required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["ltx-2.3-22b-dev-gguf",
                                         "ltx-2.3-distilled-lora", "gemma-3-12b",
                                         "ltx-2.3-video-vae", "ltx-2.3-22b-dev"]},
    # still_parallax UNREGISTERED 2026-06-30 (item 2 rip-out): no CAPABILITIES
    # row while dark -- see nodes/_otr_video_engines/__init__.py.
    # mesh_stage (0-E easy on-ramp): hy3d-2mv core-node mesher (in-process,
    # compile-free) + headless portable Blender stage. Blender renders AFTER the
    # BUG-291 reclaim barrier so the classes never co-reside. Tencent
    # community license (E-7 record gates default-on).
    # A4 audit 2026-06-11: the all-in-one hy3d checkpoint EMBEDS the DINO
    # image encoder + ShapeVAE -- no separate clip-vision requirement.
    "mesh_stage": {"required_toolchain": None, "requires_sidecar": False,
                   "cpu_ok": False,
                   "model_requirements": ["hunyuan3d-dit-v2-mv",
                                          "blender-portable"]},
    # eng_wan_i2v.render_clip passes free_after_use=True so umt5-fp8 + the 14B
    # fp8 UNET do not co-reside through the sampler on the 16 GB card; that
    # mitigation is MANDATORY, not optional. S5: model_requirements is the real
    # Wan 2.2 I2V asset id (was the stale wan2.1 label; the engine ckpt default
    # is wan2.2-i2v.safetensors).
    "wan_i2v": {"required_toolchain": None,
                "requires_sidecar": False, "cpu_ok": False,
                "model_requirements": ["wan2.2-i2v"]},
    # S2 (GO_FORWARD 4A): the 8GB-tier Wan2.2 TI2V-5B sibling. model_requirements
    # is the real 5B asset id. Apache-2.0 (commercial-clean); built 2026-06-14
    # after the live /object_info node-class capture (the registry-consistency
    # invariant forbids a row without a registered engine, so this lands WITH
    # eng_wan_ti2v).
    "wan_ti2v": {"required_toolchain": None,
                 "requires_sidecar": False, "cpu_ok": False,
                 "model_requirements": ["wan2.2-ti2v-5b"]},
    # triposg_talk / triposr / hunyuan3d_talk / trellis_talk CAPABILITIES rows
    # REMOVED 2026-06-29 (C3 -- "registry IS the menu"): these dark 3D scaffolds
    # render NotImplementedError and are now UNREGISTERED, and the
    # registry-consistency invariant forbids a CAPABILITIES row without a
    # registered engine. Restore the row WITH the @register + package import in
    # the SAME change when a real forward ships.
    # LTX-AV (audio-input) lane -- the LTX-2.3 22B audio-conditioned engines.
    # Both engines run in-process and are DEFAULT-OFF / dark (OTR_ENABLE_LTX_AV).
    # SHARP build-out (2026-06-17): the default recipe adds the distilled LoRA +
    # the projection ckpt (LTXAVTextEncoderLoader reads it) -> listed here + gated
    # in eng_ltx_av._weight_paths.
    # ltx_audio_in (2026-06-26): the ONE audio-in lane -- one engine
    # for music + announcer + character (I2V on whatever still + the shot audio,
    # music or voice). Same LTX-2.3 22B audio weights / OTR_ENABLE_LTX_AV
    # gate as the talk/music pair; accepts_still=True so the bookend still is minted.
    "ltx_audio_in": {"required_toolchain": None, "requires_sidecar": False,
                     "cpu_ok": False,
                     "model_requirements": ["ltx-2.3-22b-dev-gguf", "gemma-3-12b",
                                            "ltx-2.3-audio-vae", "ltx-2.3-video-vae",
                                            "ltx-2.3-distilled-lora", "ltx-2.3-22b-dev"]},
    # CLOUD partner video rows (S3 core, 2026-07-02, pass04 secs 5+7): the
    # render happens PROVIDER-SIDE (zero local VRAM); cpu_ok True (any box with
    # ffmpeg + credits can run them). NO enable flag (operator directive
    # 2026-07-02): rows always REGISTER + show; the dropdown pick is the enable;
    # missing credentials fail LOUD at invoke-time auth resolution.
    "cloud_kling_avatar": {"required_toolchain": None, "requires_sidecar": False,
                           "cpu_ok": True, "model_requirements": []},
    "cloud_seedance_2": {"required_toolchain": None, "requires_sidecar": False,
                         "cpu_ok": True, "model_requirements": []},
    "cloud_wan_i2v": {"required_toolchain": None, "requires_sidecar": False,
                      "cpu_ok": True, "model_requirements": []},
    "cloud_wan_i2v_audio": {"required_toolchain": None, "requires_sidecar": False,
                            "cpu_ok": True, "model_requirements": []},
    # word_razzle (Phase 1, 2026-07-03): the animated word-card cloud i2v engine
    # (Pixverse row cloud_pixverse_i2v). Provider-side render, cpu_ok. Selectable;
    # NO enable flag (dropdown pick is the enable; missing OTR_COMFY_API_KEY
    # fails LOUD at invoke).
    "word_razzle": {"required_toolchain": None, "requires_sidecar": False,
                    "cpu_ok": True, "model_requirements": []},
}
__all__.append("CAPABILITIES")


# ---------------------------------------------------------------------------
# VALIDATED_ENGINES + validated_engine_names() REMOVED 2026-06-29 (C4 -- "registry
# IS the menu"): there is NO validated-subset dropdown filter. Every REGISTERED
# engine is SELECTABLE; the per-role director COMBO is built from
# all_engine_names() (validation is the operator's MANUAL process, never a code
# gate). The "+ Add Custom Model" sentinel remains the escape hatch.
# ---------------------------------------------------------------------------
