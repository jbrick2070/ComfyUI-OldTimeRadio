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
base + the stdlib-only ``role_compat`` / ``public_engines`` helpers. No torch /
transformers / diffusers at module scope.
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
# Dep-free like the shared base (public_engines is stdlib-only, V-12 clean):
# the retired-id policy is consulted at assert_usable so a stale saved name
# gets the NAMED refusal instead of the generic not-registered error.
from .._otr_shared.public_engines import check_retired_engine, resolve_engine_id

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
    "audit_engine_roster",
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

    # --- multi-clip coverage declaration (2026-07-25, chunk 2; OPTIONAL) ---
    # ``frame_contract`` declares the STATIC legal render lengths and the
    # continuity mode (see :mod:`frame_contract`). It is optional and adapters
    # duck-type as always: an adapter that declares nothing resolves to
    # ``frame_contract.SINGLE_ONLY`` -- one render per beat, no chaining, i.e.
    # exactly today's behaviour. Multi-clip is opt-in and provable PER ENGINE;
    # it is never inherited by default.
    def frame_contract(self): ...

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
        list does not list ``role``. Never silently resolves to another engine.
        A RETIRED id raises :class:`RetiredEngineError` first (rip-sfx
        2026-08-06) -- the NAMED policy refusal, not the generic message.
        Resolved for the guard only (idempotent on internal ids); the
        registration checks below still see the caller's name unchanged."""
        check_retired_engine(resolve_engine_id(name))
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
#   required_toolchain     None, or "cu128_toolkit" (source builds; operator-
#                          blocked per the 3D plan -- keeps hunyuan/trellis dark).
#   requires_sidecar       True when the engine runs in an isolated sidecar venv.
#   device_backends        backends the engine can run on, e.g. ["cuda", "cpu",
#                          "mps"]; GPU-only rows list just ["cuda"].
#   requires_vendor        None, or "nvidia" when the adapter hard-gates on a
#                          vendor-only API (e.g. an NVML VRAM-telemetry probe).
#   needs_fp8_te           True when the engine's default weights are an
#                          fp8-scaled UNET/text-encoder artifact.
#   needs_fp4_te           True when the engine's default weights are an
#                          fp4-scaled UNET/text-encoder artifact.
#   practical_without_gpu  True when the engine can run with no GPU at all
#                          (procgen/CPU lanes; the cpu_floor tier filter).
#   sidecar_conditional    True when requires_sidecar depends on runtime
#                          config rather than being a fixed True/False.
#   model_requirements     informational model-asset ids for the S5 wizard.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    # "abstract" + "station_card" rows REMOVED 2026-06-30 (C0 -- "registry IS the
    # menu"): both engines were UNREGISTERED (abstract redundant with visualizer;
    # station_card the broken black card), and the registry-consistency invariant
    # forbids a CAPABILITIES row without a registered engine.
    "still_motion": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # viz_green (renamed from "visualizer" 2026-06-30, item 2; old saved values
    # resolve via otr_video_director._LEGACY_ENGINE_ALIASES).
    "viz_green": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # viz_mxc_cpu (2026-06-30): the OTR rainbow visualizer -- pure numpy/PIL/ffmpeg,
    # no GPU/shaders. required_toolchain None (a GL/torch toolchain would disable it on
    # every shipped profile). GPU shader tier (viz_mxc_gpu) is a deferred separate row.
    "viz_mxc_cpu": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # viz_mxc_mandala (2026-06-30): Cosmic Radio Mandala -- pycairo vector CPU
    # painter, no GPU/shaders. required_toolchain None: pycairo is NOT in
    # the main requirements (lazy-imported + probed by assert_usable so a box
    # without system libcairo never breaks any OTHER engine's install).
    "viz_mxc_mandala": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # viz_camera (2026-07-05): OTR-native Golden Flicker camera visualizer --
    # pure numpy/PIL/ffmpeg, audio-optional, no external golden-flicker import.
    "viz_camera": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    "still_flat": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    "still_pan": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # still_word (Sprint B, 2026-07-03): a still_flat sibling -- same CPU/ffmpeg
    # flat-hold render (zero VRAM), the delta is the WORD/TITLE-driven prompt its
    # base still is minted from (compose_still_word_prompt). Model-agnostic: any
    # image engine mints the still. cpu class, cpu_ok True, no model assets.
    "still_word": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # S0 portability (2026-07-10): the requirement label now names the artifact
    # family the engine DEFAULT resolves (eng_humo._HUMO_DEFAULT_UNET = Kijai's
    # Wan2_1-HuMo-14B fp8-scaled UNET, fetched by scripts/download_humo_models
    # .ps1). The old "HuMo-17B" label pointed fresh installs at Comfy-Org's
    # differently-named file the engine never looks for.
    "humo": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        # needs_fp8_te True: the default UNET is the fp8_e4m3fn-scaled Kijai
        # artifact (eng_humo._HUMO_DEFAULT_UNET).
        "needs_fp8_te": True, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["HuMo-14B-KJ"]},
    "humo_1.7B": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["HuMo-1.7B"]},
    # Same 1.7B checkpoint as humo_1.7B, just rendered 16:9 832x480 (~ same pixel
    # budget as the 480x832 portrait) -> identical VRAM class / estimate.
    "humo_1.7B_169": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["HuMo-1.7B"]},
    # Same 14B checkpoint as humo (the 2026-06-09 keystone), rendered 16:9 832x480
    # (~ same pixel budget as the 480x832 portrait) -> identical heavy VRAM class.
    # The 14B's latents match wan_2.1_vae so it is colour-correct -- the 1.7B blue
    # cast does NOT apply -- giving the operator the 06-09 quality in the 16:9 look.
    "humo_14B_169": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        # needs_fp8_te True: the default UNET is the fp8_e4m3fn-scaled Kijai
        # artifact (same Wan2_1-HuMo-14B checkpoint as humo).
        "needs_fp8_te": True, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["HuMo-14B-KJ"]},
    # GGUF splice (2026-06-15): the production LTX video recipe is the frozen
    # mini -- 22B GGUF unet + distilled LoRA @0.70 + Gemma-3 encoder + LTX video
    # VAE + projection ckpt (the 5-artifact tuple). Heavy 22B class. The 2026-06-16
    # battle adopted Q3_K_M as the default quant: measured per-clip peak ~14.8 GB
    # (at the 14.5 GB ceiling, 2.2x faster, no decode offload); Q4_K_S was ~15.8 GB
    # = over. commercial_clean (Apache GGUF + LTX-2 Community model) set True.
    "ltx_video": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
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
    "mesh_stage": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["hunyuan3d-dit-v2-mv",
                               "blender-portable"]},
    # eng_wan_i2v.render_clip passes free_after_use=True so umt5-fp8 + the 14B
    # fp8 UNET do not co-reside through the sampler on the 16 GB card; that
    # mitigation is MANDATORY, not optional. S5: model_requirements is the real
    # Wan 2.2 I2V asset id (was the stale wan2.1 label). Lane 1, 2026-08-11:
    # the ckpt default named here used to be wan2.2-i2v.safetensors, which is a
    # placeholder that exists on no box -- it is now the installed artifact,
    # eng_wan_i2v._I2V_DEFAULT_UNET, under diffusion_models. The public menu id
    # states 2.2 for the same reason this row does.
    # S2 (GO_FORWARD 4A): the 8GB-tier Wan2.2 TI2V-5B sibling. model_requirements
    # is the real 5B asset id. Apache-2.0 (commercial-clean); built 2026-06-14
    # after the live /object_info node-class capture (the registry-consistency
    # invariant forbids a row without a registered engine, so this lands WITH
    # eng_wan_ti2v).
    "wan_ti2v": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["wan2.2-ti2v-5b"]},
    # fastwan_8gb (2026-08-01): the FastWan 2.2 TI2V-5B 3-step DMD distillation.
    # Same capability shape as wan_ti2v because it is the SAME base weights --
    # cuda, no vendor gate, no fp8/fp4 (the base is the Q5_K_M GGUF wan_ti2v
    # already ships). It differs only by the rank-128 LoRA, which is a SEPARATE
    # model requirement so preflight fails CLOSED when it is absent: without it
    # the graph would render 3 steps through the UN-distilled base model, which
    # produces no error -- just ruined output wearing a FastWan receipt.
    "fastwan_8gb": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["wan2.2-ti2v-5b", "fastwan-2.2-5b-lora"]},
    # ltx_8gb (video-tiers, 2026-07-20): the 8GB-tier LTX-Video 0.9.8 distilled 2B
    # I2V engine. Modeled on wan_ti2v -- cuda, NO vendor gate, NO fp8/fp4 (the 0.9.8
    # distilled all-in-one is bf16; the LTXQ8Patch quantizer is deliberately NOT
    # adopted). requires_flag None; ordinary asset preflight only. Built after the
    # live /object_info capture + a functional in-process smoke (the
    # registry-consistency invariant forbids a row without a registered engine, so
    # this lands WITH eng_ltx_8gb + its __init__ import).
    "ltx_8gb": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["ltxv-2b-0.9.8-distilled"]},
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
    "ltx_audio_in": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"],
        # requires_vendor "nvidia": NVML VRAM-telemetry gate, table-visible now
        # (eng_ltx_av.assert_usable hard-gates on it).
        "requires_vendor": "nvidia",
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["ltx-2.3-22b-dev-gguf", "gemma-3-12b",
                               "ltx-2.3-audio-vae", "ltx-2.3-video-vae",
                               "ltx-2.3-distilled-lora", "ltx-2.3-22b-dev"]},
    # ltx25_video (LTX 2.5 Chunk A, 2026-08-19): LTX 2.5 Distilled I2V rendered
    # SILENT. cuda, no vendor gate -- and note the difference from its LTX 2.3
    # cousin ltx_audio_in, which DOES carry requires_vendor "nvidia" because it
    # hard-gates on NVML telemetry in assert_usable. This lane does not: it
    # SAMPLES a VRAM peak for the receipt via VramPeakProbe, which degrades to
    # None off a CUDA box, and it never refuses on the reading. A vendor gate
    # here would advertise an enforcement that does not run.
    #
    # needs_fp8_te / needs_fp4_te are both False: the DiT is a Q3_K_M GGUF and
    # the Gemma-4 12B text encoder is a Q5_K_M GGUF, so neither fp8 nor fp4
    # describes this stack -- GGUF k-quants are their own thing and neither
    # flag's tier filter is the right question to ask about them.
    #
    # FIVE model_requirements, and the AUDIO VAE IS ONE OF THEM even though
    # this lane emits no audio. LTXVEmptyLatentAudio mints the audio latent
    # with it and LTXVConcatAVLatent needs that latent to build the joint AV
    # tensor the sampler consumes, so preflight must fail CLOSED without it --
    # the opposite of the minimax_h3_video row above, which deliberately OMITS
    # its audio VAE because that lane genuinely never loads one.
    "ltx25_video": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["ltx-2.5-distilled-q3-gguf",
                               "gemma4-12b-ltx-2.5-proj-gguf",
                               "ltx-2.5-video-vae",
                               "ltx-2.5-audio-vae",
                               "ltx-2.5-latent-spatial-upscaler-x2"]},
    # minimax_h3_video (lane 19, 2026-08-12): MiniMax H3 FL2VA, the 33.1B packed
    # AV DiT rendered VIDEO-ONLY (this lane decodes the video half of the
    # NestedTensor latent and carries no audio VAE at all). cuda, no vendor gate.
    # needs_fp8_te is False and needs_fp4_te is TRUE: the DiT is an int8 repack
    # rather than fp8, and the Qwen3-VL-32B encoder it conditions on is the
    # NVFP4-AWQ artifact -- so the fp4 row is the one that describes this stack.
    # Three model_requirements because all three are separate ~5-21 GB fetches
    # and preflight fails CLOSED on any one of them; the audio VAE is NOT listed,
    # because listing an asset this lane never loads would make the S5 wizard ask
    # for 0.6 GB nobody needs (it belongs to lane 20).
    # THE ONE GHOST LANE (operator, 2026-08-23: "delete any animatediff that are
    # not haunted"). SD1.5 + AnimateDiff-Evolved's official v3 module through the
    # non-looped Standard Static context, plus the removable domain adapter that
    # gives the lane its degraded-transmission look. cuda, no vendor gate, no
    # sidecar, no toolchain. BOTH fp rows are False and that is the point: the
    # artifacts are FP16 and the profile pins dtype_policy="no_fp8_no_fp4" so
    # nothing silently opts them into ComfyUI's optional FP8/FP4 transformations.
    #
    # THREE artifacts, where the retired siblings had two. The adapter is named
    # here so the S5 wizard asks for it; a haunted lane without it refuses to
    # render rather than quietly producing clean output.
    #
    # NO COST ROW EXISTS FOR THIS LANE. The operator declined a measurement
    # campaign, so it is recorded admission-unenforced in the evidence manifest
    # and makes no VRAM-fit claim -- see docs/2026-08-22-ghost-signal-
    # dependency-lock.json.
    "animatediff15_v3_haunted_video": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["v1-5-pruned-emaonly-fp16.safetensors",
                               "v3_sd15_mm.ckpt",
                               "v3_sd15_adapter.ckpt"]},
    "minimax_h3_video": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": True,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["minimax-h3-fl2va-int8",
                               "qwen3vl-32b-minimax-h3-nvfp4",
                               "minimax-h3-video-vae"]},
    # minimax_h3_audio_in (lane 20, 2026-08-12): the REF2VA sibling -- the same
    # 33.1B stack conditioned on a reference portrait plus the beat's own audio.
    # Same capability shape as minimax_h3_video except for the assets: it loads
    # a DIFFERENT 21 GB DiT (ref2va, not fl2va) and it ADDS the audio VAE, which
    # its conditioner needs to ENCODE reference audio. It still decodes no
    # audio, so the lane is silent like its sibling -- "loads an audio VAE" and
    # "emits audio" are different claims and only the first is true.
    "minimax_h3_audio_in": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": True,
        "practical_without_gpu": False, "sidecar_conditional": False,
        "model_requirements": ["minimax-h3-ref2va-int8",
                               "qwen3vl-32b-minimax-h3-nvfp4",
                               "minimax-h3-video-vae",
                               "minimax-h3-audio-vae"]},
    # CLOUD partner video rows (S3 core, 2026-07-02, pass04 secs 5+7): the
    # render happens PROVIDER-SIDE (zero local VRAM); cpu_ok True (any box with
    # ffmpeg + credits can run them). NO enable flag (operator directive
    # 2026-07-02): rows always REGISTER + show; the dropdown pick is the enable;
    # missing credentials fail LOUD at invoke-time auth resolution.
    "cloud_kling_avatar": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    "cloud_seedance_2": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    "cloud_wan_i2v": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    "cloud_wan_i2v_audio": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # cloud_vidu_q2_pro_fast_720p (2026-07-09): cheap Comfy Cloud Vidu Q2
    # image-to-video row, fixed to viduq2-pro-fast at 720p. Provider-side
    # render, cpu_ok. Selectable only; missing OTR_COMFY_API_KEY fails loud at
    # invoke.
    "cloud_vidu_q2_pro_fast_720p": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # word_razzle (Phase 1, 2026-07-03): the animated word-card cloud i2v engine
    # (Pixverse row cloud_pixverse_i2v). Provider-side render, cpu_ok. Selectable;
    # NO enable flag (dropdown pick is the enable; missing OTR_COMFY_API_KEY
    # fails LOUD at invoke).
    "word_razzle": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # google_omni_video (2026-07-08): direct Google Gemini Omni Flash BYO API
    # text-to-video lane. Provider-side render, no local weights/VRAM. Selectable
    # only; missing Google key fails loud at invoke.
    "google_omni_video": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # google_veo_video (2026-07-08): direct Google Veo 3.1 BYO API text-to-video
    # lane via predictLongRunning. Provider-side render, no local weights/VRAM.
    # Selectable only; missing Google key fails loud at invoke.
    "google_veo_video": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    # (rip-sfx 2026-08-06: the five SFX-bed rows -- cloud_vidu_q2_pro_fast_720p_sfx
    # and the four google_vid_sfx_* engines -- are RETIRED. Their ids live in
    # nodes/_otr_shared/public_engines.RETIRED_ENGINE_IDS, consulted by
    # check_retired_engine at every selection boundary; they must never return
    # to this table.)
}
__all__.append("CAPABILITIES")


# ---------------------------------------------------------------------------
# VALIDATED_ENGINES + validated_engine_names() REMOVED 2026-06-29 (C4 -- "registry
# IS the menu"): there is NO validated-subset dropdown filter. Every REGISTERED
# engine is SELECTABLE; the per-role director COMBO is built from
# all_engine_names() (validation is the operator's MANUAL process, never a code
# gate). The "+ Add Custom Model" sentinel remains the escape hatch.
# ---------------------------------------------------------------------------


def audit_engine_roster():
    """Compare the EXPECTED roster against what actually registered.

    THE IMPORT-AUDIT BLINDSPOT (2026-07-25, chunk 2; both kibitz r2 seats found
    this independently). Every adapter import in
    ``_otr_video_engines/__init__.py`` is wrapped in
    ``try: ... except Exception: pass`` so a packaging quirk can never break the
    namespace import. The cost is that a BROKEN adapter fails silently: it never
    registers, it vanishes from the dropdown, and nothing says so. A
    post-registration audit that only walks the registry cannot see the hole
    either -- the missing engine is missing from both sides.

    :data:`CAPABILITIES` is the independent expected roster, maintained by hand
    next to each engine, so it survives an import failure and can be compared
    against reality.

    Returns ``{"missing": (...), "unexpected": (...)}``:

    * ``missing`` -- declared in CAPABILITIES but NOT registered. Almost always
      a swallowed import error, i.e. a real break.
    * ``unexpected`` -- registered but undeclared. The registry-consistency
      invariant already forbids this (C0, "registry IS the menu").

    Pure and side-effect-free. It REPORTS; ``tests/test_frame_contract.py``
    turns a non-empty result into a CI failure, and production logs it rather
    than refusing to boot -- a box with one broken adapter must still render
    with the other thirty.
    """
    expected = frozenset(CAPABILITIES)
    registered = frozenset(all_engine_names())
    return {
        "missing": tuple(sorted(expected - registered)),
        "unexpected": tuple(sorted(registered - expected)),
    }
