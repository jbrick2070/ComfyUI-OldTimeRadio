"""Typed video-platform schemas (A-Seam AS-4, built in W0 before adapters).

Every cross-boundary contract is a Pydantic v2 model with
``model_config = ConfigDict(extra="forbid")`` (unknown-key rejection) +
family-specific required-input rules, mirroring the shipped audio
``EngineProfile`` / ``ResolvedVoiceRequest`` discipline. Adapters validate
against these from the first family forward, so a drifted request shape fails
loudly at the boundary instead of silently rendering the wrong thing.

Pydantic is a base dependency (not a model framework) so importing this module
does not violate the cold-import invariant; the registry itself stays
pydantic-free and only adapters / the lock node import these.

The request-level ``required_inputs`` vocabulary is shared verbatim with
``nodes/_otr_shared/role_compat.py`` (tokens: ``text_prompt`` / ``init_image`` /
``audio_ref`` / ``base_clip_ref``) so the director's role filter and the
request's family validator agree on one vocabulary.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Single-sourced vocabularies
# ---------------------------------------------------------------------------

#: The video families (seven non-3D + ``character_3d`` + the LTX-AV
#: ``audio_conditioned_video`` lane for audio-reactive scene motion / music).
FAMILIES: tuple = (
    "audio_driven_face",
    "lipsync_overlay",
    "image_to_video",
    "text_to_video",
    "static_image_gen",
    "static_motion",
    "abstract",
    "character_3d",
    "audio_conditioned_video",
)

#: Request-level required-input tokens (shared with role_compat).
REQUIRED_INPUT_TOKENS: tuple = (
    "text_prompt",
    "init_image",
    "audio_ref",
    "base_clip_ref",
)

#: Hard request-level requirements per family. ``static_image_gen`` requires
#: text_prompt OR init_image (handled specially); the no-input families
#: (static_motion / abstract) accept optional asset_refs but require nothing.
#: ``character_3d`` requires an audio_ref (speech) + init_image (portrait/mesh)
#: so both the 3D pipeline and the audio-driven lip-sync chain can be built.
FAMILY_REQUIRED_INPUTS: dict = {
    "audio_driven_face": ("audio_ref", "init_image"),
    "lipsync_overlay": ("base_clip_ref", "audio_ref"),
    "image_to_video": ("init_image",),
    "text_to_video": ("text_prompt",),
    "static_image_gen": (),   # special-cased: text_prompt OR init_image
    "static_motion": (),
    "abstract": (),
    "character_3d": ("audio_ref", "init_image"),
    # LTX-AV music/scene lane: audio-reactive motion from a text prompt + the
    # per-beat slice of the frozen master (no portrait needed -- that is the
    # talk lane, which reuses audio_driven_face).
    "audio_conditioned_video": ("text_prompt", "audio_ref"),
}

# Guard: FAMILIES and FAMILY_REQUIRED_INPUTS must stay in sync (Phase 3 spec).
assert set(FAMILIES) == set(FAMILY_REQUIRED_INPUTS), (
    "schemas.py FAMILIES / FAMILY_REQUIRED_INPUTS out of sync: "
    + str(set(FAMILIES) ^ set(FAMILY_REQUIRED_INPUTS))
)


class _Forbid(BaseModel):
    """Common base: reject unknown keys (extra='forbid') everywhere."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Request sub-models
# ---------------------------------------------------------------------------


class SeedBundle(_Forbid):
    episode_seed: int = 0
    request_seed: int = 0
    variation_seed: int = 0


class Timing(_Forbid):
    source_line_ids: list[str] = Field(default_factory=list)
    target_duration_s: float = 0.0
    #: Integer frame count is the timing AUTHORITY (kills float-seconds drift).
    target_frame_count: int = 0
    start_s: float = 0.0


class Canvas(_Forbid):
    w: int
    h: int
    fps: int
    #: How a non-matching init aspect is handled: pad | crop | fit (never an
    #: implicit stretch -- canonicalize enforces it).
    aspect_policy: str = "pad"


class Strategy(_Forbid):
    #: ``unique_per_beat`` (one clip per beat) | ``pool_n_loop`` (N unique
    #: clips tiled across the whole other-beats timeline).
    mode: str = "unique_per_beat"
    pool_key: Optional[str] = None
    pool_size: Optional[int] = None


class Quality(_Forbid):
    steps: Optional[int] = None
    guidance: Optional[float] = None
    motion_strength: Optional[float] = None


class Policy(_Forbid):
    mute_generated_audio: bool = True
    allow_auto_fallback: bool = True
    strict_sync_required: bool = False


class AudioRef(_Forbid):
    path: str
    content_hash: str = ""


class VideoRequest(_Forbid):
    """One render request for one shot (the unit the render node consumes).

    ``timing.target_frame_count`` (integer) is the timing authority; the
    compositor uses frame counts, not float seconds. The model handles
    (HuMo's MODEL+CLIP+VAE+AUDIO_ENCODER, Wan's two-expert MoE, etc.) are
    adapter-internal -- NEVER request fields. The request carries only
    paths/refs + creative text + timing + seeds + policy.
    """

    request_id: str
    shot_id: str
    role: str
    family_hint: str
    profile_id: str
    text_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    asset_refs: dict[str, str] = Field(default_factory=dict)
    conditioning_refs: dict[str, str] = Field(default_factory=dict)
    audio_ref: Optional[AudioRef] = None
    base_clip_ref: Optional[str] = None
    timing: Timing
    canvas: Canvas
    strategy: Strategy = Field(default_factory=Strategy)
    seed_bundle: SeedBundle = Field(default_factory=SeedBundle)
    quality: Quality = Field(default_factory=Quality)
    policy: Policy = Field(default_factory=Policy)
    locked_against_audio_rev: Optional[str] = None
    #: Trace-only observability stamps (round-5 F2 prompt provenance +
    #: still-spine ST-4 init provenance). NEVER conditioning, never hashed
    #: into request identity; ``run_episode`` copies these onto trace rows.
    #: A REAL field (W7-pre builder migration) because the model is
    #: extra="forbid" -- the old top-level ``_prompt_*``/``_init_*`` keys
    #: made every built request schema-invalid.
    observability: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_family_required_inputs(self) -> "VideoRequest":
        fam = self.family_hint
        if fam not in FAMILIES:
            raise ValueError(
                f"VideoRequest.family_hint '{fam}' is not a known family "
                f"{FAMILIES}"
            )
        present = self._present_input_tokens()
        if fam == "static_image_gen":
            if not ({"text_prompt", "init_image"} & present):
                raise ValueError(
                    "static_image_gen requires text_prompt or "
                    "asset_refs{init_image}"
                )
            return self
        missing = [t for t in FAMILY_REQUIRED_INPUTS[fam] if t not in present]
        if missing:
            raise ValueError(
                f"VideoRequest for family '{fam}' is missing required "
                f"input(s): {missing}"
            )
        return self

    def _present_input_tokens(self) -> set:
        present = set()
        if self.text_prompt:
            present.add("text_prompt")
        if "init_image" in self.asset_refs:
            present.add("init_image")
        if self.audio_ref is not None:
            present.add("audio_ref")
        if self.base_clip_ref:
            present.add("base_clip_ref")
        return present


# ---------------------------------------------------------------------------
# Output + plan models
# ---------------------------------------------------------------------------


class CanonicalClip(_Forbid):
    """A normalized, ALWAYS-SILENT clip (or still/directory handoff).

    Integer ``frame_count`` is the timing authority; color fields are recorded
    and ffprobe-asserted (normalized to bt709 / yuv420p); ``alpha`` is explicit
    and validated. ``has_audio`` is always False -- only ``OTR_MasterAudioMux``
    may emit audio (invariant V-1).
    """

    clip_id: str
    type: str = "video"            # video | still | directory
    path: str
    container: str = "mp4"
    codec: str = "h264"
    pixel_format: str = "yuv420p"
    w: int = 0
    h: int = 0
    fps: int = 0
    frame_count: int = 0
    duration_s: float = 0.0
    has_audio: bool = False
    color_primaries: str = "bt709"
    transfer: str = "bt709"
    matrix: str = "bt709"
    range: str = "tv"
    alpha: str = "none"            # none | straight | premultiplied
    engine_id: str = ""
    profile_id: str = ""
    family: str = ""
    request_hash: str = ""
    asset_hashes: list[str] = Field(default_factory=list)
    qc: dict = Field(default_factory=dict)


class AdapterDescriptor(_Forbid):
    """Static metadata an adapter declares (cold-import-safe, no weights)."""

    engine_id: str
    adapter_api_version: int = 1
    family: str
    roles: tuple = ()
    required_inputs: tuple = ()
    optional_inputs: tuple = ()
    prepare_semantics: str = "none"
    canonicalizer: Optional[str] = None
    supports_audio_output: bool = False
    provides: tuple = ()
    allowed_consumers: tuple = ()
    provider_vram_tier: str = "radio"
    preprocess_manifest: dict = Field(default_factory=dict)
    dependency_manifest: dict = Field(default_factory=dict)
    #: 3D capability flag (3D plan section 3): True for engines that consume a
    #: clean front-facing MESH PORTRAIT (the character_3d lane). The
    #: ImageDirector granularity lock reads THIS field -- never a hard-coded
    #: family check -- so custom 3D engines declare it explicitly. A REAL
    #: schema field because the model is extra="forbid" (an ad-hoc key would
    #: be rejected).
    requires_mesh_portrait: bool = False


class VideoProfileRow(_Forbid):
    """One row of ``video_profiles.yaml`` (engine x preset policy data)."""

    profile_id: str
    engine_id: str
    family: str
    default_params: dict = Field(default_factory=dict)
    prepare_semantics: str = "none"
    provider_vram_tier: str = "radio"
    dependency_manifest: dict = Field(default_factory=dict)
    preprocess_manifest: dict = Field(default_factory=dict)
    license: dict = Field(default_factory=dict)
    #: Mirrors AdapterDescriptor.requires_mesh_portrait (3D plan section 3) so
    #: profile rows carry the 3D capability without an ad-hoc key.
    requires_mesh_portrait: bool = False


class ExecutionGroup(_Forbid):
    group_id: str
    kind: str = "consumer"         # provider | consumer
    engine_id: str = ""
    profile_id: str = ""
    depends_on: list[str] = Field(default_factory=list)
    produces_base_for: list[str] = Field(default_factory=list)


class ShotRow(_Forbid):
    shot_id: str
    source_line_ids: list[str] = Field(default_factory=list)
    group_id: str = ""
    engine_id: str = ""
    profile_id: str = ""
    family: str = ""
    strategy: dict = Field(default_factory=dict)
    request_seed: int = 0
    target_frame_count: int = 0
    render_request_hash: str = ""
    binding_hash: str = ""
    cache_keys: dict = Field(default_factory=dict)
    degradation_trail: list[str] = Field(default_factory=list)
    creative: dict = Field(default_factory=dict)


class VideoLedgerSection(_Forbid):
    """The single ``ledger['video']`` section ``OTR_ShotLock`` stamps."""

    video_revision: int = 1
    canonical_canvas: dict = Field(default_factory=dict)
    fps: int = 25
    locked_against_audio_rev: Optional[str] = None
    execution_groups: list[ExecutionGroup] = Field(default_factory=list)
    roles: dict = Field(default_factory=dict)
    shots: list[ShotRow] = Field(default_factory=list)
    #: A-S7 durable audit trail: one record per in-render LOUD fallback swap
    #: (see nodes/_otr_shared/retry_taxonomy.build_fallback_decision). Appended
    #: at the SAME video_revision -- a fallback restamp is a within-revision
    #: degradation, never a re-lock. Loose dicts (like ``roles``), not a nested
    #: model, so the render node can append without a schema round-trip.
    runtime_fallback_decisions: list = Field(default_factory=list)
