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

#: The video families (the ``character_3d`` token was RETIRED with its family
#: 2026-08-23, lean-mean order 4 -- zero live declarers; ``audio_conditioned_video``
#: is the LTX-AV lane for audio-reactive scene motion / music).
FAMILIES: tuple = (
    "audio_driven_face",
    "lipsync_overlay",
    "image_to_video",
    "text_to_video",
    "static_image_gen",
    "static_motion",
    "abstract",
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
#: (character_3d's row -- audio_ref + init_image -- retired with the family.)
FAMILY_REQUIRED_INPUTS: dict = {
    "audio_driven_face": ("audio_ref", "init_image"),
    "lipsync_overlay": ("base_clip_ref", "audio_ref"),
    "image_to_video": ("init_image",),
    "text_to_video": ("text_prompt",),
    "static_image_gen": (),   # special-cased: text_prompt OR init_image
    "static_motion": (),
    "abstract": (),
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
    #: ``unique_per_beat`` (one clip per beat) -- the only live mode since the
    #: the ``pool_n_loop`` pooling was removed 2026-07-01
    #: (rip-sfx-broll). pool_key/pool_size are retained schema slots (always
    #: None from ShotLock) so old serialized requests still parse.
    mode: str = "unique_per_beat"
    pool_key: Optional[str] = None
    pool_size: Optional[int] = None


class Quality(_Forbid):
    steps: Optional[int] = None
    guidance: Optional[float] = None
    motion_strength: Optional[float] = None


class Policy(_Forbid):
    mute_generated_audio: bool = True
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
    #: THE HONESTY RECEIPTS (no-mirror enforcement, 2026-08-06). How many of
    #: this clip's frames the engine actually RENDERED, and how the rest got
    #: there. ``native_frame_count`` is always the EMITTED count -- what the
    #: adapter handed back -- never a pre-trim decoded length, so it can never
    #: exceed ``frame_count``. ``extension_mode`` is drawn from
    #: ``acceptance.EXTENSION_MODES``; ``"none"`` means every delivered frame was
    #: rendered, and it is the only mode that may ship.
    #:
    #: DECLARED HERE EVEN THOUGH THE RENDER PATH PASSES PLAIN DICTS. This class
    #: is ``extra="forbid"``, so the day anything validates an adapter's return
    #: through it, an enriched clip carrying these two keys would be REJECTED as
    #: unknown fields -- the receipts would be dropped by the schema that exists
    #: to protect them. Adding them costs nothing today and removes a trap that
    #: would otherwise fire far from its cause.
    native_frame_count: Optional[int] = None
    extension_mode: Optional[str] = None
    #: THE CADENCE AND DELIVERY RECEIPTS (Ghost Signal, 2026-08-22). Declared
    #: here for the same reason the two above are: this model is extra="forbid",
    #: so the day anything validates an adapter's return through it, an enriched
    #: clip carrying these keys would be REJECTED as unknown fields -- the
    #: receipts would be dropped by the schema that exists to protect them.
    #:
    #: ALL SIX ARE OPTIONAL AND ABSENCE IS LOAD-BEARING. A legacy row that never
    #: carried them must keep behaving byte-for-byte, so every projection copies
    #: them present-key-only rather than stamping six nulls onto history.
    #:
    #: `delivery_scale_mode` is how a lane DECLARES the enlargement it needs, so
    #: the composite never has to ask which engine produced a row.
    #: `cadence_mode` plus the three counts are the honest accounting of a RATE
    #: CONVERSION: `model_frame_count` is every frame actually requested from and
    #: decoded by the model, `cadence_source_frame_count` is the unique prefix
    #: the selector retained, and their difference exposes source frames
    #: generated only to satisfy a node's structural minimum. `cadence_tail_trim`
    #: covers ONLY the final hold-2 surplus. Keeping the scopes apart is what
    #: stops `native_frame_count` being misread as "N distinct diffusion
    #: samples" -- on a hold-2 lane it is not.
    delivery_scale_mode: Optional[str] = None
    cadence_mode: Optional[str] = None
    cadence_source_frame_count: Optional[int] = None
    cadence_delivered_frame_count: Optional[int] = None
    cadence_tail_trim: Optional[int] = None
    model_frame_count: Optional[int] = None


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
    # (requires_mesh_portrait was here -- retired 2026-08-23 with the
    # character_3d family, lean-mean order 4. Zero declarers, zero carriers in
    # any YAML/JSON row, and its only reader -- the ImageDirector granularity
    # lock -- is removed in the same change. extra="forbid" means a stale row
    # carrying the key now fails LOUD, which is correct: that row was written
    # against a capability that no longer exists.)


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
    # (the requires_mesh_portrait mirror was here -- retired with the field
    # above, 2026-08-23.)


class ExecutionGroup(_Forbid):
    group_id: str
    kind: str = "consumer"         # provider | consumer
    engine_id: str = ""
    profile_id: str = ""
    depends_on: list[str] = Field(default_factory=list)
    produces_base_for: list[str] = Field(default_factory=list)


class ShotRow(_Forbid):
    """One row of ``ledger['video']['shots']``, as ``OTR_ShotLock`` stamps it.

    B4 (2026-07-27): this model declares ``extra="forbid"`` and was missing
    eight fields the producers genuinely write, so ``ShotRow(**real_row)``
    raised on EVERY real ledger -- which made the "live safety net" other docs
    cite unable to validate a single shipped episode, and would have turned any
    future ``VideoLedgerSection.model_validate(ledger["video"])`` into a
    hard-fail at whatever boundary first tried it.

    The field list below was derived MECHANICALLY from the producers, not from
    the bug report: the report also named ``beat_id``, and no producer stamps
    one on a shot row -- ``source_line_ids`` carries the beat. It missed
    ``jump_still_requests`` and ``motion_clause``, which are stamped.

    ABSENCE IS LOAD-BEARING for four of these, so they default to ``None``
    rather than to an empty container: an unregistered engine gets NO
    ``coverage_plan`` (and the render path then behaves as it did before chunk
    3b), a ``coverage_contract`` is stamped ONLY when the tier ceiling actually
    narrowed something (``render_driver.assert_coverage_plans`` re-derives it
    and refuses on any difference, so "stamped empty" and "never stamped" must
    not collapse), a missing ``motion_clause`` means the pass never ran, which
    is not the same as a fallback clause, and a missing ``frame_bounded`` means
    nobody ASKED -- see its own note below.
    """

    shot_id: str
    source_line_ids: list[str] = Field(default_factory=list)
    group_id: str = ""
    #: The shot's video ROLE, stamped explicitly since 2026-06-10 -- the render
    #: driver's role-scoped behaviours read it rather than parsing group_id.
    role: str = ""
    #: The NORMALIZED char_id (round 5 F5), so the portrait join never depends
    #: on the raw line scheme.
    char_id: str = ""
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
    #: Timeline position, carried by the ROW only for synthetic beats, which
    #: have no ledger LINE to read it from. dur_s is genuinely Optional: the
    #: producer forwards ``beat.get("dur_s")`` and that can be absent.
    start_s: float = 0.0
    dur_s: Optional[float] = None
    #: The durable coverage plan (chunk 3b) and its sibling ceiling receipt
    #: (B3). See the class docstring on why these are None, not {}.
    coverage_plan: Optional[dict] = None
    coverage_contract: Optional[dict] = None
    #: Per-segment still requests for a jump-cut beat (chunk 4), minted only
    #: for a lane that actually consumes a still. Empty == none owed.
    jump_still_requests: list[dict] = Field(default_factory=list)
    #: The per-beat motion clause object (text/model/fallback/source_hash).
    motion_clause: Optional[dict] = None
    #: Whether this shot's engine has a render ceiling a beat can OVERFLOW, as
    #: ``frame_contract.can_split`` answered it AT FREEZE TIME (no-mirror 7.3).
    #: The acceptance grader reads this and nothing else to decide whether the
    #: shot owes native-frame evidence, because that module imports nothing and
    #: must never consult a live registry whose clock has moved on.
    #:
    #: THREE-VALUED ON PURPOSE. ``True`` obliges the shot to prove its frames;
    #: ``False`` records that we asked and the engine is unbounded; ``None``
    #: means nobody asked -- an unregistered or unbuildable engine, or any
    #: ledger frozen before this stamp existed. Collapsing ``None`` into
    #: ``False`` would turn "unknown" into the claim "we checked", and
    #: collapsing it into ``True`` would indict every legacy episode for
    #: missing a receipt that did not exist when it rendered.
    frame_bounded: Optional[bool] = None
    #: THE DURABLE GHOST IDENTITY (2026-08-22), stamped once by
    #: `otr_shot_lock.build_execution_plan` for a `character_video` beat whose
    #: resolved engine is a prompt-owned lane, and read by the pure composer.
    #:
    #: Optional, and None means "this beat never needed one" -- a non-Ghost
    #: episode must not acquire a new seed or style requirement just because
    #: this field exists.
    subject_sigil: Optional[str] = None
    #: THE AUTHORED GHOST PROMPT v2 OBJECT (2026-08-22), stamped by
    #: `otr_shot_lock.build_execution_plan` for every beat whose resolved engine
    #: declares the Ghost prompt profile, and read by the render driver as the
    #: whole content authority for that beat.
    #:
    #: THREE-STATE, exactly like `frame_bounded` above and for the same reason.
    #: A valid object means Prompt v2; ABSENCE means a legacy row frozen before
    #: this sprint, which the driver composes through the explicit v1
    #: compatibility path; and a present-but-malformed object is NEITHER -- it
    #: fails closed rather than downgrading, because a silent v1 downgrade would
    #: be indistinguishable from a healthy legacy replay while reproducing the
    #: name-leaking fragment v2 exists to remove.
    #:
    #: Deliberately a plain dict on this schema: `ghost_signal_author`
    #: `validate_ghost_prompt_object` is the field-set authority and it is
    #: importable with nothing installed, so the contract lives in one place
    #: rather than being half-restated here as a second model that can drift.
    ghost_prompt: Optional[dict] = None


class VideoLedgerSection(_Forbid):
    """The single ``ledger['video']`` section ``OTR_ShotLock`` stamps."""

    video_revision: int = 1
    canonical_canvas: dict = Field(default_factory=dict)
    fps: int = 25
    execution_groups: list[ExecutionGroup] = Field(default_factory=list)
    roles: dict = Field(default_factory=dict)
    shots: list[ShotRow] = Field(default_factory=list)
    #: RETAINED SCHEMA SLOT, STAMPED NEVER (Sprint A rip, 2026-07-02): the
    #: fallback machinery is deleted, so nothing appends here anymore. The slot
    #: survives (always []) so old serialized ledgers still parse -- no schema
    #: churn (A5).
    runtime_fallback_decisions: list = Field(default_factory=list)
