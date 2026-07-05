"""Strict contracts for the upstream story lab v2 (transplant workspace).

JSON owns content and configuration. Python owns validation, routing,
execution, and fail-loud errors. No fallbacks. No hidden models or engines.

Vocabulary lock (kibitz r1 2026-07-02): `story_model` is the dramatic/tonal
lane; LLM engines are SLOT MODELS (creative/technical) resolved at runtime by
the production node widgets. The lab declares a slot plan, never engine ids.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "v2.0"

#: Slot roles an LLM-touching pass may declare. Engine ids are runtime-owned.
SlotRole = Literal["creative", "technical"]

#: Prompt template seams a story pack may own (kibitz r1: templates only -
#: generated content fields like casting_brief are interpreter OUTPUT, not
#: seams). Keys match v1 `prompt_stages` for add-only migration.
#:
#: Phase A split (2026-07-04): the seams the production writer consumes at
#: runtime live in PRODUCTION_TEMPLATE_SEAMS; the four
#: simple_4_prompt_experimental pipeline seams live in
#: EXPERIMENTAL_PIPELINE_SEAMS. Callers that need "any allowed pack seam"
#: (the SourceBankSpec / StoryPack / PassDecl validators) use
#: ALL_TEMPLATE_SEAMS. TEMPLATE_SEAMS stays as a back-compat alias so any
#: external importer keeps working.
PRODUCTION_TEMPLATE_SEAMS = (
    "outline_system",
    "pitch_room_system",
    "story_select_system",
    "dramatic_state_system",
    "line_grounding",
    "coda_system",
    "title_system",
    "style_pick_inventor",
    "style_pick_chooser",
    "style_pick_chooser_user_template",
    # Phase A additions (2026-07-04): the outline macro/phase/beat sub-passes
    # and the line composer system prompt. Production defaults to its Python
    # literals; a pack that supplies one of these overrides the default.
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
    "line_composer_system",
)
EXPERIMENTAL_PIPELINE_SEAMS = (
    # simple_4_prompt_experimental pass prompts (pipeline-executable seams;
    # adaptive-cleanup was CUT to documentation-only in kibitz r1).
    "pass_1_creative_story",
    "pass_2_creative_ledger_fill",
    "pass_3_technical_schema_cleanup",
    "pass_4_technical_ledger_audit",
)
ALL_TEMPLATE_SEAMS = PRODUCTION_TEMPLATE_SEAMS + EXPERIMENTAL_PIPELINE_SEAMS
#: Back-compat alias -- validators below at :185, :232, :351 keep using this name.
TEMPLATE_SEAMS = ALL_TEMPLATE_SEAMS

#: Label variables usable in ANY seam template (substituted from the resolved
#: profile). Kibitz r2 (Codex, confirmed): some production templates carry
#: RUNTIME variables filled at the call site - those are declared per seam
#: below, sourced from production_mirror/nodes/_otr_style_picker.py:301,:334.
LABEL_TEMPLATE_VARIABLES = frozenset(
    {
        "story_form_label",
        "source_material_label",
        "source_develop_verb",
        "source_grounding_label",
        "key_terms_label",
        "close_brief_label",
        "title_form_label",
    }
)
SEAM_RUNTIME_VARIABLES: dict[str, frozenset[str]] = {
    "style_pick_inventor": frozenset(
        {"n_required", "seed_sample_block", "article_excerpt"}
    ),
    "style_pick_chooser": frozenset(
        {"article_excerpt", "candidates_block", "story_summary"}
    ),
    "style_pick_chooser_user_template": frozenset(
        {"article_excerpt", "candidates_block", "story_summary"}
    ),
}


def allowed_seam_variables(seam: str) -> frozenset[str]:
    """Per-seam template variable allowlist (labels + declared runtime)."""

    return LABEL_TEMPLATE_VARIABLES | SEAM_RUNTIME_VARIABLES.get(seam, frozenset())

#: Motion-prompt role keys allowed in a visual policy. Pinned to the
#: production vocabulary (render_driver._LTX_MOTION_PROMPT_BY_ROLE) plus the
#: forward-declared lipsync/character keys the visual stage will define.
#: Dead roles (sfx, scene_broll, background_abstract) must stay dead.
ALLOWED_MOTION_ROLE_KEYS = frozenset(
    {"announcer", "music_open", "music_close", "music_inter"}
)

CODA_MODES = ("real_news_report", "archive_source_note", "source_attribution", "none")


class SourceMaterialPacket(BaseModel):
    """Raw source material before interpretation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    packet_version: int = 1
    source_bank_id: str
    source_mode: str = "fixture"
    source_kind: str = ""
    source_label: str = ""
    rights_status: Literal[
        "unknown", "public_domain", "licensed", "fair_use_research"
    ] = "unknown"
    source_title: str = ""
    source_author: str = ""
    source_url: str = ""
    source_hash: str = ""
    source_text_ref: str = ""
    source_summary: str = ""
    raw_text: str = ""
    # Fixture-brief block (kibitz FABLE review SHOULD-FIX: fixture briefs are
    # content and live in the packet JSON, not in Python).
    fixture_casting_brief: str = ""
    fixture_script_brief: str = ""
    fixture_close_brief: str = ""
    fixture_key_terms: list[str] = Field(default_factory=list)
    fixture_fidelity_rules: list[str] = Field(default_factory=list)


class PublicDomainSourceManifest(BaseModel):
    """Source-folder manifest for public-domain books, comics, plays, etc.

    PD packets are FILE-BACKED: `text_files`/`image_files` are relative
    POSIX-style paths inside the manifest folder; absolute paths and `..`
    components are rejected by the registry (v1 rule preserved).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    source_id: str
    source_bank_id: Literal["public_domain_story"] = "public_domain_story"
    source_kind: str
    title: str
    author: str = ""
    publication_year: int | None = None
    rights_status: Literal["public_domain"] = "public_domain"
    source_url: str = ""
    text_files: list[str] = Field(default_factory=list)
    image_files: list[str] = Field(default_factory=list)
    adaptation_mode: str
    required_fidelity: list[str] = Field(default_factory=list)


class BankDefaults(BaseModel):
    """Bank-level label/coda defaults; packs override single-level."""

    model_config = ConfigDict(extra="forbid")

    story_form_label: str = ""
    source_material_label: str = ""
    source_develop_verb: str = ""
    source_grounding_label: str = ""
    key_terms_label: str = "KEY TERMS"
    close_brief_label: str = "source note"
    title_form_label: str = ""
    coda_mode: str = ""


class SourceBankSpec(BaseModel):
    """One source bank, declared in fixtures/banks.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    source_bank_id: str
    label: str
    dropdown_label: str = ""
    source_kind: str = ""
    #: Named Python behavior bindings, resolved through an explicit
    #: allowlist registry. Unknown binding = hard error.
    interpreter: str
    #: Fetcher binding; empty string means packet-driven only (v1 scope:
    #: only science_news declares a fetcher).
    fetcher: str = ""
    default_story_model: str
    default_visual_style: str
    default_story_pipeline: str = "legacy_many_pass"
    defaults: BankDefaults = Field(default_factory=BankDefaults)
    #: Seams every pack of this bank MUST supply (no invented default prose).
    required_seams: list[str] = Field(default_factory=list)
    runnable: bool = True
    guide_ref: str = ""

    @model_validator(mode="after")
    def _seams_known_and_runnable_bound(self) -> "SourceBankSpec":
        unknown = [s for s in self.required_seams if s not in TEMPLATE_SEAMS]
        if unknown:
            raise ValueError(
                f"bank {self.source_bank_id!r} requires unknown seams: {unknown}; "
                f"known seams: {sorted(TEMPLATE_SEAMS)}"
            )
        if self.runnable and not self.interpreter.strip():
            raise ValueError(
                f"bank {self.source_bank_id!r} is runnable but declares no "
                "interpreter binding"
            )
        return self


class StoryPack(BaseModel):
    """Cloneable prompt/content pack for one (bank, model, pipeline) lane.

    v1-compatible: `prompt_stages` keeps its name and v1 keys; v2 adds
    labels/coda fields so the pack is the single content source the resolved
    StoryPromptProfile is built from (no Python prose anywhere).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    source_bank_id: str
    story_model_id: str
    story_pipeline_id: str = "legacy_many_pass"
    label: str
    status: Literal["ready_fixture", "experimental", "not_implemented"] = (
        "ready_fixture"
    )
    prompt_stages: dict[str, str] = Field(default_factory=dict)
    #: Label overrides (single-level over bank defaults; empty = inherit).
    labels: BankDefaults = Field(default_factory=BankDefaults)
    coda_mode: str = ""
    coda_examples: list[str] = Field(default_factory=list)
    outline_rules_extra: str = ""
    examples: list[str] = Field(default_factory=list)
    tone_guardrails: list[str] = Field(default_factory=list)
    forbidden_plot_patterns: list[str] = Field(default_factory=list)
    forbidden_leakage_terms: list[str] = Field(default_factory=list)
    source_requirements: list[str] = Field(default_factory=list)
    ledger_validation_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _stages_known_and_coda_valid(self) -> "StoryPack":
        unknown = sorted(set(self.prompt_stages) - set(TEMPLATE_SEAMS))
        if unknown:
            raise ValueError(
                f"pack {self.story_model_id!r} has unknown prompt_stages keys: "
                f"{unknown}; allowed: {sorted(TEMPLATE_SEAMS)}"
            )
        if self.coda_mode and self.coda_mode not in CODA_MODES:
            raise ValueError(
                f"pack {self.story_model_id!r} coda_mode {self.coda_mode!r} "
                f"not in {CODA_MODES}"
            )
        return self


class StoryPromptProfile(BaseModel):
    """RESOLVED view of (bank defaults + pack). Field names are locked to the
    v1/pre-SFX-r4 vocabulary because production OutlineRequest/adapter code
    is planned against them. Built by profiles.resolve_profile(); contains
    no Python-authored prose."""

    model_config = ConfigDict(extra="forbid")

    source_bank_id: str
    story_model_id: str
    story_form_label: str
    source_material_label: str
    source_develop_verb: str
    source_grounding_label: str
    key_terms_label: str = "KEY TERMS"
    close_brief_label: str = "source note"
    coda_mode: Literal[
        "real_news_report", "archive_source_note", "source_attribution", "none"
    ]
    title_form_label: str
    line_grounding_instruction: str
    outline_rules_extra: str = ""
    tone_guardrails: list[str] = Field(default_factory=list)
    forbidden_plot_patterns: list[str] = Field(default_factory=list)
    outline_system_prompt: str | None = None
    pitch_room_system_prompt: str | None = None
    story_select_system_prompt: str | None = None
    dramatic_state_system_prompt: str | None = None
    coda_system_prompt: str | None = None
    coda_examples: list[str] = Field(default_factory=list)
    title_system_prompt: str | None = None
    style_picker_inventor_system_prompt: str | None = None
    style_picker_chooser_system_prompt: str | None = None
    style_picker_chooser_user_template: str | None = None
    # Phase A additions (2026-07-04): the four new production seams. None
    # means "no pack override -- production falls back to its Python literal
    # at _otr_outline._MACRO_SYSTEM_PROMPT / _PHASE_SYSTEM_PROMPT /
    # _BEAT_SYSTEM_PROMPT / _otr_line_composer._SYSTEM_PROMPT".
    outline_macro_system_prompt: str | None = None
    outline_phase_system_prompt: str | None = None
    outline_beat_system_prompt: str | None = None
    line_composer_system_prompt: str | None = None


class StoryInputPacket(BaseModel):
    """Interpreted story material ready for a ledger-writing spec.

    casting_brief/script_brief/close_brief/key_terms are GENERATED CONTENT
    FIELDS (interpreter output), never pack templates (kibitz r1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    packet_version: int = 1
    source_bank_id: str
    story_model_id: str
    source_label: str = ""
    casting_brief: str = ""
    script_brief: str = ""
    close_brief: str = ""
    key_terms: list[str] = Field(default_factory=list)
    source_fidelity_rules: list[str] = Field(default_factory=list)
    adaptation_trace: dict[str, Any] = Field(default_factory=dict)
    source_material: SourceMaterialPacket


class VisualStylePolicy(BaseModel):
    """Still/video rendering policy; independent axis from source/story."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    style_id: str
    label: str = ""
    positive_tail: str = ""
    image_grade_tail: str = ""
    broadcast_tail: str = ""
    era_tail: str = ""
    allow_radio_tails: bool = True
    forbidden_terms: list[str] = Field(default_factory=list)
    announcer_visual_subject: str = ""
    music_visual_subject: str = ""
    scene_open_subject: str = ""
    character_portrait_style: str = ""
    character_scene_style: str = ""
    motion_prompts: dict[str, str] = Field(default_factory=dict)
    ledger_directives: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _motion_keys_allowed(self) -> "VisualStylePolicy":
        unknown = sorted(set(self.motion_prompts) - ALLOWED_MOTION_ROLE_KEYS)
        if unknown:
            raise ValueError(
                f"visual style {self.style_id!r} motion_prompts has unknown "
                f"role keys {unknown}; allowed: {sorted(ALLOWED_MOTION_ROLE_KEYS)} "
                "(dead roles sfx/scene_broll/background_abstract must stay dead)"
            )
        return self


class PassDecl(BaseModel):
    """One declared pipeline pass. Descriptive passes document the
    production-native sequence; executable passes run in the lab runner."""

    model_config = ConfigDict(extra="forbid")

    pass_id: str
    slot: SlotRole
    seam_refs: list[str] = Field(default_factory=list)
    description: str = ""

    @model_validator(mode="after")
    def _seams_known(self) -> "PassDecl":
        unknown = [s for s in self.seam_refs if s not in TEMPLATE_SEAMS]
        if unknown:
            raise ValueError(
                f"pass {self.pass_id!r} references unknown seams: {unknown}"
            )
        return self


class PipelineSpec(BaseModel):
    """Declared pass structure. legacy_many_pass is DESCRIPTIVE (the
    production writer owns the real sequence; this is stamping/audit
    metadata). simple_4_prompt_experimental is lab-executable."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    story_pipeline_id: str
    label: str
    executable_in_lab: bool = False
    passes: list[PassDecl] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pass_ids_unique(self) -> "PipelineSpec":
        ids = [p.pass_id for p in self.passes]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"pipeline {self.story_pipeline_id!r} has duplicate pass ids"
            )
        return self


class ResolutionDecision(BaseModel):
    """One resolved id: what was asked, what was chosen, why, from where."""

    model_config = ConfigDict(extra="forbid")

    axis: Literal["source_bank", "story_model", "story_pipeline", "visual_style"]
    requested: str
    resolved: str
    default_applied: bool
    decided_by: str  # e.g. "explicit", "banks.json:default_story_model"


class Resolution(BaseModel):
    """Auditable record of every id decision (kibitz r1: 'auto' is data,
    not invisible behavior)."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[ResolutionDecision] = Field(default_factory=list)

    def resolved(self, axis: str) -> str:
        for d in self.decisions:
            if d.axis == axis:
                return d.resolved
        raise KeyError(f"no resolution recorded for axis {axis!r}")


class Provenance(BaseModel):
    """Content hashes tying a spec to the exact JSON bytes that shaped it."""

    model_config = ConfigDict(extra="forbid")

    production_baseline: str = ""
    lab_state_digest: str = ""
    banks_sha256: str = ""
    pipelines_sha256: str = ""
    pack_sha256: str = ""
    style_sha256: str = ""
    packet_sha256: str = ""


class LedgerWritingSpec(BaseModel):
    """Control plane for filling the existing production ledger."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    source_bank_id: str
    story_model_id: str
    story_pipeline_id: str = "legacy_many_pass"
    visual_style_id: str = "sci_fi_radio"
    source_material: SourceMaterialPacket
    story_input: StoryInputPacket
    prompt_profile: StoryPromptProfile
    visual_policy: VisualStylePolicy
    slot_plan: list[PassDecl] = Field(default_factory=list)
    resolution: Resolution = Field(default_factory=Resolution)
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _ids_are_consistent(self) -> "LedgerWritingSpec":
        checks = [
            ("visual_policy.style_id", self.visual_policy.style_id, self.visual_style_id),
            ("story_input.source_bank_id", self.story_input.source_bank_id, self.source_bank_id),
            ("story_input.story_model_id", self.story_input.story_model_id, self.story_model_id),
            ("prompt_profile.source_bank_id", self.prompt_profile.source_bank_id, self.source_bank_id),
            ("prompt_profile.story_model_id", self.prompt_profile.story_model_id, self.story_model_id),
            ("source_material.source_bank_id", self.source_material.source_bank_id, self.source_bank_id),
        ]
        for name, got, want in checks:
            if got != want:
                raise ValueError(f"{name} must match spec: {got!r} != {want!r}")
        return self


class MetaMirrors(BaseModel):
    """The two production META compatibility shapes (kibitz r2: news_used is
    an output socket production derives from the outline at
    OTR_LedgerScriptWriter._build_news_payload - the bridge never fakes it).
    Key sets are pinned in compat.py and drift-tested against the mirror."""

    model_config = ConfigDict(extra="forbid")

    news: dict[str, Any]
    news_seed: dict[str, Any]


class FetchRequest(BaseModel):
    """Runtime fetch hints for a bank fetcher binding (science-only in v1)."""

    model_config = ConfigDict(extra="forbid")

    style_hint: str = ""
    model_hint: str = ""


class BridgeArtifact(BaseModel):
    """The one frozen JSON file production consumes (translator-head output).
    Emitted only when every content/config JSON validated; never partial.

    adapter_news_article is the writer-internal article-dict input shape -
    the same dict `_resolve_inputs` synthesizes on the custom_premise branch
    (writer :1338-1346) - so packet-driven lanes enter production through an
    existing, tested seam with seed_source='bridge_packet'."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    created_utc: str
    ledger_writing_spec: LedgerWritingSpec
    meta_mirrors: MetaMirrors
    adapter_news_article: dict[str, Any]
