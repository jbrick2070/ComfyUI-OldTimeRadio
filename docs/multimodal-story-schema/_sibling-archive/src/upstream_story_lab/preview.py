"""Prompt preview rendering + pack-driven leakage scanning.

Forbidden terms are pack content (JSON), scanned against RENDERED previews;
they are never rendered into live prompts (models copy negated terms - the
metadata-not-prompt rule, kept from v1).
"""

from __future__ import annotations

from .contracts import LedgerWritingSpec


def render_prompt_preview(spec: LedgerWritingSpec) -> str:
    """A flat preview of the prose the ledger-writing stages would see."""

    profile = spec.prompt_profile
    parts = [
        profile.outline_system_prompt or "",
        profile.pitch_room_system_prompt or "",
        profile.story_select_system_prompt or "",
        profile.dramatic_state_system_prompt or "",
        profile.coda_system_prompt or "",
        profile.title_system_prompt or "",
        f"Story form: {profile.story_form_label}",
        f"Source material: {profile.source_material_label}",
        f"Develop verb: {profile.source_develop_verb}",
        f"Grounding: {profile.source_grounding_label}",
        f"Line instruction: {profile.line_grounding_instruction}",
        "Tone guardrails: " + "; ".join(profile.tone_guardrails),
        # Forbidden patterns stay metadata: count only, never the phrases.
        f"Forbidden pattern count: {len(profile.forbidden_plot_patterns)}",
    ]
    return "\n".join(p for p in parts if p)


def render_visual_preview(spec: LedgerWritingSpec) -> str:
    """A flat preview of the visual language a policy would emit."""

    policy = spec.visual_policy
    parts = [
        policy.positive_tail,
        policy.image_grade_tail,
        policy.broadcast_tail,
        policy.era_tail,
        policy.announcer_visual_subject,
        policy.music_visual_subject,
        policy.scene_open_subject,
        policy.character_portrait_style,
        policy.character_scene_style,
        *policy.motion_prompts.values(),
    ]
    return "\n".join(p for p in parts if p)


def find_forbidden_terms(text: str, terms: list[str]) -> list[str]:
    """Case-insensitive containment scan; the term list comes from the pack
    or policy JSON, never from a Python constant."""

    low = (text or "").lower()
    return [term for term in terms if term.lower() in low]


def scan_story_leakage(registry, spec: LedgerWritingSpec) -> list[str]:
    """Leakage terms found in the story prompt preview. The term list is the
    ACTIVE PACK's forbidden_leakage_terms (pack JSON metadata) - scanned for
    every bank that declares terms, not just media archive."""

    pack = registry.pack(
        spec.source_bank_id, spec.story_model_id, spec.story_pipeline_id
    )
    return find_forbidden_terms(
        render_prompt_preview(spec), list(pack.forbidden_leakage_terms)
    )


def scan_visual_leakage(spec: LedgerWritingSpec) -> list[str]:
    """Forbidden visual terms found in the rendered visual preview (the term
    list is the policy's own forbidden_terms)."""

    return find_forbidden_terms(
        render_visual_preview(spec), list(spec.visual_policy.forbidden_terms)
    )
