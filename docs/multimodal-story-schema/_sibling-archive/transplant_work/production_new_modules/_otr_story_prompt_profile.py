"""Production-side StoryPromptProfile consumption helpers.

DESTINATION: ComfyUI-OldTimeRadio/nodes/_otr_story_prompt_profile.py
STATUS: staged in the lab (transplant_work/); NOT installed yet.

Field names are locked to the pre-SFX r4 vocabulary. The helpers return
exactly the kwargs the locked transplant plan threads into production seams:

- OutlineRequest fields (labels/verbs/rules) - profile persona prompts stay
  separate (locked decision: profile holds persona overrides, OutlineRequest
  holds label fields).
- pick_style(...) override kwargs (empty string = keep science/module
  constants; non-empty = pack-owned persona).
- compose_source_coda routing by coda_mode (science implementation remains
  compose_news_coda; other modes fail loud until their composers land).
"""

from __future__ import annotations

from typing import Any

VALID_CODA_MODES = (
    "real_news_report", "archive_source_note", "source_attribution", "none",
)


class ProfileInputError(ValueError):
    """The supplied profile dict is unusable; message names the field."""


REQUIRED_PROFILE_FIELDS = (
    "source_bank_id",
    "story_model_id",
    "story_form_label",
    "source_material_label",
    "source_develop_verb",
    "source_grounding_label",
    "coda_mode",
    "title_form_label",
    "line_grounding_instruction",
)


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ProfileInputError("prompt_profile must be an object")
    missing = [
        f for f in REQUIRED_PROFILE_FIELDS if not str(profile.get(f, "")).strip()
    ]
    if missing:
        raise ProfileInputError(f"prompt_profile missing required fields: {missing}")
    if profile["coda_mode"] not in VALID_CODA_MODES:
        raise ProfileInputError(
            f"prompt_profile.coda_mode {profile['coda_mode']!r} not in "
            f"{VALID_CODA_MODES}"
        )
    return profile


def outline_request_fields(profile: dict[str, Any]) -> dict[str, Any]:
    """Kwargs for OutlineRequest (label/verb/rule fields; keyword-defaulted
    only per the locked plan - never positional)."""

    profile = validate_profile(profile)
    return {
        "story_form_label": profile["story_form_label"],
        "source_material_label": profile["source_material_label"],
        "source_develop_verb": profile["source_develop_verb"],
        "source_grounding_label": profile["source_grounding_label"],
        "outline_rules_extra": profile.get("outline_rules_extra", ""),
        "forbidden_plot_patterns": list(profile.get("forbidden_plot_patterns", [])),
        "outline_system_prompt": profile.get("outline_system_prompt") or "",
    }


def style_picker_overrides(profile: dict[str, Any]) -> dict[str, str]:
    """Kwargs for _otr_style_picker.pick_style (locked signature). Empty
    string preserves the science/module constants; the science profile leaves
    these unset so its behavior is byte-identical."""

    profile = validate_profile(profile)
    return {
        "inventor_system_prompt": profile.get("style_picker_inventor_system_prompt") or "",
        "chooser_system_prompt": profile.get("style_picker_chooser_system_prompt") or "",
        "chooser_user_template": profile.get("style_picker_chooser_user_template") or "",
    }


def coda_mode(profile: dict[str, Any]) -> str:
    return validate_profile(profile)["coda_mode"]


def line_grounding_instruction(profile: dict[str, Any]) -> str:
    return validate_profile(profile)["line_grounding_instruction"]


def dramatic_state_labels(profile: dict[str, Any]) -> dict[str, str]:
    """Labels for _otr_dramatic_state_llm (replaces hardcoded NEWS KEY TERMS /
    NEWS PREMISE for non-science lanes)."""

    profile = validate_profile(profile)
    return {
        "key_terms_label": profile.get("key_terms_label", "KEY TERMS"),
        "premise_label": f"{profile['source_material_label'].upper()} PREMISE",
        "grounding_label": profile["source_grounding_label"],
    }
