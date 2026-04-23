"""
flux_prompt_extractor.py  --  OTR_VisualExtractFluxPrompt ComfyUI node
=====================================================================
Adapter node between OTR_VisualPromptCoercion (whose outputs are
JSON-stringified token lists) and ComfyUI's stock CLIPTextEncode
(which needs a plain-text STRING).

Required because:
  - OTR_VisualPromptCoercion emits ``cleaned_script_json`` as a
    JSON-stringified Audio Token array (environment + dialogue +
    sfx + scene_break + title + pause tokens, all mixed).
  - ComfyUI's KSampler can't be scheduled before the Mistral polish
    finishes unless there is an actual data link between them. A
    link from PromptCoercion -> CLIPTextEncode fixes the schedule
    order AND feeds the polished text into FLUX.
  - But sending raw JSON into CLIPTextEncode would encode 8 KB of
    braces and quotes as the prompt, producing garbage imagery.

This node does one job: parse the JSON, pick the N-th environment
token's polished description, return it as a plain string suitable
for CLIPTextEncode.

Zero torch, zero diffusers, zero model load. Pure string transform
so there's no bottleneck risk (see memory: custom_node_weakness).

Usage wiring in the TEST workflow::

    OTR_VisualPromptCoercion.cleaned_script_json  -- STRING -->
        OTR_VisualExtractFluxPrompt.script_json
    OTR_VisualExtractFluxPrompt.prompt            -- STRING -->
        CLIPTextEncode (positive).text

Audio / dialogue / sfx tokens pass through untouched; only the
environment token at ``shot_index`` is selected. If the index is
out of range the fallback string is returned so the pipeline still
renders something recognizable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("OTR.visual.flux_prompt_extractor")


_DEFAULT_FALLBACK = (
    "cinematic 35mm film still, dimly lit starship bridge, red alert "
    "lighting, holographic console glow, tense crew silhouettes, "
    "volumetric haze, shallow depth of field"
)


class VisualExtractFluxPrompt:
    """Pick the N-th environment token out of the polished script JSON."""

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Connect from OTR_VisualPromptCoercion's "
                        "cleaned_script_json output."
                    ),
                }),
                "shot_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99,
                    "step": 1,
                    "tooltip": (
                        "Which environment token to pick (0 = first). "
                        "Matches flux_anchor's single-shot mode default."
                    ),
                }),
            },
            "optional": {
                "fallback_prompt": ("STRING", {
                    "multiline": True,
                    "default": _DEFAULT_FALLBACK,
                    "tooltip": (
                        "Used when the JSON has no environment token at "
                        "the requested index, or when parsing fails. "
                        "Keeps FLUX rendering something instead of "
                        "erroring the whole graph."
                    ),
                }),
                "style_suffix": ("STRING", {
                    "multiline": False,
                    "default": (
                        "cinematic, 35mm film, anamorphic lens, "
                        "volumetric lighting, heavy vignette, "
                        "muted color grade, sharp focus"
                    ),
                    "tooltip": (
                        "Appended to the selected env prompt. Matches "
                        "visual/backends/flux_anchor.py::_STYLE_SUFFIX "
                        "so native FLUX output stays stylistically "
                        "consistent with the sidecar path."
                    ),
                }),
            },
        }

    def execute(
        self,
        script_json: str,
        shot_index: int = 0,
        fallback_prompt: str = _DEFAULT_FALLBACK,
        style_suffix: str = "",
    ) -> tuple[str]:
        env_prompt = self._pick_env_prompt(
            script_json, shot_index, fallback_prompt
        )
        parts = [env_prompt]
        if style_suffix and style_suffix.strip():
            parts.append(style_suffix.strip())
        final = ", ".join(parts)
        log.info(
            "[ExtractFluxPrompt] shot_index=%d len=%d: %s",
            shot_index,
            len(final),
            final[:200] + ("..." if len(final) > 200 else ""),
        )
        return (final,)

    @staticmethod
    def _pick_env_prompt(
        script_json: str,
        shot_index: int,
        fallback: str,
    ) -> str:
        """Return the N-th environment token's description, or fallback."""
        if not script_json or not script_json.strip():
            log.info("[ExtractFluxPrompt] empty script_json; using fallback")
            return fallback
        try:
            payload: Any = json.loads(script_json)
        except json.JSONDecodeError as exc:
            log.warning(
                "[ExtractFluxPrompt] script_json JSONDecodeError: %s; "
                "using fallback",
                exc,
            )
            return fallback

        # Accept either a bare list of tokens or {"tokens": [...]} shape.
        if isinstance(payload, dict):
            tokens = payload.get("tokens") or payload.get("script") or []
        elif isinstance(payload, list):
            tokens = payload
        else:
            log.warning(
                "[ExtractFluxPrompt] unexpected script_json root type "
                "%s; using fallback",
                type(payload).__name__,
            )
            return fallback

        env_tokens = [
            t for t in tokens
            if isinstance(t, dict) and t.get("type") == "environment"
        ]
        if not env_tokens:
            log.info(
                "[ExtractFluxPrompt] no environment tokens found "
                "(total tokens=%d); using fallback",
                len(tokens),
            )
            return fallback

        idx = max(0, min(shot_index, len(env_tokens) - 1))
        token = env_tokens[idx]
        description = (token.get("description") or "").strip()
        if not description:
            log.info(
                "[ExtractFluxPrompt] env token %d has empty description; "
                "using fallback",
                idx,
            )
            return fallback

        return description


__all__ = ["VisualExtractFluxPrompt"]
