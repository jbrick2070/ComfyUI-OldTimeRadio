"""
llm_selector.py  --  OTR_VisualLLMSelector ComfyUI node
=================================================================
Single switcher node that picks the LLM used by every downstream
video-prompt cleanup in the visual path.  Wire this node once and
fan its ``model_id`` output into every consumer (VisualPromptCoercion,
future polish nodes, sidecar planners, etc.) so Jeffrey never has
to remember to change the model in more than one place.

Design intent:
    - One widget, one source of truth.
    - Output is plain STRING ``model_id`` so it composes cleanly
      with any downstream node that accepts a string input socket.
    - Default is Mistral-Nemo-Instruct-2407 -- the canonical OTR
      story-writing LLM (22.84 GB cache on disk, already warm for the
      audio path).  Downstream nodes still accept ``"none"`` for a
      rule-based fallback, but a fresh node placed on the graph or
      a freshly-loaded workflow with an empty widget now defaults to
      real LLM polish instead of silently falling back.
    - Choices match OTR_LLMScriptWriter exactly, so the same
      canonical list drives both the audio-story and visual paths.

Wiring pattern (typical):
    OTR_VisualLLMSelector.model_id
        -> OTR_VisualPromptCoercion.model_id
        -> (future) OTR_VisualDirector.model_id
        -> (future) OTR_VisualCaptioner.model_id
"""

from __future__ import annotations

import logging

log = logging.getLogger("OTR.visual.llm_selector")


# ---------------------------------------------------------------------------
# Canonical LLM model list
# ---------------------------------------------------------------------------
# These choices mirror the dropdown in nodes.story_orchestrator so the
# audio and visual paths share one catalog.  Update both lists together
# if a new model is added to the project.
_LLM_MODEL_CHOICES: list[str] = [
    "none",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-4-E4B-it",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
]
_LLM_DEFAULT: str = "mistralai/Mistral-Nemo-Instruct-2407"


class VisualLLMSelector:
    """
    One-widget node that broadcasts a model_id string downstream.

    Output:
        model_id: STRING -- plain model identifier; "none" means
                             rule-based fallback, no LLM pass.

    Notes:
        - This node does NOT load any model.  Loading happens in the
          sidecar/backend nodes that accept ``model_id`` as an input.
        - Keeping load decoupled from selection means the selector
          is free to change during a run without flushing VRAM.
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "select"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_id",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (_LLM_MODEL_CHOICES, {
                    "default": _LLM_DEFAULT,
                    "tooltip": (
                        "LLM chosen for every downstream visual-prompt "
                        "cleanup.  Default = Mistral-Nemo-Instruct-2407 "
                        "(canonical OTR writer).  Pick 'none' to disable "
                        "the LLM pass and run rule-based fallback only. "
                        "Matches OTR_LLMScriptWriter's model list."
                    ),
                }),
            },
        }

    def select(self, model_id: str) -> tuple[str]:
        # Sanity normalise: strip whitespace, leave casing intact so
        # the HuggingFace ID resolves correctly downstream.
        resolved = (model_id or _LLM_DEFAULT).strip()
        if resolved not in _LLM_MODEL_CHOICES:
            log.warning(
                "[LLMSelector] Unknown model_id %r, falling back to %r",
                resolved,
                _LLM_DEFAULT,
            )
            resolved = _LLM_DEFAULT
        log.info("[LLMSelector] visual path LLM = %s", resolved)
        return (resolved,)


__all__ = [
    "VisualLLMSelector",
    "_LLM_MODEL_CHOICES",
    "_LLM_DEFAULT",
]
