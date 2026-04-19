"""
prompt_coercion.py  --  OTR_VisualPromptCoercion ComfyUI node
=================================================================
Rule-based + optional-LLM normalizer that sits between a raw
script_json source (file, Prompt Builder, or hand-authored widget)
and the OTR_VisualBridge.  Purpose: clean each parsed value to the
strict format that FLUX / LTX / Wan2.1 expect.

Architecture note:
    Phase 2 of the visual test guide says the upstream Prompt
    Builder's ``llm_model`` input should be optional, with a
    rule-based fallback when no LLM is connected.  This node
    implements that contract as a standalone bolt-on: it operates
    on the raw JSON token array and applies per-token-type cleanup
    rules.  If an LLM is connected the node still runs the rules
    first, then hands the LLM a nearly-clean payload (so the LLM
    can't regress basics like stripping [SFX:] tags).

Design intent:
    - Default behaviour is pure-Python, deterministic, fast.
    - The ``llm_model`` slot is optional.  If left disconnected,
      the node behaves as a rule-based cleaner only.
    - An ``enable_llm`` toggle exists so LLMs can be wired but
      temporarily disabled during iteration without unwiring.
    - Never fatal: coercion errors log + pass through the original
      value so the audio path is never threatened.

Cleanup rules (v1, deterministic):
    environment:
        - Strip embedded ``[SFX: ...]`` markers + stray bracketed
          stage directions that bleed through from the treatment.
        - Strip parenthetical emotion tags like ``(frustrated)``
          when they appear mid-description (they belong on
          dialogue, not environment).
        - Collapse whitespace + trailing punctuation.
        - Clamp to ~80 words so it fits FLUX / LTX prompt budgets.
    dialogue.line:
        - Preserve the spoken text verbatim (TTS parity matters).
        - Only strip stray ``[SFX: ...]`` bleed-through.
    sfx.description:
        - Strip leading ``[SFX]``/``[SFX:`` noise, flatten
          commas/semicolons to a single comma-separated list of
          concrete cues.
    dialogue.character_name:
        - Upper-case, strip surrounding whitespace, collapse
          repeated spaces.
    dialogue.voice_traits:
        - Lowercase, split on '*' or ',' and rejoin as
          ``'trait_a * trait_b'`` canonical form.

Design doc:
    uploads/ltx_visual_test_guide.md (Phase 2)
    workflows/test_context/silent_uprising.json (reference payload)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("OTR.visual.prompt_coercion")


# ---------------------------------------------------------------------------
# Rule library
# ---------------------------------------------------------------------------

# Matches "[SFX: ...]" or "[SFX:" or "[SFX ... ]" -- tolerant of treatment
# truncations like "[SFX: faint breathing of RAINN" (no closing bracket).
_SFX_EMBED_RE = re.compile(r"\[\s*sfx\s*[:\-]?[^\]]*\]?", re.IGNORECASE)

# Matches parenthetical emotion cues: (frustrated), (determined), (over comms)
_EMOTION_PARENS_RE = re.compile(r"\(\s*[a-z][a-z\s,/'-]+\s*\)", re.IGNORECASE)

# Matches stray bracketed stage directions other than SFX: [PAUSE], [BEAT]
_STAGE_BRACKET_RE = re.compile(r"\[\s*(pause|beat|music|direction)\b[^\]]*\]?", re.IGNORECASE)

# Multiple consecutive whitespace runs
_WS_RE = re.compile(r"\s+")

# Trailing punctuation soup like ",,,,.." or " . ."
_PUNCT_TAIL_RE = re.compile(r"[\s,;.:]+$")

# Env prompt word cap.  FLUX handles ~77 tokens for CLIP-L; LTX handles
# longer via T5 but most concise prompts under ~80 words both perform
# well and read cleanly to a human during debugging.
_ENV_WORD_CAP = 80


def _clean_environment(raw: str) -> str:
    """Strip SFX / emotion / bracket bleed-through from an environment token."""
    if not raw:
        return ""
    s = _SFX_EMBED_RE.sub(" ", raw)
    s = _STAGE_BRACKET_RE.sub(" ", s)
    s = _EMOTION_PARENS_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _PUNCT_TAIL_RE.sub("", s)
    # Word clamp -- never truncate a word mid-character
    words = s.split(" ")
    if len(words) > _ENV_WORD_CAP:
        s = " ".join(words[:_ENV_WORD_CAP])
    return s


def _clean_dialogue_line(raw: str) -> str:
    """Spoken text: preserve meaning + TTS-parity, only strip SFX bleed."""
    if not raw:
        return ""
    s = _SFX_EMBED_RE.sub(" ", raw)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _clean_sfx_description(raw: str) -> str:
    """SFX description: strip leading [SFX]/[SFX: markers, flatten list."""
    if not raw:
        return ""
    s = raw
    # If the string starts with a stray [SFX...] noise marker, drop it.
    s = re.sub(r"^\s*\[\s*sfx\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\]\s*$", "", s)
    # Normalise separators to commas.
    s = re.sub(r"\s*;\s*", ", ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _PUNCT_TAIL_RE.sub("", s)
    return s


def _clean_character_name(raw: str) -> str:
    """Canonicalise a character name: upper-case, collapsed whitespace."""
    if not raw:
        return "UNKNOWN"
    s = _WS_RE.sub(" ", raw).strip().upper()
    return s


def _clean_voice_traits(raw: str) -> str:
    """Canonical voice-traits form:  'trait_a * trait_b * trait_c'."""
    if not raw:
        return ""
    # Split on either '*' or ','
    parts = re.split(r"[*,]", raw)
    parts = [p.strip().lower() for p in parts if p.strip()]
    if not parts:
        return ""
    return " * ".join(parts)


# ---------------------------------------------------------------------------
# Token-array coercion pass
# ---------------------------------------------------------------------------

def coerce_script_lines(script_lines: list[dict]) -> tuple[list[dict], dict]:
    """Run rule-based cleanup across every token.

    Returns (cleaned_tokens, stats_dict).  Tokens are deep-copied; the
    input list is never mutated.
    """
    stats = {
        "total_tokens": len(script_lines),
        "environments_cleaned": 0,
        "dialogues_cleaned": 0,
        "sfx_cleaned": 0,
        "characters_canonicalised": 0,
    }
    cleaned: list[dict] = []
    for tok in script_lines:
        if not isinstance(tok, dict):
            cleaned.append(tok)
            continue
        ttype = tok.get("type", "")
        new_tok = dict(tok)

        if ttype == "environment":
            original = new_tok.get("description", "") or new_tok.get("text", "")
            new_tok["description"] = _clean_environment(original)
            if "text" in new_tok and "description" in new_tok:
                # Drop duplicate legacy key once description is populated.
                new_tok.pop("text", None)
            if new_tok["description"] != original:
                stats["environments_cleaned"] += 1

        elif ttype == "dialogue":
            orig_line = new_tok.get("line", "") or new_tok.get("text", "")
            orig_name = new_tok.get("character_name", "") or new_tok.get("character", "")
            orig_traits = new_tok.get("voice_traits", "")

            new_tok["line"] = _clean_dialogue_line(orig_line)
            new_tok["character_name"] = _clean_character_name(orig_name)
            new_tok["voice_traits"] = _clean_voice_traits(orig_traits)
            # Drop legacy duplicates
            new_tok.pop("text", None)
            new_tok.pop("character", None)

            if (
                new_tok["line"] != orig_line
                or new_tok["character_name"] != orig_name
                or new_tok["voice_traits"] != orig_traits
            ):
                stats["dialogues_cleaned"] += 1
            if new_tok["character_name"] != orig_name:
                stats["characters_canonicalised"] += 1

        elif ttype == "sfx":
            original = new_tok.get("description", "") or new_tok.get("text", "")
            new_tok["description"] = _clean_sfx_description(original)
            new_tok.pop("text", None)
            if new_tok["description"] != original:
                stats["sfx_cleaned"] += 1

        cleaned.append(new_tok)

    return cleaned, stats


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class VisualPromptCoercion:
    """
    Clean raw script_json before it reaches OTR_VisualBridge.

    Wire:
        OTR_VisualLLMSelector.model_id ──┐
        raw_script_json ─► OTR_VisualPromptCoercion ─► OTR_VisualBridge

    Required input:
        script_json: STRING (Canonical Audio Token array as JSON)

    Optional inputs:
        model_id:       STRING ("none" or a HF id from the central
                                OTR_VisualLLMSelector switcher).  When
                                != "none", the LLM polish pass slot is
                                activated; v1 is a no-op stub.
        llm_model:      any    (pre-loaded LLM handle; v2 seam)
        max_env_words:  INT    (override environment word cap)

    Outputs:
        cleaned_script_json: STRING (drop-in replacement for VisualBridge)
        coercion_stats_json: STRING (diagnostic counts + model_id audit)
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "coerce"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cleaned_script_json", "coercion_stats_json")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "Raw Canonical Audio Token array from a Prompt "
                        "Builder, script file, or hand-authored widget."
                    ),
                }),
            },
            "optional": {
                # Feed this from OTR_VisualLLMSelector.model_id so the
                # whole visual path shares one LLM choice.  Default
                # "none" keeps the node in rule-based-only mode, which
                # is the safe behaviour when the selector is not wired.
                "model_id": ("STRING", {
                    "default": "none",
                    "forceInput": True,
                    "tooltip": (
                        "Wire from OTR_VisualLLMSelector.model_id. "
                        "'none' (default) = rule-based cleanup only, "
                        "no LLM pass."
                    ),
                }),
                # Optional loaded-model handle -- reserved for v2 when
                # a shared loader feeds the whole visual path.  Accepts
                # any type so a Gemma/Ollama handle can plug in without
                # forcing transformers at node-scan time.
                "llm_model": ("*", {"tooltip": "Optional pre-loaded LLM handle (v2 seam; leave unconnected)."}),
                "max_env_words": ("INT", {
                    "default": _ENV_WORD_CAP,
                    "min": 10,
                    "max": 200,
                    "tooltip": "Word cap for environment prompts.",
                }),
            },
        }

    def coerce(
        self,
        script_json: str,
        model_id: str = "none",
        llm_model: Any = None,
        max_env_words: int = _ENV_WORD_CAP,
    ) -> tuple[str, str]:
        # Temporarily override the module cap for this call
        global _ENV_WORD_CAP
        previous_cap = _ENV_WORD_CAP
        _ENV_WORD_CAP = int(max_env_words)

        try:
            try:
                tokens = json.loads(script_json)
            except json.JSONDecodeError as exc:
                log.error("[PromptCoercion] Invalid JSON: %s", exc)
                return (script_json, json.dumps({"error": f"parse_error: {exc}"}))

            if not isinstance(tokens, list):
                log.error("[PromptCoercion] Expected top-level JSON array, got %s", type(tokens).__name__)
                return (script_json, json.dumps({"error": "not_an_array"}))

            cleaned, stats = coerce_script_lines(tokens)

            # Record the LLM decision in stats regardless of whether the
            # pass ran.  Downstream diagnostic tooling can audit which
            # model was selected for which episode.
            resolved_model = (model_id or "none").strip() or "none"
            stats["llm_model_id"] = resolved_model
            stats["llm_pass_ran"] = False

            if resolved_model != "none":
                # LLM polish pass is intentionally a no-op in v1.  The
                # model_id is wired through so v2 can flip this branch
                # on without touching the workflow graph.  When a
                # pre-loaded llm_model handle is also supplied, the
                # sidecar planner can short-circuit loading.
                log.info(
                    "[PromptCoercion] LLM polish reserved (v1 stub). model_id=%s handle=%s",
                    resolved_model,
                    "provided" if llm_model is not None else "none",
                )
                stats["llm_pass_note"] = "v1_stub_reserved_seam"
                stats["llm_handle_provided"] = llm_model is not None

            cleaned_json = json.dumps(cleaned, indent=2)
            stats_json = json.dumps(stats, indent=2)
            log.info("[PromptCoercion] cleaned %d tokens (env=%d dlg=%d sfx=%d) llm=%s",
                     stats["total_tokens"],
                     stats["environments_cleaned"],
                     stats["dialogues_cleaned"],
                     stats["sfx_cleaned"],
                     resolved_model)
            return (cleaned_json, stats_json)
        finally:
            _ENV_WORD_CAP = previous_cap


__all__ = ["VisualPromptCoercion", "coerce_script_lines"]
