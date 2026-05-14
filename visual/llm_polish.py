"""
llm_polish.py  --  Real LLM polish pass for visual-path prompts
=================================================================
Consumed by OTR_VisualPromptCoercion. Receives its model_id from the
writer's `creative_writing_model` broadcast output socket (S30 B5;
the legacy OTR_VisualLLMSelector picker node was deleted).

Scope (v1 live):
    - Polish ENVIRONMENT tokens only.  Dialogue text is TTS-critical
      and must never be rewritten.  SFX descriptions are already
      compact and pass through the rule pass cleanly.
    - One LLM call per environment token, short prompt, max 80 words.
    - Deterministic (do_sample=False) so the same input yields the
      same polished output across runs.

Design intent (S30 B5 onward):
    - Model acquisition routes through
      `_otr_model_loader.request_slot("creative", model_id)`. ONE
      LLM_CACHE in the loader holds at most one resident model
      regardless of how many surfaces request it (writer +
      cascade + visual all share the cache).
    - HF_TOKEN is resolved via _hf_token.ensure_hf_token() before
      request_slot so gated models (Gemma/Mistral) work without
      manual setup.
    - Graceful fallback: any exception returns the rule-cleaned
      input untouched + a diagnostic note.  Audio is never threatened.

Not in v1 (deferred):
    - Multi-token polish of dialogue (TTS parity too risky).
    - Sampling-config respect for the model's own generation_config.json
      (currently uses do_sample=False; the audio-intentional sprint
      will address this for the polish path).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ._hf_token import ensure_hf_token

log = logging.getLogger("OTR.visual.llm_polish")


# ---------------------------------------------------------------------------
# S30 B5: _POLISH_CACHE module-level dict + _load_model() function
# DELETED. Three caches existed before this commit (writer-side,
# orchestrator-side, polish-side); on the 16 GB card any one
# combination could double-load Mistral-Nemo and OOM (Prime Directive
# 2). Visual polish now acquires its model_id via the shared
# _otr_model_loader.request_slot("creative", model_id) entry point so
# the single LLM_CACHE in the loader holds at most ONE resident model
# regardless of how many surfaces request it.
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a prompt editor for a cinematic diffusion model. "
    "Rewrite the given environment description into one single line of "
    "concrete visual detail: subject, setting, lighting, composition. "
    "Keep it under 80 words. Do not add dialogue, story, or character names. "
    "Do not use quotation marks. Return only the rewritten description."
)


def _acquire_polish_entry(model_id: str) -> tuple[Any, Any, str] | None:
    """Acquire (or reuse cached) LLM cache_entry for visual-prompt
    polish via the shared loader's slot scheduler.

    LLM slot: creative -- visual prompt cleanup is narrative-style
    rewrite (one-line cinematic descriptions for diffusion); routes
    to the creative slot per the S30 routing table.

    Returns (model, tokenizer, device) on success, None on failure.
    Failures are logged but never raise -- the polish path falls
    back to rule-based cleanup whenever this returns None.
    """
    try:
        # Lazy import: visual/ depends on torch via the loader chain
        # but llm_polish is consumed at module-load time by node-scan;
        # the import lives inside the function so importing this
        # module never pulls torch.
        from nodes import _otr_model_loader as _OTRML  # type: ignore
    except ImportError as exc:
        log.warning(
            "[llm_polish] _otr_model_loader not importable: %s", exc,
        )
        return None
    # Make sure HF_TOKEN is live in os.environ before request_slot
    # hits any gated repo (Mistral-Nemo / Gemma family).
    ensure_hf_token()
    try:
        cache_entry = _OTRML.request_slot("creative", model_id)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[llm_polish] request_slot failed for %s: %s", model_id, exc,
        )
        return None
    model = cache_entry.get("model")
    tokenizer = cache_entry.get("tokenizer")
    device = cache_entry.get("device", "cuda")
    if model is None or tokenizer is None:
        log.warning(
            "[llm_polish] request_slot returned incomplete entry for %s",
            model_id,
        )
        return None
    log.info(
        "[llm_polish] using cache_entry for %s on %s (single LLM_CACHE)",
        model_id, device,
    )
    return (model, tokenizer, device)


def _build_prompt(env_description: str, tokenizer: Any) -> str:
    """Assemble a chat-template prompt if the tokenizer supports one,
    otherwise fall back to a bare system+user concatenation.
    """
    user_msg = (
        "Rewrite this environment description for a diffusion model. "
        "Keep it visual, concrete, under 80 words, one line, no quotes.\n\n"
        f"DESCRIPTION:\n{env_description}"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without a chat template.
        return f"{_SYSTEM_PROMPT}\n\n{user_msg}\n\nREWRITE:"


def _generate_single(
    env_description: str,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int = 100,
) -> str | None:
    """Run one polish call.  Returns the cleaned string or None on error."""
    try:
        import torch
    except ImportError:
        return None

    prompt = _build_prompt(env_description, tokenizer)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Slice off the prompt tokens so we only decode the new output.
        new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Strip surrounding quotes if the model ignored the instruction.
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1].strip()
        return text or None
    except Exception as exc:
        log.warning("[llm_polish] generate failed: %s", exc)
        return None


def polish_environment_prompts(
    tokens: list[dict],
    model_id: str,
    max_env_words: int = 80,
) -> tuple[list[dict], dict]:
    """Polish the ``description`` field of every environment token.

    Non-environment tokens pass through untouched.  The function
    deep-copies input tokens and never raises; on any load/inference
    failure it returns the input unchanged with a diagnostic note.

    Returns (polished_tokens, polish_stats).
    """
    stats = {
        "polish_attempted": 0,
        "polish_succeeded": 0,
        "polish_skipped": 0,
        "polish_fallback": False,
        "model_id": model_id,
    }

    # S30 B5: legacy "none" sentinel deleted (it tied to the deleted
    # OTR_VisualLLMSelector picker). An empty / unwired model_id
    # still skips the LLM pass and routes to rule-based-only cleanup.
    if not model_id:
        stats["polish_skipped"] = sum(
            1 for t in tokens if isinstance(t, dict) and t.get("type") == "environment"
        )
        return list(tokens), stats

    loaded = _acquire_polish_entry(model_id)
    if loaded is None:
        stats["polish_fallback"] = True
        stats["polish_skipped"] = sum(
            1 for t in tokens if isinstance(t, dict) and t.get("type") == "environment"
        )
        stats["error"] = "model_load_failed"
        return list(tokens), stats

    model, tokenizer, device = loaded

    # Mirror flux_anchor's single-shot default: when OTR_FLUX_ALL_SHOTS
    # is not set, polish only the FIRST environment token and pass the
    # rest through untouched. Keeps the isolation-debug cycle fast --
    # no point polishing 4 prompts when flux_anchor will only render
    # shot 1 anyway.
    _all_shots_mode = os.environ.get("OTR_FLUX_ALL_SHOTS", "").strip() == "1"
    _single_shot_mode = not _all_shots_mode
    if _single_shot_mode:
        log.info(
            "[llm_polish] SINGLE-SHOT MODE (default) -- polishing only "
            "prompt #1; prompts 2..N pass through unchanged. Set "
            "OTR_FLUX_ALL_SHOTS=1 to polish all environment tokens."
        )

    polished: list[dict] = []
    for tok in tokens:
        if not isinstance(tok, dict) or tok.get("type") != "environment":
            polished.append(tok)
            continue

        original = tok.get("description", "")
        if not original:
            polished.append(tok)
            continue

        # Single-shot: after the first polish, pass remaining env tokens
        # through untouched.
        if _single_shot_mode and stats["polish_attempted"] >= 1:
            polished.append(tok)
            continue

        stats["polish_attempted"] += 1
        idx = stats["polish_attempted"]
        # Log the INPUT prompt so the main ComfyUI console shows exactly
        # what the LLM is being asked to rewrite.  Truncated to 200 chars
        # for readability; full text always lives in the polished token's
        # description_original field.
        log.info(
            "[llm_polish] #%d IN  (len=%d): %s",
            idx, len(original), original[:200] + ("..." if len(original) > 200 else ""),
        )
        t0 = time.time()
        new_text = _generate_single(original, model, tokenizer, device)
        elapsed = time.time() - t0
        if not new_text:
            log.warning(
                "[llm_polish] #%d OUT (failed in %.1fs, keeping original)",
                idx, elapsed,
            )
            # Leave original intact on failure.
            polished.append(tok)
            continue

        # Enforce word cap even if the LLM ignores it.
        words = new_text.split()
        if len(words) > max_env_words:
            new_text = " ".join(words[:max_env_words])

        log.info(
            "[llm_polish] #%d OUT (len=%d, %.1fs): %s",
            idx, len(new_text), elapsed,
            new_text[:200] + ("..." if len(new_text) > 200 else ""),
        )

        new_tok = dict(tok)
        new_tok["description"] = new_text
        new_tok["description_source"] = "llm_polished"
        new_tok["description_original"] = original
        polished.append(new_tok)
        stats["polish_succeeded"] += 1

    log.info(
        "[llm_polish] model=%s attempted=%d succeeded=%d",
        model_id,
        stats["polish_attempted"],
        stats["polish_succeeded"],
    )
    return polished, stats


def unload() -> None:
    """Release cached model (test hook + manual VRAM flush path).

    S30 B5: delegates to the shared _otr_model_loader.unload_llm()
    since visual polish no longer owns a separate cache. Any
    consumer that was calling visual.llm_polish.unload() continues
    to work; the teardown now releases the single LLM_CACHE in the
    loader (and the legacy orchestrator stack via the loader's
    best-effort fallback).
    """
    try:
        from nodes import _otr_model_loader as _OTRML  # type: ignore
        _OTRML.unload_llm()
    except Exception as exc:  # noqa: BLE001
        log.debug("[llm_polish] unload delegation failed: %s", exc)


__all__ = ["polish_environment_prompts", "unload"]
