"""
llm_polish.py  --  Real LLM polish pass for visual-path prompts
=================================================================
Consumed by OTR_VisualPromptCoercion when the central
OTR_VisualLLMSelector hands it a model_id other than "none".

Scope (v1 live):
    - Polish ENVIRONMENT tokens only.  Dialogue text is TTS-critical
      and must never be rewritten.  SFX descriptions are already
      compact and pass through the rule pass cleanly.
    - One LLM call per environment token, short prompt, max 80 words.
    - Deterministic (do_sample=False) so the same input yields the
      same polished output across runs.

Design intent:
    - Single module-level model cache keyed by (model_id, dtype)
      so swapping model_id on the selector reloads lazily.
    - HF_TOKEN is resolved via _hf_token.ensure_hf_token() on every
      load so gated models (Gemma/Mistral) work without manual setup.
    - local_files_only=True first, Hub fallback only if the cache
      lookup raises -- matches story_orchestrator's pattern.
    - 1-token warmup after load to absorb CUDA JIT stall (Blackwell
      sm_120 + SDPA hits this on first generate).
    - Graceful fallback: any exception returns the rule-cleaned
      input untouched + a diagnostic note.  Audio is never threatened.

Not in v1 (deferred):
    - Quantization dispatch (story_orchestrator owns this for now).
    - max_memory / device_map "auto" -- visual polish is small enough
      that plain .to(device) is fine.
    - Multi-token polish of dialogue (TTS parity too risky).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ._hf_token import ensure_hf_token

log = logging.getLogger("OTR.visual.llm_polish")


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_POLISH_CACHE: dict[str, Any] = {
    "model_id": None,
    "model": None,
    "tokenizer": None,
    "device": None,
    "loaded_at": 0.0,
}


_SYSTEM_PROMPT = (
    "You are a prompt editor for a cinematic diffusion model. "
    "Rewrite the given environment description into one single line of "
    "concrete visual detail: subject, setting, lighting, composition. "
    "Keep it under 80 words. Do not add dialogue, story, or character names. "
    "Do not use quotation marks. Return only the rewritten description."
)


def _load_model(model_id: str) -> tuple[Any, Any, str] | None:
    """Load (or reuse) the LLM for visual-prompt polish.

    Returns (model, tokenizer, device) on success, None on failure.
    Failures are logged but never raise.
    """
    if _POLISH_CACHE["model_id"] == model_id and _POLISH_CACHE["model"] is not None:
        return (
            _POLISH_CACHE["model"],
            _POLISH_CACHE["tokenizer"],
            _POLISH_CACHE["device"],
        )

    # Heavy imports are lazy so node-scan never pays transformers cost.
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        log.warning("[llm_polish] transformers/torch not available: %s", exc)
        return None

    # Make sure HF_TOKEN is live in os.environ before load attempts.
    token = ensure_hf_token()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cache_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )

    # Tokenizer first -- cheaper, proves the cache works before we
    # download multi-GB weights.
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=False,
                cache_dir=cache_dir,
                token=token,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=False,
                cache_dir=cache_dir,
                token=token,
            )
    except Exception as exc:
        log.warning("[llm_polish] tokenizer load failed for %s: %s", model_id, exc)
        return None

    # Now the model weights.  Visual polish generates short outputs
    # (<= 80 words), so no KV-cache hardening is required here.
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=load_dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=token,
            )
        except Exception as cache_err:
            log.info(
                "[llm_polish] local cache miss for %s (%s); trying Hub",
                model_id,
                type(cache_err).__name__,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=False,
                torch_dtype=load_dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=token,
            )
    except Exception as exc:
        log.warning("[llm_polish] model load failed for %s: %s", model_id, exc)
        return None

    try:
        model = model.to(device)
        model = model.eval()
    except Exception as exc:
        log.warning("[llm_polish] model.to(%s) failed: %s", device, exc)
        return None

    # 1-token warmup -- absorbs the Blackwell SDPA JIT stall so the
    # first real polish call does not block for 30s+.
    try:
        warm_ids = tokenizer("Test.", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model.generate(warm_ids, max_new_tokens=1, do_sample=False)
        del warm_ids
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        log.debug("[llm_polish] warmup failed (non-fatal): %s", exc)

    _POLISH_CACHE["model_id"] = model_id
    _POLISH_CACHE["model"] = model
    _POLISH_CACHE["tokenizer"] = tokenizer
    _POLISH_CACHE["device"] = device
    _POLISH_CACHE["loaded_at"] = time.time()

    log.info("[llm_polish] loaded %s on %s", model_id, device)
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
    max_new_tokens: int = 160,
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

    if model_id == "none" or not model_id:
        stats["polish_skipped"] = sum(
            1 for t in tokens if isinstance(t, dict) and t.get("type") == "environment"
        )
        return list(tokens), stats

    loaded = _load_model(model_id)
    if loaded is None:
        stats["polish_fallback"] = True
        stats["polish_skipped"] = sum(
            1 for t in tokens if isinstance(t, dict) and t.get("type") == "environment"
        )
        stats["error"] = "model_load_failed"
        return list(tokens), stats

    model, tokenizer, device = loaded

    polished: list[dict] = []
    for tok in tokens:
        if not isinstance(tok, dict) or tok.get("type") != "environment":
            polished.append(tok)
            continue

        original = tok.get("description", "")
        if not original:
            polished.append(tok)
            continue

        stats["polish_attempted"] += 1
        new_text = _generate_single(original, model, tokenizer, device)
        if not new_text:
            # Leave original intact on failure.
            polished.append(tok)
            continue

        # Enforce word cap even if the LLM ignores it.
        words = new_text.split()
        if len(words) > max_env_words:
            new_text = " ".join(words[:max_env_words])

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
    """Release cached model (test hook + manual VRAM flush path)."""
    model = _POLISH_CACHE.get("model")
    if model is not None:
        try:
            import gc
            import torch
            del _POLISH_CACHE["model"]
            _POLISH_CACHE["model"] = None
            _POLISH_CACHE["tokenizer"] = None
            _POLISH_CACHE["model_id"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            _POLISH_CACHE["model"] = None
            _POLISH_CACHE["tokenizer"] = None
            _POLISH_CACHE["model_id"] = None


__all__ = ["polish_environment_prompts", "unload"]
