"""Dedicated LOCAL llama.cpp / Ollama writer lane (2026-06-04).

A first-class LOCAL model lane, deliberately SEPARATE from the cloud
OpenRouter lane and the credit-billed Comfy Credits lane. It speaks the same
OpenAI-compatible ``/v1`` chat-completions protocol that Ollama (and
llama.cpp's ``server``) expose, so architecturally it sits in the same
"reached over HTTP, zero ComfyUI-process VRAM, provider-tagged cache_entry,
no weight load" camp as those lanes -- which is exactly why it reuses their
loader seam (so ``request_slot`` never tries to HF-download the model or
evict the resident local model). But the endpoint is a LOCAL daemon on
``127.0.0.1:11434`` (override with ``OLLAMA_BASE_URL``), it needs NO API key,
it has NO credit/cost guard (local inference is free), and it is FAIL-CLOSED:
it never falls back to any cloud URL. "Reached over HTTP" is not "remote" --
Ollama is local infrastructure.

This module is self-contained on purpose (its own HTTP POST, its own
``OLLAMA_*`` env, no import from ``_otr_openrouter_backend``) so the local
lane can never be coupled to or confused with the cloud lane.

It carries one lever the local Ollama daemon honours over ``/v1``:
``reasoning_effort`` (``OLLAMA_REASONING_EFFORT=none`` disables a thinking
model's ``<think>`` preamble so it does not burn the output budget). Ollama's
OpenAI-compatible ``/v1`` does NOT accept a raw top-level ``grammar`` (GBNF)
field: structured output is expressed as a JSON-schema ``response_format`` that
Ollama converts to GBNF internally. This lane therefore REJECTS a raw
``grammar`` fail-loud -- the style-picker inventor never sends one here (its
always-on parser net handles overgeneration), and a genuine llama.cpp
``server`` would be a separate future lane. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("OTR")

# The CuratedModel.loader_backend literal + provider tag this lane keys on.
OLLAMA_BACKEND_KEY = "ollama_local_http"
PROVIDER = "ollama"

# LOCAL-only default. Override OLLAMA_BASE_URL for a llama.cpp server on a
# different local port (e.g. http://localhost:8080/v1) or another box on the
# LAN. There is NO cloud default and NO cloud fallback: if the local server is
# unreachable the call fails closed with a NAMED error.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_OUTPUT_TOKENS_CAP = 6144
DEFAULT_MIN_OUTPUT_TOKENS = 512
DEFAULT_TIMEOUT_S = 600
DEFAULT_MAX_RETRIES = 2

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_VALID_REASONING_EFFORT = frozenset({"high", "medium", "low", "none"})


class OllamaError(RuntimeError):
    """Base error for the local Ollama lane."""


class OllamaConfigError(OllamaError):
    """The local lane is selected but misconfigured (no tag, bad env)."""


class OllamaCallFailedError(OllamaError):
    """The local /v1 call failed (server down, bad model tag, empty body).

    Fail-closed: the lane never silently falls back to a cloud endpoint."""


# --------------------------------------------------------------------------- #
# Pure helpers (env + transport). No import from the OpenRouter module.
# --------------------------------------------------------------------------- #
def _base_url() -> str:
    """The LOCAL /v1 base URL. Never a cloud endpoint."""
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _reasoning_effort_from_env() -> str:
    """``OLLAMA_REASONING_EFFORT`` -> a validated effort string (default "none").

    This lane exists for gemma-4-12b, a *thinking* model: left to reason it
    spends the whole output budget on a ``<think>`` preamble and returns empty
    content (finish_reason='length'), which aborts the style inventor. So the
    lane DEFAULTS to ``none`` (no thinking) when the env is unset or invalid;
    set ``OLLAMA_REASONING_EFFORT=high|medium|low`` to re-enable reasoning.
    Ollama ignores the field for non-thinking models, so the default is safe."""
    raw = (os.environ.get("OLLAMA_REASONING_EFFORT", "") or "").strip().lower()
    return raw if raw in _VALID_REASONING_EFFORT else "none"


def ollama_enabled() -> bool:
    """The local lane is always available: it is local infrastructure, with no
    cloud API key and no opt-in flag. A misreachable server fails closed at
    generate time with a NAMED error (the C-7 named-error path)."""
    return True


def _post_chat_completion(*, base_url: str, payload: dict, timeout_s: int) -> dict:
    """POST one chat-completion to the LOCAL /v1 endpoint; return parsed JSON.

    Isolated so tests can patch it (no network in CI). ``requests`` is imported
    lazily to keep the module import-safe + network-free. Ollama ignores the
    Authorization header, so a placeholder bearer token keeps OpenAI clients
    happy without implying any cloud credential."""
    import requests  # lazy: keep module import-safe + network-free

    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": "Bearer ollama-local",  # ignored by Ollama/llama.cpp
            "Content-Type": "application/json",
        },
        data=json.dumps(payload).encode("utf-8"),
        timeout=timeout_s,
    )
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 -- a non-JSON error body is tolerated
        body = None
    return {"status_code": resp.status_code, "json": body, "text": resp.text}


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class OllamaBackend:
    """LoaderBackend adapter for a LOCAL llama.cpp / Ollama model.

    ``load()`` builds a provider-tagged cache_entry (no weights, no tokenizer,
    zero ComfyUI-process VRAM) carrying the LOCAL base_url + the Ollama model
    tag. ``generate()`` posts one chat completion to the LOCAL /v1 endpoint --
    no API key, no cost guard, fail-closed. ``unload()`` is a no-op."""

    # --- protocol: load -------------------------------------------------
    def load(self, repo_id: str, row: Any) -> dict[str, Any]:
        tag = str(getattr(row, "ollama_model_tag", "") or "").strip()
        if not tag:
            raise OllamaConfigError(
                f"{repo_id!r} routes to the local Ollama lane but its catalog "
                f"row carries no ollama_model_tag (the Ollama model name, e.g. "
                f"'hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M')."
            )
        context_window = int(
            getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW)
            or DEFAULT_CONTEXT_WINDOW
        )
        cache_entry: dict[str, Any] = {
            "provider": PROVIDER,
            "model_id": repo_id,         # the curated dropdown id
            "slug": tag,                 # the Ollama model tag
            "base_url": _base_url(),     # LOCAL only -- never cloud
            "context_cap": context_window,
            "context_window": context_window,
            # no "model" / "tokenizer" keys: the generate-fn factory branches
            # on provider BEFORE requiring them.
        }
        log.info(
            "[Ollama] load %s -> tag=%s base_url=%s ctx=%d (LOCAL, 0 process VRAM)",
            repo_id, tag, cache_entry["base_url"], context_window,
        )
        return cache_entry

    # --- protocol: generate --------------------------------------------
    def generate(
        self,
        model: Any,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        stop: Any = None,
        response_format: dict | None = None,
        grammar: str | None = None,
        **_ignored: Any,
    ) -> str:
        """Run one LOCAL chat completion and return the decoded string.

        ``model`` is the cache_entry from ``load()``. No cost guard (local
        inference is free); fail-closed on transport / non-200."""
        cache_entry = model
        slug = cache_entry.get("slug") or ""
        base_url = cache_entry.get("base_url") or _base_url()

        # Output budget: a thinking model must not be starved. Floor, then clamp
        # to the cap. max_tokens is a ceiling only, so a generous floor is free.
        cap = _int_env("OLLAMA_MAX_OUTPUT_TOKENS", DEFAULT_OUTPUT_TOKENS_CAP)
        floor = _int_env("OLLAMA_MIN_OUTPUT_TOKENS", DEFAULT_MIN_OUTPUT_TOKENS)
        out_tokens = max(int(max_new_tokens or 0), floor)
        if out_tokens > cap:
            out_tokens = cap

        payload: dict[str, Any] = {
            "model": slug,
            "messages": messages,
            "max_tokens": out_tokens,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if stop:
            payload["stop"] = [s for s in stop if s]
        # Disable a thinking model's <think> preamble when asked (OLLAMA_
        # REASONING_EFFORT=none); unset -> omitted -> byte-identical.
        reasoning_effort = _reasoning_effort_from_env()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format
        if grammar:
            # Ollama's OpenAI-compatible /v1 does NOT accept a raw top-level
            # `grammar` (GBNF) field -- only a real llama.cpp `server` does.
            # Structured output is a JSON-schema `response_format` Ollama converts
            # to GBNF internally. Fail loud rather than POST an unsupported field
            # or silently drop it: a lane flag must never mislead the operator.
            raise OllamaConfigError(
                "Ollama /v1 does not accept a raw GBNF `grammar`; express the "
                "constraint as a JSON-schema `response_format` instead."
            )

        return self._post_with_retries(base_url=base_url, payload=payload, slug=slug)

    # --- protocol: unload ----------------------------------------------
    def unload(self, model: Any) -> None:  # noqa: ARG002
        """No-op: nothing resident in this process. The local lane must never
        touch the resident local transformers model (no-evict)."""
        return None

    # --- internals ------------------------------------------------------
    def _post_with_retries(self, *, base_url: str, payload: dict, slug: str) -> str:
        timeout_s = _int_env("OLLAMA_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        max_retries = _int_env("OLLAMA_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                result = _post_chat_completion(
                    base_url=base_url, payload=payload, timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 -- transport error
                last_err = f"transport error: {type(exc).__name__}: {exc}"
                log.warning(
                    "[Ollama] attempt %d/%d failed (%s)",
                    attempt + 1, max_retries + 1, last_err,
                )
                time.sleep(min(2 ** attempt, 8))
                continue
            status = int(result.get("status_code") or 0)
            if status == 200:
                return self._extract_text(result, slug=slug)
            last_err = f"HTTP {status}: {(result.get('text') or '')[:300]}"
            if status not in _RETRYABLE_STATUS:
                break
            log.warning(
                "[Ollama] attempt %d/%d got retryable %s",
                attempt + 1, max_retries + 1, last_err,
            )
            time.sleep(min(2 ** attempt, 8))
        raise OllamaCallFailedError(
            f"Local Ollama call failed for tag={slug!r} at {base_url} after "
            f"{max_retries + 1} attempt(s): {last_err}. Is 'ollama serve' "
            f"running and the model pulled (check 'ollama list')? This lane is "
            f"LOCAL-only and never falls back to a cloud endpoint."
        )

    @staticmethod
    def _extract_text(result: dict, *, slug: str) -> str:
        data = result.get("json") or {}
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise OllamaCallFailedError(
                f"Local Ollama response for tag={slug!r} had no choices: "
                f"{str(data)[:300]}"
            )
        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content:
            raise OllamaCallFailedError(
                f"Local Ollama {slug} returned empty message content "
                f"(finish_reason={choices[0].get('finish_reason')!r})."
            )
        return content


# --------------------------------------------------------------------------- #
# generate-fn factory (mirrors make_openrouter_generate_fn / comfy sibling)
# --------------------------------------------------------------------------- #
def make_ollama_generate_fn(
    cache_entry: dict, *, response_format: dict | None = None,
):
    """Return a generate_fn closure for a provider-tagged Ollama cache_entry.

    Matches the signature every OTR generate_fn uses --
    ``(messages, *, temperature, max_new_tokens, stop=None, response_format=None,
    grammar=None) -> str`` -- so the loader factories, the writer's
    ``_build_truncating_generate_fn``, and the constrained-generate seam return
    it unchanged when ``cache_entry["provider"] == "ollama"``.

    This lane deliberately does NOT advertise ``_otr_supports_grammar``: Ollama's
    /v1 cannot honour a raw top-level GBNF ``grammar`` (only a JSON-schema
    ``response_format``). The style-picker inventor gates on
    ``getattr(fn, "_otr_supports_grammar", False)``, so the marker's absence
    keeps its exactly-N grammar away from this lane (the inventor's always-on
    parser net handles overgeneration). A ``grammar`` forced in anyway is
    rejected fail-loud by ``OllamaBackend.generate``."""
    backend = OllamaBackend()
    bound_rf = response_format

    def generate_fn(messages, *, temperature=None, max_new_tokens=None,
                    stop=None, response_format=None, grammar=None):
        rf = response_format if response_format is not None else bound_rf
        return backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
            grammar=grammar,
        )

    generate_fn._otr_ollama = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    return generate_fn


__all__ = [
    "OLLAMA_BACKEND_KEY",
    "PROVIDER",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_CONTEXT_WINDOW",
    "OllamaError",
    "OllamaConfigError",
    "OllamaCallFailedError",
    "OllamaBackend",
    "ollama_enabled",
    "make_ollama_generate_fn",
]
