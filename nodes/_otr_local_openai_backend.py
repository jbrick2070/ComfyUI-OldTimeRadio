"""Generic local OpenAI-compatible LLM backend.

This is the opt-in Gemma 4 12B lane. It speaks the same
``/v1/chat/completions`` wire protocol as llama.cpp ``llama-server``,
Google LiteRT-LM serve, and other local OpenAI-compatible servers, but it
does not start or manage any daemon. If the endpoint is unavailable the
selected lane fails closed with a named error.

Enable it with ``GEMMA4_12B_ENABLED=1`` and select the virtual catalog handle
``local_gemma4_12b``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("OTR")

LOCAL_OPENAI_BACKEND_KEY = "local_openai_http"
PROVIDER = "local_openai"
ROW_ID = "local_gemma4_12b"

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL_ID = "ggml-org/gemma-4-12B-it-GGUF:Q4_K_M"
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_OUTPUT_TOKENS_CAP = 512
DEFAULT_TIMEOUT_S = 600
DEFAULT_MAX_RETRIES = 1

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_VALID_REASONING_EFFORT = frozenset({"high", "medium", "low", "none"})


class LocalOpenAIError(RuntimeError):
    """Base error for the local OpenAI-compatible lane."""


class LocalOpenAIConfigError(LocalOpenAIError):
    """The lane was selected but its endpoint/model config is unusable."""


class LocalOpenAICallFailedError(LocalOpenAIError):
    """The local OpenAI-compatible call failed after bounded retries."""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def local_gemma4_enabled() -> bool:
    """Whether the virtual ``local_gemma4_12b`` catalog row is visible."""
    return _env_flag("GEMMA4_12B_ENABLED", False)


def _base_url() -> str:
    return (os.environ.get("GEMMA4_12B_BASE_URL") or DEFAULT_BASE_URL).strip()


def _model_id() -> str:
    return (os.environ.get("GEMMA4_12B_MODEL_ID") or DEFAULT_MODEL_ID).strip()


def _api_key() -> str:
    return (os.environ.get("GEMMA4_12B_API_KEY") or "local-openai").strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _reasoning_effort_from_env() -> str:
    raw = (os.environ.get("GEMMA4_12B_REASONING_EFFORT", "") or "").strip().lower()
    return raw if raw in _VALID_REASONING_EFFORT else "none"


def _post_chat_completion(*, base_url: str, payload: dict, timeout_s: int) -> dict:
    import requests

    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload).encode("utf-8"),
        timeout=timeout_s,
    )
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 -- non-JSON local server error body
        body = None
    return {"status_code": resp.status_code, "json": body, "text": resp.text}


def _get_models(*, base_url: str, timeout_s: int) -> dict:
    import requests

    url = f"{base_url.rstrip('/')}/models"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=timeout_s,
    )
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = None
    return {"status_code": resp.status_code, "json": body, "text": resp.text}


def validate_local_openai_server(
    base_url: str | None = None,
    model_id: str | None = None,
    *,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Return structured health for a local OpenAI-compatible endpoint.

    This helper is explicit and side-effect free: it never starts a process and
    never falls back to a sidecar or cloud. ``/v1/models`` is treated as optional
    because llama.cpp builds vary; the tiny completion is the real proof.
    """
    resolved_base = (base_url or _base_url()).strip()
    resolved_model = (model_id or _model_id()).strip()
    result: dict[str, Any] = {
        "ok": False,
        "base_url": resolved_base,
        "model_id": resolved_model,
        "models_status": None,
        "completion_status": None,
        "error": "",
    }
    if not resolved_base or not resolved_model:
        result["error"] = "missing base_url or model_id"
        return result
    try:
        models = _get_models(base_url=resolved_base, timeout_s=timeout_s)
        result["models_status"] = int(models.get("status_code") or 0)
    except Exception as exc:  # noqa: BLE001 -- optional endpoint
        result["models_status"] = f"{type(exc).__name__}: {exc}"
    try:
        completion = _post_chat_completion(
            base_url=resolved_base,
            timeout_s=timeout_s,
            payload={
                "model": resolved_model,
                "messages": [{"role": "user", "content": "Return exactly: OK"}],
                "max_tokens": 8,
                "reasoning_effort": "none",
            },
        )
        status = int(completion.get("status_code") or 0)
        result["completion_status"] = status
        if status == 200:
            text = LocalOpenAIBackend._extract_text(completion, slug=resolved_model)
            result["ok"] = bool(text.strip())
            return result
        result["error"] = f"HTTP {status}: {(completion.get('text') or '')[:200]}"
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


class LocalOpenAIBackend:
    """LoaderBackend adapter for an external local OpenAI-compatible server."""

    def load(self, repo_id: str, row: Any) -> dict[str, Any]:
        if not local_gemma4_enabled():
            raise LocalOpenAIConfigError(
                f"{repo_id!r} selected but GEMMA4_12B_ENABLED is not true."
            )
        slug = _model_id()
        if not slug:
            raise LocalOpenAIConfigError("GEMMA4_12B_MODEL_ID is empty.")
        base_url = _base_url()
        if not base_url:
            raise LocalOpenAIConfigError("GEMMA4_12B_BASE_URL is empty.")
        context_window = int(
            getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW)
            or DEFAULT_CONTEXT_WINDOW
        )
        cache_entry = {
            "provider": PROVIDER,
            "model_id": repo_id,
            "slug": slug,
            "base_url": base_url,
            "context_cap": context_window,
            "context_window": context_window,
        }
        log.info(
            "[LocalOpenAI] load %s -> model=%s base_url=%s ctx=%d "
            "(external local server, 0 process VRAM)",
            repo_id, slug, base_url, context_window,
        )
        return cache_entry

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
        if grammar:
            raise LocalOpenAIConfigError(
                "The local OpenAI-compatible lane does not accept raw GBNF "
                "`grammar`; use JSON-schema `response_format`."
            )
        cache_entry = model
        slug = cache_entry.get("slug") or _model_id()
        base_url = cache_entry.get("base_url") or _base_url()
        cap = _int_env("GEMMA4_12B_MAX_NEW_TOKENS", DEFAULT_OUTPUT_TOKENS_CAP)
        out_tokens = int(max_new_tokens or cap)
        if out_tokens > cap:
            out_tokens = cap
        payload: dict[str, Any] = {
            "model": slug,
            "messages": messages,
            "max_tokens": out_tokens,
            "reasoning_effort": _reasoning_effort_from_env(),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if stop:
            payload["stop"] = [s for s in stop if s]
        if response_format is not None:
            payload["response_format"] = response_format
        return self._post_with_retries(base_url=base_url, payload=payload, slug=slug)

    def unload(self, model: Any) -> None:  # noqa: ARG002
        return None

    def _post_with_retries(self, *, base_url: str, payload: dict, slug: str) -> str:
        timeout_s = _int_env("GEMMA4_12B_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        max_retries = _int_env("GEMMA4_12B_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                result = _post_chat_completion(
                    base_url=base_url, payload=payload, timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"transport error: {type(exc).__name__}: {exc}"
                log.warning(
                    "[LocalOpenAI] attempt %d/%d failed (%s)",
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
                "[LocalOpenAI] attempt %d/%d got retryable %s",
                attempt + 1, max_retries + 1, last_err,
            )
            time.sleep(min(2 ** attempt, 8))
        raise LocalOpenAICallFailedError(
            f"Local OpenAI-compatible call failed for model={slug!r} at "
            f"{base_url} after {max_retries + 1} attempt(s): {last_err}. "
            f"Start the external local server or disable {ROW_ID}; this lane "
            f"never falls back to a sidecar or cloud."
        )

    @staticmethod
    def _extract_text(result: dict, *, slug: str) -> str:
        data = result.get("json") or {}
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise LocalOpenAICallFailedError(
                f"Local OpenAI-compatible response for model={slug!r} had "
                f"no choices: {str(data)[:300]}"
            )
        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content:
            raise LocalOpenAICallFailedError(
                f"Local OpenAI-compatible model {slug} returned empty "
                f"message content (finish_reason="
                f"{choices[0].get('finish_reason')!r})."
            )
        return content


def make_local_openai_generate_fn(
    cache_entry: dict, *, response_format: dict | None = None,
):
    backend = LocalOpenAIBackend()
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

    generate_fn._otr_local_openai = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    return generate_fn


__all__ = [
    "LOCAL_OPENAI_BACKEND_KEY",
    "PROVIDER",
    "ROW_ID",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_ID",
    "DEFAULT_CONTEXT_WINDOW",
    "LocalOpenAIError",
    "LocalOpenAIConfigError",
    "LocalOpenAICallFailedError",
    "LocalOpenAIBackend",
    "local_gemma4_enabled",
    "validate_local_openai_server",
    "make_local_openai_generate_fn",
]
