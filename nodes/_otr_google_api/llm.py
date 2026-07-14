"""LoaderBackend adapter and generate_fn for direct Google Gemini LLM calls."""

from __future__ import annotations

import json
from typing import Any

from .._otr_generation_budget import (
    GenerationContextOverflowError,
    estimate_prompt_tokens,
    fit_output_tokens,
)
from .client import (
    GoogleAPIError,
    GoogleAPIRequestShapeError,
    create_interaction,
)
from .models import (
    DEFAULT_CONTEXT_WINDOW,
    GOOGLE_API_PROVIDER,
    resolve_model_for_slot,
    supports_structured_output,
)


def _messages_to_google(messages: list[dict]) -> tuple[str | None, str]:
    if not isinstance(messages, list) or not messages:
        raise GoogleAPIRequestShapeError("Google LLM messages must be a non-empty list.")
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise GoogleAPIRequestShapeError("Each Google LLM message must be a dict.")
        role = str(msg.get("role") or "user").strip().lower()
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(p.get("text", "")) for p in content
                if isinstance(p, dict) and p.get("text")
            )
        if not isinstance(content, str):
            raise GoogleAPIRequestShapeError("Google LLM message content must be text.")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            prefix = "Assistant: " if role == "assistant" else ""
            user_parts.append(prefix + content)
        else:
            raise GoogleAPIRequestShapeError(f"Unsupported Google LLM role: {role!r}.")
    user_text = "\n\n".join(p for p in user_parts if p.strip()).strip()
    if not user_text:
        raise GoogleAPIRequestShapeError("Google LLM request has no user text.")
    system_text = "\n\n".join(p for p in system_parts if p.strip()).strip() or None
    return system_text, user_text


def _response_format_payload(response_format: Any | None) -> dict[str, Any] | None:
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise GoogleAPIRequestShapeError("Google response_format must be a dict.")
    if response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
    elif response_format.get("type") == "json_object":
        schema = response_format.get("schema")
    else:
        schema = response_format.get("schema")
    if schema is None:
        return {"type": "text", "mime_type": "application/json"}
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }


def _extract_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str) and body["output_text"]:
        return body["output_text"]
    texts: list[str] = []
    for step in body.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("type") not in (None, "model_output"):
            continue
        for content in step.get("content") or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                texts.append(content["text"])
                continue
            text_obj = content.get("text")
            if isinstance(text_obj, dict) and isinstance(text_obj.get("text"), str):
                texts.append(text_obj["text"])
    out = "\n".join(t for t in texts if t).strip()
    if not out:
        raise GoogleAPIRequestShapeError(
            f"Google API response contained no text output: {json.dumps(body)[:500]}"
        )
    return out


class GoogleAPIBackend:
    """Zero-VRAM LoaderBackend adapter for Google Gemini Interactions API."""

    def load(self, repo_id: str, row: Any, policy: Any = None) -> dict[str, Any]:
        # S1 policy contract: remote lane -- asserts the lane_allowlist
        # admits it (defense in depth behind request_slot's backstop) and
        # deliberately IGNORES every hardware field; zero local compute.
        if policy is not None and not policy.admits_lane("google_api"):
            raise GoogleAPIError(
                f"{repo_id}: 'google_api' lane is not admitted by the "
                f"profile lane_allowlist {list(policy.lane_allowlist)}."
            )
        model_id = resolve_model_for_slot(repo_id)
        context_window = int(
            getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW)
            or DEFAULT_CONTEXT_WINDOW
        )
        return {
            "provider": GOOGLE_API_PROVIDER,
            "model_id": repo_id,
            "google_model": model_id,
            "context_cap": context_window,
            "context_window": context_window,
        }

    def generate(
        self,
        model: Any,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        stop: Any = None,
        response_format: dict | None = None,
        **_ignored: Any,
    ) -> str:
        cache_entry = model
        google_model = str(cache_entry.get("google_model") or "")
        if not google_model:
            raise GoogleAPIRequestShapeError("Google cache entry missing concrete model id.")
        rf = _response_format_payload(response_format)
        if rf is not None and not supports_structured_output(google_model):
            raise GoogleAPIRequestShapeError(
                f"Google model {google_model!r} is not marked structured-output "
                "capable in OTR's catalog/cache. No request was sent."
            )
        system_text, input_text = _messages_to_google(messages)
        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = float(temperature)
        if max_new_tokens is not None:
            requested_tokens = max(1, int(max_new_tokens))
            try:
                generation_config["max_output_tokens"] = fit_output_tokens(
                    requested_tokens,
                    context_cap=int(
                        cache_entry.get("context_cap") or DEFAULT_CONTEXT_WINDOW
                    ),
                    prompt_tokens=estimate_prompt_tokens(messages),
                    label=f"Google API {google_model}",
                )
            except GenerationContextOverflowError as exc:
                raise GoogleAPIRequestShapeError(str(exc)) from exc
        if stop:
            generation_config["stop_sequences"] = [str(s) for s in stop if s]
        payload: dict[str, Any] = {
            "model": google_model,
            "input": input_text,
            "store": False,
        }
        if system_text:
            payload["system_instruction"] = system_text
        if generation_config:
            payload["generation_config"] = generation_config
        if rf is not None:
            payload["response_format"] = rf
        body = create_interaction(payload)
        return _extract_text(body)

    def unload(self, model: Any) -> None:  # noqa: ARG002
        return None


def make_google_api_generate_fn(cache_entry: dict, *, response_format: dict | None = None):
    backend = GoogleAPIBackend()
    bound_rf = response_format

    def generate_fn(messages, *, temperature=None, max_new_tokens=None,
                    stop=None, response_format=None, grammar=None):
        if grammar is not None:
            raise GoogleAPIRequestShapeError(
                "Google API LLM lane does not support GBNF grammar payloads. "
                "Use response_format/json schema or a local grammar-capable model."
            )
        rf = response_format if response_format is not None else bound_rf
        return backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
        )

    generate_fn._otr_google_api = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    return generate_fn


__all__ = [
    "GoogleAPIBackend",
    "make_google_api_generate_fn",
]
