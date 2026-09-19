"""LoaderBackend adapter and generate_fn for direct Google Gemini LLM calls."""

from __future__ import annotations

import json
import logging
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
    otr_env,
)

_LOG = logging.getLogger("OTR.google_api")

# THINKING EATS THE OUTPUT BUDGET (measured 2026-09-19, first live leg of
# google_veo_low_1act). The Interactions API counts thought tokens against
# `max_output_tokens`, and `gemini-flash-latest` thinks by default: at a
# 256-token budget it spent 242 on thoughts and returned 7 visible tokens
# (`status: incomplete`, truncated JSON, the writer's 250-token description
# call died after 3 attempts). Measured levels, same prompt:
#   gemini-flash-latest       default -> 861-897 thoughts; "low" -> 0; "minimal" REJECTED (400)
#   gemini-flash-lite-latest  default -> 0;              "low" -> 241 (turns thinking ON)
#   gemini-pro-latest         "low"   -> 241 (cannot be switched off)
# So the level is per MODEL, and headroom is the belt under the braces: the
# visible budget the caller asked for is what fit_output_tokens sizes; the
# thinking headroom is added on top (still clamped to the context room) so a
# model that thinks anyway cannot starve the answer it was asked for.
THINKING_HEADROOM_TOKENS = 1024
_THINKING_LEVEL_BY_FAMILY = (
    ("flash-lite", None),   # already silent; a level would switch it ON
    ("flash", "low"),       # measured 0 thought tokens at "low"
)
# gemini-pro-latest cannot stop thinking and spent 1148 thought tokens on the
# same prompt at its default level (2048 budget, completed), so its headroom
# is doubled. Not a shipped pick; a user can select it.
_THINKING_HEADROOM_BY_FAMILY = (
    ("pro", 2048),
)


def thinking_level_for(google_model: str) -> str | None:
    """The `generation_config.thinking_level` to send for a model, or None
    to leave the provider default. `OTR_GOOGLE_THINKING_LEVEL` overrides for
    every model (`default` = send nothing)."""
    override = str(otr_env.get("OTR_GOOGLE_THINKING_LEVEL") or "").strip().lower()
    if override:
        return None if override in ("default", "none", "provider") else override
    name = str(google_model or "").lower()
    for family, level in _THINKING_LEVEL_BY_FAMILY:
        if family in name:
            return level
    return None


def thinking_headroom_tokens(google_model: str = "") -> int:
    """Thought-token headroom added on top of the caller's visible budget.
    `OTR_GOOGLE_THINKING_HEADROOM` overrides for every model."""
    raw = str(otr_env.get("OTR_GOOGLE_THINKING_HEADROOM") or "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    name = str(google_model or "").lower()
    for family, headroom in _THINKING_HEADROOM_BY_FAMILY:
        if family in name:
            return headroom
    return THINKING_HEADROOM_TOKENS


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


def _output_limit_reason(body: Any) -> str:
    """Return a provider limit reason from common interaction response shapes.

    The Interactions API reports a budget-truncated answer as TOP-LEVEL
    ``status: "incomplete"`` (no finish_reason anywhere in the body -- measured
    2026-09-19). Only the top-level status counts: a nested object carrying
    its own ``status`` (a step, a tool call) says nothing about the output
    budget, so that check lives here and not in the recursive walk below."""
    if isinstance(body, dict) and (
            str(body.get("status") or "").strip().lower() == "incomplete"):
        return "incomplete"
    return _nested_finish_limit(body)


def _nested_finish_limit(body: Any) -> str:
    limit_values = {"length", "max_tokens", "max_output_tokens", "max_tokens_reached"}
    if isinstance(body, dict):
        for key, value in body.items():
            if key in {"finish_reason", "finishReason", "stop_reason", "incomplete_reason"}:
                normalized = str(value or "").strip().casefold()
                if normalized in limit_values:
                    return normalized
            found = _nested_finish_limit(value)
            if found:
                return found
    elif isinstance(body, list):
        for value in body:
            found = _nested_finish_limit(value)
            if found:
                return found
    return ""


def _extract_text(
    body: dict[str, Any], *, fail_on_output_limit: bool = False,
) -> str:
    limit_reason = _output_limit_reason(body)
    if limit_reason:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        thought = usage.get("total_thought_tokens")
        visible = usage.get("total_output_tokens")
        if fail_on_output_limit:
            raise GoogleAPIRequestShapeError(
                "Google API exhausted the provider output capacity "
                f"({limit_reason}; thought_tokens={thought} output_tokens={visible}); "
                "the partial artifact is not eligible for reroll"
            )
        # The caller asked for whatever came back, so return it -- but say
        # so, because a truncated JSON body otherwise surfaces three calls
        # later as a bare "no decodable JSON object" with no cause attached.
        # Blame thinking only when the usage shows thought tokens.
        cause = ("a thinking model spent the output budget (see "
                 "thinking_level_for / OTR_GOOGLE_THINKING_HEADROOM)"
                 if isinstance(thought, (int, float)) and thought > 0
                 else "the output budget was exhausted")
        _LOG.warning(
            "[OTR.google_api] response truncated (%s): thought_tokens=%s "
            "output_tokens=%s -- returning the partial text; %s",
            limit_reason, thought, visible, cause)
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
        require_full_output = bool(getattr(
            messages, "_otr_require_full_output_budget", False,
        ))
        reserve_remaining = bool(getattr(
            messages, "_otr_reserve_remaining_output_capacity", False,
        ))
        bounded_capacity = reserve_remaining and max_new_tokens is not None
        fail_on_output_limit = bool(getattr(
            messages, "_otr_fail_on_output_limit", False,
        ))
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
        level = thinking_level_for(google_model)
        if level:
            generation_config["thinking_level"] = level
        if max_new_tokens is not None or reserve_remaining:
            context_cap = int(
                cache_entry.get("context_cap") or DEFAULT_CONTEXT_WINDOW
            )
            prompt_tokens = estimate_prompt_tokens(messages)
            requested_tokens = (
                context_cap if reserve_remaining else max(1, int(max_new_tokens))
            )
            try:
                visible_budget = fit_output_tokens(
                    requested_tokens,
                    context_cap=context_cap,
                    prompt_tokens=prompt_tokens,
                    label=f"Google API {google_model}",
                    require_full=require_full_output or bounded_capacity,
                )
            except GenerationContextOverflowError as exc:
                raise GoogleAPIRequestShapeError(str(exc)) from exc
            # Thought tokens are billed against max_output_tokens, so the
            # visible budget the caller sized gets thinking headroom on top,
            # clamped to the room the context leaves (fit_output_tokens has
            # already guaranteed visible_budget <= that room).
            room = context_cap - prompt_tokens
            generation_config["max_output_tokens"] = min(
                visible_budget + thinking_headroom_tokens(google_model), room)
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
        return _extract_text(
            body, fail_on_output_limit=fail_on_output_limit,
        )

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
        _reply = backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
        )
        # CLOUD RUNAWAY GUARD (2026-08-13) -- see nodes/_otr_decode_guard.py.
        # Post-hoc, because an HTTP call has no token loop to watch. It cannot
        # save the spend; it stops a degenerate reply reaching the ledger and
        # raises the same rerollable phase the local guard raises.
        try:
            from .._otr_decode_guard import assert_no_verbatim_cycle
        except ImportError:  # pragma: no cover - flat/standalone import path
            from _otr_decode_guard import (  # type: ignore
                assert_no_verbatim_cycle,
            )
        assert_no_verbatim_cycle(_reply, label="google_api")
        return _reply

    generate_fn._otr_google_api = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    return generate_fn


__all__ = [
    "GoogleAPIBackend",
    "make_google_api_generate_fn",
]
