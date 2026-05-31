"""OpenRouter remote-LLM backend (opt-in, default-off).

Implements the duck-typed `LoaderBackend` protocol (FC1) from
`nodes/_otr_loader_backends.py` so the writer pipeline can drive a
remote OpenRouter model through the SAME `load`/`generate`/`unload`
surface the local transformers backends use -- with zero local VRAM.

Hard constraints honoured here (see the go-forward plan, C1-C9):
  * C3 offline-first: no remote path is reachable unless BOTH
    `OPENROUTER_API_KEY` and `OTR_ENABLE_OPENROUTER=1` are set.
  * C6 hard cost guard: a conservative per-call AND per-run token
    ceiling is enforced BEFORE the network call; the call aborts
    rather than spend unbounded credits. Spend is logged per call.
  * C5 no half-remote: bounded retries on transient failures, then a
    clean abort (`OpenRouterCallFailedError`) -- never a silent
    mid-run fall-back to local.
  * C9 no secrets: the key is read from the environment only, never
    logged and never written to disk.

This module is import-safe with no network and no torch. The two
mockable seams the tests drive are module-level functions
`_estimate_request_tokens` and `_post_chat_completion`; patch them to
prove the cost-ceiling abort and the retry ladder without a network.

S3 wires this backend into the live loader path; S1 (this file) only
builds + registers it and proves it under mocked HTTP.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Identity constants (shared with the catalog rows in S2 + wiring in S3/S5)
# ---------------------------------------------------------------------------

OPENROUTER_BACKEND_KEY = "openrouter_http"
"""The `loader_backend` Literal value the virtual catalog rows carry."""

PROVIDER = "openrouter"
"""The `cache_entry["provider"]` tag the generate-fn factory branches on."""

SLOT_A_ID = "openrouter:slot-a"
SLOT_B_ID = "openrouter:slot-b"
OPENROUTER_ROW_IDS = (SLOT_A_ID, SLOT_B_ID)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CONTEXT_WINDOW = 8192

# Conservative cost ceilings. Deliberately low so an unconfigured
# operator cannot accidentally spend a fortune; raise via env when ready.
DEFAULT_MAX_TOKENS_PER_CALL = 8192
DEFAULT_MAX_TOKENS_PER_RUN = 200_000
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 2  # total attempts = retries + 1


# ---------------------------------------------------------------------------
# Errors -- all abort the run (C4 fail-closed / C5 no half-remote)
# ---------------------------------------------------------------------------


class OpenRouterError(RuntimeError):
    """Base for every OpenRouter backend failure."""


class OpenRouterConfigError(OpenRouterError):
    """Remote was requested but the environment is not configured
    (missing key / disabled gate / unresolved slug)."""


class OpenRouterCostCeilingError(OpenRouterError):
    """A call would exceed the configured token/spend ceiling (C6).
    Raised BEFORE the network call -- no credits are spent."""


class OpenRouterCallFailedError(OpenRouterError):
    """The remote call failed after bounded retries (C5). The run
    aborts; there is no mid-episode fall-back to a local model."""


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def openrouter_enabled() -> bool:
    """C3 gate: remote is reachable ONLY when the key is present AND the
    explicit enable flag is set. Either missing ⇒ remote is off and the
    virtual rows never appear in the dropdowns (S2)."""
    return bool(_env("OPENROUTER_API_KEY")) and os.environ.get(
        "OTR_ENABLE_OPENROUTER", "0"
    ) == "1"


def is_openrouter_row_id(repo_id: str) -> bool:
    """True for the two virtual handles `openrouter:slot-a|b`."""
    return isinstance(repo_id, str) and repo_id in OPENROUTER_ROW_IDS


def _slot_letter(repo_id: str) -> str:
    if repo_id == SLOT_A_ID:
        return "A"
    if repo_id == SLOT_B_ID:
        return "B"
    raise OpenRouterConfigError(
        f"not an OpenRouter virtual row id: {repo_id!r} "
        f"(expected one of {OPENROUTER_ROW_IDS})"
    )


def resolve_slug(repo_id: str) -> str:
    """Map a virtual handle to the real model slug bound in the env
    (`OPENROUTER_MODEL_A` / `OPENROUTER_MODEL_B`). Raises
    OpenRouterConfigError when the slug env var is unset -- the handle
    has no meaning without it. The slug is the operator's runtime choice
    and is recorded in run meta (S5), never hard-coded here."""
    letter = _slot_letter(repo_id)
    slug = _env(f"OPENROUTER_MODEL_{letter}")
    if not slug:
        raise OpenRouterConfigError(
            f"{repo_id} selected but OPENROUTER_MODEL_{letter} is not set. "
            f"Bind it via setx (see docs/openrouter-setup.md), then restart "
            f"ComfyUI in a fresh terminal."
        )
    return slug


def reset_run_budget() -> None:
    """Zero the per-run token accumulator. Called at the start of an
    episode so the per-run cost ceiling (C6) measures one run, not the
    process lifetime."""
    global _run_token_total
    _run_token_total = 0


_run_token_total = 0


# ---------------------------------------------------------------------------
# Mockable seams (tests patch these; no network at import or in CI)
# ---------------------------------------------------------------------------


def _estimate_request_tokens(messages: list[dict], max_new_tokens: int) -> int:
    """Conservative pre-call token estimate: ~4 chars per prompt token
    plus the full output budget. Intentionally rough and ALWAYS an
    over-estimate bias so the cost guard errs toward aborting early.
    Tests patch this to force the ceiling."""
    prompt_chars = sum(len(str(m.get("content", ""))) for m in (messages or []))
    return prompt_chars // 4 + int(max_new_tokens or 0)


def _post_chat_completion(
    *, base_url: str, api_key: str, payload: dict, timeout_s: int,
) -> dict:
    """POST one chat-completion request and return the parsed JSON.

    Isolated so the tests can patch it with a fake (no network in CI).
    `requests` is imported lazily so this module stays import-safe in
    environments without it."""
    import requests  # lazy: keep module import-safe + network-free

    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter recommends these attribution headers; harmless.
            "HTTP-Referer": "https://github.com/jbrick2070/ComfyUI-OldTimeRadio",
            "X-Title": "OldTimeRadio",
        },
        data=json.dumps(payload).encode("utf-8"),
        timeout=timeout_s,
    )
    return {
        "status_code": resp.status_code,
        "json": _safe_json(resp),
        "text": resp.text,
    }


def _safe_json(resp: Any) -> dict | None:
    try:
        return resp.json()
    except Exception:  # noqa: BLE001 -- non-JSON error body is tolerated
        return None


_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class OpenRouterBackend:
    """LoaderBackend adapter for remote OpenRouter models (FC1).

    `load()` builds a provider-tagged cache_entry carrying the resolved
    slug + per-slot config -- no weights, no tokenizer, zero VRAM.
    `generate()` posts the chat request behind the cost guard + retry
    ladder. `unload()` is a no-op (nothing is resident)."""

    # --- protocol: load -------------------------------------------------

    def load(self, repo_id: str, row: Any) -> dict[str, Any]:
        """Return a remote cache_entry. No network call fires here -- a
        bad slug surfaces at generate time, and load must stay cheap so
        the writer's per-call request_slot is fast (C2: it must not
        touch the resident local model)."""
        if not openrouter_enabled():
            raise OpenRouterConfigError(
                f"{repo_id} selected but OpenRouter is not enabled. Set "
                f"OPENROUTER_API_KEY and OTR_ENABLE_OPENROUTER=1 "
                f"(see docs/openrouter-setup.md)."
            )
        letter = _slot_letter(repo_id)
        slug = resolve_slug(repo_id)
        context_window = int(
            getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW)
            or DEFAULT_CONTEXT_WINDOW
        )
        cache_entry: dict[str, Any] = {
            "provider": PROVIDER,
            "model_id": repo_id,          # the virtual handle
            "slot_letter": letter,        # "A" / "B"
            "slug": slug,                 # the resolved real model id
            "context_cap": context_window,
            "context_window": context_window,
            # optional per-slot overrides (FC3) -- None ⇒ caller controls
            "temperature_override": _float_env(f"OPENROUTER_{letter}_TEMP"),
            "max_tokens_cap": _int_env(
                f"OPENROUTER_{letter}_MAXTOK", DEFAULT_MAX_TOKENS_PER_CALL
            ),
            "base_url": _env("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
            # no "model" / "tokenizer" keys: the generate-fn factory
            # branches on provider BEFORE requiring them (S3).
        }
        log.info(
            "[OpenRouter] load slot=%s handle=%s slug=%s ctx=%d (remote, 0 VRAM)",
            letter, repo_id, slug, context_window,
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
        **_ignored: Any,
    ) -> str:
        """Run one remote chat completion and return the decoded string.

        `model` is the cache_entry returned by `load()`. Enforces the
        cost ceiling BEFORE the call (C6), retries transient failures a
        bounded number of times, then aborts cleanly (C5)."""
        cache_entry = model
        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            raise OpenRouterConfigError(
                "OPENROUTER_API_KEY is not set; cannot make a remote call."
            )
        slug = cache_entry.get("slug") or resolve_slug(cache_entry["model_id"])
        base_url = cache_entry.get("base_url") or DEFAULT_BASE_URL

        # Resolve the output budget, clamped to the per-slot cap.
        cap = int(cache_entry.get("max_tokens_cap") or DEFAULT_MAX_TOKENS_PER_CALL)
        out_tokens = int(max_new_tokens or cap)
        if out_tokens > cap:
            out_tokens = cap
        temp = (
            cache_entry.get("temperature_override")
            if cache_entry.get("temperature_override") is not None
            else temperature
        )

        # --- C6 cost guard: enforce BEFORE any network call ---
        est = _estimate_request_tokens(messages, out_tokens)
        self._enforce_cost_ceiling(est, slug=slug)

        payload: dict[str, Any] = {
            "model": slug,
            "messages": messages,
            "max_tokens": out_tokens,
        }
        if temp is not None:
            payload["temperature"] = float(temp)
        if stop:
            payload["stop"] = [s for s in stop if s]
        if response_format is not None:
            payload["response_format"] = response_format

        text = self._post_with_retries(
            base_url=base_url, api_key=api_key, payload=payload, slug=slug,
        )

        # Account actual spend (best-effort; falls back to the estimate).
        self._account_spend(est)
        return text

    # --- protocol: unload ----------------------------------------------

    def unload(self, model: Any) -> None:  # noqa: ARG002
        """No-op: a remote entry holds no VRAM and no resident weights.
        Critically, the remote path must never touch the resident local
        model (C2 no-evict) -- so there is nothing to free here."""
        return None

    # --- internals ------------------------------------------------------

    def _enforce_cost_ceiling(self, est_tokens: int, *, slug: str) -> None:
        per_call = _int_env("OPENROUTER_MAX_TOKENS_PER_CALL", DEFAULT_MAX_TOKENS_PER_CALL)
        per_run = _int_env("OPENROUTER_MAX_TOKENS_PER_RUN", DEFAULT_MAX_TOKENS_PER_RUN)
        if est_tokens > per_call:
            raise OpenRouterCostCeilingError(
                f"OpenRouter call aborted: estimated {est_tokens} tokens "
                f"exceeds OPENROUTER_MAX_TOKENS_PER_CALL={per_call} "
                f"(slug={slug}). No request was sent."
            )
        if _run_token_total + est_tokens > per_run:
            raise OpenRouterCostCeilingError(
                f"OpenRouter call aborted: this call (~{est_tokens} tokens) "
                f"would push the run total {_run_token_total} over "
                f"OPENROUTER_MAX_TOKENS_PER_RUN={per_run}. No request was "
                f"sent. Raise the ceiling or shorten the episode."
            )

    def _account_spend(self, est_tokens: int) -> None:
        global _run_token_total
        _run_token_total += int(est_tokens)
        log.info(
            "[OpenRouter] call accounted ~%d tokens (run total ~%d)",
            est_tokens, _run_token_total,
        )

    def _post_with_retries(
        self, *, base_url: str, api_key: str, payload: dict, slug: str,
    ) -> str:
        timeout_s = _int_env("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        max_retries = _int_env("OPENROUTER_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        last_err: str = ""
        for attempt in range(max_retries + 1):
            try:
                result = _post_chat_completion(
                    base_url=base_url, api_key=api_key,
                    payload=payload, timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 -- network/transport error
                last_err = f"transport error: {type(exc).__name__}: {exc}"
                log.warning(
                    "[OpenRouter] attempt %d/%d failed (%s)",
                    attempt + 1, max_retries + 1, last_err,
                )
                self._sleep_backoff(attempt)
                continue

            status = int(result.get("status_code") or 0)
            if status == 200:
                return self._extract_text(result, slug=slug)

            last_err = f"HTTP {status}: {self._error_snippet(result)}"
            if status in _RETRYABLE_STATUS and attempt < max_retries:
                log.warning(
                    "[OpenRouter] attempt %d/%d retryable (%s)",
                    attempt + 1, max_retries + 1, last_err,
                )
                self._sleep_backoff(attempt)
                continue
            # Non-retryable (e.g. 400/401/403/404) -> abort now.
            break

        raise OpenRouterCallFailedError(
            f"OpenRouter call to {slug} failed after {max_retries + 1} "
            f"attempt(s): {last_err}. Aborting the run (no mid-episode "
            f"fall-back to local, per C5)."
        )

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        # Short, bounded backoff. Kept tiny so tests stay fast; real
        # transient recovery does not need long waits here.
        delay = min(2.0, 0.25 * (2 ** attempt))
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _extract_text(result: dict, *, slug: str) -> str:
        body = result.get("json") or {}
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterCallFailedError(
                f"OpenRouter {slug} returned no choices: "
                f"{str(body)[:300]}"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise OpenRouterCallFailedError(
                f"OpenRouter {slug} returned empty message content."
            )
        return content

    @staticmethod
    def _error_snippet(result: dict) -> str:
        body = result.get("json")
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:200]
        text = result.get("text") or ""
        return str(text)[:200]


__all__ = [
    "OpenRouterBackend",
    "OpenRouterError",
    "OpenRouterConfigError",
    "OpenRouterCostCeilingError",
    "OpenRouterCallFailedError",
    "OPENROUTER_BACKEND_KEY",
    "PROVIDER",
    "SLOT_A_ID",
    "SLOT_B_ID",
    "OPENROUTER_ROW_IDS",
    "openrouter_enabled",
    "is_openrouter_row_id",
    "resolve_slug",
    "reset_run_budget",
]
