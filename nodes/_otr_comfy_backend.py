"""Comfy Credits remote-LLM backend (2026-06-01, four-dropdown router).

The Comfy Credits lane is the sibling of the OpenRouter own-key lane in
`_otr_openrouter_backend`. ComfyUI's credit-billed text path is the
OpenRouter **partner node**: ComfyUI proxies the chat-completion to its
API service and bills the logged-in account's prepaid credits, exposing
the same frontier catalog (Claude / GPT / Gemini / Grok / DeepSeek /
Qwen / Mistral / GLM / Kimi / Perplexity). So this backend is, in shape,
"OpenRouter over Comfy's proxy + Comfy's auth":

  * Catalog: a PINNED constant (`COMFY_LLM_MODELS`) -- the partner node's
    curated model list. No disk cache / refresh script (unlike the
    own-key OpenRouter lane, whose catalog is fetched).
  * Auth: NOT an env key. ComfyUI injects the configured Comfy API key
    into a node via the hidden input `api_key_comfy_org`. The writer node
    captures it at run() and hands it here via `set_auth(...)`. The
    logged-in account's session bearer is never requested: the Comfy
    Registry scan treats a third-party pack declaring that hidden input
    as credential access and flags the version (PBUG-20260902-04).
  * Gate: `OTR_ENABLE_COMFY_CREDITS=1` (opt-in, default-off) -- mirrors
    the OpenRouter enable gate so the offline baseline + the dropdowns
    stay untouched until the operator opts in (C3 parity, no surprise
    charges).
  * Endpoint: the Comfy API base + chat path are env-overridable
    (`OTR_COMFY_API_BASE` / `OTR_COMFY_CHAT_PATH`) so the operator
    confirms the exact proxy surface at the first credit-billed run
    (operator live-smoke), exactly as the OpenRouter lane shipped. A
    mismatch fails closed with a clear, named error -- it never crashes
    Node 1 and never splices a half-written episode.

Isolation: every entry point swallows import/attr hiccups so a Comfy
break degrades (Auto falls back to OpenRouter / local) rather than
taking down the writer (PD1).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from ._otr_generation_budget import (
    GenerationContextOverflowError,
    estimate_prompt_tokens,
    fit_output_tokens,
)

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity + constants
# ---------------------------------------------------------------------------

COMFY_BACKEND_KEY = "comfy_credits_http"
"""The `row.loader_backend` literal the dispatch table keys on."""

PROVIDER = "comfy_credits"
"""The `cache_entry["provider"]` tag the generate-fn factory branches on."""

SLOT_A_ID = "comfy:slot-a"
SLOT_B_ID = "comfy:slot-b"
COMFY_ROW_IDS = frozenset({SLOT_A_ID, SLOT_B_ID})

DEFAULT_CONTEXT_WINDOW = 8192

# The Comfy API proxy surface. Verified 2026-06-01 against the bundled
# partner-node source: path from `comfy_api_nodes/nodes_openrouter.py`
# (OPENROUTER_CHAT_ENDPOINT) and base from `comfy_api_nodes/util/_helpers.py`
# (default_base_url -> args.comfy_api_base or https://api.comfy.org). The
# proxy path carries an `/api/` segment that the pre-verification best-guess
# omitted (BUG-LOCAL-297). Still env-overridable (OTR_COMFY_API_BASE /
# OTR_COMFY_CHAT_PATH) so a future proxy move needs no code change; a
# mismatch fails closed with a named error (never crashes Node 1).
DEFAULT_BASE_URL = "https://api.comfy.org"
DEFAULT_CHAT_PATH = "/proxy/openrouter/api/v1/chat/completions"

# Recommended slugs (the dropdown lead + the resolution fallback). Both are
# present in the Comfy partner-node catalog below. Anthropic tops out at
# claude-opus-4.7 on Comfy's curated list (the own-key OpenRouter lane uses
# 4.8) -- so the Comfy lane carries its own creative default.
COMFY_RECOMMENDED_CREATIVE_DEFAULT = "anthropic/claude-opus-4.7"
COMFY_RECOMMENDED_TECHNICAL_DEFAULT = "google/gemini-3.5-flash"

# CURATED (operator directive 2026-07-04): ONE recognizable, NON-REASONING model
# per major brand, ordered cheapest -> premium. Reasoning models (deepseek-*-pro,
# perplexity sonar-reasoning/deep-research, *-thinking, gpt-5.5-pro, qwen *-max)
# are DELIBERATELY EXCLUDED -- they return empty / no-choices / time out on the
# writer's STRUCTURED-JSON slot calls (proven 2026-07-04: perplexity/sonar-
# reasoning-pro upstream-502'd the news_interpreter; the technical DEFAULT was
# also on a reasoning deepseek-v4-pro). Keep the writer's JSON path on
# text-completion models. gemini-3.5-flash is LIVE-PROVEN (Desktop, 2026-07-04).
#
# RE-CONFIRMED LIVE 2026-08-09, and read this before "modernising" the rule.
# GO_FORWARD had recorded this block as STALE -- the reasoning being that
# essentially every frontier SKU now advertises reasoning, so an exclusion of
# "reasoning models" supposedly no longer describes what the list does. That
# reading conflates two different things. The rule excludes reasoning-BRANDED
# SKUs (the -pro / -thinking / sonar-reasoning tiers) because they EMPIRICALLY
# BREAK STRUCTURED JSON; it never claimed the survivors cannot reason. The list
# still performs exactly that, which is why deepseek-v3.2 sits here labelled
# NON-reasoning while deepseek-*-pro does not.
# The re-confirmation was accidental and therefore worth trusting: an
# unrelated OpenRouter roundtable run on 2026-08-09 put `deepseek/deepseek-v4-pro`
# on a 3-model panel and it returned EMPTY CONTENT with finish_reason=length,
# having spent its entire token budget on hidden reasoning -- the exact failure
# this block was written about, on the exact SKU pattern it names, thirteen
# months of model churn later. The rule is load-bearing. Do not delete it; if
# the slugs need re-dating, that is a separate mechanical pass.
COMFY_LLM_MODELS: tuple[str, ...] = (
    "google/gemini-3.5-flash",        # Gemini   -- LOW/cheap, fast, PROVEN 2026-07-04
    "deepseek/deepseek-v3.2",         # DeepSeek -- cheap general chat (NON-reasoning)
    "mistralai/mistral-large-2512",   # Mistral  -- mid, capable general model
    "x-ai/grok-4.20",                 # Grok     -- mid
    "openai/gpt-5.5",                 # ChatGPT  -- upper-mid (the non-pro, non-reasoning tier)
    "anthropic/claude-opus-4.7",      # Claude   -- UPPER, strongest writer (creative default)
)

# Cost-guard defaults (mirror the OpenRouter lane). Prepaid credits already
# cap spend account-side; these are a second belt-and-suspenders ceiling.
DEFAULT_MAX_TOKENS_PER_CALL = 32768
DEFAULT_MAX_TOKENS_PER_RUN = 300000
DEFAULT_OUTPUT_TOKENS_CAP = 8192
# BUG-LOCAL-301: 1024 (was 512). Parity with the OpenRouter lane's BUG-294
# output-token floor. max_tokens is a CEILING -- a higher floor costs nothing
# on short replies (the model stops at finish_reason=stop and bills only actual
# tokens) but stops a verbose / reasoning technical model from being cut off
# mid-object. The recommended technical default deepseek-v4-pro is a reasoning
# model: it emits chain-of-thought before the JSON, so a 512 floor truncated
# build_news_briefs (finish_reason=length -> JSONDecodeError -> the writer
# halted at news_interpreter). Reasoning-heavy models can still need more --
# bump OTR_COMFY_MIN_OUTPUT_TOKENS, or use a non-reasoning / local technical
# model for the structured-JSON passes.
DEFAULT_MIN_OUTPUT_TOKENS = 1024
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 2

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ComfyCreditsError(RuntimeError):
    """Base for every Comfy Credits backend failure."""


class ComfyCreditsConfigError(ComfyCreditsError):
    """Comfy Credits was requested but the environment is not configured
    (disabled gate / missing auth token / unresolved slug / unset
    endpoint)."""


class ComfyCreditsCostCeilingError(ComfyCreditsError):
    """A call would exceed the configured token ceiling. Raised BEFORE the
    network call -- no credits are spent."""


class ComfyCreditsCallFailedError(ComfyCreditsError):
    """The remote call failed after bounded retries. The run aborts; there
    is no mid-episode fall-back to a local model (the writer-level fallback
    only applies before the first validated call)."""


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _env(name: str) -> str | None:
    v = otr_env.get(name)
    return v if v else None


def _int_env(name: str, default: int) -> int:
    raw = otr_env.get(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


def _float_env(name: str) -> float | None:
    raw = otr_env.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def comfy_credits_enabled() -> bool:
    """C3 gate: the Comfy Credits lane is visible ONLY when the explicit
    enable flag is set. Login + credits are not detectable at INPUT_TYPES
    time, so the env flag is the load-visible opt-in; the actual auth token
    is checked at generate() time. Default-off keeps the offline baseline
    and the dropdowns untouched (mirrors openrouter_enabled)."""
    return otr_env.get("OTR_ENABLE_COMFY_CREDITS", "0") == "1"


def is_comfy_row_id(repo_id: str) -> bool:
    """True for the two virtual handles `comfy:slot-a|b`."""
    return isinstance(repo_id, str) and repo_id in COMFY_ROW_IDS


def _slot_letter(repo_id: str) -> str:
    if repo_id == SLOT_A_ID:
        return "A"
    if repo_id == SLOT_B_ID:
        return "B"
    raise ComfyCreditsConfigError(
        f"not a Comfy Credits virtual row id: {repo_id!r} "
        f"(expected one of {sorted(COMFY_ROW_IDS)})"
    )


def recommended_slug_for_slot(letter: str) -> str:
    """The recommended catalog slug for slot A (creative) / B (technical).
    Honours a per-slot env override, else the recommended constant."""
    up = letter.strip().upper()
    if up == "A":
        return (
            _env("OTR_COMFY_SLOT_A_DEFAULT")
            or COMFY_RECOMMENDED_CREATIVE_DEFAULT
        )
    if up == "B":
        return (
            _env("OTR_COMFY_SLOT_B_DEFAULT")
            or COMFY_RECOMMENDED_TECHNICAL_DEFAULT
            or COMFY_RECOMMENDED_CREATIVE_DEFAULT
        )
    return COMFY_RECOMMENDED_CREATIVE_DEFAULT


# ---------------------------------------------------------------------------
# Slot bindings (module-global, fed from the writer's widget values)
# ---------------------------------------------------------------------------

_slot_bindings: dict[str, str | None] = {"A": None, "B": None}


def _clean_slot_value(v: Any) -> str | None:
    """Normalize a slot-picker widget value to a real slug or None.

    Empty, whitespace, or a parenthesized UI sentinel such as
    ``(enable Comfy Credits)`` means "unset" for this run. This mirrors the
    OpenRouter lane and prevents a long-lived ComfyUI process from reusing a
    previous run's bound slug after the operator turns a slot back to the
    sentinel.
    """
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or (s.startswith("(") and s.endswith(")")):
        return None
    return s


def set_slot_bindings(*, slot_a: Any = None, slot_b: Any = None) -> None:
    """Record this run's slot-picker slug bindings.

    The writer calls this once at the start of every run. Sentinel / empty
    values explicitly clear the per-run binding so resolution falls through to
    the env / recommended chain instead of leaking a previous run's slug.
    """
    _slot_bindings["A"] = _clean_slot_value(slot_a)
    _slot_bindings["B"] = _clean_slot_value(slot_b)


def clear_slot_bindings() -> None:
    _slot_bindings["A"] = None
    _slot_bindings["B"] = None


def resolve_slug(repo_id: str) -> str:
    """Resolve a `comfy:slot-a|b` handle to a real catalog slug.

    Chain: the operator's bound slot pick (verbatim, preserved even if a
    future catalog drops it) -> per-slot env override -> recommended
    constant. A bound slug not in the current catalog is kept with a warn
    (never silently swapped)."""
    letter = _slot_letter(repo_id)
    bound = _slot_bindings.get(letter)
    if bound:
        if bound not in COMFY_LLM_MODELS:
            log.warning(
                "[ComfyCredits] slot %s bound slug %r is not in the pinned "
                "catalog; using it verbatim (catalog may be stale).",
                letter, bound,
            )
        return bound
    return recommended_slug_for_slot(letter)


# ---------------------------------------------------------------------------
# Auth (module-global, fed from the writer's hidden ComfyUI inputs)
# ---------------------------------------------------------------------------

_auth: dict[str, str] = {}


def set_auth(*, api_key: Any = None) -> None:
    """Record the ComfyUI-injected credential for the credit-billed call.

    ComfyUI fills the writer's hidden input `api_key_comfy_org` (a configured
    Comfy API key) at execution time. The writer threads it here. Best-effort:
    a non-string / empty value is ignored. The key is never logged or stamped
    into run meta."""
    if isinstance(api_key, str) and api_key:
        _auth["api_key"] = api_key


def clear_auth() -> None:
    _auth.clear()


def _bearer() -> str | None:
    """The credential to send: the configured Comfy API key (it works on
    non-whitelisted hosts too).

    API-KEY-RESPECTFUL (2026-07-04): the Comfy Credits LLM chat proxy authenticates
    ONLY via ComfyUI's own injected credential (the writer's hidden input
    api_key_comfy_org). We do NOT repurpose the media-lane OTR_COMFY_API_KEY here
    -- it is rejected by the chat proxy (HTTP 401) and mixing key surfaces
    confuses users. Headless cloud LLM = the OpenRouter own-key lane; Comfy
    Credits LLM = the Desktop app signed in with a Comfy API key. The session
    bearer is no longer requested (PBUG-20260902-04)."""
    return _auth.get("api_key")


# ---------------------------------------------------------------------------
# Cost guard (per-run accumulator; reset by the writer each episode)
# ---------------------------------------------------------------------------

_run_token_total = 0


def reset_run_budget() -> None:
    global _run_token_total
    _run_token_total = 0


def _estimate_request_tokens(messages: list[dict], out_tokens: int) -> int:
    """A cheap ~4-chars-per-token estimate of prompt + output, used only by
    the pre-call ceiling. Never bills; the real spend is account-side."""
    chars = 0
    for m in messages or ():
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            chars += len(c)
    return (chars // 4) + int(out_tokens or 0)


def catalog_models() -> list[dict]:
    """The pinned catalog as id-dicts (shape parity with the OpenRouter
    cache so the dropdown builder can share filtering logic)."""
    return [{"id": slug} for slug in COMFY_LLM_MODELS]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _post_comfy_chat_completion(
    *, url: str, bearer: str, payload: dict, timeout_s: int,
) -> dict:
    """POST one chat-completion to the Comfy proxy and return parsed JSON.

    Isolated so tests patch it with a fake (no network in CI). `requests`
    is imported lazily so the module stays import-safe + network-free."""
    import requests  # lazy: keep module import-safe + network-free

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
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
    except Exception:  # noqa: BLE001 -- a non-JSON error body is tolerated
        return None


def _chat_url() -> str:
    base = (_env("OTR_COMFY_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
    path = _env("OTR_COMFY_CHAT_PATH") or DEFAULT_CHAT_PATH
    if not path.startswith("/"):
        path = "/" + path
    return base + path


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class ComfyCreditsBackend:
    """LoaderBackend adapter for credit-billed Comfy partner-node models.

    `load()` builds a provider-tagged cache_entry carrying the resolved
    slug -- no weights, no tokenizer, zero local VRAM. `generate()` posts
    the chat request behind the cost guard + bounded retries. `unload()`
    is a no-op (nothing is resident; the resident local model is never
    touched -- C2 no-evict)."""

    def load(self, repo_id: str, row: Any, policy: Any = None) -> dict[str, Any]:
        # S1 policy contract: remote lane -- asserts the lane_allowlist
        # admits it (defense in depth behind request_slot's backstop) and
        # deliberately IGNORES every hardware field; zero local compute.
        # Auth stays the hidden comfy-org API-key input captured by set_auth()
        # from the writer's run() -- the policy carries NO credentials.
        if policy is not None and not policy.admits_lane("comfy_credits"):
            raise ComfyCreditsConfigError(
                f"{repo_id}: 'comfy_credits' lane is not admitted by the "
                f"profile lane_allowlist {list(policy.lane_allowlist)}."
            )
        if not comfy_credits_enabled():
            raise ComfyCreditsConfigError(
                f"{repo_id} selected but Comfy Credits is not enabled. Set "
                f"OTR_ENABLE_COMFY_CREDITS=1 and log in to a Comfy account "
                f"with credits (see "
                f"https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
                f"v2.0-alpha/docs/comfy-credits-setup.md)."
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
            "temperature_override": _float_env(f"OTR_COMFY_{letter}_TEMP"),
            "max_tokens_cap": _int_env(
                f"OTR_COMFY_{letter}_MAXTOK", DEFAULT_OUTPUT_TOKENS_CAP
            ),
            # no "model" / "tokenizer" keys: the generate-fn factory
            # branches on provider BEFORE requiring them.
        }
        log.info(
            "[ComfyCredits] load slot=%s handle=%s slug=%s ctx=%d "
            "(credit-billed, 0 local VRAM)",
            letter, repo_id, slug, context_window,
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
        **_ignored: Any,
    ) -> str:
        """Run one credit-billed chat completion and return the decoded
        string. `model` is the cache_entry from load(). Enforces the token
        ceiling BEFORE the call, retries transient failures a bounded
        number of times, then aborts cleanly."""
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
        bearer = _bearer()
        if not bearer:
            raise ComfyCreditsConfigError(
                "No Comfy credential available. Log in to a Comfy account "
                "(or set a Comfy API key) so ComfyUI injects the hidden "
                "auth into the writer node. See "
                "https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
                "v2.0-alpha/docs/comfy-credits-setup.md."
            )
        slug = cache_entry.get("slug") or resolve_slug(cache_entry["model_id"])

        cap = int(cache_entry.get("max_tokens_cap") or DEFAULT_OUTPUT_TOKENS_CAP)
        floor = _int_env("OTR_COMFY_MIN_OUTPUT_TOKENS", DEFAULT_MIN_OUTPUT_TOKENS)
        requested_tokens = max(1, int(max_new_tokens or 0))
        if (require_full_output or bounded_capacity) and requested_tokens > cap:
            capacity = GenerationContextOverflowError(
                f"Comfy Credits {slug} cannot fit the complete requested "
                f"output: requested_output={requested_tokens}, "
                f"provider_output_cap={cap}"
            )
            raise ComfyCreditsConfigError(str(capacity)) from capacity
        if reserve_remaining:
            out_tokens = cap
        elif bool(getattr(messages, "_otr_strict_remote_output_budget", False)):
            out_tokens = requested_tokens
        else:
            out_tokens = max(requested_tokens, floor)
        if out_tokens > cap:
            out_tokens = cap
        try:
            fitted_tokens = fit_output_tokens(
                out_tokens,
                context_cap=int(cache_entry.get("context_cap") or DEFAULT_CONTEXT_WINDOW),
                prompt_tokens=estimate_prompt_tokens(messages),
                min_output_tokens=min(floor, cap),
                label=f"Comfy Credits {slug}",
                require_full=require_full_output or bounded_capacity,
            )
        except GenerationContextOverflowError as exc:
            raise ComfyCreditsConfigError(str(exc)) from exc
        if fitted_tokens != out_tokens:
            log.warning(
                "[ComfyCredits] output budget clamped: requested=%d effective=%d "
                "context_cap=%d",
                out_tokens, fitted_tokens,
                int(cache_entry.get("context_cap") or DEFAULT_CONTEXT_WINDOW),
            )
        out_tokens = fitted_tokens
        temp = (
            cache_entry.get("temperature_override")
            if cache_entry.get("temperature_override") is not None
            else temperature
        )

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
            bearer=bearer, payload=payload, slug=slug,
            fail_on_output_limit=fail_on_output_limit,
        )
        self._account_spend(est)
        return text

    def unload(self, model: Any) -> None:  # noqa: ARG002
        """No-op: a credit-billed entry holds no VRAM and no resident
        weights, and must never touch the resident local model (C2)."""
        return None

    # --- internals ------------------------------------------------------

    def _enforce_cost_ceiling(self, est_tokens: int, *, slug: str) -> None:
        per_call = _int_env("OTR_COMFY_MAX_TOKENS_PER_CALL", DEFAULT_MAX_TOKENS_PER_CALL)
        per_run = _int_env("OTR_COMFY_MAX_TOKENS_PER_RUN", DEFAULT_MAX_TOKENS_PER_RUN)
        if est_tokens > per_call:
            raise ComfyCreditsCostCeilingError(
                f"Comfy Credits call aborted: estimated {est_tokens} tokens "
                f"exceeds OTR_COMFY_MAX_TOKENS_PER_CALL={per_call} "
                f"(slug={slug}). No request was sent."
            )
        if _run_token_total + est_tokens > per_run:
            raise ComfyCreditsCostCeilingError(
                f"Comfy Credits call aborted: this call (~{est_tokens} tokens) "
                f"would push the run total {_run_token_total} over "
                f"OTR_COMFY_MAX_TOKENS_PER_RUN={per_run}. No request was sent."
            )

    def _account_spend(self, est_tokens: int) -> None:
        global _run_token_total
        _run_token_total += int(est_tokens)
        log.info(
            "[ComfyCredits] call accounted ~%d tokens (run total ~%d)",
            est_tokens, _run_token_total,
        )

    def _post_with_retries(
        self, *, bearer: str, payload: dict, slug: str,
        fail_on_output_limit: bool = False,
    ) -> str:
        url = _chat_url()
        timeout_s = _int_env("OTR_COMFY_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        max_retries = _int_env("OTR_COMFY_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                result = _post_comfy_chat_completion(
                    url=url, bearer=bearer, payload=payload, timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 -- network/transport error
                last_err = f"transport error: {type(exc).__name__}: {exc}"
                log.warning(
                    "[ComfyCredits] attempt %d/%d failed (%s)",
                    attempt + 1, max_retries + 1, last_err,
                )
                self._sleep_backoff(attempt)
                continue

            status = int(result.get("status_code") or 0)
            if status == 200:
                return self._extract_text(
                    result, slug=slug,
                    fail_on_output_limit=fail_on_output_limit,
                )

            last_err = f"HTTP {status}: {self._error_snippet(result)}"
            if status in _RETRYABLE_STATUS and attempt < max_retries:
                log.warning(
                    "[ComfyCredits] attempt %d/%d retryable (%s)",
                    attempt + 1, max_retries + 1, last_err,
                )
                self._sleep_backoff(attempt)
                continue
            break

        raise ComfyCreditsCallFailedError(
            f"Comfy Credits call to {slug} failed after {max_retries + 1} "
            f"attempt(s): {last_err}. Endpoint was {url!r} -- if this is the "
            f"first credit-billed run, confirm OTR_COMFY_API_BASE / "
            f"OTR_COMFY_CHAT_PATH against the live Comfy proxy. Aborting the "
            f"run (no mid-episode fall-back to local)."
        )

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = min(2.0, 0.25 * (2 ** attempt))
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _extract_text(
        result: dict, *, slug: str, fail_on_output_limit: bool = False,
    ) -> str:
        body = result.get("json") or {}
        if otr_env.get("OTR_COMFY_DEBUG_RAW") == "1":
            log.info("[ComfyCredits] raw %s response: %s", slug, json.dumps(body)[:1500])
        choices = body.get("choices") or []
        if not choices:
            raise ComfyCreditsCallFailedError(
                f"Comfy Credits {slug} returned no choices: {str(body)[:300]}"
            )
        if choices[0].get("finish_reason") == "length":
            if fail_on_output_limit:
                raise ComfyCreditsCallFailedError(
                    f"Comfy Credits {slug} exhausted the provider output "
                    "capacity; the partial artifact is not eligible for reroll"
                )
            log.warning(
                "[ComfyCredits] %s hit finish_reason=length -- output "
                "truncated at the token ceiling; raise "
                "OTR_COMFY_MIN_OUTPUT_TOKENS or the slot max-tokens cap.",
                slug,
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") in (None, "text")
            )
        if (not isinstance(content, str) or not content) and message.get("reasoning"):
            content = str(message.get("reasoning"))
        if not isinstance(content, str) or not content:
            raise ComfyCreditsCallFailedError(
                f"Comfy Credits {slug} returned empty message content "
                f"(finish_reason={choices[0].get('finish_reason')!r})."
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


def make_comfy_credits_generate_fn(cache_entry: dict, *, response_format: dict | None = None):
    """Return a generate_fn closure for a provider-tagged Comfy cache_entry.

    Matches the signature every OTR generate_fn uses --
    ``(messages, *, temperature, max_new_tokens, stop=None) -> str`` -- so
    the loader factories and the writer's `_build_truncating_generate_fn`
    return it unchanged when ``cache_entry["provider"] == "comfy_credits"``.
    """
    backend = ComfyCreditsBackend()
    bound_rf = response_format

    def generate_fn(messages, *, temperature=None, max_new_tokens=None,
                    stop=None, response_format=None):
        rf = response_format if response_format is not None else bound_rf
        _reply = backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
        )
        # CLOUD RUNAWAY GUARD (2026-08-13) -- see _otr_decode_guard.
        # Post-hoc, because an HTTP call has no token loop to watch. It cannot
        # save the spend; it stops a degenerate reply reaching the ledger and
        # raises the same rerollable phase the local guard raises.
        try:
            from ._otr_decode_guard import assert_no_verbatim_cycle
        except ImportError:  # pragma: no cover - flat/standalone import path
            from _otr_decode_guard import (  # type: ignore
                assert_no_verbatim_cycle,
            )
        assert_no_verbatim_cycle(_reply, label="comfy_credits")
        return _reply

    generate_fn._otr_comfy_credits = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    return generate_fn


def comfy_run_meta() -> dict[str, Any]:
    """Run-meta provenance: the slug each Comfy-bound slot resolves to.
    The resolved slug is a PUBLIC model id (not a secret); the auth token
    is never read or stamped. Never raises (PD1)."""
    out: dict[str, Any] = {}
    for letter, handle in (("A", SLOT_A_ID), ("B", SLOT_B_ID)):
        try:
            out[f"slot_{letter.lower()}_resolved_slug"] = resolve_slug(handle)
        except ComfyCreditsError:
            out[f"slot_{letter.lower()}_resolved_slug"] = "<unresolved>"
    return out


__all__ = [
    "COMFY_BACKEND_KEY",
    "PROVIDER",
    "SLOT_A_ID",
    "SLOT_B_ID",
    "COMFY_ROW_IDS",
    "COMFY_LLM_MODELS",
    "COMFY_RECOMMENDED_CREATIVE_DEFAULT",
    "COMFY_RECOMMENDED_TECHNICAL_DEFAULT",
    "DEFAULT_CONTEXT_WINDOW",
    "ComfyCreditsError",
    "ComfyCreditsConfigError",
    "ComfyCreditsCostCeilingError",
    "ComfyCreditsCallFailedError",
    "comfy_credits_enabled",
    "is_comfy_row_id",
    "recommended_slug_for_slot",
    "resolve_slug",
    "set_slot_bindings",
    "clear_slot_bindings",
    "set_auth",
    "clear_auth",
    "reset_run_budget",
    "catalog_models",
    "ComfyCreditsBackend",
    "make_comfy_credits_generate_fn",
    "comfy_run_meta",
]
