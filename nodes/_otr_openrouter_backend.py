"""OpenRouter remote-LLM backend (opt-in, default-off).

Implements the duck-typed `LoaderBackend` protocol (FC1) from
`nodes/_otr_loader_backends.py` so the writer pipeline can drive a
remote OpenRouter model through the SAME `load`/`generate`/`unload`
surface the local transformers backends use -- with zero local VRAM.

Hard constraints honoured here (see the go-forward plan, C1-C9):
  * offline-first: no remote path is reachable unless `OPENROUTER_API_KEY`
    is set (C6: the separate OTR_ENABLE_OPENROUTER opt-in flag gate was removed).
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

import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ._otr_generation_budget import (
    CAPACITY_PHASE_OUTPUT_LIMIT,
    GenerationContextOverflowError,
    PromptContextOverflowError,
    estimate_prompt_tokens,
    fit_output_tokens,
)

log = logging.getLogger(__name__)

#: One-shot guard so the "reasoning_effort ACTIVE" evidence line logs once/process.
_REASONING_EFFORT_LOGGED = False
_MANDATORY_REASONING_SLUGS: set[str] = set()
_MANDATORY_REASONING_OVERRIDE_LOGGED: set[str] = set()
# An upstream can select a different provider endpoint for a structured request
# than for an ordinary completion. Capability learning belongs to the resolved
# slug and structured-output mode, not the slug globally.
_TEMPERATURE_UNSUPPORTED_ROUTES: set[tuple[str, bool]] = set()

# ---------------------------------------------------------------------------
# Reasoning-wrapper strip (BUG-306 / BUG-LOCAL-308 family)
# ---------------------------------------------------------------------------
# Thinking-mode models (gemma-4, DeepSeek-R1, QwQ, ...) prepend their
# chain-of-thought to the reply, wrapped in <think>...</think> -- and some
# emit OpenAI-"harmony" <|channel|>analysis<|message|>...<|channel|>final
# <|message|> markers. Some OpenAI-compatible endpoints put that scaffolding
# inline in message.content, so the writer's structured (JSON / GBNF) passes
# parse the reasoning preamble and abort the episode.
# We strip it at the single response chokepoint (_extract_text) so every
# downstream parser sees the clean answer. The strip is a strict no-op when
# none of the markers are present, so non-thinking models (mistral-nemo,
# claude, ...) are byte-for-byte unaffected.
_THINK_PAIR_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|(?:end|return|channel)\|>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_HARMONY_HEADER_RE = re.compile(
    r"<\|channel\|>.*?<\|message\|>", re.DOTALL | re.IGNORECASE
)


def _strip_reasoning_tags(text: str) -> str:
    """Remove thinking-mode reasoning scaffolding from a model reply so the
    writer's structured passes parse clean output (BUG-306/308 family).

    Handles, in order:
      1. OpenAI-"harmony" channels -- keep only the *final* channel's
         message when present; otherwise drop analysis headers + sentinels.
      2. Well-formed ``<think>...</think>`` blocks (removed entirely).
      3. A dangling ``</think>`` with no open -- some servers prefill the
         opening ``<think>`` in the chat template, so the completion carries
         only the close; keep everything after the LAST ``</think>``.
      4. A leading dangling ``<think>`` with no close.

    Returns the input unchanged when none of the markers appear.
    """
    if not isinstance(text, str) or not text:
        return text
    low = text.lower()
    if "<think" not in low and "</think" not in low and "<|channel|>" not in low:
        return text

    out = text

    # 1. Harmony channels: keep the final channel's message if present.
    if "<|channel|>" in out.lower():
        finals = _HARMONY_FINAL_RE.findall(out)
        if finals:
            out = finals[-1]
        else:
            out = _HARMONY_HEADER_RE.sub("", out)
            for sentinel in ("<|end|>", "<|return|>", "<|start|>", "<|channel|>"):
                out = out.replace(sentinel, "")

    # 2. Balanced <think>...</think> blocks.
    out = _THINK_PAIR_RE.sub("", out)

    # 3. Dangling close (template pre-filled the open): keep text after it.
    low2 = out.lower()
    if "</think" in low2:
        idx = low2.rfind("</think")
        gt = out.find(">", idx)
        out = out[gt + 1:] if gt != -1 else out

    # 4. Leading dangling open with no close.
    out = re.sub(r"^\s*<think\b[^>]*>", "", out, flags=re.IGNORECASE)

    return out.strip()


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

# Recommended OpenRouter default slugs (the drift-prone constants flagged in
# the 2026-06-01 go-forward plan, open-question 2). A fresh node offers these
# as the leading slot picks when the operator has set no per-slot
# OTR_OPENROUTER_SLOT_x_DEFAULT override. They are recommended STARTING points,
# never a cage -- any cached slug can be chosen. Creative favours a strong
# narrative model; technical favours a reliably structured-output model.
# CREATIVE is an ALIAS (chunk B, 2026-08-09). It was pinned to
# `anthropic/claude-opus-4.8`, which was ALREADY a version behind -- opus-5 was
# live at the identical price ($5/$25 per M) while the pin still said 4.8. That
# is the whole failure mode the curation policy exists to stop, and a pin cannot
# notice it has gone stale. Replay is unaffected: the ledger stamps the RESOLVED
# concrete model, proven by tests/test_openrouter_resolved.py.
# BOTH SLOTS DEFAULT TO THE AUTO-ROUTER (operator, 2026-08-10: "lets make auto
# default -- for both openrouter A/B").
#
# THIS OVERRIDES THE RULE DIRECTLY ABOVE IT, and the rule was not wrong. A router
# picks by criteria this pack does not control, so a router default is a config
# that resolves differently week to week -- which is exactly what the previous
# defaults were chosen to avoid. The operator was told that, twice, and decided
# for it anyway. It is his call and it is recorded here rather than argued with.
#
# WHAT IT COSTS, so nobody rediscovers it as a bug:
#   * The writer model can differ between episodes, and between two passes of one
#     episode. The 2026-08-04 "story quality is DONE, stop chasing it" directive
#     settled prose variance; a router default reopens that door by design.
#   * Price is `-1` -- not published, not discoverable in advance. Unlike a
#     `~latest` alias, whose real price IS in the catalog and merely moves when
#     the vendor ships, there is nothing to look up.
#   * `response_format` on the router row says the ROUTER accepts the parameter,
#     not that today's route honours it. An unparseable structured pass on a
#     router run is expected variance, not a new defect.
#
# WHAT MAKES IT SURVIVABLE: this pack stamps the RESOLVED model. A router run is
# fully auditable after the fact (`meta["resolved_models"]`,
# tests/test_openrouter_resolved.py), so "which model wrote this episode, and
# what did it cost" always has an answer -- just not before the run.
#
# The previous defaults, kept here because reverting should be a one-line edit
# rather than an archaeology exercise:
#   creative  ~anthropic/claude-opus-latest
#   technical deepseek/deepseek-v4-pro   (concrete on purpose -- the only
#             DeepSeek pointer is the FLASH tier, so aliasing it would be a
#             capability DROP on the slot that most needs structured output)
OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT = "openrouter/auto"
OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT = "openrouter/auto"

# Conservative cost ceilings. Deliberately low so an unconfigured
# operator cannot accidentally spend a fortune; raise via env when ready.
# Defect C fix: the COST per-call ceiling must sit ABOVE the OUTPUT cap, or a
# reply near the output cap (+ the prompt added by the estimate) spuriously
# trips OpenRouterCostCeilingError. So they are two separate numbers.
DEFAULT_OUTPUT_TOKENS_CAP = 16384     # ceiling on OUTPUT tokens (max_tokens) per call
# R3 (2026-06-22): bumped 8192 -> 16384 as the 0-line GUARD for the frontier
# reasoning-effort default (below). A reasoning model spends part of the output
# budget on hidden reasoning; at the old 8192 ceiling a low-effort run could
# starve the story body (finish_reason=length -> empty/truncated -> 0-line).
# Still well under the per-call cost ceiling. Override per slot via
# OPENROUTER_<A|B>_MAXTOK.
DEFAULT_MAX_TOKENS_PER_CALL = 32768   # COST per-call ceiling (prompt+output estimate)
DEFAULT_MAX_TOKENS_PER_RUN = 300_000
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 2  # total attempts = retries + 1

# Minimum output budget for a remote call. The writer's per-call
# max_new_tokens are sized for the LOCAL grammar-constrained path, where
# lm-format-enforcer forces a compact bare JSON object that fits in
# ~150-200 tokens. A free-form remote model (no token grammar) writes a
# fuller object + a ```json fence and needs more room; at the local
# budget it truncates mid-object (finish_reason=length) -> unparseable
# JSON -> fail-closed abort. This floor (max_tokens is a CEILING -- the
# model still stops at finish_reason=stop and bills only actual tokens)
# lets the remote model finish. Overridable via OPENROUTER_MIN_OUTPUT_TOKENS.
DEFAULT_MIN_OUTPUT_TOKENS = 1024

# ...and the SAME floor is nowhere near enough when the model REASONS.
#
# `max_tokens` bounds reasoning tokens + content tokens TOGETHER, and the hidden
# reasoning is emitted FIRST. So a reasoning model spends the floor thinking and
# gets cut before it writes a single content token -> finish_reason=length -> an
# empty/truncated body -> unparseable JSON -> fail-closed abort. This is not
# hypothetical: DEFAULT_REASONING_EFFORT is "low", so reasoning is ON for EVERY
# remote call by default.
#
# The OUTPUT CAP already learned this (16384, R3 2026-06-22: "a reasoning model
# spends part of the output budget on hidden reasoning ... could starve the story
# body"). That fix protected the BIG calls -- the story body asks for thousands of
# tokens, so it clears a reasoning preamble on its way past. It never protected
# the SMALL ones. A short call (the announcer's news-coda bridge asks for ~150
# tokens, sized for the local grammar-constrained path) is floored to 1024 and
# dies, because 1024 is a reasoning budget, not a reasoning budget PLUS a bridge
# line.
#
# Live 2026-07-14, 420w `public_domain_story` leg: aion-3.0-mini (mandatory
# reasoning) hit finish_reason=length on BOTH news-coda bridge attempts and the
# episode aborted -- correctly, since the no-fallback rip (2026-07-03) refuses to
# paper a dead LLM over with canned text. The model was never given room to
# answer.
#
# So the floor is reasoning-aware: when reasoning is active, it must cover the
# preamble AND the answer. max_tokens is a CEILING -- the model still stops at
# finish_reason=stop and bills only what it actually emits -- so a generous floor
# costs nothing on a short reply. Overridable via
# OPENROUTER_MIN_OUTPUT_TOKENS_REASONING.
DEFAULT_MIN_OUTPUT_TOKENS_REASONING = 4096


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


class OpenRouterModelGoneError(OpenRouterCallFailedError):
    """A DELETED model -- OpenRouter answered with a deletion-class 404
    ("No endpoints found" / "not found" / "no allowed providers"). This is
    a *renderability* failure, not a quality choice, so -- UNLIKE every other
    failure -- the dispatch layer is permitted to fall over to the slot's
    REMOTE fallback (env/recommended) exactly once, LOUDLY (2026-06-18,
    ChatGPT-reviewed). It subclasses OpenRouterCallFailedError so any caller
    that does not special-case it still aborts cleanly (C5). The fallback is
    remote->remote ONLY: a deleted remote model never silently becomes a
    LOCAL model (that stays the operator's explicit opt-in)."""

    def __init__(self, message: str, *, slug: str = "") -> None:
        super().__init__(message)
        self.slug = slug


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


#: R3 frontier-writer config default (2026-06-22). The live re-measure showed
#: grok-4.3 at reasoning_effort=low HALVED the critic flatness score (14/18 ->
#: 8/18) vs reasoning OFF -- a real near-free craft win for the frontier writers.
#: So the OpenRouter default is now "low" (paired with the bumped output cap as
#: the 0-line guard). Set OPENROUTER_REASONING_EFFORT explicitly to override --
#: "none" restores the no-reasoning behaviour (the gemma-via-/v1 lane) and any
#: high|medium|low tunes it.
DEFAULT_REASONING_EFFORT = "low"


def _reasoning_effort_from_env() -> str | None:
    """Resolve the OpenAI-standard reasoning control (high|medium|low|none).

    Reads OPENROUTER_REASONING_EFFORT; UNSET/empty -> ``DEFAULT_REASONING_EFFORT``
    ("low", the R3 frontier-writer default). An explicit value always wins --
    set "none" to suppress reasoning (thinking /v1 lanes: emit the structured
    answer directly instead of spending the output budget on a <think> preamble
    -> finish_reason=length -> unparseable JSON). reasoning_effort is the
    portable lever across OpenAI-compatible transports."""
    raw = _env("OPENROUTER_REASONING_EFFORT")
    if raw is None or not raw.strip():
        return DEFAULT_REASONING_EFFORT
    return raw.strip().lower() or DEFAULT_REASONING_EFFORT


def _mandatory_reasoning_effort(
    slug: str, reasoning_meta: Any = None,
) -> str | None:
    """Return the effective effort for a model whose reasoning is mandatory.

    OpenRouter's model catalog declares ``reasoning.mandatory`` and lists
    supported efforts in descending order. An operator-wide ``none`` remains
    valid for ordinary models, but a mandatory model rejects that value. In
    that case use its lowest declared non-``none`` effort, or the project low
    default when a stale/cold catalog has no capability detail.
    """
    requested = _reasoning_effort_from_env()
    meta = reasoning_meta if isinstance(reasoning_meta, dict) else {}
    mandatory = bool(meta.get("mandatory")) or slug in _MANDATORY_REASONING_SLUGS
    if not mandatory or requested != "none":
        return requested

    supported = [
        str(value).strip().lower()
        for value in (meta.get("supported_efforts") or [])
        if str(value).strip().lower() != "none"
    ]
    effective = supported[-1] if supported else DEFAULT_REASONING_EFFORT
    if slug not in _MANDATORY_REASONING_OVERRIDE_LOGGED:
        _MANDATORY_REASONING_OVERRIDE_LOGGED.add(slug)
        log.warning(
            "[OpenRouter] %s requires reasoning; overriding the incompatible "
            "global effort 'none' with %r for this model only.",
            slug,
            effective,
        )
    return effective


def openrouter_enabled() -> bool:
    """Remote is reachable when the API KEY is present -- a creds-present check,
    NOT a promotion gate. C6 (2026-06-29 -- "registry IS the menu"): the separate
    OTR_ENABLE_OPENROUTER opt-in flag is GONE; any CONFIGURED LLM (its credentials
    present) is selectable, and the virtual rows appear in the dropdowns whenever
    the key is set. No key ⇒ remote is off (the rows never appear, S2)."""
    return bool(_env("OPENROUTER_API_KEY"))


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
    """Map a virtual handle (openrouter:slot-a/b) to the real model slug,
    resolving on the STORED slot-picker widget value (S3 / plan §5). Three
    cases:

      (2) A slug is bound for the slot (the operator's slot-picker pick) ->
          USE IT VERBATIM. If it is absent from the (possibly stale/cold)
          local cache, WARN but still attempt it -- a stale cache is not a
          gone model, and we never silently swap a saved slug.
      (1) No binding (empty / unset / placeholder sentinel) -> fallback
          chain: OTR_OPENROUTER_SLOT_x_DEFAULT -> OPENROUTER_MODEL_x (env,
          now DEMOTED to a fallback) -> recommended default -> config error.

    (Case 3 -- a selected call FAILS -- is handled at generate time by the
    retry ladder raising OpenRouterCallFailedError; there is no remote->remote
    swap.) The resolved slug is a public model id, recorded in run meta (S3),
    never hard-coded for a live pick and never a secret."""
    letter = _slot_letter(repo_id)

    # Case 2: explicit saved slug from the slot-picker widget -> verbatim.
    bound = _slot_bindings.get(letter)
    if bound:
        if not _slug_in_cache(bound):
            log.warning(
                "[OpenRouter] slot-%s slug %r is not in the local catalog "
                "cache (stale or cold cache?). Using it as saved and "
                "attempting the call -- run the refresh script to update "
                "discovery. No substitution.",
                letter, bound,
            )
        return bound

    # Case 1: unbound -> fallback chain (env is now a fallback, not primary).
    slot_default = _env(f"OTR_OPENROUTER_SLOT_{letter}_DEFAULT")
    if slot_default:
        return slot_default
    env_slug = _env(f"OPENROUTER_MODEL_{letter}")
    if env_slug:
        return env_slug
    recommended = recommended_slug_for_slot(letter)
    if recommended:
        return recommended
    raise OpenRouterConfigError(
        f"{repo_id} selected but no slug is bound: set the "
        f"openrouter_slot_{letter.lower()}_model widget, OPENROUTER_MODEL_{letter}, "
        f"or OTR_OPENROUTER_SLOT_{letter}_DEFAULT. See docs/openrouter-setup.md."
    )


#: Substrings (lower-cased) that mark a DELETION-class 404 -- the model is GONE,
#: not merely throttled/down. OpenRouter's wording varies (the documented
#: fallback-classifier bug is that the 404 JSON text is inconsistent), so the
#: match is deliberately tolerant. Kept tight enough that an outage 404 with a
#: generic body does NOT match (a bare "404" with no gone-phrase still aborts).
_MODEL_GONE_MARKERS: tuple = (
    "no endpoints found",
    "no allowed providers",
    "not found",
    "does not exist",
    "no longer available",
    "is not a valid model",
)


def is_model_gone_error(status: int, snippet: str) -> bool:
    """True iff an HTTP error is the DELETION class (a permanently gone model),
    distinct from a transient/auth failure. Only a 404 whose body carries a
    gone-marker qualifies -- 401/403/422/429/5xx and a bodyless 404 do NOT, so
    an outage surfaces (and aborts per C5) instead of being misread as deletion
    and silently downgraded to the fallback."""
    if int(status or 0) != 404:
        return False
    low = (snippet or "").lower()
    return any(m in low for m in _MODEL_GONE_MARKERS)


def is_mandatory_reasoning_error(status: int, snippet: str) -> bool:
    """Recognize OpenRouter's exact capability rejection for reasoning-off.

    This is deliberately narrower than a generic HTTP 400 retry: only a
    reasoning-specific message that says the feature is mandatory or cannot be
    disabled qualifies for one same-model capability renegotiation.
    """
    if int(status or 0) != 400:
        return False
    low = (snippet or "").lower()
    return "reasoning" in low and (
        "mandatory" in low
        or "cannot be disabled" in low
        or "can't be disabled" in low
        or "must be enabled" in low
    )


def is_temperature_parameter_error(status: int, snippet: str) -> bool:
    """Recognize an upstream's exact rejection of the optional temperature.

    OpenRouter's model-level catalog can advertise ``temperature`` while the
    only endpoint satisfying a simultaneous structured-output requirement has
    deprecated it.  That disagreement is safe to negotiate by omitting only
    the optional sampling knob and retrying the same model once.  Generic 400s
    remain fail-fast.
    """
    if int(status or 0) != 400:
        return False
    low = (snippet or "").lower()
    return "temperature" in low and (
        "deprecated" in low
        or "not supported" in low
        or "unsupported" in low
        or "does not support" in low
    )


def _strip_route_suffix(slug: str) -> str:
    base, sep, suffix = str(slug or "").rpartition(":")
    if sep and base and suffix.lower() in ("nitro", "floor"):
        return base
    return str(slug or "")


def fallback_slug_for_slot(letter: str, *, exclude: str = "") -> str | None:
    """The REMOTE fallback slug for a slot when its chosen model was DELETED:
    the same precedence as resolve_slug's unbound chain (per-slot default ->
    OPENROUTER_MODEL_x env -> recommended default), but SKIPPING any candidate
    equal to the dead slug (so we never retry straight back into the 404).
    Returns None when no DISTINCT live-candidate slug exists (caller then aborts
    cleanly: "all candidates deleted"). Remote->remote only -- never a local
    model (C5)."""
    dead = _strip_route_suffix(exclude).lower()
    candidates = (
        _env(f"OTR_OPENROUTER_SLOT_{letter}_DEFAULT"),
        _env(f"OPENROUTER_MODEL_{letter}"),
        recommended_slug_for_slot(letter),
    )
    for cand in candidates:
        if cand and _strip_route_suffix(cand).lower() != dead:
            return cand
    return None


# ---------------------------------------------------------------------------
# Provider routing -- speed / cost control (":nitro" fastest / ":floor" cheapest)
# ---------------------------------------------------------------------------
#
# A single slug (e.g. "anthropic/claude-3.5-sonnet") is served by several
# upstream providers; OpenRouter picks one per call. `provider.sort` biases
# that choice: "throughput" routes to the fastest provider (so the writer's
# many internal LLM calls return as quickly as possible), "price" routes to
# the cheapest, "latency" to the lowest time-to-first-token. OpenRouter also
# accepts ":nitro" (== throughput) and ":floor" (== price) as slug shortcuts.
#
# We normalize BOTH the slug shortcut and the env knobs to one `provider.sort`
# value so there is a single code path and the resolved choice is recorded in
# run meta (S5). Default (no knob, no suffix) is OpenRouter's normal
# load-balanced routing -- no `sort` is sent.

_SORT_BY_ALIAS = {
    # fastest provider (highest throughput) -- ":nitro"
    "nitro": "throughput",
    "fast": "throughput",
    "fastest": "throughput",
    "throughput": "throughput",
    "speed": "throughput",
    # cheapest provider -- ":floor"
    "floor": "price",
    "cheap": "price",
    "cheapest": "price",
    "price": "price",
    "cost": "price",
    # lowest time-to-first-token
    "latency": "latency",
}


def _normalize_sort(token: str | None) -> str | None:
    """Map a friendly route token (nitro / floor / fast / cheap / ...) or a
    raw OpenRouter sort (throughput / price / latency) to a `provider.sort`
    value. An empty or unrecognized token resolves to None (OpenRouter's
    default load-balanced routing)."""
    if not token:
        return None
    return _SORT_BY_ALIAS.get(token.strip().lower())


def _sort_from_env(name: str) -> str | None:
    """Read a routing env var and normalize it, warning (never raising) when
    a non-empty value is not a recognized route so a typo is visible."""
    raw = _env(name)
    if not raw:
        return None
    sort = _normalize_sort(raw)
    if sort is None:
        log.warning(
            "[OpenRouter] %s=%r is not a recognized route "
            "(use nitro/floor or throughput/price/latency); using default routing.",
            name, raw,
        )
    return sort


def _split_slug_suffix(slug: str) -> tuple[str, str | None]:
    """Honour OpenRouter's native ':nitro' / ':floor' slug shortcuts. Returns
    (clean_slug, sort): the suffix is stripped so it is sent as an explicit
    `provider.sort` (one code path) instead of relying on the wire shortcut,
    and the clean slug is what gets stamped into run meta."""
    if not isinstance(slug, str):
        return slug, None
    base, sep, suffix = slug.rpartition(":")
    if sep and base and suffix.lower() in ("nitro", "floor"):
        return base, _normalize_sort(suffix)
    return slug, None


def resolve_route(letter: str, slug: str) -> tuple[str, str | None]:
    """Resolve the provider-routing sort for a slot, most-specific first:

      1. a ':nitro' / ':floor' suffix on the slug (the operator typed it onto
         the model id -- the most explicit signal), else
      2. the per-slot env ``OPENROUTER_<L>_ROUTE``, else
      3. the global env ``OPENROUTER_SORT``, else
      4. None -- OpenRouter's default load-balanced routing.

    Returns ``(clean_slug_without_suffix, sort_or_None)``. The slug suffix is
    always stripped from the returned slug regardless of which rule wins, so
    the wire payload and run meta never carry the shortcut."""
    clean_slug, suffix_sort = _split_slug_suffix(slug)
    if suffix_sort is not None:
        return clean_slug, suffix_sort
    per_slot = _sort_from_env(f"OPENROUTER_{letter}_ROUTE")
    if per_slot is not None:
        return clean_slug, per_slot
    return clean_slug, _sort_from_env("OPENROUTER_SORT")


def reset_run_budget() -> None:
    """Zero the per-run token accumulator. Called at the start of an
    episode so the per-run cost ceiling (C6) measures one run, not the
    process lifetime. Also clears the per-run resolved-model ledger so the
    treatment records THIS episode's `~latest` resolutions, not stale ones."""
    global _run_token_total
    _run_token_total = 0
    _resolved_models.clear()


_run_token_total = 0


# ---------------------------------------------------------------------------
# Per-run resolved-model ledger -- historical accuracy for the credits sheet.
# A `~latest` alias resolves to a CONCRETE model SERVER-SIDE at call time; the
# only place that truth surfaces is the response body's `model` field. We
# accumulate {requested_slug: {resolved, calls, cost_usd}} per episode so the
# episode treatment can record exactly which concrete model served the run.
# ---------------------------------------------------------------------------
_resolved_models: dict[str, dict] = {}


def _record_resolved(requested_slug: str, body: dict) -> None:
    """Record the concrete model + cost the upstream reported for a call.
    Best-effort; never raises (provenance must not break a run, PD1)."""
    try:
        req = str(requested_slug or "").strip() or "(unknown)"
        resolved = str((body or {}).get("model") or "").strip()
        usage = (body or {}).get("usage") or {}
        cost = usage.get("cost")
        try:
            cost = float(cost) if cost is not None else 0.0
        except (TypeError, ValueError):
            cost = 0.0
        slot = _resolved_models.setdefault(
            req, {"resolved": "", "calls": 0, "cost_usd": 0.0})
        if resolved:
            slot["resolved"] = resolved
        slot["calls"] += 1
        slot["cost_usd"] += cost
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        pass


def resolved_models_snapshot() -> dict:
    """Return a deep-ish copy of the per-run resolved-model ledger:
    ``{requested_slug: {"resolved": str, "calls": int, "cost_usd": float}}``.
    Empty when no OpenRouter call was made this run."""
    return {k: dict(v) for k, v in _resolved_models.items()}


# ---------------------------------------------------------------------------
# S3 per-run slot bindings -- the slot-picker widget values, demoting env
# ---------------------------------------------------------------------------
#
# The writer records the openrouter_slot_a/b_model widget values here at the
# start of run() (the same single per-episode entry that resets the budget),
# so resolve_slug can map a handle to the OPERATOR'S chosen slug instead of the
# env. Process-global because the loader->backend call chain does not thread
# widget values. A None / empty / placeholder-sentinel binding means "unset"
# -> resolve_slug falls through to the env / recommended chain (so headless
# runs and old workflows keep working). The plan's preservation rule lives in
# resolve_slug: a bound slug is used verbatim and never silently swapped.

_slot_bindings: dict[str, str | None] = {"A": None, "B": None}


def _clean_slot_value(v: Any) -> str | None:
    """Normalize a slot-picker widget value to a real slug or None. Empty,
    whitespace, or a parenthesized UI placeholder sentinel (e.g.
    '(enable OpenRouter)') is treated as 'unset' (-> None)."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or (s.startswith("(") and s.endswith(")")):
        return None
    return s


def set_slot_bindings(*, slot_a: Any = None, slot_b: Any = None) -> None:
    """Record the per-run slot->slug bindings from the writer's slot-picker
    widgets. Normalized via _clean_slot_value; the env (OPENROUTER_MODEL_A/B)
    is demoted to a fallback used only when a slot is unbound."""
    _slot_bindings["A"] = _clean_slot_value(slot_a)
    _slot_bindings["B"] = _clean_slot_value(slot_b)


def clear_slot_bindings() -> None:
    """Drop both bindings (back to env / recommended fallback). Called by
    tests; the writer overwrites via set_slot_bindings each run()."""
    _slot_bindings["A"] = None
    _slot_bindings["B"] = None


def recommended_slug_for_slot(letter: str) -> str:
    """The recommended default slug for slot 'A'/'B': the per-slot env
    override OTR_OPENROUTER_SLOT_x_DEFAULT if set, else the built-in
    OPENROUTER_RECOMMENDED_*_DEFAULT constant. This is the last rung of the
    resolve_slug fallback chain (§5 case 1) before the config error."""
    L = (letter or "").strip().upper()
    if L == "A":
        return (_env("OTR_OPENROUTER_SLOT_A_DEFAULT")
                or OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT)
    if L == "B":
        return (_env("OTR_OPENROUTER_SLOT_B_DEFAULT")
                or OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT
                or OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT)
    raise OpenRouterConfigError(f"slot letter must be 'A' or 'B', got {letter!r}")


def _slug_in_cache(slug: str) -> bool:
    """True if `slug` is an id in the S0 disk cache. Used only to decide
    whether to WARN about a saved slug absent from a stale/cold cache --
    never to reject or swap it (cache staleness != a gone model)."""
    try:
        return any(m.get("id") == slug for m in cached_models())
    except Exception:  # noqa: BLE001 -- a cache read must never break resolution
        return False


# ---------------------------------------------------------------------------
# Catalog cache (S0) -- disk-cached OpenRouter model list for the dropdowns
# ---------------------------------------------------------------------------
#
# INPUT_TYPES() must NEVER touch the network (hard rule): the four model
# dropdowns build from THIS on-disk cache only. The cache is refreshed
# EXPLICITLY (an operator / refresh-script call to refresh_catalog_cache),
# never at import and never inside INPUT_TYPES. A missing / corrupt / empty
# cache degrades safely to an empty model list -- discovery is empty, but a
# saved slug is still preserved + attempted (S3). The cache governs
# DISCOVERY only; it must never mutate a saved slot slug value.

CATALOG_SCHEMA_VERSION = 2
_CATALOG_FILENAME = "openrouter_models.json"
_CATALOG_STALE_AFTER_S = 7 * 24 * 3600  # older than a week -> "stale" (still usable)


def _catalog_cache_path() -> Path:
    """``<repo>/models/openrouter_models.json`` (in-repo, git-ignored). This
    module lives in ``nodes/``, so the repo root is two parents up. Override
    the directory via ``OTR_OPENROUTER_CACHE_DIR`` (tests / relocation)."""
    override = _env("OTR_OPENROUTER_CACHE_DIR")
    base = Path(override) if override else Path(__file__).resolve().parent.parent / "models"
    return base / _CATALOG_FILENAME


def _empty_catalog(source: str) -> dict:
    """A safe, well-formed empty catalog. ``source`` records WHY it is empty
    (missing / corrupt) for the staleness log + run meta."""
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "fetched_at": None,
        "source": source,
        "count": 0,
        "models": [],
    }


def load_catalog_cache() -> dict:
    """Read the on-disk catalog. NEVER raises, NEVER blocks, NEVER touches the
    network. Missing file -> empty(source='missing'); unreadable / corrupt /
    wrong-shape -> empty(source='corrupt'). On success: schema_version,
    fetched_at, source, count, models[]."""
    path = _catalog_cache_path()
    try:
        if not path.is_file():
            return _empty_catalog("missing")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            log.warning("[OpenRouter] catalog cache %s malformed; treating as empty.", path)
            return _empty_catalog("corrupt")
        data.setdefault("schema_version", CATALOG_SCHEMA_VERSION)
        data.setdefault("fetched_at", None)
        data.setdefault("source", "cache")
        data["count"] = len(data["models"])
        return data
    except Exception as exc:  # noqa: BLE001 -- a cache read must never break a dropdown build
        log.warning("[OpenRouter] catalog cache read failed (%s); treating as empty.", exc)
        return _empty_catalog("corrupt")


def cached_models() -> list[dict]:
    """The model dicts from the cache (or [] when missing/corrupt). Each entry
    carries at least ``id``; S1 derives provider / supports_json / recency
    from the stored fields."""
    models = load_catalog_cache().get("models") or []
    return [m for m in models if isinstance(m, dict) and m.get("id")]


def _cached_model(slug: str) -> dict:
    """Return cached capability metadata for *slug*, or an empty mapping."""
    clean = _strip_route_suffix(slug)
    try:
        for model in cached_models():
            if model.get("id") == clean:
                return model
    except Exception:  # noqa: BLE001 -- cache metadata is advisory
        pass
    return {}


def resolve_context_window(slug: str, *, row_default: int | None = None) -> int:
    """The REAL context window of the resolved remote model.

    The virtual catalog rows (`openrouter:slot-a|b`) are STATIC: one row stands
    in for every slug an operator may bind to it, so its `context_window` cannot
    describe the model actually selected. It carried
    ``DEFAULT_CONTEXT_WINDOW`` (8192) -- a LOCAL, VRAM-shaped number that is
    simply false for a remote model. `fit_output_tokens` then clamped every
    remote call against that fiction: `aion-labs/aion-3.0-mini` advertises
    131,072 tokens and was handed 8,192, so a 9,520-token artifact request
    (codex56sol P6 at 720 words) was silently reduced to whatever 8,192 minus
    the prompt left, and the script came back cut off mid-JSON. The model that
    could trivially write it was strangled by a constant.

    The catalog cache ALREADY stores each slug's advertised `context_length`
    (see `_slim_model`), so the truth was on disk the whole time and simply was
    not read. Read it.

    A cold / stale / corrupt cache has no entry for the slug. That is a
    genuinely unknown window, so fall back to the row default and say so
    LOUDLY -- a conservative 8,192 that the operator can see beats a confident
    number nobody measured. Never raises: a cache miss must not break a load.
    """
    fallback = int(row_default or DEFAULT_CONTEXT_WINDOW)
    advertised = _cached_model(slug).get("context_length")
    try:
        window = int(advertised)
    except (TypeError, ValueError):
        window = 0
    if window > 0:
        return window
    log.warning(
        "[OpenRouter] %s has no context_length in the catalog cache (cold or "
        "stale?); falling back to %d tokens. A long artifact request will be "
        "clamped against that floor. Refresh the catalog to use the model's "
        "real window.",
        slug, fallback,
    )
    return fallback


def _parse_iso(ts: Any) -> datetime.datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def catalog_meta() -> dict:
    """Compact staleness view for the dropdown-build log + run meta:
    ``{source, fetched_at, count, staleness}``. staleness is one of
    live | cache | stale | empty. Never raises."""
    try:
        data = load_catalog_cache()
        count = int(data.get("count") or 0)
        fetched_at = data.get("fetched_at")
        source = data.get("source") or "cache"
        if count == 0:
            staleness = "empty"
        elif source == "live":
            staleness = "live"
        else:
            staleness = "cache"
            ts = _parse_iso(fetched_at)
            if ts is not None:
                age = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds()
                if age > _CATALOG_STALE_AFTER_S:
                    staleness = "stale"
        return {"source": source, "fetched_at": fetched_at,
                "count": count, "staleness": staleness}
    except Exception:  # noqa: BLE001
        return {"source": "corrupt", "fetched_at": None, "count": 0, "staleness": "empty"}


def _slim_model(raw: dict) -> dict | None:
    """Keep only the fields the dropdowns + filters need; None for a row with
    no usable id. ``supports_json`` is derived from OpenRouter's
    ``supported_parameters`` (a model that lists ``structured_outputs`` or
    ``response_format`` can be schema-constrained -- the REQUIRE_JSON filter,
    S1)."""
    if not isinstance(raw, dict):
        return None
    mid = raw.get("id")
    if not isinstance(mid, str) or not mid:
        return None
    sp = raw.get("supported_parameters")
    sp = [str(x) for x in sp] if isinstance(sp, list) else []
    pricing = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
    # Output modality (OpenRouter `architecture.output_modalities`, e.g.
    # ["text"] for an LLM, ["image"] for an image generator). Captured so the
    # writer slot pickers can hide non-text models (an image model in the
    # creative slot returns a monologue, never a usable line -- 2026-06-18).
    arch = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
    out_mods = arch.get("output_modalities")
    out_mods = [str(x).lower() for x in out_mods] if isinstance(out_mods, list) else []
    raw_reasoning = raw.get("reasoning") if isinstance(raw.get("reasoning"), dict) else {}
    supported_efforts = raw_reasoning.get("supported_efforts")
    supported_efforts = (
        [str(value).strip().lower() for value in supported_efforts]
        if isinstance(supported_efforts, list)
        else []
    )
    # DISPOSITION OF THE UNREAD KEYS (settled 2026-08-09 -- KEEP, do not prune).
    # Only `supported_efforts` and `mandatory` are consulted today (see :324 and
    # :330). `default_effort`, `default_enabled` and `supports_max_tokens` are
    # captured and never read -- GO_FORWARD flagged one of the three as a
    # possible BUG-12.86 sibling (a field that reads as though it informs a
    # decision and informs nothing) and asked delete-or-consume.
    # The ruling is KEEP, for a reason that stopped being hypothetical on
    # 2026-08-09: an OpenRouter roundtable run had `deepseek/deepseek-v4-pro`
    # return EMPTY CONTENT with finish_reason=length, having spent its whole
    # budget on hidden reasoning. That is precisely the condition this sub-dict
    # describes, and it is the same failure class that got reasoning-branded
    # SKUs excluded from the Comfy lane (see _otr_comfy_backend.py). So these
    # keys are not decoration -- they are the data a future guard needs to
    # refuse a reasoning-heavy model in a structured-JSON slot.
    # They also differ from BUG-12.86 in the way that matters: that class is
    # about a field keyed on a producer string the producer NEVER EMITS, so it
    # reads empty forever. These are faithfully populated from the upstream row;
    # they are simply not consulted yet. Pruning them would also change the
    # on-disk shape of every cached row in models/openrouter_models.json for no
    # gain.
    reasoning = {
        "supported_efforts": supported_efforts,
        "default_effort": (
            str(raw_reasoning.get("default_effort")).strip().lower()
            if raw_reasoning.get("default_effort") is not None
            else None
        ),
        "default_enabled": bool(raw_reasoning.get("default_enabled")),
        "mandatory": bool(raw_reasoning.get("mandatory")),
        "supports_max_tokens": bool(raw_reasoning.get("supports_max_tokens")),
    }
    return {
        "id": mid,
        "name": str(raw.get("name") or mid),
        "provider": mid.split("/", 1)[0] if "/" in mid else "",
        "created": raw.get("created"),
        "context_length": raw.get("context_length"),
        "pricing": {"prompt": pricing.get("prompt"),
                    "completion": pricing.get("completion")},
        "supported_parameters": sp,
        "supports_json": ("structured_outputs" in sp) or ("response_format" in sp),
        "output_modalities": out_mods,
        "reasoning": reasoning,
    }


def _fetch_models_json(*, base_url: str, api_key: str | None, timeout_s: int) -> list[dict]:
    """Mockable network seam (tests patch this): GET ``/models`` and return the
    raw ``data`` list. ``requests`` is imported lazily so the module stays
    import-safe + network-free. ONLY ``refresh_catalog_cache`` calls this --
    never INPUT_TYPES, never at import."""
    import requests  # lazy: keep module import-safe + network-free
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data") if isinstance(body, dict) else body
    return data if isinstance(data, list) else []


def _atomic_write_catalog(catalog: dict) -> None:
    """Write the cache atomically (temp file + ``os.replace``) so a crash
    mid-write never leaves a half-written / corrupt cache. Creates ``models/``
    if absent. Never raises into the caller."""
    try:
        path = _catalog_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("[OpenRouter] catalog atomic write failed (%s).", exc)


def refresh_catalog_cache(*, force: bool = False) -> dict:  # noqa: ARG001 -- force reserved
    """Fetch the live OpenRouter model list and atomically write the cache.
    EXPLICIT call only (operator / refresh script) -- never import, never
    INPUT_TYPES. A fetch failure NEVER destroys a good cache and never raises:
    it logs and returns the existing cache. Returns the catalog dict in
    effect after the call."""
    base_url = _env("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
    api_key = _env("OPENROUTER_API_KEY")
    timeout_s = _int_env("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S)
    try:
        raw_models = _fetch_models_json(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 -- a failed refresh keeps the old cache
        log.warning("[OpenRouter] catalog refresh failed (%s); keeping existing cache.", exc)
        return load_catalog_cache()
    models = [m for m in (_slim_model(r) for r in raw_models) if m is not None]
    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "live",
        "count": len(models),
        "models": models,
    }
    _atomic_write_catalog(catalog)
    log.info("[OpenRouter] catalog refreshed: %d models, source=live", len(models))
    return catalog


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

    def load(self, repo_id: str, row: Any, policy: Any = None) -> dict[str, Any]:
        """Return a remote cache_entry. No network call fires here -- a
        bad slug surfaces at generate time, and load must stay cheap so
        the writer's per-call request_slot is fast (C2: it must not
        touch the resident local model).

        S1 policy contract: this remote lane ASSERTS the profile
        lane_allowlist admits it (defense in depth behind request_slot's
        backstop) and deliberately IGNORES every hardware field
        (device/attn/quant/gguf) -- the lane uses zero local compute."""
        if policy is not None and not policy.admits_lane("openrouter"):
            raise OpenRouterConfigError(
                f"{repo_id}: 'openrouter' lane is not admitted by the "
                f"profile lane_allowlist {list(policy.lane_allowlist)}."
            )
        if not openrouter_enabled():
            raise OpenRouterConfigError(
                f"{repo_id} selected but OpenRouter is not enabled. Set "
                f"OPENROUTER_API_KEY "
                f"(see docs/openrouter-setup.md)."
            )
        letter = _slot_letter(repo_id)
        # Resolve the slug AND its provider-routing sort together: a ':nitro'/
        # ':floor' suffix on the bound slug is stripped here and carried as an
        # explicit sort, so the wire payload + run meta hold the clean slug.
        slug, provider_sort = resolve_route(letter, resolve_slug(repo_id))
        cached_model = _cached_model(slug)
        # The window belongs to the SLUG, not to the static virtual row. Reading
        # it from the row capped every remote model at the local 8192 default.
        context_window = resolve_context_window(
            slug,
            row_default=getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW),
        )
        cache_entry: dict[str, Any] = {
            "provider": PROVIDER,
            "model_id": repo_id,          # the virtual handle
            "slot_letter": letter,        # "A" / "B"
            "slug": slug,                 # the resolved real model id
            "provider_sort": provider_sort,  # throughput|price|latency|None
            "reasoning": cached_model.get("reasoning"),
            "context_cap": context_window,
            "context_window": context_window,
            # optional per-slot overrides (FC3) -- None ⇒ caller controls
            "temperature_override": _float_env(f"OPENROUTER_{letter}_TEMP"),
            "max_tokens_cap": _int_env(
                f"OPENROUTER_{letter}_MAXTOK", DEFAULT_OUTPUT_TOKENS_CAP
            ),
            "base_url": _env("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
            # no "model" / "tokenizer" keys: the generate-fn factory
            # branches on provider BEFORE requiring them (S3).
        }
        log.info(
            "[OpenRouter] load slot=%s handle=%s slug=%s route=%s ctx=%d "
            "(remote, 0 VRAM)",
            letter, repo_id, slug, provider_sort or "default", context_window,
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
        """Run one remote chat completion and return the decoded string.

        `model` is the cache_entry returned by `load()`. Enforces the
        cost ceiling BEFORE the call (C6), retries transient failures a
        bounded number of times, then aborts cleanly (C5)."""
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
        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            raise OpenRouterConfigError(
                "OPENROUTER_API_KEY is not set; cannot make a remote call."
            )
        slug = cache_entry.get("slug") or resolve_slug(cache_entry["model_id"])
        base_url = cache_entry.get("base_url") or DEFAULT_BASE_URL
        provider_sort = cache_entry.get("provider_sort")

        # Resolve the output budget. The remote model has NO token grammar,
        # so it must not inherit the local grammar-era per-call budget (which
        # truncates a free-form object mid-JSON). Floor at
        # DEFAULT_MIN_OUTPUT_TOKENS, then clamp to the per-slot cap. max_tokens
        # is a ceiling only -- the model stops at finish_reason=stop and bills
        # actual tokens, so a generous floor costs nothing on short replies.
        cap = int(cache_entry.get("max_tokens_cap") or DEFAULT_OUTPUT_TOKENS_CAP)
        # The BASE floor is also the HARD minimum: the value below which
        # fit_output_tokens refuses the call outright (context overflow). It stays
        # lean on purpose -- see the reasoning floor below.
        base_floor = _int_env(
            "OPENROUTER_MIN_OUTPUT_TOKENS", DEFAULT_MIN_OUTPUT_TOKENS,
        )
        floor = base_floor

        # Resolve reasoning BEFORE the budget: `max_tokens` bounds reasoning +
        # content together, and the reasoning is emitted FIRST. A floor that does
        # not cover the preamble hands the model a budget it burns before it can
        # answer (see DEFAULT_MIN_OUTPUT_TOKENS_REASONING). The same value is
        # reused for the payload below -- resolve it once.
        reasoning_effort = _mandatory_reasoning_effort(
            slug, cache_entry.get("reasoning")
        )
        if reasoning_effort and reasoning_effort != "none":
            floor = max(floor, _int_env(
                "OPENROUTER_MIN_OUTPUT_TOKENS_REASONING",
                DEFAULT_MIN_OUTPUT_TOKENS_REASONING,
            ))

        requested_tokens = max(1, int(max_new_tokens or 0))
        if (require_full_output or bounded_capacity) and requested_tokens > cap:
            capacity = GenerationContextOverflowError(
                f"OpenRouter {slug} cannot fit the complete requested output: "
                f"requested_output={requested_tokens}, provider_output_cap={cap}"
            )
            raise OpenRouterConfigError(str(capacity)) from capacity
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
                # The HARD minimum stays the BASE floor, never the reasoning floor.
                # The reasoning floor is a DESIRED minimum -- it lifts the budget
                # when the window can afford it; it must never turn a survivable
                # call into an abort. On a small-context model a long prompt can
                # leave less than the reasoning floor (8192 cap - a ~7000-token
                # prompt = 1192 tokens): that call should still RUN with what is
                # left, degraded, not refuse to start. Gating the hard minimum on
                # the reasoning floor would abort it.
                min_output_tokens=min(base_floor, cap),
                label=f"OpenRouter {slug}",
                require_full=require_full_output or bounded_capacity,
            )
        except GenerationContextOverflowError as exc:
            raise OpenRouterConfigError(str(exc)) from exc
        if fitted_tokens != out_tokens:
            log.warning(
                "[OpenRouter] output budget clamped: requested=%d effective=%d "
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

        # --- C6 cost guard: enforce BEFORE any network call ---
        est = _estimate_request_tokens(messages, out_tokens)
        self._enforce_cost_ceiling(est, slug=slug)

        payload: dict[str, Any] = {
            "model": slug,
            "messages": messages,
            "max_tokens": out_tokens,
        }

        if stop:
            payload["stop"] = [s for s in stop if s]
        # Reasoning control (gemma-4 / thinking-model lane). The OpenAI-standard
        # `reasoning_effort` (high|medium|low|none) is the portable lever used
        # to bound or disable the <think> preamble; set
        # OPENROUTER_REASONING_EFFORT=none so a reasoning model emits the
        # structured answer directly instead of burning the output budget on
        # reasoning (-> finish_reason=length -> unparseable JSON). Unset -> the
        # field is omitted, so non-thinking models / OpenRouter-proper are
        # byte-identical. Not require_parameters-gated: a backend that ignores it
        # simply reasons as before -- the output-token floor is the safety net,
        # and that floor is now reasoning-aware (resolved with the budget above,
        # because a net sized without the preamble is a net with a hole in it).
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
            # EVIDENT (operator 2026-06-22): log ONCE per process so any server
            # log -- the soak's AND a manual workflow run -- visibly shows the
            # reasoning lever is active (the fix for the finish_reason=length
            # truncation epidemic on reasoning ~latest writers).
            global _REASONING_EFFORT_LOGGED
            if not _REASONING_EFFORT_LOGGED:
                _REASONING_EFFORT_LOGGED = True
                log.info("[OpenRouter] reasoning_effort=%s ACTIVE "
                         "(OPENROUTER_REASONING_EFFORT) -- reasoning models emit "
                         "the answer instead of burning the output budget on "
                         "hidden reasoning", reasoning_effort)
        # Build ONE provider-routing object so the speed/cost sort and the
        # require_parameters guard coexist (a second `payload["provider"]`
        # assignment would clobber the first).
        provider_opts: dict[str, Any] = {}
        if provider_sort:
            # ":nitro"/throughput = fastest provider, ":floor"/price =
            # cheapest, latency = lowest time-to-first-token (C6: the cost
            # guard still applies; a faster upstream is not an uncapped one).
            provider_opts["sort"] = provider_sort
        if response_format is not None:
            payload["response_format"] = response_format
            # Defect A: only route to upstreams that actually honour the
            # requested parameters, so response_format can't be silently
            # dropped on the wire (which would make the enforcement a no-op).
            provider_opts["require_parameters"] = True
        if grammar:
            # A (2026-06-04): GBNF decode constraint for the local
            # llama-server lane -- llama.cpp accepts a top-level `grammar`
            # over its OpenAI-compatible /v1 endpoint. Used by the style-
            # picker inventor to hard-cap output at exactly N descriptors so
            # an overgenerating model (gemma's 63-vs-5) cannot break the
            # exact-count gate. require_parameters keeps it fail-closed: a
            # backend that ignores `grammar` is filtered out rather than
            # left silently unconstrained.
            payload["grammar"] = grammar
            provider_opts["require_parameters"] = True
        if provider_opts:
            payload["provider"] = provider_opts
        if (
            temp is not None
            and (slug, response_format is not None)
            not in _TEMPERATURE_UNSUPPORTED_ROUTES
        ):
            payload["temperature"] = float(temp)

        try:
            text = self._post_with_retries(
                base_url=base_url, api_key=api_key, payload=payload, slug=slug,
                fail_on_output_limit=fail_on_output_limit,
            )
        except OpenRouterModelGoneError as gone:
            # The operator's chosen model was DELETED. If this call came through
            # a slot handle, fall over to that slot's REMOTE fallback exactly
            # once, LOUDLY -- a deletion is a renderability failure, not the
            # quality swap C5 forbids. A directly-pinned slug (no slot) or an
            # exhausted fallback chain still aborts cleanly.
            model_id = cache_entry.get("model_id")
            fb_slug = None
            if is_openrouter_row_id(model_id):
                letter = _slot_letter(model_id)
                fb_slug = fallback_slug_for_slot(letter, exclude=gone.slug or slug)
            if not fb_slug:
                raise
            log.warning(
                "[OpenRouter] LOUD model-gone fallback: slot %s model %r was "
                "DELETED (gone-404); falling over to the remote fallback %r for "
                "this call. Update the openrouter_slot_%s_model widget. "
                "Remote->remote only; never local (C5).",
                model_id, gone.slug or slug, fb_slug, _slot_letter(model_id).lower(),
            )
            fb_clean, _fb_sort = _split_slug_suffix(fb_slug)
            payload["model"] = fb_clean
            text = self._post_with_retries(
                base_url=base_url, api_key=api_key, payload=payload, slug=fb_clean,
                fail_on_output_limit=fail_on_output_limit,
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
        fail_on_output_limit: bool = False,
    ) -> str:
        timeout_s = _int_env("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        max_retries = _int_env("OPENROUTER_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        last_err: str = ""
        last_status: int = 0
        last_snippet: str = ""
        transient_retries_left = max_retries
        reasoning_retry_used = False
        temperature_retry_used = False
        pending_temperature_route: tuple[str, bool] | None = None
        total_attempts = 0
        payload = dict(payload)
        while True:
            total_attempts += 1
            try:
                result = _post_chat_completion(
                    base_url=base_url, api_key=api_key,
                    payload=payload, timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 -- network/transport error
                last_err = f"transport error: {type(exc).__name__}: {exc}"
                last_status, last_snippet = 0, ""
                log.warning(
                    "[OpenRouter] attempt %d/%d failed (%s)",
                    total_attempts, max_retries + 1, last_err,
                )
                if transient_retries_left > 0:
                    transient_retries_left -= 1
                    self._sleep_backoff(max_retries - transient_retries_left - 1)
                    continue
                break

            status = int(result.get("status_code") or 0)
            if status == 200:
                if pending_temperature_route is not None:
                    _TEMPERATURE_UNSUPPORTED_ROUTES.add(
                        pending_temperature_route
                    )
                return self._extract_text(
                    result, slug=slug,
                    fail_on_output_limit=fail_on_output_limit,
                )

            last_snippet = self._error_snippet(result)
            last_status = status
            last_err = f"HTTP {status}: {last_snippet}"
            if (
                not reasoning_retry_used
                and payload.get("reasoning_effort") == "none"
                and is_mandatory_reasoning_error(status, last_snippet)
            ):
                reasoning_retry_used = True
                _MANDATORY_REASONING_SLUGS.add(slug)
                effective = _mandatory_reasoning_effort(slug)
                payload["reasoning_effort"] = effective
                log.warning(
                    "[OpenRouter] %s rejected reasoning_effort='none' as "
                    "mandatory; retrying the same model once with %r and "
                    "remembering the capability for this process.",
                    slug,
                    effective,
                )
                continue
            if (
                not temperature_retry_used
                and "temperature" in payload
                and is_temperature_parameter_error(status, last_snippet)
            ):
                temperature_retry_used = True
                pending_temperature_route = (
                    slug,
                    payload.get("response_format") is not None,
                )
                payload.pop("temperature", None)
                log.warning(
                    "[OpenRouter] %s rejected the optional temperature "
                    "parameter; retrying the same model once without it.",
                    slug,
                )
                continue
            if status in _RETRYABLE_STATUS and transient_retries_left > 0:
                transient_retries_left -= 1
                log.warning(
                    "[OpenRouter] attempt %d/%d retryable (%s)",
                    total_attempts, max_retries + 1, last_err,
                )
                self._sleep_backoff(max_retries - transient_retries_left - 1)
                continue
            # Non-retryable (e.g. 400/401/403/404) -> abort now.
            break

        # A DELETION-class 404 ("model gone") is a renderability failure, not a
        # transient one: raise the distinct subclass so the dispatch layer can
        # fall over to the slot's REMOTE fallback once (LOUD). Every other
        # failure stays a plain abort (C5) -- including a bodyless 404.
        if is_model_gone_error(last_status, last_snippet):
            raise OpenRouterModelGoneError(
                f"OpenRouter model {slug} is gone (HTTP {last_status}: "
                f"{last_snippet}).",
                slug=slug,
            )
        raise OpenRouterCallFailedError(
            f"OpenRouter call to {slug} failed after {total_attempts} "
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
    def _extract_text(
        result: dict, *, slug: str, fail_on_output_limit: bool = False,
    ) -> str:
        body = result.get("json") or {}
        if os.environ.get("OPENROUTER_DEBUG_RAW") == "1":
            log.info("[OpenRouter] raw %s response: %s", slug, json.dumps(body)[:1500])
        # Record the concrete model the upstream actually served (a `~latest`
        # alias resolves server-side) + its cost, for the episode credits sheet.
        _record_resolved(slug, body)
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterCallFailedError(
                f"OpenRouter {slug} returned no choices: "
                f"{str(body)[:300]}"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        # Defect D: content may be a list of typed parts (Anthropic-style
        # content blocks); join their text. Some reasoning models also put
        # the answer only in `reasoning` when `content` is empty -- fall
        # back to that rather than aborting silently.
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") in (None, "text")
            )
        if (not isinstance(content, str) or not content) and message.get("reasoning"):
            content = str(message.get("reasoning"))
        # Strip thinking-mode reasoning scaffolding (<think>...</think>,
        # harmony channels) so the writer's structured passes parse clean
        # output -- a no-op for non-thinking models (BUG-306/308 family).
        if isinstance(content, str):
            content = _strip_reasoning_tags(content)
        if choices[0].get("finish_reason") == "length":
            if fail_on_output_limit:
                raise PromptContextOverflowError(
                    f"OpenRouter {slug} exhausted the provider output capacity; "
                    "the partial artifact is discarded rather than repaired",
                    phase=CAPACITY_PHASE_OUTPUT_LIMIT,
                    raw_completion=content if isinstance(content, str) else None,
                    generated_tokens=(body.get("usage") or {}).get(
                        "completion_tokens"
                    ),
                    ended_with_eos=False,
                )
            log.warning(
                "[OpenRouter] %s hit finish_reason=length -- output truncated "
                "at the token ceiling; a downstream JSON parse may fail. Raise "
                "OPENROUTER_MIN_OUTPUT_TOKENS or the slot max-tokens cap.",
                slug,
            )
        if not isinstance(content, str) or not content:
            raise OpenRouterCallFailedError(
                f"OpenRouter {slug} returned empty message content "
                f"(finish_reason={choices[0].get('finish_reason')!r})."
            )
        return content

    @staticmethod
    def _error_snippet(result: dict) -> str:
        body = result.get("json")
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                metadata = err.get("metadata")
                raw = metadata.get("raw") if isinstance(metadata, dict) else None
                if isinstance(raw, str) and raw:
                    try:
                        provider_body = json.loads(raw)
                    except (TypeError, ValueError):
                        provider_body = None
                    if isinstance(provider_body, dict):
                        provider_error = provider_body.get("error")
                        if (
                            isinstance(provider_error, dict)
                            and provider_error.get("message")
                        ):
                            return str(provider_error["message"])[:200]
                if err.get("message"):
                    return str(err["message"])[:200]
        text = result.get("text") or ""
        return str(text)[:200]


def make_openrouter_generate_fn(cache_entry: dict, *, response_format: dict | None = None,
                                grammar: str | None = None):
    """Return a generate_fn closure for a provider-tagged remote
    cache_entry (FC2 seam 2).

    The closure matches the signature every OTR generate_fn uses --
    ``(messages, *, temperature, max_new_tokens, stop=None, ...) -> str`` --
    so the loader factories (`make_generate_fn`,
    `make_polish_generate_fn`) and the writer's
    `_build_truncating_generate_fn` can all return it unchanged when
    ``cache_entry["provider"] == "openrouter"``.

    `response_format` is None for free-form creative + the plain
    structured_call path; S4 passes an OpenRouter json_schema
    response_format for the grammar-constrained technical path so remote
    technical output is schema-enforced (fail-closed via structured_call's
    validate + bounded-repair ladder).

    `grammar` (A, 2026-06-04) is an optional GBNF string for the local
    llama-server lane. It is None for every normal call; the style-picker
    inventor passes its exactly-N grammar per-call so an overgenerating
    model cannot break the exact-count gate. The `_otr_supports_grammar`
    marker lets a caller pass a grammar ONLY to backends that honour it --
    local backends lack the marker and stay byte-identical."""
    backend = OpenRouterBackend()
    bound_rf = response_format
    bound_grammar = grammar

    def generate_fn(messages, *, temperature=None, max_new_tokens=None,
                    stop=None, response_format=None, grammar=None):
        # A per-call response_format (e.g. structured_call passing
        # json_object on an un-schema'd creative call) overrides the bound
        # one; otherwise the closure's bound response_format (S4 json_schema)
        # is used. Free-form calls pass neither and get plain generation.
        rf = response_format if response_format is not None else bound_rf
        # A: a per-call GBNF grammar (the style-picker inventor passing its
        # exactly-N grammar) overrides the bound one; both None = free-form.
        g = grammar if grammar is not None else bound_grammar
        _reply = backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
            grammar=g,
        )
        # CLOUD RUNAWAY GUARD (2026-08-13). A remote call has no token loop to
        # attach a StoppingCriteria to, so this cannot save the spend -- but it
        # stops a degenerate reply reaching the ledger and raises the same
        # rerollable phase the local guard raises. Without it, local lanes were
        # protected and cloud lanes silently were not.
        try:
            from ._otr_decode_guard import assert_no_verbatim_cycle
        except ImportError:  # pragma: no cover - flat/standalone import path
            from _otr_decode_guard import (  # type: ignore
                assert_no_verbatim_cycle,
            )
        assert_no_verbatim_cycle(_reply, label="openrouter")
        return _reply

    # Markers so structured_call can detect a remote fn and whether it
    # already carries a schema-bound response_format (don't override that).
    # _otr_supports_grammar (A) lets the style-picker inventor pass a per-
    # call GBNF only to backends that honour it (the remote llama-server
    # lane); local backends lack the marker and stay byte-identical.
    generate_fn._otr_openrouter = True  # type: ignore[attr-defined]
    # Backend accepts a json_object response_format -> invoke_structured_slot
    # forces it for a schema-less structured pass (mirrors the GGUF lane).
    generate_fn._otr_supports_json_object = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    generate_fn._otr_supports_grammar = True  # type: ignore[attr-defined]
    return generate_fn


def schema_to_response_format(schema_model: Any, *, name: str = "otr_schema") -> dict:
    """Map a Pydantic model class to an OpenRouter
    ``response_format={type: json_schema, ...}`` payload (S4). Kept here
    so the remote constrained-generate path mirrors the local
    `make_constrained_generate_fn(cache_entry, schema_model)` entry."""
    schema_dict = schema_model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema_dict,
        },
    }


def openrouter_meta_for(creative_id: str, technical_id: str) -> dict[str, Any]:
    """Build run-meta remote-LLM provenance (S5).

    For each slot bound to an OpenRouter virtual handle, records the
    provider, the virtual handle, the RESOLVED slug, and basic params
    (per-slot max-tokens cap + optional temperature override). Adds a
    run-level `llm_remote_provider` and `llm_remote_schema_mode` (schema
    mode = the json_schema response_format path, used when the TECHNICAL
    slot is remote). Returns ``{}`` when neither slot is remote, so a
    local run gets no extra meta keys and stays byte-identical (C1).

    The resolved slug is a PUBLIC model id (e.g. "openai/gpt-4o"), not a
    secret -- recording it makes the env-side binding auditable in the
    run. The API key is never read here and never stamped (C9). Never
    raises (PD1: provenance must not break a run); an unresolved slug is
    stamped as "<unresolved>" rather than blowing up."""
    meta: dict[str, Any] = {}
    for slot_name, model_id in (("creative", creative_id), ("technical", technical_id)):
        if not is_openrouter_row_id(model_id):
            continue
        try:
            letter = _slot_letter(model_id)
            slug, sort = resolve_route(letter, resolve_slug(model_id))
        except OpenRouterError:
            letter = "?"
            slug = "<unresolved>"
            sort = None
        meta[f"llm_{slot_name}_provider"] = PROVIDER
        meta[f"llm_{slot_name}_handle"] = model_id
        meta[f"llm_{slot_name}_slug"] = slug
        meta[f"llm_{slot_name}_route"] = sort or "default"
        meta[f"llm_{slot_name}_max_tokens_cap"] = _int_env(
            f"OPENROUTER_{letter}_MAXTOK", DEFAULT_MAX_TOKENS_PER_CALL
        ) if letter != "?" else DEFAULT_MAX_TOKENS_PER_CALL
        meta[f"llm_{slot_name}_temperature_override"] = (
            _float_env(f"OPENROUTER_{letter}_TEMP") if letter != "?" else None
        )
    if meta:
        meta["llm_remote_provider"] = PROVIDER
        meta["llm_remote_schema_mode"] = is_openrouter_row_id(technical_id)
    return meta


def openrouter_run_meta() -> dict[str, Any]:
    """S3 run-meta: the slug each slot RESOLVES to (on the current bindings /
    fallback chain) plus catalog staleness, so a run records exactly which
    remote model would serve each slot and how fresh discovery was. Returns
    {} when remote is disabled, keeping a local run byte-identical (C1).
    Never raises (PD1): an unresolvable slot stamps '<unresolved>'."""
    if not openrouter_enabled():
        return {}
    out: dict[str, Any] = {}
    for letter, handle in (("a", SLOT_A_ID), ("b", SLOT_B_ID)):
        try:
            out[f"slot_{letter}_resolved_slug"] = resolve_slug(handle)
        except OpenRouterError:
            out[f"slot_{letter}_resolved_slug"] = "<unresolved>"
    cm = catalog_meta()
    out["openrouter_catalog_source"] = cm.get("source")
    out["openrouter_catalog_fetched_at"] = cm.get("fetched_at")
    out["openrouter_catalog_staleness"] = cm.get("staleness")
    return out


__all__ = [
    "OpenRouterBackend",
    "make_openrouter_generate_fn",
    "schema_to_response_format",
    "openrouter_meta_for",
    "resolved_models_snapshot",
    "reset_run_budget",
    "OpenRouterError",
    "OpenRouterConfigError",
    "OpenRouterCostCeilingError",
    "OpenRouterCallFailedError",
    "OpenRouterModelGoneError",
    "is_model_gone_error",
    "fallback_slug_for_slot",
    "OPENROUTER_BACKEND_KEY",
    "PROVIDER",
    "SLOT_A_ID",
    "SLOT_B_ID",
    "OPENROUTER_ROW_IDS",
    "OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT",
    "OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT",
    "openrouter_enabled",
    "is_openrouter_row_id",
    "resolve_slug",
    "resolve_route",
    "reset_run_budget",
    "set_slot_bindings",
    "clear_slot_bindings",
    "recommended_slug_for_slot",
    "openrouter_run_meta",
    # catalog cache (S0)
    "CATALOG_SCHEMA_VERSION",
    "load_catalog_cache",
    "cached_models",
    "catalog_meta",
    "refresh_catalog_cache",
]


# ---------------------------------------------------------------------------
# Self-test (S0 cache) -- no network, no GPU. Drives the cache through
# missing / live / corrupt / offline and proves it never raises or blocks,
# and that load_catalog_cache stays network-free (INPUT_TYPES-safe).
# Prints "SELF-TEST PASS: N/N". Run: python nodes\_otr_openrouter_backend.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    _passed = 0
    _total = 0

    def _check(name: str, cond: bool) -> None:
        global _passed, _total
        _total += 1
        if cond:
            _passed += 1
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name}")

    _mod = sys.modules[__name__]
    os.environ["OTR_OPENROUTER_CACHE_DIR"] = tempfile.mkdtemp(prefix="otr_orcat_")

    # 1. Missing cache -> safe empty, never raises.
    c0 = load_catalog_cache()
    _check("missing -> source=missing, empty",
           c0["source"] == "missing" and c0["models"] == [] and cached_models() == [])
    _check("catalog_meta(missing) -> staleness=empty",
           catalog_meta()["staleness"] == "empty")

    # 2. Refresh with a mocked fetch -> live cache; slimmed; supports_json derived.
    _fake_models = [
        {"id": "anthropic/claude-opus-4.8", "name": "Claude Opus 4.8",
         "created": 1, "context_length": 200000,
         "pricing": {"prompt": "0.000005", "completion": "0.000025"},
         "supported_parameters": ["tools", "response_format", "structured_outputs"]},
        {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro",
         "created": 2, "context_length": 128000,
         "pricing": {"prompt": "0.00000043", "completion": "0.00000087"},
         "supported_parameters": ["tools"]},
        {"no_id": True},  # malformed row -> dropped by _slim_model
    ]
    _mod._fetch_models_json = lambda **kw: _fake_models
    cat = refresh_catalog_cache()
    _check("refresh -> source=live, count=2 (malformed row dropped)",
           cat["source"] == "live" and cat["count"] == 2)
    _by_id = {m["id"]: m for m in cached_models()}
    _check("supports_json True when structured_outputs/response_format present",
           _by_id["anthropic/claude-opus-4.8"]["supports_json"] is True)
    _check("supports_json False when absent",
           _by_id["deepseek/deepseek-v4-pro"]["supports_json"] is False)
    _check("provider derived from slug prefix",
           _by_id["anthropic/claude-opus-4.8"]["provider"] == "anthropic")
    _check("catalog_meta(live) -> staleness=live, count=2",
           catalog_meta()["staleness"] == "live" and catalog_meta()["count"] == 2)

    # 3. Corrupt cache file -> safe empty, never raises.
    _catalog_cache_path().write_text("{ this is not valid json", encoding="utf-8")
    c3 = load_catalog_cache()
    _check("corrupt -> source=corrupt, empty",
           c3["source"] == "corrupt" and c3["models"] == [])

    # 4. Offline / failed fetch -> keeps the existing good cache, never raises.
    _mod._fetch_models_json = lambda **kw: _fake_models
    refresh_catalog_cache()  # lay down a good cache again

    def _boom(**kw):
        raise RuntimeError("simulated network failure")

    _mod._fetch_models_json = _boom
    try:
        cat4 = refresh_catalog_cache()
        _check("offline refresh -> kept existing cache (count 2), no raise",
               cat4["count"] == 2)
    except Exception as exc:  # noqa: BLE001 -- must NOT raise
        _check(f"offline refresh raised: {exc!r}", False)

    # 5. load_catalog_cache stays network-free even with the fetch seam broken
    #    (INPUT_TYPES safety -- the dropdowns read this, never the network).
    _check("load_catalog_cache works with fetch seam broken (no network)",
           isinstance(load_catalog_cache(), dict))

    os.environ.pop("OTR_OPENROUTER_CACHE_DIR", None)
    print(f"SELF-TEST PASS: {_passed}/{_total}")
    sys.exit(0 if _passed == _total else 1)
