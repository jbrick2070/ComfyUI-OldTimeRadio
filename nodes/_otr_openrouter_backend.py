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

import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reasoning-wrapper strip (BUG-306 / BUG-LOCAL-308 family)
# ---------------------------------------------------------------------------
# Thinking-mode models (gemma-4, DeepSeek-R1, QwQ, ...) prepend their
# chain-of-thought to the reply, wrapped in <think>...</think> -- and some
# emit OpenAI-"harmony" <|channel|>analysis<|message|>...<|channel|>final
# <|message|> markers. Through Ollama's OpenAI-compatible endpoint that
# scaffolding lands inline in message.content, so the writer's structured
# (JSON / GBNF) passes parse the reasoning preamble and abort the episode.
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
      3. A dangling ``</think>`` with no open -- Ollama pre-fills the
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
OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT = "anthropic/claude-opus-4.8"
OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT = "deepseek/deepseek-v4-pro"

# Conservative cost ceilings. Deliberately low so an unconfigured
# operator cannot accidentally spend a fortune; raise via env when ready.
# Defect C fix: the COST per-call ceiling must sit ABOVE the OUTPUT cap, or a
# reply near the output cap (+ the prompt added by the estimate) spuriously
# trips OpenRouterCostCeilingError. So they are two separate numbers.
DEFAULT_OUTPUT_TOKENS_CAP = 8192      # ceiling on OUTPUT tokens (max_tokens) per call
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


def _reasoning_effort_from_env() -> str | None:
    """Read OPENROUTER_REASONING_EFFORT, the OpenAI-standard reasoning control
    (high|medium|low|none). Ollama's OpenAI-compatible /v1 honours it to bound
    or DISABLE a thinking model's <think> preamble: the gemma-4 lane sets
    'none' so the model emits the structured answer directly instead of
    spending the output budget on reasoning (-> finish_reason=length -> an
    empty/truncated body the JSON passes cannot parse). Native `think:false`
    and `chat_template_kwargs` are NOT honoured on /v1 (ollama#14820 / #16240),
    so reasoning_effort is the portable lever. Unset/empty -> None (the field
    is omitted, so non-thinking models and OpenRouter-proper are unaffected)."""
    raw = _env("OPENROUTER_REASONING_EFFORT")
    if not raw:
        return None
    return raw.strip().lower() or None


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
    process lifetime."""
    global _run_token_total
    _run_token_total = 0


_run_token_total = 0


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

CATALOG_SCHEMA_VERSION = 1
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
        # Resolve the slug AND its provider-routing sort together: a ':nitro'/
        # ':floor' suffix on the bound slug is stripped here and carried as an
        # explicit sort, so the wire payload + run meta hold the clean slug.
        slug, provider_sort = resolve_route(letter, resolve_slug(repo_id))
        context_window = int(
            getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW)
            or DEFAULT_CONTEXT_WINDOW
        )
        cache_entry: dict[str, Any] = {
            "provider": PROVIDER,
            "model_id": repo_id,          # the virtual handle
            "slot_letter": letter,        # "A" / "B"
            "slug": slug,                 # the resolved real model id
            "provider_sort": provider_sort,  # throughput|price|latency|None
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
        floor = _int_env("OPENROUTER_MIN_OUTPUT_TOKENS", DEFAULT_MIN_OUTPUT_TOKENS)
        out_tokens = max(int(max_new_tokens or 0), floor)
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
        # Reasoning control (gemma-4 / thinking-model lane). The OpenAI-standard
        # `reasoning_effort` (high|medium|low|none) is the portable lever Ollama's
        # /v1 honours to bound or disable the <think> preamble; set
        # OPENROUTER_REASONING_EFFORT=none so a reasoning model emits the
        # structured answer directly instead of burning the output budget on
        # reasoning (-> finish_reason=length -> unparseable JSON). Unset -> the
        # field is omitted, so non-thinking models / OpenRouter-proper are
        # byte-identical. Not require_parameters-gated: a backend that ignores it
        # simply reasons as before -- the output-token floor is the safety net.
        reasoning_effort = _reasoning_effort_from_env()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
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
        if os.environ.get("OPENROUTER_DEBUG_RAW") == "1":
            log.info("[OpenRouter] raw %s response: %s", slug, json.dumps(body)[:1500])
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterCallFailedError(
                f"OpenRouter {slug} returned no choices: "
                f"{str(body)[:300]}"
            )
        if choices[0].get("finish_reason") == "length":
            log.warning(
                "[OpenRouter] %s hit finish_reason=length -- output truncated "
                "at the token ceiling; a downstream JSON parse may fail. Raise "
                "OPENROUTER_MIN_OUTPUT_TOKENS or the slot max-tokens cap.",
                slug,
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
            if isinstance(err, dict) and err.get("message"):
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
        return backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
            grammar=g,
        )

    # Markers so structured_call can detect a remote fn and whether it
    # already carries a schema-bound response_format (don't override that).
    # _otr_supports_grammar (A) lets the style-picker inventor pass a per-
    # call GBNF only to backends that honour it (the remote llama-server
    # lane); local backends lack the marker and stay byte-identical.
    generate_fn._otr_openrouter = True  # type: ignore[attr-defined]
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
    "OpenRouterError",
    "OpenRouterConfigError",
    "OpenRouterCostCeilingError",
    "OpenRouterCallFailedError",
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
