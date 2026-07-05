"""Bridge-artifact validator for the OTR upstream transplant.

DESTINATION: ComfyUI-OldTimeRadio/nodes/_otr_ledger_input_adapter.py
STATUS: staged in the lab (transplant_work/); NOT installed in production yet.

Validates one frozen bridge artifact (the upstream lab's translator-head
output) before any production code consumes it. Pure stdlib + optional
injected model classes - it never imports the lab and only touches production
models the caller explicitly passes in (production wires NewsBriefs; lab
tests pass None and get pinned key-set validation instead - an explicit
parameter, not a hidden fallback).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

#: Pinned to production news_interpreter.NewsBriefs (13 fields). The caller
#: may pass the real model class for full validation; the pin guards key-set
#: shape when the model is not supplied (lab-side tests).
NEWS_BRIEFS_FIELDS = (
    "casting_brief", "script_brief", "news_close_brief", "key_terms",
    "source_hash", "source_chars", "prompt_version", "schema_version",
    "model_id", "decoder_profile", "seed", "attempts", "attempt_failures",
)
NEWS_SEED_KEYS = (
    "headline", "source", "url", "date", "body_chars", "style", "selected_at",
)
NEWS_ARTICLE_KEYS = (
    "headline", "summary", "full_text", "source", "date", "link", "seed_text",
)
REQUIRED_TOP_LEVEL = (
    "schema_version", "created_utc", "ledger_writing_spec", "meta_mirrors",
    "adapter_news_article",
)
REQUIRED_SPEC_IDS = (
    "source_bank_id", "story_model_id", "story_pipeline_id", "visual_style_id",
)
SUPPORTED_SCHEMA_VERSIONS = ("v2.0",)


class BridgeInputError(ValueError):
    """The bridge artifact is unusable. Message says exactly why. There is no
    partial acceptance and no fallback payload."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeInputError(message)


def load_bridge_artifact(
    path: str | Path,
    *,
    news_briefs_model: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load a bridge artifact JSON file from an EXPLICIT path (the writer's
    forceInput socket value - never a scanned folder) and validate it.
    This is the one production entry point for file handoff (kibitz r3:
    emit writes a path, production consumes a path; one contract)."""

    p = Path(path)
    if not p.is_file():
        raise BridgeInputError(f"bridge artifact file not found: {p}")
    raw = p.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        raise BridgeInputError(f"bridge artifact has a UTF-8 BOM: {p}")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeInputError(f"bridge artifact is not valid UTF-8 JSON: {p}: {exc}") from exc
    return validate_bridge_artifact(data, news_briefs_model=news_briefs_model)


def validate_bridge_artifact(
    data: dict[str, Any],
    *,
    news_briefs_model: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate and return the artifact dict; raise BridgeInputError on any
    defect. `news_briefs_model` is the production NewsBriefs class when
    called from production code (strongest validation path)."""

    _require(isinstance(data, dict), "bridge artifact must be a JSON object")
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in data]
    _require(not missing, f"bridge artifact missing top-level keys: {missing}")
    _require(
        data["schema_version"] in SUPPORTED_SCHEMA_VERSIONS,
        f"unsupported bridge schema_version {data['schema_version']!r}; "
        f"supported: {SUPPORTED_SCHEMA_VERSIONS}",
    )

    spec = data["ledger_writing_spec"]
    _require(isinstance(spec, dict), "ledger_writing_spec must be an object")
    for key in REQUIRED_SPEC_IDS:
        _require(
            bool(str(spec.get(key, "")).strip()),
            f"ledger_writing_spec.{key} is required and empty",
        )
    profile = spec.get("prompt_profile")
    _require(isinstance(profile, dict), "ledger_writing_spec.prompt_profile missing")
    for key in ("source_bank_id", "story_model_id"):
        _require(
            profile.get(key) == spec.get(key),
            f"prompt_profile.{key} != spec.{key} - refusing inconsistent spec",
        )
    policy = spec.get("visual_policy")
    _require(isinstance(policy, dict), "ledger_writing_spec.visual_policy missing")
    _require(
        policy.get("style_id") == spec.get("visual_style_id"),
        "visual_policy.style_id != spec.visual_style_id",
    )

    mirrors = data["meta_mirrors"]
    _require(isinstance(mirrors, dict), "meta_mirrors must be an object")
    news = mirrors.get("news")
    _require(isinstance(news, dict), "meta_mirrors.news missing")
    if news_briefs_model is not None:
        try:
            news_briefs_model(**news)
        except Exception as exc:
            raise BridgeInputError(
                f"meta_mirrors.news failed NewsBriefs validation: {exc}"
            ) from exc
    else:
        _require(
            set(news) == set(NEWS_BRIEFS_FIELDS),
            "meta_mirrors.news keys != NewsBriefs fields: "
            f"{sorted(set(news) ^ set(NEWS_BRIEFS_FIELDS))}",
        )
    _require(isinstance(news.get("key_terms"), list),
             "meta_mirrors.news.key_terms must be a list (freeze invariant)")

    seed = mirrors.get("news_seed")
    _require(isinstance(seed, dict), "meta_mirrors.news_seed missing")
    _require(
        set(seed) == set(NEWS_SEED_KEYS),
        f"meta_mirrors.news_seed keys drifted: {sorted(set(seed) ^ set(NEWS_SEED_KEYS))}",
    )
    _require(bool(str(seed.get("headline", "")).strip()),
             "meta_mirrors.news_seed.headline is required (consumer-read key)")

    article = data["adapter_news_article"]
    _require(isinstance(article, dict), "adapter_news_article must be an object")
    _require(
        set(article) == set(NEWS_ARTICLE_KEYS),
        f"adapter_news_article keys drifted: {sorted(set(article) ^ set(NEWS_ARTICLE_KEYS))}",
    )
    _require(bool(str(article.get("seed_text", "")).strip()),
             "adapter_news_article.seed_text is required and empty")
    return data
