"""Model ids, slot bindings, and cache helpers for the Google BYO LLM lane."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .client import GoogleAPIKeyMissingError, resolve_api_key

GOOGLE_API_BACKEND_KEY = "google_api_http"
GOOGLE_API_PROVIDER = "google_api"
GOOGLE_API_SLOT_A_ID = "google_api:slot-a"
GOOGLE_API_SLOT_B_ID = "google_api:slot-b"
GOOGLE_API_ROW_IDS = frozenset({GOOGLE_API_SLOT_A_ID, GOOGLE_API_SLOT_B_ID})

GOOGLE_API_MODEL_UNSELECTED = "(select Google API model)"
GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT = "gemini-flash-latest"
GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT = "gemini-flash-lite-latest"
GOOGLE_API_LEGACY_TEXT_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)
#: RETIRED 2026-08-10: `gemini-2.0-flash` and `gemini-2.0-flash-lite` were
#: removed from this tuple because they are GONE from Google's own catalog.
#: Measured, not assumed -- `scripts/verify_google_slugs.py` fetched a COMPLETE
#: terminal-page listing of 52 ids and neither was in it
#: (`docs/2026-08-10-MEASUREMENT-google-catalog-slug-provenance.md`). Offering a
#: model the provider no longer lists is a dropdown entry that can only fail at
#: request time, which is the exact `tencent/hy3:free` shape this pack's slug
#: guards exist to prevent. The lane's DEFAULTS were never affected: both are
#: `*-latest` pointers and both are still listed.
GOOGLE_API_STABLE_TEXT_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)
GOOGLE_API_STATIC_TEXT_MODELS = (
    GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT,
    GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT,
    *GOOGLE_API_STABLE_TEXT_MODELS,
    *GOOGLE_API_LEGACY_TEXT_MODELS,
)
GOOGLE_API_STRUCTURED_OUTPUT_MODELS = frozenset(GOOGLE_API_STATIC_TEXT_MODELS)
DEFAULT_CONTEXT_WINDOW = 8192

_slot_bindings: dict[str, str | None] = {"A": None, "B": None}


class GoogleAPISlotBindingError(GoogleAPIKeyMissingError):
    """A Google slot handle was selected without a concrete model binding."""


def google_api_enabled() -> bool:
    try:
        resolve_api_key()
        return True
    except GoogleAPIKeyMissingError:
        return False


def is_google_api_row_id(repo_id: str) -> bool:
    return isinstance(repo_id, str) and repo_id in GOOGLE_API_ROW_IDS


def _slot_letter(repo_id: str) -> str:
    if repo_id == GOOGLE_API_SLOT_A_ID:
        return "A"
    if repo_id == GOOGLE_API_SLOT_B_ID:
        return "B"
    raise GoogleAPISlotBindingError(
        f"not a Google API virtual row id: {repo_id!r} "
        f"(expected one of {sorted(GOOGLE_API_ROW_IDS)})"
    )


def _clean_slot_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s == GOOGLE_API_MODEL_UNSELECTED:
        return None
    if s.startswith("(") and s.endswith(")"):
        return None
    return s


def set_slot_bindings(*, slot_a: Any = None, slot_b: Any = None) -> None:
    _slot_bindings["A"] = _clean_slot_value(slot_a)
    _slot_bindings["B"] = _clean_slot_value(slot_b)


def clear_slot_bindings() -> None:
    _slot_bindings["A"] = None
    _slot_bindings["B"] = None


def _cache_path() -> Path:
    env = os.environ.get("OTR_GOOGLE_API_MODEL_CACHE")
    if env:
        return Path(env).expanduser()
    try:
        import folder_paths  # type: ignore

        user_dir = getattr(folder_paths, "get_user_directory", lambda: "")()
        if user_dir:
            return Path(user_dir) / "old_time_radio" / "google_api_model_cache.json"
    except Exception:
        pass
    try:
        from .._otr_paths import otr_shared_tmp_dir
    except Exception:
        from nodes._otr_paths import otr_shared_tmp_dir  # type: ignore
    return otr_shared_tmp_dir() / "google_api_model_cache.json"


def load_model_cache(path: Path | None = None) -> dict[str, Any]:
    p = path or _cache_path()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"models": [], "source": "missing"}
    except Exception:
        return {"models": [], "source": "corrupt"}
    if not isinstance(data, dict):
        return {"models": [], "source": "corrupt"}
    models = data.get("models")
    if not isinstance(models, list):
        models = []
    return {"models": models, "source": data.get("source") or "cache"}


def _cached_text_models() -> list[str]:
    out: list[str] = []
    for row in load_model_cache().get("models") or []:
        if isinstance(row, str):
            mid = row
        elif isinstance(row, dict):
            mid = str(row.get("id") or row.get("name") or "")
            output = row.get("output") or row.get("output_modalities")
            if output and "text" not in [str(v).lower() for v in output]:
                continue
        else:
            continue
        if mid and mid not in out:
            out.append(mid)
    return out


def google_api_model_choices(slot: str) -> list[str]:
    """Network-free choices for google_api_slot_<slot>_model."""
    s = slot.strip().lower()
    if s not in ("a", "b"):
        raise ValueError(f"slot must be 'a' or 'b', got {slot!r}")
    choices = [GOOGLE_API_MODEL_UNSELECTED]
    if google_api_enabled():
        lead = (
            GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT if s == "a"
            else GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT
        )
        for mid in (lead, *GOOGLE_API_STATIC_TEXT_MODELS, *_cached_text_models()):
            if mid and mid not in choices:
                choices.append(mid)
    else:
        for mid in _cached_text_models():
            if mid and mid not in choices:
                choices.append(mid)
    return choices


def resolve_model_for_slot(repo_id: str) -> str:
    letter = _slot_letter(repo_id)
    bound = _slot_bindings.get(letter)
    if not bound:
        raise GoogleAPISlotBindingError(
            f"{repo_id} selected but google_api_slot_{letter.lower()}_model is "
            f"unset. Pick a concrete Gemini model in the Google API slot picker. "
            "No request was sent and no local fallback was attempted."
        )
    return bound


def supports_structured_output(model_id: str) -> bool:
    if model_id in GOOGLE_API_STRUCTURED_OUTPUT_MODELS:
        return True
    for row in load_model_cache().get("models") or []:
        if not isinstance(row, dict):
            continue
        if row.get("id") != model_id and row.get("name") != model_id:
            continue
        val = row.get("structured_outputs")
        if val is None:
            val = row.get("supports_json")
        return bool(val)
    return False


def google_api_run_meta(creative_id: str, technical_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for slot_name, repo_id in (("creative", creative_id), ("technical", technical_id)):
        if not is_google_api_row_id(repo_id):
            continue
        out[f"llm_{slot_name}_provider"] = GOOGLE_API_PROVIDER
        out[f"llm_{slot_name}_handle"] = repo_id
        try:
            out[f"llm_{slot_name}_google_model"] = resolve_model_for_slot(repo_id)
        except GoogleAPISlotBindingError:
            out[f"llm_{slot_name}_google_model"] = "<unbound>"
    if out:
        out["llm_remote_provider"] = GOOGLE_API_PROVIDER
        out["google_api_model_bindings"] = {
            "slot_a": _slot_bindings.get("A"),
            "slot_b": _slot_bindings.get("B"),
        }
    return out


__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "GOOGLE_API_BACKEND_KEY",
    "GOOGLE_API_MODEL_UNSELECTED",
    "GOOGLE_API_PROVIDER",
    "GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT",
    "GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT",
    "GOOGLE_API_ROW_IDS",
    "GOOGLE_API_SLOT_A_ID",
    "GOOGLE_API_SLOT_B_ID",
    "GOOGLE_API_STATIC_TEXT_MODELS",
    "GoogleAPISlotBindingError",
    "clear_slot_bindings",
    "google_api_enabled",
    "google_api_model_choices",
    "google_api_run_meta",
    "is_google_api_row_id",
    "load_model_cache",
    "resolve_model_for_slot",
    "set_slot_bindings",
    "supports_structured_output",
]
