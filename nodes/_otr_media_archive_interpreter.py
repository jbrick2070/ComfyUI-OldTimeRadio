"""Media archive source brain.

Turns the normalized media-archive source payload into the same briefs-like
object that the legacy_many_pass writer already consumes. This is intentionally
parallel to ``news_interpreter`` without importing or reusing its science/news
validators.
"""
from __future__ import annotations

import hashlib
from typing import Callable

from pydantic import BaseModel, Field, field_validator

try:
    from ._otr_structured_call import StructuredCallFailedError, structured_call
except ImportError:  # pragma: no cover - flat test imports
    from _otr_structured_call import StructuredCallFailedError, structured_call  # type: ignore


PROMPT_VERSION = "media_archive_interpreter_v1"
SCHEMA_VERSION = "media_archive_briefs_v1"

_MAX_CASTING_BRIEF_CHARS = 900
_MAX_SCRIPT_BRIEF_CHARS = 1200
_MAX_CLOSE_BRIEF_CHARS = 500
_MAX_KEY_TERMS = 7
_MAX_KEY_TERM_CHARS = 80


class MediaArchiveInterpreterError(RuntimeError):
    """Raised when the media archive source brain cannot produce briefs."""

    def __init__(self, *, attempts: int, reason: str) -> None:
        self.attempts = attempts
        self.reason = reason
        super().__init__(
            f"media archive interpreter failed after {attempts} attempt(s): "
            f"{reason}"
        )


class MediaArchiveBriefs(BaseModel):
    """Briefs contract consumed by ``_otr_source_payload`` and the writer."""

    casting_brief: str = Field(..., max_length=_MAX_CASTING_BRIEF_CHARS)
    script_brief: str = Field(..., max_length=_MAX_SCRIPT_BRIEF_CHARS)
    news_close_brief: str = Field(..., max_length=_MAX_CLOSE_BRIEF_CHARS)
    key_terms: list[str] = Field(..., min_length=1, max_length=_MAX_KEY_TERMS)

    source_hash: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    attempts: int = 0

    @field_validator("key_terms", mode="before")
    @classmethod
    def _trim_term_count(cls, value):
        if isinstance(value, list) and len(value) > _MAX_KEY_TERMS:
            return value[:_MAX_KEY_TERMS]
        return value

    @field_validator("key_terms")
    @classmethod
    def _coerce_term_lengths(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for term in value:
            if not isinstance(term, str):
                raise ValueError(
                    f"key_term must be str, got {type(term).__name__}: {term!r}"
                )
            clean = " ".join(term.split()).strip()
            if len(clean) > _MAX_KEY_TERM_CHARS:
                clean = clean[:_MAX_KEY_TERM_CHARS].rsplit(" ", 1)[0].rstrip()
            if clean:
                out.append(clean)
        if not out:
            raise ValueError("key_terms must contain at least one non-empty term")
        return out


def _source_hash(payload: dict) -> str:
    h = hashlib.sha256()
    for key in ("headline", "summary", "full_text", "source", "date", "link"):
        h.update(str(payload.get(key, "")).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _build_prompt(payload: dict) -> list[dict[str, str]]:
    headline = str(payload.get("headline", "")).strip()
    source = str(payload.get("source", "")).strip()
    date = str(payload.get("date", "")).strip()
    link = str(payload.get("link", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    full_text = str(payload.get("full_text", "")).strip()
    source_block = "\n".join(
        part for part in (
            f"Title: {headline}",
            f"Source: {source}",
            f"Date: {date}" if date else "",
            f"URL: {link}" if link else "",
            f"Summary: {summary}",
            "Body:",
            full_text[:4000],
        )
        if part
    )
    instruction = (
        "You are the source brain for an optimistic old-time-radio drama lane "
        "about media archives and preservation.\n\n"
        "Turn the media-history source below into JSON briefs for a fictional "
        "curious archive mystery/adventure. Center discovering, restoring, "
        "researching, or preserving media history: film archives, forgotten "
        "recordings, lost broadcasts, restoration projects, librarians, "
        "historians, projectionists, collectors, and archivists.\n\n"
        "Tone: curious, humane, optimistic, public-broadcast mystery. Think "
        "National Treasure or Nancy Drew by way of archive documentaries. "
        "No crime thriller, murder, weapons, horror, sci-fi anthology, "
        "spaceship, mission control, laboratory containment, or generic "
        "experiment emergency.\n\n"
        "Return ONE JSON object only with exactly these keys:\n"
        "{\n"
        "  \"casting_brief\": \"80-700 chars; likely human roles and voices\",\n"
        "  \"script_brief\": \"120-1000 chars; fictional radio story premise "
        "grounded in the archive source\",\n"
        "  \"news_close_brief\": \"80-420 chars; archive/source note for the "
        "announcer, not a news report\",\n"
        "  \"key_terms\": [\"2-7 concise archive/source terms\"]\n"
        "}\n\n"
        "The story may be fictional, but the archive object, collection, "
        "preservation labor, or media-history hook must clearly come from the "
        "source material.\n\n"
        f"{source_block}"
    )
    return [{"role": "user", "content": instruction}]


def build_media_archive_briefs(
    *,
    technical_fn: Callable[..., str],
    payload: dict,
    model_id: str = "",
    max_attempts: int = 3,
    base_temperature: float = 0.45,
    max_new_tokens: int = 520,
) -> MediaArchiveBriefs:
    """Run the media-archive source brain through the structured JSON ladder."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    slot_calls = 0

    def _counting_slot_fn(msgs, *, temperature, max_new_tokens):
        nonlocal slot_calls
        slot_calls += 1
        # LLM slot: technical -- delegated source-brief generation call.
        return technical_fn(
            msgs, temperature=temperature, max_new_tokens=max_new_tokens,
        )

    def _content_validator(brief: MediaArchiveBriefs) -> str | None:
        if len(brief.key_terms) < 2:
            return "media archive briefs require at least two key_terms"
        hay = " ".join(
            [brief.casting_brief, brief.script_brief, brief.news_close_brief]
            + list(brief.key_terms)
        ).casefold()
        for term in (
            "spaceship", "mission control", "laboratory containment",
            "alien invasion", "murder victim", "body count",
            "haunting", "cursed object", "ransom", "corpse",
            "ghost", "phantom", "emergency broadcast",
            "serial killer", "murder weapon",
        ):
            if term in hay:
                return f"media archive brief drifted into forbidden term {term!r}"
        return None

    try:
        # LLM slot: technical -- structured JSON source-brief extraction for
        # the media-archive lane, routed through the writer's technical slot.
        brief = structured_call(
            prompt=_build_prompt(payload),
            schema=MediaArchiveBriefs,
            slot_fn=_counting_slot_fn,
            base_temperature=float(base_temperature),
            structural_retry_temperature=float(base_temperature) / 2.0,
            post_validator=_content_validator,
            max_new_tokens=int(max_new_tokens),
            max_attempts=int(max_attempts),
            helper_name="build_media_archive_briefs",
        )
    except StructuredCallFailedError as exc:
        raise MediaArchiveInterpreterError(
            attempts=exc.attempts,
            reason=(
                f"{type(exc.last_error).__name__}: {exc.last_error}"
                if exc.last_error is not None
                else "no error captured"
            ),
        ) from exc
    # Any other exception (coding bugs, contract violations, unexpected
    # backend errors) propagates HARD -- matching the science wrapper's
    # non-NewsInterpreterError path in _otr_source_payload.py.

    brief.source_hash = _source_hash(payload)
    brief.model_id = model_id
    brief.attempts = slot_calls
    return brief


__all__ = [
    "MediaArchiveBriefs",
    "MediaArchiveInterpreterError",
    "build_media_archive_briefs",
]
