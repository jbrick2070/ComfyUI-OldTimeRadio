"""Media archive source brain.

Turns the normalized media-archive source payload into the same briefs-like
object that the legacy_many_pass writer already consumes. This is intentionally
parallel to ``news_interpreter`` without importing or reusing its science/news
validators.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

try:
    from ._otr_structured_call import StructuredCallFailedError, structured_call
except ImportError:  # pragma: no cover - flat test imports
    from _otr_structured_call import StructuredCallFailedError, structured_call  # type: ignore


PROMPT_VERSION = "media_archive_interpreter_v1"
SCHEMA_VERSION = "media_archive_briefs_v1"
DRAMA_SEED_SCHEMA_VERSION = "media_archive_drama_seeds_v1"

_DRAMA_SEEDS_PATH = (
    Path(__file__).resolve().parent
    / "story_packs"
    / "media_archive"
    / "drama_seeds.json"
)

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
    """Briefs contract consumed by the source adapter and writer."""

    casting_brief: str
    script_brief: str
    news_close_brief: str
    key_terms: list[str] = Field(default_factory=list)

    source_hash: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    attempts: int = 0


def _source_hash(payload: dict) -> str:
    h = hashlib.sha256()
    for key in ("headline", "summary", "full_text", "source", "date", "link"):
        h.update(str(payload.get(key, "")).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def load_drama_seeds(path: str | Path | None = None) -> tuple[str, ...]:
    """Load and validate the JSON-owned media archive dramatic seed deck."""
    seed_path = str(path or _DRAMA_SEEDS_PATH)
    return _load_drama_seeds_cached(seed_path)


@lru_cache(maxsize=4)
def _load_drama_seeds_cached(seed_path: str) -> tuple[str, ...]:
    path = Path(seed_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason=f"could not read media archive drama seeds: {exc}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason=f"malformed media archive drama seeds JSON: {exc}",
        ) from exc

    if not isinstance(raw, dict):
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason="media archive drama seeds root must be an object",
        )
    if raw.get("schema_version") != DRAMA_SEED_SCHEMA_VERSION:
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason=(
                "media archive drama seeds schema_version must be %r"
                % DRAMA_SEED_SCHEMA_VERSION
            ),
        )
    seeds_raw = raw.get("seeds")
    if not isinstance(seeds_raw, list):
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason="media archive drama seeds must have a seeds list",
        )

    seeds: list[str] = []
    for idx, seed in enumerate(seeds_raw):
        if not isinstance(seed, str):
            raise MediaArchiveInterpreterError(
                attempts=0,
                reason=f"media archive drama seed {idx} must be a string",
            )
        title = " ".join(seed.split()).strip()
        if not title:
            raise MediaArchiveInterpreterError(
                attempts=0,
                reason=f"media archive drama seed {idx} is empty",
            )
        seeds.append(title)

    if len(seeds) != len(set(seeds)):
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason="media archive drama seeds must be unique",
        )
    if not seeds:
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason="media archive drama seeds cannot be empty",
        )
    return tuple(seeds)


def select_drama_seed(payload: dict, seeds: tuple[str, ...] | None = None) -> str:
    """Select one seed deterministically from the RSS/source payload hash."""
    deck = seeds or load_drama_seeds()
    if not deck:
        raise MediaArchiveInterpreterError(
            attempts=0,
            reason="media archive drama seeds cannot be empty",
        )
    return deck[int(_source_hash(payload), 16) % len(deck)]


def _build_prompt(payload: dict) -> list[dict[str, str]]:
    headline = str(payload.get("headline", "")).strip()
    source = str(payload.get("source", "")).strip()
    date = str(payload.get("date", "")).strip()
    link = str(payload.get("link", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    full_text = str(payload.get("full_text", "")).strip()
    drama_seed = select_drama_seed(payload)
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
        "Tone: curious, humane, optimistic, public-broadcast mystery. Use the "
        "selected dramatic seed as the compact adventure lens. "
        "No crime thriller, murder, weapons, horror, sci-fi anthology, "
        "spaceship, mission control, laboratory containment, or generic "
        "experiment emergency.\n\n"
        f"Dramatic seed lens: {drama_seed}\n"
        "Source-truth rule: RSS/source material remains the source of truth. "
        "The dramatic seed is only a fictional lens; the seed is not source "
        "fact and must not create source facts.\n\n"
        "Return ONE JSON object only with exactly these keys:\n"
        "{\n"
        "  \"casting_brief\": \"source-grounded human roles and voices\",\n"
        "  \"script_brief\": \"fictional radio story premise grounded in the archive source\",\n"
        "  \"news_close_brief\": \"archive/source note for the announcer, not a "
        "news report -- name what was found and who found or reported it "
        "(a person, archive, or institution), stated briefly, then end on "
        "one short reflective thought. One to three sentences total, never "
        "a written explainer.\",\n"
        "  \"key_terms\": [\"optional concise archive/source terms\"]\n"
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
    "DRAMA_SEED_SCHEMA_VERSION",
    "MediaArchiveBriefs",
    "MediaArchiveInterpreterError",
    "build_media_archive_briefs",
    "load_drama_seeds",
    "select_drama_seed",
]
