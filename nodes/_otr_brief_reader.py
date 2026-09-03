"""nodes/_otr_brief_reader.py -- central reader for the story-brief
meta delta (Sprint 8.1).

The brief producer (`_otr_story_brief.py`) emits the eight v1 meta
keys plus the five v2 additive fields onto `meta` (decision A1, flat
additive -- see `downstream_brief_consumer_followup.md`). Every
downstream consumer that wants to read one of those fields routes
through `_read_brief_field` here instead of poking the meta dict
directly. Centralizing the access path means:

  * Consumers cannot drift on what counts as "no signal" -- the
    helper treats absent / non-dict meta / wrong-type values
    uniformly as "fall through to caller-supplied default", and the
    caller hands in the default that matches its v1 path.
  * If a future v3 ever nests fields under a sub-object, the
    dotted-path navigator handles it without any consumer-side
    change (decision B: dotted-path API, forward-compat).
  * One unit test pins the contract for every consumer at once,
    rather than N nearly-identical reader tests per consumer file.

Module is PURE: no I/O, no GPU, no ComfyUI imports, no producer
import. The dependency direction is consumer -> reader, NEVER the
reverse. The producer (`_otr_story_brief.py`) stamps the v2 fields;
the reader only ever reads them.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
"""
from __future__ import annotations

from typing import Any


__all__ = ["_read_brief_field", "spoken_term"]


def spoken_term(term: Any) -> str:
    """One brief term as a human would say it, for a prompt a model reads.

    The brief is authored by an LLM against a schema that names its list fields
    and, for `setting_terms`, defined nothing -- so a share of terms arrive in
    identifier case: ``control_room``, ``film_reel``, ``petri_dish``,
    ``concrete_floors``. Nothing downstream undid that, and every consumer that
    joins those terms into model-facing text emitted them verbatim. Measured
    2026-09-03 across the 1,955 episodes on disk, ``snake_case`` was 690 of the
    773 mechanically-bad setting terms (PBUG-20260903-04).

    An underscore is never part of what the author meant to say -- it is the
    schema's punctuation leaking into prose -- so this is a normalisation, not a
    judgement about the term. Prose is returned unchanged, which is every term on
    a healthy episode.

    IT LIVES HERE, beside `_read_brief_field`, for the reason this module was
    written: *consumers cannot be allowed to drift on how they read a brief
    field.* Four consumers join `story_brief_terms.setting` into model-facing
    text -- the ghost video composer, the still-image fallback, the ShotLock
    derivation prompt and the music prompt -- and fixing one of them is how the
    same episode ends up spelled two ways in two prompts.
    """
    return " ".join(str(term or "").replace("_", " ").split())


def _meta(obj: Any) -> dict:
    """Accept either a meta dict directly OR a parent dict carrying a
    `meta` sub-key. Mirrors `_otr_story_brief_helpers._meta` so a
    consumer that hands in either shape gets uniform behavior.

    Returns an empty dict when the input is not a dict or carries
    nothing recognizable -- the caller's default then flows through.
    """
    if isinstance(obj, dict):
        # Brief-shaped meta dicts always carry at least story_brief or
        # story_brief_status; if neither is present the caller probably
        # handed in a parent dict (ledger or script_json) with a `meta`
        # sub-key.
        if "story_brief_status" in obj or "story_brief" in obj:
            return obj
        sub = obj.get("meta")
        if isinstance(sub, dict):
            return sub
        # Treat the input as already-meta-shaped (callers may pass a
        # bare meta dict early in the lifecycle before story_brief is
        # stamped); the dotted-path walker handles missing keys.
        return obj
    return {}


def _read_brief_field(
    meta: Any,
    dotted_path: str,
    default: Any = None,
) -> Any:
    """Read a value from the brief meta delta via dotted-path navigation.

    Sprint 8.1 (decision B: dotted-path forward-compat). Under A1
    (flat additive), every v2 field is a top-level meta key and the
    dotted path is a single segment -- e.g.

        _read_brief_field(meta, "music_mood_terms", default=[])
        _read_brief_field(meta, "atmosphere_line",  default="")

    The dotted-path navigator also reads the existing v1 nested
    object `story_brief_terms` for callers that prefer one helper
    over two access patterns -- e.g.

        _read_brief_field(meta, "story_brief_terms.atmosphere",
                          default=[])

    Args:
      meta:
        Either the brief-shaped meta dict (the one carrying
        `story_brief_status`), OR a parent dict (ledger / script_json)
        that carries `meta` as a sub-key. `_meta` normalizes both.
      dotted_path:
        Segments joined by ".". Each segment is a dict key. Leading /
        trailing / empty segments raise ValueError so a caller cannot
        silently degrade a typo to the default.
      default:
        Returned when any segment is missing, when an intermediate
        value is not a dict, or when the meta dict is empty. The
        caller picks the type that matches its v1 fallback path
        (typically `[]` for list-shaped fields and `""` for string-
        shaped fields).

    Returns:
      The terminal value at `dotted_path` if every segment is present
      and the intermediates are dicts; otherwise `default`. Returns
      `default` for `None` terminal values too -- a v2 field stamped
      explicitly to None (no producer path does this today, but a
      future writer regression might) reads as "no signal" rather
      than as a literal None.

    Raises:
      ValueError: dotted_path is empty, has empty segments, or
        contains only whitespace. Catches the most common consumer-
        side typo before it silently returns the default forever.
    """
    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise ValueError(
            f"_read_brief_field: dotted_path must be a non-empty string; "
            f"got {dotted_path!r}"
        )
    segments = dotted_path.split(".")
    if any(not seg or not seg.strip() for seg in segments):
        raise ValueError(
            f"_read_brief_field: dotted_path {dotted_path!r} has an empty "
            f"segment (leading/trailing/double dot)"
        )

    cursor: Any = _meta(meta)
    if not cursor:
        return default
    for seg in segments:
        if not isinstance(cursor, dict):
            return default
        if seg not in cursor:
            return default
        cursor = cursor[seg]
    if cursor is None:
        return default
    return cursor
