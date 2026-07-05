"""Visual-style packs -- Stage 3 of the multi-modal story schema.

Style packs live in JSON (nodes/visual_styles/<style_id>.json); this module is
pure BEHAVIOR: strict fail-loud pack loading + validation + resolution. Zero
content literals. Stdlib only, same posture as _otr_story_routing.py.

Law (Stage 3, kibitz r1-r4 converged): JSON owns VISUAL-STYLE DELTAS (the
prompt tails + declared overrides); Python owns geometry contracts, core
prompt assembly, validation, routing. No fallbacks; unknown id = hard error.

LAZY: importing this module performs ZERO file I/O (ComfyUI custom-node import
isolation). The style directory sweep runs on the first resolve call, then
caches. `_clear_caches()` is the test hook.

v1 slice: tails + allow_radio_tails only. Subject overrides, motion-register
overrides, and STYLE_ANCHOR styling are NOT in the v1 schema (STAGE3_SUBPLAN
section 8 checklist).

forbidden_terms is LOAD-TIME LINT ONLY in v1 (case-insensitive substring over
the pack's own four tail fields -- a pack must not violate its own bans). No
compose-time scrub, no compose-time warn state (r1 M1 + r2 CUT-1).

The byte-identity contract: sci_fi_radio.json's four tails are byte-identical
to the extraction-fixture constants in _otr_story_brief_helpers.py
(ERA_TAIL_DEFAULT / STYLE_TAIL_DEFAULT / IMAGE_GRADE_TAIL /
RADIO_BROADCAST_TAIL) -- pinned by tests/test_visual_styles_3a.py. Production
code reads the PACK; the constants survive only as that fixture + the
legacy no-style lane of get_era_tail.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_VISUAL_STYLES_ROOT = Path(__file__).resolve().parent / "visual_styles"

#: The production default -- the lane every pre-Stage-3 episode ran.
DEFAULT_STYLE_ID = "sci_fi_radio"

KNOWN_STYLE_SCHEMA_VERSIONS = frozenset({"v1"})

_STYLE_ID_RE = re.compile(r"^[a-z0-9_]+$")

#: Exact v1 row schema: field name -> required python type.
_REQUIRED_FIELDS: "dict[str, type]" = {
    "style_id": str,
    "label": str,
    "positive_tail": str,
    "image_grade_tail": str,
    "broadcast_tail": str,
    "allow_radio_tails": bool,
    "forbidden_terms": list,
    "era_tail": str,
    "schema_version": str,
}

#: The four tail fields the load-time forbidden-terms lint sweeps.
_LINTED_TAIL_FIELDS = ("positive_tail", "image_grade_tail",
                       "broadcast_tail", "era_tail")


class VisualStyleError(Exception):
    """Base: any fail-loud visual-style problem."""


class UnknownVisualStyleError(VisualStyleError):
    """style_id has no pack under nodes/visual_styles/."""


class VisualStyleValidationError(VisualStyleError):
    """A pack file violates the v1 schema / lint / path contract."""


@dataclass(frozen=True)
class VisualStyle:
    """One loaded, validated style pack (attribute access only -- r2 OPT)."""
    style_id: str
    label: str
    positive_tail: str
    image_grade_tail: str
    broadcast_tail: str
    allow_radio_tails: bool
    forbidden_terms: "tuple[str, ...]"
    era_tail: str
    schema_version: str


# Lazy singleton -- built on first access, never at import time.
_STYLES: "dict[str, VisualStyle] | None" = None


def _validate_row(raw: dict, path: Path) -> VisualStyle:
    if not isinstance(raw, dict):
        raise VisualStyleValidationError(
            f"visual style {path}: top level must be an object, "
            f"got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))
    if unknown:
        raise VisualStyleValidationError(
            f"visual style {path}: unknown key(s) {unknown!r} -- the v1 "
            f"schema is exact (subject/motion/anchor overrides are NOT in "
            f"v1; see STAGE3_SUBPLAN section 8)")
    missing = sorted(set(_REQUIRED_FIELDS) - set(raw))
    if missing:
        raise VisualStyleValidationError(
            f"visual style {path}: missing required key(s) {missing!r}")
    for key, typ in _REQUIRED_FIELDS.items():
        val = raw[key]
        if typ is bool:
            if not isinstance(val, bool):
                raise VisualStyleValidationError(
                    f"visual style {path}: {key} must be a bool, "
                    f"got {type(val).__name__}")
        elif not isinstance(val, typ) or isinstance(val, bool):
            raise VisualStyleValidationError(
                f"visual style {path}: {key} must be {typ.__name__}, "
                f"got {type(val).__name__}")
    style_id = raw["style_id"]
    if not _STYLE_ID_RE.match(style_id):
        raise VisualStyleValidationError(
            f"visual style {path}: style_id {style_id!r} must match "
            f"{_STYLE_ID_RE.pattern}")
    if style_id != path.stem:
        raise VisualStyleValidationError(
            f"visual style {path}: header style_id {style_id!r} does not "
            f"match the filename {path.stem!r} -- the path IS the coordinate")
    if raw["schema_version"] not in KNOWN_STYLE_SCHEMA_VERSIONS:
        raise VisualStyleValidationError(
            f"visual style {path}: unknown schema_version "
            f"{raw['schema_version']!r}; known: "
            f"{sorted(KNOWN_STYLE_SCHEMA_VERSIONS)}")
    terms: "list[str]" = []
    for i, t in enumerate(raw["forbidden_terms"]):
        if not isinstance(t, str) or not t.strip():
            raise VisualStyleValidationError(
                f"visual style {path}: forbidden_terms[{i}] must be a "
                f"non-empty string")
        terms.append(t)
    # Load-time lint (r4: case-insensitive substring over the 4 tail fields).
    for term in terms:
        needle = term.lower()
        for fld in _LINTED_TAIL_FIELDS:
            if needle in str(raw[fld]).lower():
                raise VisualStyleValidationError(
                    f"visual style {path}: forbidden term {term!r} appears "
                    f"in its own {fld} -- a pack must not violate its own "
                    f"bans")
    return VisualStyle(
        style_id=style_id,
        label=raw["label"],
        positive_tail=raw["positive_tail"],
        image_grade_tail=raw["image_grade_tail"],
        broadcast_tail=raw["broadcast_tail"],
        allow_radio_tails=raw["allow_radio_tails"],
        forbidden_terms=tuple(terms),
        era_tail=raw["era_tail"],
        schema_version=raw["schema_version"],
    )


def _load_all() -> "dict[str, VisualStyle]":
    if not _VISUAL_STYLES_ROOT.is_dir():
        raise VisualStyleValidationError(
            f"visual styles directory missing: {_VISUAL_STYLES_ROOT}")
    styles: "dict[str, VisualStyle]" = {}
    entries = sorted(_VISUAL_STYLES_ROOT.iterdir(), key=lambda p: p.name)
    for entry in entries:
        if entry.is_dir():
            raise VisualStyleValidationError(
                f"unexpected subdirectory in visual_styles/: {entry} -- "
                f"packs are flat <style_id>.json files")
        if entry.suffix != ".json":
            raise VisualStyleValidationError(
                f"unexpected non-pack file in visual_styles/: {entry}")
        try:
            raw = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VisualStyleValidationError(
                f"visual style {entry}: unreadable/malformed JSON: {exc}"
            ) from exc
        row = _validate_row(raw, entry)
        styles[row.style_id] = row
    if not styles:
        raise VisualStyleValidationError(
            f"no visual style packs found under {_VISUAL_STYLES_ROOT}")
    if DEFAULT_STYLE_ID not in styles:
        raise VisualStyleValidationError(
            f"the default style {DEFAULT_STYLE_ID!r} is missing from "
            f"{_VISUAL_STYLES_ROOT} -- the production lane must exist")
    return styles


def _ensure_loaded() -> "dict[str, VisualStyle]":
    global _STYLES
    if _STYLES is None:
        _STYLES = _load_all()
    return _STYLES


def list_style_ids() -> "tuple[str, ...]":
    """Registered style ids, deterministic filename order (dropdowns)."""
    return tuple(_ensure_loaded())


def resolve_visual_style(style_id: str) -> VisualStyle:
    """id -> validated VisualStyle. Unknown id = hard error, no fallback."""
    styles = _ensure_loaded()
    try:
        return styles[style_id]
    except KeyError:
        raise UnknownVisualStyleError(
            f"unknown visual_style {style_id!r}; registered: "
            f"{sorted(styles)}") from None


def get_visual_style(meta: object) -> VisualStyle:
    """The ONE meta-driven resolver composer entry points call (once per
    entry; pass the result down -- helpers never re-resolve).

    meta["visual_style"] absent/empty => the production default
    (sci_fi_radio) -- every pre-Stage-3 ledger and every default run.
    A PRESENT-but-unknown id fails LOUD (UnknownVisualStyleError)."""
    m = meta if isinstance(meta, dict) else {}
    raw = m.get("visual_style")
    style_id = str(raw).strip() if raw is not None else ""
    if not style_id:
        style_id = DEFAULT_STYLE_ID
    return resolve_visual_style(style_id)


def _clear_caches() -> None:
    """Test hook: drop the lazy registry (mirrors _otr_story_routing)."""
    global _STYLES
    _STYLES = None


__all__ = [
    "DEFAULT_STYLE_ID",
    "KNOWN_STYLE_SCHEMA_VERSIONS",
    "UnknownVisualStyleError",
    "VisualStyle",
    "VisualStyleError",
    "VisualStyleValidationError",
    "get_visual_style",
    "list_style_ids",
    "resolve_visual_style",
]
