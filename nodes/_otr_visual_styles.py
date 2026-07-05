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

SCHEMA v2 (TOTAL-COVERAGE slice, STAGE3_TOTAL_COVERAGE_SUBPLAN v5 FINAL,
kibitz r1-r4 converged 2026-07-05): the v1 tail fields stay, PLUS 11 new str
fields + 4 new dict fields carrying every pack-owned LOOK/SUBJECT surface
(geometry-vs-look law: framing/headroom/mouth-safety GEOMETRY stays Python;
only LOOK/SUBJECT text lives here). The v2 schema is FINAL from chunk A1 --
A2/C surfaces LOAD now and are consumed by their own chunks. A v1 pack fails
load LOUD naming the path (no back-compat defaults -- fail-loud law).

Non-empty rule: every NEW str field must be non-empty EXCEPT
`scene_instruction_look` (the char-scene builder has no existing look text,
so sci_fi ships "" and the composer appends only-when-non-empty -- r4 AG M1).
Template fields carry EXACTLY one placeholder and no other brace tokens:
`announcer_subject_ltx_mouth` + every `open_subjects` value take `{form}`;
`non_character_emblem_fallback` takes `{base}`. `announcer_subject_ltx_mouth`
must additionally carry mouth-prominence vocabulary (the ia2v lip-sync
contract). `motion_registers` values are budgeted at 240 chars at LOAD
(BUG-LOCAL-112). The forbidden-terms lint sweeps the 4 tail fields plus ALL
new string leaves + dict values (r2 codex S2).

The byte-identity contract: sci_fi_radio.json's fields are byte-identical to
the extraction-fixture constants (tails in _otr_story_brief_helpers.py; look/
subject fixtures in otr_meta_brief_image_prompt.py + the open-subject
templates in _otr_story_brief_helpers.py; motion registers in
render_driver.py; still_word maps in otr_meta_brief_image_prompt.py) --
pinned by tests. Production code reads the PACK; the constants survive only
as those fixtures + the designated legacy no-style lanes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_VISUAL_STYLES_ROOT = Path(__file__).resolve().parent / "visual_styles"

#: The production default -- the lane every pre-Stage-3 episode ran.
DEFAULT_STYLE_ID = "sci_fi_radio"

KNOWN_STYLE_SCHEMA_VERSIONS = frozenset({"v2"})

_STYLE_ID_RE = re.compile(r"^[a-z0-9_]+$")

#: v1 tail fields (kept in v2; empty-string stays legal on the 4 tails).
_V1_FIELDS: "dict[str, type]" = {
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

#: NEW v2 str fields (r4 authoritative inventory). All REQUIRED; all
#: non-empty except scene_instruction_look (r4 AG M1 exemption).
_V2_STR_FIELDS = (
    "portrait_look",
    "portrait_look_talking",
    "portrait_instruction_look",
    "scene_instruction_look",
    "announcer_subject_face",
    "announcer_subject_ltx_mouth",
    "announcer_subject_object",
    "radio_object_look",
    "plate_look",
    "non_character_emblem_fallback",
    "still_word_title_mood_style",
)
_V2_EMPTY_LEGAL_STR_FIELDS = frozenset({"scene_instruction_look"})

#: NEW v2 dict fields -> their EXACT (case-sensitive) key sets.
_V2_DICT_FIELDS: "dict[str, frozenset]" = {
    "open_subjects": frozenset({"synthetic", "announcer", "default"}),
    "motion_registers": frozenset(
        {"announcer", "music_open", "music_close", "music_inter"}),
    "still_word_typography": frozenset(
        {"noir", "sci-fi", "western", "pulp", "default"}),
    "still_word_backdrop": frozenset(
        {"noir", "sci-fi", "western", "pulp", "default"}),
}

_REQUIRED_FIELDS: "dict[str, type]" = dict(_V1_FIELDS)
_REQUIRED_FIELDS.update({name: str for name in _V2_STR_FIELDS})
_REQUIRED_FIELDS.update({name: dict for name in _V2_DICT_FIELDS})

#: The four tail fields the load-time forbidden-terms lint has always swept;
#: v2 extends the sweep over all new string leaves + dict values.
_LINTED_TAIL_FIELDS = ("positive_tail", "image_grade_tail",
                       "broadcast_tail", "era_tail")

#: Template fields: field -> the exactly-once placeholder name.
_TEMPLATE_STR_FIELDS = {
    "announcer_subject_ltx_mouth": "form",
    "non_character_emblem_fallback": "base",
}
#: open_subjects values are {form} templates too (each value exactly once).
_OPEN_SUBJECTS_PLACEHOLDER = "form"

#: Mouth-prominence vocabulary the ltx_mouth subject must carry (ia2v
#: lip-sync contract: LTX drives whatever READS as a mouth).
_MOUTH_VOCAB = ("mouth", "lips")

#: BUG-LOCAL-112: motion prompts are budgeted; enforced at LOAD on pack
#: motion_registers values (mirrors render_driver._LTX_MOTION_PROMPT_MAX).
_MOTION_REGISTER_MAX_CHARS = 240


class VisualStyleError(Exception):
    """Base: any fail-loud visual-style problem."""


class UnknownVisualStyleError(VisualStyleError):
    """style_id has no pack under nodes/visual_styles/."""


class VisualStyleValidationError(VisualStyleError):
    """A pack file violates the v2 schema / lint / path contract."""


@dataclass(frozen=True)
class VisualStyle:
    """One loaded, validated style pack (attribute access only -- r2 OPT).
    Dict fields are immutable mappings (r2 codex OPT)."""
    style_id: str
    label: str
    positive_tail: str
    image_grade_tail: str
    broadcast_tail: str
    allow_radio_tails: bool
    forbidden_terms: "tuple[str, ...]"
    era_tail: str
    schema_version: str
    # -- v2 LOOK/SUBJECT surfaces (geometry stays Python) --
    portrait_look: str
    portrait_look_talking: str
    portrait_instruction_look: str
    scene_instruction_look: str
    announcer_subject_face: str
    announcer_subject_ltx_mouth: str
    announcer_subject_object: str
    radio_object_look: str
    plate_look: str
    non_character_emblem_fallback: str
    still_word_title_mood_style: str
    open_subjects: "MappingProxyType"
    motion_registers: "MappingProxyType"
    still_word_typography: "MappingProxyType"
    still_word_backdrop: "MappingProxyType"


# Lazy singleton -- built on first access, never at import time.
_STYLES: "dict[str, VisualStyle] | None" = None


def _lint_template(path: Path, label: str, value: str,
                   placeholder: str) -> None:
    """EXACTLY one ``{placeholder}`` and no other brace tokens (a stray
    ``{x}`` would KeyError at compose time -- fail at load instead)."""
    token = "{%s}" % placeholder
    if value.count("{") != 1 or value.count("}") != 1 or token not in value:
        raise VisualStyleValidationError(
            f"visual style {path}: {label} must contain the placeholder "
            f"{token} exactly once and no other brace tokens; got {value!r}")


def _validate_row(raw: dict, path: Path) -> VisualStyle:
    if not isinstance(raw, dict):
        raise VisualStyleValidationError(
            f"visual style {path}: top level must be an object, "
            f"got {type(raw).__name__}")
    declared = raw.get("schema_version")
    if declared == "v1":
        raise VisualStyleValidationError(
            f"visual style {path}: schema_version 'v1' is retired -- "
            f"upgrade to v2 (total-coverage slice 2026-07-05: add the "
            f"{len(_V2_STR_FIELDS)} look/subject str fields + "
            f"{len(_V2_DICT_FIELDS)} dict fields; see "
            f"STAGE3_TOTAL_COVERAGE_SUBPLAN.md section 1a)")
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))
    if unknown:
        raise VisualStyleValidationError(
            f"visual style {path}: unknown key(s) {unknown!r} -- the v2 "
            f"schema is exact (subject/motion/ledger_directives lab fields "
            f"are NOT schema fields; see STAGE3_TOTAL_COVERAGE_SUBPLAN)")
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
    # -- v2 str fields: non-empty (one exemption) --
    for name in _V2_STR_FIELDS:
        if (name not in _V2_EMPTY_LEGAL_STR_FIELDS
                and not str(raw[name]).strip()):
            raise VisualStyleValidationError(
                f"visual style {path}: {name} must be non-empty (only "
                f"scene_instruction_look may be empty -- r4 AG M1)")
    # -- template lints --
    for name, placeholder in _TEMPLATE_STR_FIELDS.items():
        _lint_template(path, name, raw[name], placeholder)
    if not any(w in raw["announcer_subject_ltx_mouth"].lower()
               for w in _MOUTH_VOCAB):
        raise VisualStyleValidationError(
            f"visual style {path}: announcer_subject_ltx_mouth must carry "
            f"mouth-prominence vocabulary ({'/'.join(_MOUTH_VOCAB)}) -- the "
            f"ia2v lip-sync contract drives whatever READS as a mouth")
    # -- dict fields: exact keys, non-empty str values --
    dict_values: "list[tuple[str, str]]" = []
    for name, keys in _V2_DICT_FIELDS.items():
        got = raw[name]
        if set(got) != keys:
            raise VisualStyleValidationError(
                f"visual style {path}: {name} keys must be EXACTLY "
                f"{sorted(keys)} (case-sensitive); got {sorted(got)}")
        for k, v in got.items():
            if not isinstance(v, str) or not v.strip():
                raise VisualStyleValidationError(
                    f"visual style {path}: {name}[{k!r}] must be a "
                    f"non-empty string")
            dict_values.append((f"{name}[{k!r}]", v))
    for k, v in raw["open_subjects"].items():
        _lint_template(path, f"open_subjects[{k!r}]", v,
                       _OPEN_SUBJECTS_PLACEHOLDER)
    for k, v in raw["motion_registers"].items():
        if len(v) > _MOTION_REGISTER_MAX_CHARS:
            raise VisualStyleValidationError(
                f"visual style {path}: motion_registers[{k!r}] is "
                f"{len(v)} chars, over the {_MOTION_REGISTER_MAX_CHARS}-char "
                f"motion budget (BUG-LOCAL-112)")
    terms: "list[str]" = []
    for i, t in enumerate(raw["forbidden_terms"]):
        if not isinstance(t, str) or not t.strip():
            raise VisualStyleValidationError(
                f"visual style {path}: forbidden_terms[{i}] must be a "
                f"non-empty string")
        terms.append(t)
    # Load-time lint (r2 codex S2): case-insensitive substring over the 4
    # tail fields + ALL new string leaves + dict values -- a pack must not
    # violate its own bans anywhere.
    linted: "list[tuple[str, str]]" = [
        (fld, str(raw[fld])) for fld in _LINTED_TAIL_FIELDS]
    linted.extend((name, str(raw[name])) for name in _V2_STR_FIELDS)
    linted.extend(dict_values)
    for term in terms:
        needle = term.lower()
        for label, text in linted:
            if needle in text.lower():
                raise VisualStyleValidationError(
                    f"visual style {path}: forbidden term {term!r} appears "
                    f"in its own {label} -- a pack must not violate its own "
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
        portrait_look=raw["portrait_look"],
        portrait_look_talking=raw["portrait_look_talking"],
        portrait_instruction_look=raw["portrait_instruction_look"],
        scene_instruction_look=raw["scene_instruction_look"],
        announcer_subject_face=raw["announcer_subject_face"],
        announcer_subject_ltx_mouth=raw["announcer_subject_ltx_mouth"],
        announcer_subject_object=raw["announcer_subject_object"],
        radio_object_look=raw["radio_object_look"],
        plate_look=raw["plate_look"],
        non_character_emblem_fallback=raw["non_character_emblem_fallback"],
        still_word_title_mood_style=raw["still_word_title_mood_style"],
        open_subjects=MappingProxyType(dict(raw["open_subjects"])),
        motion_registers=MappingProxyType(dict(raw["motion_registers"])),
        still_word_typography=MappingProxyType(
            dict(raw["still_word_typography"])),
        still_word_backdrop=MappingProxyType(
            dict(raw["still_word_backdrop"])),
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
