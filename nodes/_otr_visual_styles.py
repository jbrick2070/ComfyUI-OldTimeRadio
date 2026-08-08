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
contract). `motion_registers` values are budgeted at 240 chars at load
(BUG-LOCAL-112). Authored style vocabulary is never classified or banned.

The byte-identity contract: sci_fi_radio.json's fields are byte-identical to
the extraction-fixture constants (tails in _otr_story_brief_helpers.py; look/
subject fixtures in otr_meta_brief_image_prompt.py + the open-subject
templates in _otr_story_brief_helpers.py; motion registers in
render_driver.py; still_word maps in otr_meta_brief_image_prompt.py) --
pinned by tests. Production code reads the PACK; the constants survive only
as those fixtures + the designated legacy no-style lanes.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, Field, field_validator

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
        {"announcer", "music_open", "music_close", "music_inter"}
    ),
    "still_word_typography": frozenset(
        {"noir", "sci-fi", "western", "pulp", "default"}
    ),
    "still_word_backdrop": frozenset(
        {"noir", "sci-fi", "western", "pulp", "default"}
    ),
}

_REQUIRED_FIELDS: "dict[str, type]" = {
    **_V1_FIELDS,
    **{name: str for name in _V2_STR_FIELDS},
    **{name: dict for name in _V2_DICT_FIELDS},
}

#: Template fields: field -> the exactly-once placeholder name.
_TEMPLATE_STR_FIELDS: "dict[str, str]" = {
    "announcer_subject_ltx_mouth": "form",
    "non_character_emblem_fallback": "base",
}
#: open_subjects values are {form} templates too (each value exactly once).
_OPEN_SUBJECTS_PLACEHOLDER = "form"

#: Mouth-prominence vocabulary the ltx_mouth subject must carry (ia2v
#: lip-sync contract: LTX drives whatever READS as a mouth).
_MOUTH_VOCAB: "tuple[str, ...]" = ("mouth", "lips")

#: BUG-LOCAL-112: motion prompts are budgeted; enforced at LOAD on pack
#: motion_registers values (mirrors render_driver._LTX_MOTION_PROMPT_MAX).
_MOTION_REGISTER_MAX_CHARS: int = 240


class VisualStyleError(Exception):
    """Base: any fail-loud visual-style problem."""


class UnknownVisualStyleError(VisualStyleError):
    """Raised when an unknown style_id is requested."""


class VisualStyleValidationError(VisualStyleError):
    """Raised when a pack fails v2 schema or template lints."""


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


# ---------------------------------------------------------------------------
# Visual Style Card & Card-to-Pack Composer
# ---------------------------------------------------------------------------


class VisualStyleCardModel(BaseModel):
    """Pydantic model for the 9-field Visual Style Card emitted by LLM reflection."""

    medium_short: str = Field(
        max_length=40, description="1-3 word compact art medium token"
    )
    medium: str = Field(
        description="Primary visual art medium descriptive phrase"
    )
    radio_material: str = Field(
        description="Physical material and craftsmanship of radio console"
    )
    character_art_style: str = Field(
        description="Color-neutral portrait rendering style"
    )
    texture: str = Field(description="Tactile surface texture and weathering")
    linework: str = Field(description="Line quality and edge definition")
    lighting_character: str = Field(description="Atmospheric lighting quality")
    typography_voice: str = Field(
        description="Letterform character for title cards"
    )
    motion_temperament: str = Field(
        max_length=60, description="Surface motion descriptor"
    )

    @field_validator(
        "medium_short",
        "medium",
        "radio_material",
        "character_art_style",
        "texture",
        "linework",
        "lighting_character",
        "typography_voice",
        "motion_temperament",
    )
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("card field must be a non-blank string")
        return value.strip()


def _sanitize_card_token(text: str) -> str:
    """Strip any brace characters from card field strings to prevent template corruption."""
    return text.replace("{", "").replace("}", "").strip()


def _fit_motion_slot(text: str, max_len: int = 60) -> str:
    """Truncate motion temperament on word boundary to fit camera template budget."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    idx = cut.rfind(" ")
    return (cut[:idx] if idx >= 15 else cut).rstrip(" .,;")


def compose_pack_from_card(card: VisualStyleCardModel | dict) -> dict:
    """Compose a 23-field v2 VisualStyle pack dictionary from a VisualStyleCard.

    Card inputs are sanitized to ensure placeholders and brace contracts remain exact.
    Returns a plain Python dict ready for validate_pack validation.
    """
    if isinstance(card, dict):
        card_obj = VisualStyleCardModel.model_validate(card)
    else:
        card_obj = card

    m_short = _sanitize_card_token(card_obj.medium_short)
    m_long = _sanitize_card_token(card_obj.medium)
    r_mat = _sanitize_card_token(card_obj.radio_material)
    c_art = _sanitize_card_token(card_obj.character_art_style)
    tex = _sanitize_card_token(card_obj.texture)
    line = _sanitize_card_token(card_obj.linework)
    light = _sanitize_card_token(card_obj.lighting_character)
    typo = _sanitize_card_token(card_obj.typography_voice)
    m_temp = _fit_motion_slot(_sanitize_card_token(card_obj.motion_temperament), 60)

    med_for_gradient = m_short if m_short else m_long

    return {
        "style_id": "visual_storybased",
        "label": "Visual Story-Based (Dynamic)",
        "positive_tail": f"{m_long}, {tex}, {line}",
        "image_grade_tail": f"{light}, {tex} texture grade",
        "broadcast_tail": f"{m_long} texture, broadcast-distressed cinematic aesthetic, centered composition",
        "allow_radio_tails": True,
        "era_tail": "",
        "schema_version": "v2",
        "portrait_look": f"{c_art}, {m_long} character rendering, {line}, {light}",
        "portrait_look_talking": f"{c_art}, {line}, bright even key light",
        "portrait_instruction_look": f"rendered in {m_long} style, tactile and era-consistent",
        "scene_instruction_look": f"{m_long} scene with {line} and a {tex} background",
        "announcer_subject_face": f"its tuning dial crafted as an expressive face -- two dial-eyes and a speaker-grille mouth, an anthropomorphic {r_mat} radio console hosting the broadcast",
        "announcer_subject_ltx_mouth": f"{{form}} as a living {r_mat} appliance face: its speaker grille bends into a huge expressive mouth, big dark painted lips curling open mid-speech, and its two tuning dials are its eyes -- a face-forward anthropomorphic radio built from {r_mat}, filling the frame",
        "announcer_subject_object": f"presented as a tabletop radio constructed from {r_mat}, its {m_long} cabinet, woven grille, and a glowing tuning dial",
        "radio_object_look": f"{light}, gentle {tex} shadows",
        "plate_look": f"{m_long} background set, {tex} scenery",
        "non_character_emblem_fallback": f"a single emblematic object rendered in {m_long} style representing {{base}}",
        "still_word_title_mood_style": f"abstract {m_long} mood composition, {tex} textures in a symbolic non-literal arrangement, no lettering",
        "open_subjects": {
            "synthetic": f"{{form}} warming up on a table in a {m_long} set, {tex} dials and glowing vacuum tubes",
            "announcer": f"{{form}} in a broadcast booth rendered in {m_long} style, {light}",
            "default": f"{{form}} glowing warmly, {tex} dials and vacuum tubes",
        },
        "motion_registers": {
            "announcer": f"Continuous shot, same console throughout. Dial needle sweeps in gentle arcs. {m_temp}. Slow handheld dolly forward.",
            "music_open": f"Continuous shot, same console throughout. Dial whip-pans across frequencies. {m_temp}. Dynamic dolly push forward.",
            "music_close": f"Continuous shot, same console throughout. Dial settles. {m_temp}. Slow dolly pull back.",
            "music_inter": f"Continuous shot, same console throughout. Dial steady. {m_temp}. Slow orbit around the speaker.",
        },
        "still_word_typography": {
            "noir": f"{typo} lettering, deep shadow",
            "sci-fi": f"{typo} lettering, clean angular lines",
            "western": f"{typo} lettering, sturdy weather-beaten edges",
            "pulp": f"{typo} lettering, dramatic layered outline",
            "default": f"{typo} lettering, clean even strokes",
        },
        "still_word_backdrop": {
            "noir": f"deep indigo {med_for_gradient}-toned gradient, dim raking side light",
            "sci-fi": f"cool slate-blue {med_for_gradient}-toned gradient, crisp cold studio light",
            "western": f"warm sand-amber {med_for_gradient}-toned gradient, low golden raking light",
            "pulp": f"rich crimson {med_for_gradient}-toned gradient, bold dramatic side light",
            "default": f"soft dark {med_for_gradient}-toned gradient, gentle even studio light",
        },
    }


# ---------------------------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------------------------


def _lint_template(label: str, value: str, placeholder: str) -> None:
    """EXACTLY one ``{placeholder}`` and no other brace tokens."""
    token = "{%s}" % placeholder
    if value.count("{") != 1 or value.count("}") != 1 or token not in value:
        raise VisualStyleValidationError(
            f"{label} must contain the placeholder "
            f"{token} exactly once and no other brace tokens; got {value!r}"
        )


def validate_pack(raw: dict, expected_style_id: str | None = None) -> VisualStyle:
    """Path-independent validator for a VisualStyle pack mapping.

    Validates structure, exact schema keys, types, string non-emptiness,
    placeholders, mouth vocabulary, dict keys, and motion register char limits.
    If expected_style_id is provided, verifies raw["style_id"] == expected_style_id.
    """
    if not isinstance(raw, dict):
        raise VisualStyleValidationError(
            f"visual style pack: top level must be an object, "
            f"got {type(raw).__name__}"
        )
    declared = raw.get("schema_version")
    if declared == "v1":
        raise VisualStyleValidationError(
            "visual style pack: schema_version 'v1' is retired -- upgrade to v2"
        )
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))
    if unknown:
        raise VisualStyleValidationError(
            f"visual style pack: unknown key(s) {unknown!r}"
        )
    missing = sorted(set(_REQUIRED_FIELDS) - set(raw))
    if missing:
        raise VisualStyleValidationError(
            f"visual style pack: missing required key(s) {missing!r}"
        )
    for key, typ in _REQUIRED_FIELDS.items():
        val = raw[key]
        if typ is bool:
            if not isinstance(val, bool):
                raise VisualStyleValidationError(
                    f"visual style pack: {key} must be a bool, "
                    f"got {type(val).__name__}"
                )
        elif not isinstance(val, typ) or isinstance(val, bool):
            raise VisualStyleValidationError(
                f"visual style pack: {key} must be {typ.__name__}, "
                f"got {type(val).__name__}"
            )
    style_id = raw["style_id"]
    if not _STYLE_ID_RE.match(style_id):
        raise VisualStyleValidationError(
            f"visual style pack: style_id {style_id!r} must match "
            f"{_STYLE_ID_RE.pattern}"
        )
    if expected_style_id is not None and style_id != expected_style_id:
        raise VisualStyleValidationError(
            f"visual style pack: style_id {style_id!r} does not match "
            f"expected {expected_style_id!r}"
        )
    if raw["schema_version"] not in KNOWN_STYLE_SCHEMA_VERSIONS:
        raise VisualStyleValidationError(
            f"visual style pack: unknown schema_version "
            f"{raw['schema_version']!r}; known: "
            f"{sorted(KNOWN_STYLE_SCHEMA_VERSIONS)}"
        )
    # -- v2 str fields: non-empty (one exemption) --
    for name in _V2_STR_FIELDS:
        if (
            name not in _V2_EMPTY_LEGAL_STR_FIELDS
            and not str(raw[name]).strip()
        ):
            raise VisualStyleValidationError(
                f"visual style pack: {name} must be non-empty"
            )
    # -- template lints --
    for name, placeholder in _TEMPLATE_STR_FIELDS.items():
        _lint_template(f"field {name}", raw[name], placeholder)
    if not any(
        w in raw["announcer_subject_ltx_mouth"].lower() for w in _MOUTH_VOCAB
    ):
        raise VisualStyleValidationError(
            f"visual style pack: announcer_subject_ltx_mouth must carry "
            f"mouth-prominence vocabulary ({'/'.join(_MOUTH_VOCAB)})"
        )
    # -- dict fields: exact keys, non-empty str values --
    for name, keys in _V2_DICT_FIELDS.items():
        got = raw[name]
        if set(got) != keys:
            raise VisualStyleValidationError(
                f"visual style pack: {name} keys must be EXACTLY "
                f"{sorted(keys)} (case-sensitive); got {sorted(got)}"
            )
        for k, v in got.items():
            if not isinstance(v, str) or not v.strip():
                raise VisualStyleValidationError(
                    f"visual style pack: {name}[{k!r}] must be a "
                    f"non-empty string"
                )
    for k, v in raw["open_subjects"].items():
        _lint_template(
            f"open_subjects[{k!r}]", v, _OPEN_SUBJECTS_PLACEHOLDER
        )
    for k, v in raw["motion_registers"].items():
        if len(v) > _MOTION_REGISTER_MAX_CHARS:
            raise VisualStyleValidationError(
                f"visual style pack: motion_registers[{k!r}] is "
                f"{len(v)} chars, over the {_MOTION_REGISTER_MAX_CHARS}-char "
                f"motion budget (BUG-LOCAL-112)"
            )
    return VisualStyle(
        style_id=style_id,
        label=raw["label"],
        positive_tail=raw["positive_tail"],
        image_grade_tail=raw["image_grade_tail"],
        broadcast_tail=raw["broadcast_tail"],
        allow_radio_tails=raw["allow_radio_tails"],
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
            dict(raw["still_word_typography"])
        ),
        still_word_backdrop=MappingProxyType(
            dict(raw["still_word_backdrop"])
        ),
    )


def _validate_row(raw: dict, path: Path) -> VisualStyle:
    try:
        style_obj = validate_pack(raw)
    except VisualStyleValidationError as exc:
        raise VisualStyleValidationError(f"visual style {path}: {exc}") from exc
    if style_obj.style_id != path.stem:
        raise VisualStyleValidationError(
            f"visual style {path}: header style_id {style_obj.style_id!r} does not "
            f"match the filename {path.stem!r} -- the path IS the coordinate"
        )
    return style_obj


def _load_all() -> "dict[str, VisualStyle]":
    if not _VISUAL_STYLES_ROOT.is_dir():
        raise VisualStyleValidationError(
            f"visual styles directory missing: {_VISUAL_STYLES_ROOT}"
        )
    styles: "dict[str, VisualStyle]" = {}
    entries = sorted(_VISUAL_STYLES_ROOT.iterdir(), key=lambda p: p.name)
    for entry in entries:
        if entry.is_dir():
            raise VisualStyleValidationError(
                f"unexpected subdirectory in visual_styles/: {entry} -- "
                f"packs are flat <style_id>.json files"
            )
        if entry.suffix != ".json":
            raise VisualStyleValidationError(
                f"unexpected non-pack file in visual_styles/: {entry}"
            )
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
            f"no visual style packs found under {_VISUAL_STYLES_ROOT}"
        )
    if DEFAULT_STYLE_ID not in styles:
        raise VisualStyleValidationError(
            f"the default style {DEFAULT_STYLE_ID!r} is missing from "
            f"{_VISUAL_STYLES_ROOT} -- the production lane must exist"
        )
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
            f"{sorted(styles)}"
        ) from None


def get_visual_style(meta: object) -> VisualStyle:
    """The ONE meta-driven resolver composer entry points call (once per
    entry; pass the result down -- helpers never re-resolve).

    meta["visual_style"] absent/empty => the production default
    (sci_fi_radio) -- every pre-Stage-3 ledger and every default run.
    A PRESENT-but-unknown id fails LOUD (UnknownVisualStyleError).
    visual_storybased => reads embedded_visual_style_pack and validates it."""
    m = meta if isinstance(meta, dict) else {}
    raw = m.get("visual_style")
    style_id = str(raw).strip() if raw is not None else ""
    if not style_id:
        style_id = DEFAULT_STYLE_ID

    if style_id == "visual_storybased":
        receipt = m.get("visual_style_receipt")
        if isinstance(receipt, dict) and receipt.get("status") == "pending":
            raise VisualStyleError(
                "visual_storybased pack generation is pending"
            )

        embedded = m.get("embedded_visual_style_pack")
        if not isinstance(embedded, dict):
            raise VisualStyleError(
                "visual_storybased requires embedded_visual_style_pack dict in meta"
            )

        style_obj = validate_pack(embedded, expected_style_id="visual_storybased")

        if isinstance(receipt, dict) and "sha256" in receipt:
            expected_sha = receipt["sha256"]
            canonical_bytes = json.dumps(
                embedded, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            derived_sha = hashlib.sha256(canonical_bytes).hexdigest()
            if derived_sha != expected_sha:
                raise VisualStyleError(
                    f"embedded visual_storybased pack sha256 mismatch: "
                    f"got {derived_sha}, expected {expected_sha}"
                )
        return style_obj

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
    "VisualStyleCardModel",
    "VisualStyleError",
    "VisualStyleValidationError",
    "compose_pack_from_card",
    "get_visual_style",
    "list_style_ids",
    "resolve_visual_style",
    "validate_pack",
]
