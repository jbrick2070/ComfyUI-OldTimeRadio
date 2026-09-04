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

ONE STYLE AUTHORITY (2026-08-17, PBUG-20260817-01): a pack also owns its own
NEGATIVE (`negative_tail`) and this module owns the single derivation of the
style TOKEN (`compact_style_cue`) and the single way it is applied
(`prefix_style_cue`). Both used to live in `render_driver` serving video
prompts only, while the negative lived hardcoded in `z_image_turbo` where it
was style-blind and vetoed four of the nine packs on every mint. `negative_tail`
is KNOWN-but-OPTIONAL rather than required so that FROZEN embedded packs, whose
sha256 receipt forbids injecting a default, still validate.

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

#: KNOWN-but-OPTIONAL fields: accepted when present, never demanded.
#:
#: `negative_tail` (2026-08-17, PBUG-20260817-01) is the pack's own NEGATIVE
#: conditioning -- the style half of what `z_image_turbo` used to hardcode
#: engine-side, where it was style-blind and vetoed four illustration-family
#: packs on every mint.
#:
#: It is OPTIONAL rather than required for one hard reason: `get_visual_style`
#: re-validates `embedded_visual_style_pack` out of FROZEN ledgers and then
#: sha256s the exact canonical bytes against the stored receipt. A REQUIRED key
#: would fail every pre-existing `visual_storybased` ledger on the missing-key
#: path, and injecting a default to compensate would change those bytes and
#: trip the sha instead -- the receipt structurally forbids back-compat
#: defaults. Optional protects frozen history; a test pins that all nine
#: SHIPPED packs carry a non-empty value, which protects the present.
_OPTIONAL_FIELDS: "dict[str, type]" = {
    "negative_tail": str,
}

#: Every key a pack may legally carry. The unknown-key guard reads THIS;
#: the missing-key guard still reads `_REQUIRED_FIELDS`. Splitting the two is
#: what makes a known-but-optional field expressible at all.
_KNOWN_FIELDS: "dict[str, type]" = {**_REQUIRED_FIELDS, **_OPTIONAL_FIELDS}

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
    # -- v2 NEGATIVE surface (optional; see _OPTIONAL_FIELDS) --
    # Empty means "this pack expresses no style negative"; the engine then
    # falls back to its hygiene default rather than to nothing.
    negative_tail: str = ""


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
        # Deliberately EMPTY, and stated rather than omitted. The card
        # describes an ARBITRARY medium, so the anti-style terms would have to
        # be inferred from the positive text -- a prompt-scanning heuristic,
        # which is exactly what this build is forbidden to add. An empty value
        # routes the mint to the engine's hygiene fallback (anti-artifact only,
        # no style opinion), which is the correct neutral answer here.
        "negative_tail": "",
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
    unknown = sorted(set(raw) - set(_KNOWN_FIELDS))
    if unknown:
        raise VisualStyleValidationError(
            f"visual style pack: unknown key(s) {unknown!r}"
        )
    missing = sorted(set(_REQUIRED_FIELDS) - set(raw))
    if missing:
        raise VisualStyleValidationError(
            f"visual style pack: missing required key(s) {missing!r}"
        )
    # Type-check every key PRESENT, not merely every required one -- an
    # optional key that skipped this would let an int or list reach the
    # dataclass and fail much later inside a string join.
    for key, typ in ((k, t) for k, t in _KNOWN_FIELDS.items() if k in raw):
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
        # Optional: absent on every pre-2026-08-17 frozen embedded pack.
        negative_tail=raw.get("negative_tail", ""),
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


#: The one place a style TOKEN is derived, and the one place it is applied.
#:
#: ONE STYLE AUTHORITY (2026-08-17): this pair used to live only in
#: `render_driver` as `_compact_style_talking_cue` / `_prefix_video_style_cue`,
#: serving video prompts. Stills reached the engines with their style only
#: TAIL-appended by `finish_visual_prompt`, i.e. in the weakest position on
#: models that weight early tokens. Both families now share this derivation --
#: a second, disagreeing definition of "the style word" is the failure mode
#: this consolidation exists to prevent.


def compact_style_cue(vstyle) -> str:
    """The blunt 2-4 word style token for a pack, or "" for the default.

    `sci_fi_radio` returns "" on purpose: it IS the house look, its tails are
    byte-pinned to the shared constants, and prefixing it would churn every
    default-lane golden for no visual gain.
    """
    if str(getattr(vstyle, "style_id", "") or "") == DEFAULT_STYLE_ID:
        return ""
    raw = str(getattr(vstyle, "positive_tail", "") or "").strip()
    if not raw:
        raw = str(getattr(vstyle, "portrait_look_talking", "") or "").strip()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", raw)
    lowered = [word.lower() for word in words]
    if lowered[:4] == ["recursive", "fractal", "light", "field"]:
        return " ".join(words[:4]).strip()
    limit = 2
    for i, word in enumerate(words[:4]):
        if word.lower() == "style":
            limit = i + 1
            break
    return " ".join(words[:limit]).strip()


#: How many WORDS of a pack's own style vocabulary may ride at the END of a
#: prompt, beyond the blunt front cue. Sized against the nine shipped packs
#: (2026-09-03): their trailing remainders run 4 to 22 words, median 9, so ten
#: takes the whole remainder for five packs and trims the four long ones. In
#: tokens that is roughly 13 against `GHOST_AUTHOR_TOKEN_TARGET` of 69 on a
#: prompt measuring about 32 today -- deliberate headroom, because the fitter
#: still has to fit the banana route after this.
TRAILING_STYLE_MAX_WORDS = 10


def trailing_style_cue(vstyle, *, max_words: int = TRAILING_STYLE_MAX_WORDS) -> str:
    """The pack's own style vocabulary for the END of a prompt, or "".

    The front cue (:func:`compact_style_cue`) is blunt on purpose -- two to four
    words, stopping at "style" -- so a pack that authored
    "anime style, expressive linework, cel-shaded color" contributes only
    "anime style" today and the rest of what it authored is never asked for.
    This returns that remainder, bounded.

    **THE DEFAULT STYLE RETURNS "" EXPLICITLY, AND THAT GUARD IS THE POINT.**
    `sci_fi_radio` IS the house look; its tails are byte-pinned to the shared
    constants and it deliberately emits no front cue. Deriving the remainder by
    subtracting the front cue would therefore hand back its ENTIRE
    `positive_tail` ("cinematic, 35mm film look, subtle film grain, volumetric
    lighting") and emit it at the back -- style text on the one style that must
    have none, churning every default-lane golden. Caught in review before it
    shipped; the guard is checked FIRST, never inferred from an empty cue.

    **WHOLE COMMA UNITS ONLY, never a word slice.** Trimming "cel-shaded color"
    to "cel-shaded" changes what is being asked for, which is the same rule
    `GHOST_V3_DROP_ORDER` already follows for whole slots. A unit that does not
    fit is dropped entire, and everything after it with it, so the result always
    reads as authored prose rather than a truncated fragment.
    """
    if str(getattr(vstyle, "style_id", "") or "") == DEFAULT_STYLE_ID:
        return ""
    raw = str(getattr(vstyle, "positive_tail", "") or "").strip()
    if not raw:
        return ""
    # compact_style_cue builds its words FROM positive_tail, so the front cue is
    # a word-prefix of it and dropping that many words is exact rather than a
    # string search that could match again later in the tail.
    front_words = len(compact_style_cue(vstyle).split())
    rest = " ".join(raw.split()[front_words:]).strip().strip(",").strip()
    if not rest:
        return ""
    kept, used = [], 0
    for unit in (u.strip().strip(",").strip() for u in rest.split(",")):
        if not unit:
            continue
        length = len(unit.split())
        if used + length > max(int(max_words), 0):
            break
        kept.append(unit)
        used += length
    return ", ".join(kept)


#: How many WORDS of a pack's motion register may ride in a bookend prompt.
#: The registers were authored for the LTX lane and run 20-29 words across the
#: nine packs, which is 26-37 CLIP tokens -- enough on its own to push an
#: AnimateDiff prompt past the ~75-token point where that model stops moving
#: altogether. Fourteen buys the two sentences that carry the movement.
#:
#: FOURTEEN, NOT TWELVE (2026-09-03). At twelve, the over-budget branch below
#: falls back to `text = kinetic`, which discards the CAMERA CLAUSE -- the one
#: part of the register that makes the shot travel rather than shimmer in
#: place. Measured across all 32 registers: exactly 2 exceeded twelve, both by
#: a single word, and both are packs the operator names as his interest --
#: `video_art/music_open` ("...red blue green halos, measured push forward")
#: and `paper_origami/announcer` ("...in gentle arcs, slow steady dolly
#: forward"). Losing the camera move to save one word defeats the entire
#: purpose of a function that exists to restore movement, and 14 words is
#: still comfortably the "one subject, one action, one speed" shape the wan
#: family's directive asks for. The other 30 registers are byte-identical at
#: either value.
MOTION_REGISTER_MAX_WORDS = 14


def bounded_motion_register(register, *, max_words: int = MOTION_REGISTER_MAX_WORDS) -> str:
    """The MOVEMENT out of a pack's motion register, compacted, or "".

    Every register is authored as five sentences in the same shape: a static
    framing constraint, three clauses of subject motion, and a CAMERA move --
    e.g. storybook_engraving's announcer reads "Continuous shot, same console
    throughout. Etched dial needle glides. Hand-tinted highlights shimmer.
    Paper grain breathes softly. Slow illustrated dolly forward."

    Two deliberate choices, both from the operator's 2026-09-03 report that the
    announcer and music beats "had basically no movement":

    * **The leading "Continuous shot" constraint is DROPPED.** It is a shot rule
      rather than a movement, and telling a model to hold still is the opposite
      of the defect this exists to fix.
    * **The CAMERA sentence is kept even when the middle ones are not.** It is
      the last sentence in every pack and the one an image-to-video model
      actually acts on; "slow dolly push forward" moves a shot that "paper
      texture shimmers" only decorates.

    Returns "" for an absent, empty or unusable register -- a bookend then keeps
    the generic world-motion clause it has today rather than losing its motion
    slot entirely.
    """
    parts = [p.strip().rstrip(".").strip()
             for p in " ".join(str(register or "").split()).split(".")]
    parts = [p for p in parts if p]

    # THE FRAMING SENTENCE CARRIES THE ANTECEDENT, so dropping it silently
    # orphans the noun that follows. Every pack authors "Continuous shot, same
    # console throughout. Dial needle sweeps in crisp arcs..." -- the console
    # establishes what the dial BELONGS to. Strip the sentence and the clause
    # reads "dial needle sweeps", which an image model may attach to a
    # telephone, a clock face or a gauge. Operator, 2026-09-03: *"'dial' -- you'd
    # have to say radio system dial or such."*
    #
    # So the static camera instruction still goes (telling a model to hold still
    # is the defect this exists to fix), and the SUBJECT it named is re-anchored
    # onto the kinetic clause instead of being thrown away with it.
    # THE FRAMING SENTENCE IS NOT ALWAYS FIRST. Seven packs open with
    # "Continuous shot, same console throughout."; `recur_frac` and `video_art`
    # prefix a MOTTO ("Recursive fractal light field.", "Video-art feedback.").
    # An index-0 check therefore stripped the motto, kept the motto as the
    # kinetic clause, and dropped the dial motion sitting behind the framing
    # line -- so Ghost was sent "recursive fractal light field, slow recursive
    # push forward": the front cue restated, with no movement in it. On exactly
    # the two packs the operator names as his interest.
    #
    # Search for the framing sentence at ANY index and drop everything up to and
    # including it. On the seven standard packs the index is 0 and the result is
    # byte-identical.
    subject = ""
    framing_idx = next(
        (i for i, part in enumerate(parts)
         if part.lower().startswith("continuous shot")), None)
    if framing_idx is not None:
        head = parts[framing_idx]
        for noun in ("console", "radio set", "radio"):
            if noun in head.lower():
                subject = "radio console" if noun == "console" else noun
                break
        parts = parts[framing_idx + 1:]
    if not parts:
        return ""
    # WHOLE WORDS. A substring test here got "dial settles" wrong on its first
    # run -- "set" is inside "set-tles" -- which is the identical mistake that
    # made `"close" in "closing"` fail in the driver's register selector, made
    # minutes after fixing that one. Membership by substring is how a token test
    # quietly stops meaning what it says.
    _already_anchored = re.search(r"\b(radio|console|set)\b", parts[0].lower())
    if subject and not _already_anchored:
        parts[0] = "%s %s" % (subject, parts[0][:1].lower() + parts[0][1:])
    def _lead_lower(phrase):
        # Sentence case becomes clause case: these were authored as standalone
        # sentences and are being joined into one comma-separated clause, where
        # a mid-phrase capital reads as a proper noun to a text encoder.
        return phrase[:1].lower() + phrase[1:] if phrase else ""

    kinetic = _lead_lower(parts[0])
    camera = _lead_lower(parts[-1]) if len(parts) > 1 else ""
    text = ", ".join([kinetic] + ([camera] if camera and camera != kinetic else []))
    if len(text.split()) > max(int(max_words), 1):
        text = kinetic
    if len(text.split()) > max(int(max_words), 1):
        return ""
    return text


def prefix_style_cue(vstyle, prompt: str) -> str:
    """Front-anchor the pack's style token on ``prompt``. ADDITIVE ONLY.

    Never rewrites, strips, reorders or rejects -- it only ever prepends, so it
    stays on the permitted side of the authored-prompt law
    (`otr_shot_lock.py`, "no Python vocabulary or token-overlap judge may
    replace an authored non-empty visual prompt", whose own next line is an
    unconditional additive prepend).

    IDEMPOTENT, and deliberately POSITIONAL rather than membership-based: the
    token is usually already present at the TAIL, so an "is it in the string"
    test would find it and do nothing on exactly the prompts that need it.
    """
    prompt = str(prompt or "").strip()
    cue = compact_style_cue(vstyle)
    if not prompt or not cue:
        return prompt
    low_prompt = prompt.lower()
    low_cue = cue.lower().rstrip(".")
    if low_prompt.startswith(low_cue):
        return prompt
    if low_cue.endswith(" style"):
        base_cue = low_cue[: -len(" style")].strip()
        if base_cue and low_prompt.startswith(base_cue):
            # Strip whatever punctuation followed the bare token. Stripping "."
            # alone left `"Cartoon, a man"` -> `"cartoon style. , a man"`.
            rest = prompt[len(base_cue):].lstrip(".,:; ")
            return ("%s. %s" % (cue.rstrip("."), rest)
                    if rest else "%s." % cue.rstrip("."))
    return "%s. %s" % (cue.rstrip("."), prompt)


#: The pack surfaces a still prompt is actually built from. A negative phrase
#: that one of these asks for is a self-veto, by definition.
_POSITIVE_SURFACES = (
    "positive_tail", "image_grade_tail", "broadcast_tail", "era_tail",
    "portrait_look", "portrait_look_talking", "portrait_instruction_look",
    "scene_instruction_look", "plate_look",
    "announcer_subject_face", "announcer_subject_ltx_mouth",
    "announcer_subject_object", "radio_object_look",
    "still_word_title_mood_style", "non_character_emblem_fallback",
)
#: DICT-valued surfaces whose VALUES also land in real prompt text -- the
#: still-word card joins `still_word_typography[genre]` and
#: `still_word_backdrop[genre]` straight into its prompt, and `open_subjects`
#: feeds the opens. Omitting them left a hole: no shipped pack collides on them
#: today, so nothing was broken, but a future edit to a pack's typography or
#: backdrop wording could silently reintroduce a self-veto that neither the
#: runtime resolution nor the traceroute would catch.
_POSITIVE_DICT_SURFACES = (
    "open_subjects", "still_word_typography", "still_word_backdrop",
)
#: Anti-ARTIFACT, never anti-STYLE. Kept even if the word appears in a
#: positive surface -- "text" in a title-card description is not a request for
#: a watermark.
_ARTIFACT_PHRASES = frozenset({"text", "watermark"})


def effective_negative(style) -> str:
    """The pack's negative with any phrase its OWN positive asks for removed.

    Operator ruling 2026-08-17: *"we can't have negative prompts conflicting
    with any visual style."* PBUG-20260817-01 was one instance of that -- a
    style-blind negative vetoing "cartoon, illustration" on a cartoon episode
    -- and moving the negative into the packs fixed the cross-pack case but
    not the self-veto case. `sci_fi_radio` still shipped "cartoon, illustration"
    while its own `announcer_subject_ltx_mouth` asks for "a living cartoon
    appliance face" and its `still_word_title_mood_style` asks for "atmospheric
    period illustration".

    Resolving it HERE rather than by hand-editing that one string is the root
    fix: it holds for all ten identities, for the dynamic pack the composer
    builds at runtime, and for any pack authored later. A pack can no longer
    veto itself, by construction.

    This reads OUR OWN CONFIG against itself and edits OUR OWN negative. It
    never inspects or alters an authored prompt, so it is not the prompt-
    scanning injector the build is forbidden to add, and THE LAW is untouched.

    Comparison is on comma-separated PHRASES, not bare words: "plastic skin"
    must not be dropped just because a portrait asks for realistic skin.
    """
    raw = str(getattr(style, "negative_tail", "") or "")
    if not raw:
        return ""
    parts = [str(getattr(style, f, "") or "") for f in _POSITIVE_SURFACES]
    for f in _POSITIVE_DICT_SURFACES:
        values = getattr(style, f, None) or {}
        try:
            parts.extend(str(v or "") for v in values.values())
        except AttributeError:  # not a mapping -- ignore rather than raise
            pass
    positive = " ".join(parts).lower()
    kept = []
    for chunk in raw.split(","):
        phrase = chunk.strip()
        low = phrase.lower()
        if not phrase:
            continue
        if low not in _ARTIFACT_PHRASES and re.search(
                r"\b%s\b" % re.escape(low), positive):
            continue  # this pack asks for it -- never suppress it
        kept.append(phrase)
    return ", ".join(kept)


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
    "MOTION_REGISTER_MAX_WORDS",
    "TRAILING_STYLE_MAX_WORDS",
    "bounded_motion_register",
    "compose_pack_from_card",
    "get_visual_style",
    "list_style_ids",
    "resolve_visual_style",
    "trailing_style_cue",
    "validate_pack",
]
