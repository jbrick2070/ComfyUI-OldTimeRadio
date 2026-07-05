"""Story-pack loader -- Stage 1 of the multi-modal story schema.

Content lives in JSON packs (nodes/story_packs/<bank>/<model>.json); this module
is pure BEHAVIOR: strict fail-loud loading + validation + seam lookup. Zero content
literals. Stdlib only (no pydantic) -- a small validator needs no dependency and
stays agnostic to the repo's pydantic v1/v2 split.

Law: JSON owns content/config; Python owns validation/routing/execution; no
fallbacks; unknown id = hard error.

STAGE 1 IS DORMANT: this module is loaded + tested but NOT yet consumed by any
production node. A migrated seam (Stage 1b+) uses get_pack_prompt (raises on
missing -- no hidden fallback); get_pack_prompt_or_none is the pre-migration
passthrough (None -> caller keeps its Python constant).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# --- known schema versions (unknown version = hard error, no silent migration) --
KNOWN_SCHEMA_VERSIONS = frozenset({"v2.0"})

# --- the EXACT set of seam keys Stage 1 authors + validates. Not a superset: a
#     reserved-but-unpinned name would let an unauthored seam pass silently.
PRODUCTION_SEAM_ALLOWLIST = frozenset({
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
    "line_composer_system",
    # Lane-enablement chunk 2 (2026-07-05): the Build-4 grouped-exchange
    # static system prompt (use_exchange). Science extraction fixture =
    # _otr_compose_exchange.EXCHANGE_SYSTEM_PROMPT (byte-identity pinned).
    "exchange_system",
    "coda_system",
    "announcer_intro_system",
    "announcer_intro_safe_system",
    "announcer_outro_system",
    # style_pick_inventor_system/_user and style_pick_chooser_system/_user
    # REMOVED (style-engine consolidation, 2026-07-05): the two-pass LLM
    # style picker they seamed is deleted; style is the single
    # deterministic engine call now, with no pack-overridable prompts.
})

_REQUIRED_TOP_LEVEL = (
    "source_bank_id",
    "story_model_id",
    "story_pipeline_id",
    "schema_version",
    "prompt_stages",
)


class StoryPackError(Exception):
    """Base: any fail-loud story-pack problem."""


class StoryPackNotFoundError(StoryPackError):
    """The pack file does not exist / cannot be read."""


class StoryPackParseError(StoryPackError):
    """The pack file is not valid JSON (incl. duplicate keys)."""


class StoryPackValidationError(StoryPackError):
    """The parsed pack violates the contract (missing/unknown/typed wrong)."""


class UnknownSeamError(StoryPackError):
    """A seam key is not in PRODUCTION_SEAM_ALLOWLIST."""


@dataclass(frozen=True)
class StoryPack:
    source_bank_id: str
    story_model_id: str
    story_pipeline_id: str
    schema_version: str
    prompt_stages: "dict[str, str]"
    label: str = ""
    status: str = ""
    examples: list = field(default_factory=list)
    tone_guardrails: list = field(default_factory=list)
    forbidden_plot_patterns: list = field(default_factory=list)
    forbidden_leakage_terms: list = field(default_factory=list)
    source_requirements: list = field(default_factory=list)
    ledger_validation_notes: list = field(default_factory=list)


_KNOWN_FIELDS = frozenset(StoryPack.__dataclass_fields__.keys())

# Load-once cache keyed by resolved absolute path (STRICT production-only
# validation -- load_pack).
_PACK_CACHE: "dict[str, StoryPack]" = {}

# Stage 2: SEPARATE cache for pipeline-seam-aware loads, keyed by
# (resolved path, extra_seams). Sharing _PACK_CACHE would poison the strict
# path: a pack cached under a permissive seam union would later satisfy a
# strict load_pack call without re-validation (kibitz r2 M1).
_PACK_CACHE_WITH_SEAMS: "dict[tuple[str, frozenset], StoryPack]" = {}


def _reject_dup_keys(pairs):
    """json object_pairs_hook: raise on any duplicate key (fires on nested
    objects too, e.g. within prompt_stages). Plain json.loads keeps the last."""
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise StoryPackParseError(f"duplicate key {key!r} in story pack JSON")
        seen[key] = value
    return seen


def _validate(data: dict, origin: str,
              extra_seams: frozenset = frozenset()) -> StoryPack:
    if not isinstance(data, dict):
        raise StoryPackValidationError(f"{origin}: top-level JSON must be an object")

    unknown = sorted(set(data) - _KNOWN_FIELDS)
    if unknown:
        raise StoryPackValidationError(
            f"{origin}: unknown top-level key(s) {unknown}; "
            f"known: {sorted(_KNOWN_FIELDS)}"
        )
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in data]
    if missing:
        raise StoryPackValidationError(f"{origin}: missing required key(s) {missing}")

    for key in ("source_bank_id", "story_model_id", "story_pipeline_id",
                "schema_version"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise StoryPackValidationError(f"{origin}: {key} must be a non-empty string")

    if data["schema_version"] not in KNOWN_SCHEMA_VERSIONS:
        raise StoryPackValidationError(
            f"{origin}: unknown schema_version {data['schema_version']!r}; "
            f"known: {sorted(KNOWN_SCHEMA_VERSIONS)}"
        )

    stages = data["prompt_stages"]
    if not isinstance(stages, dict):
        raise StoryPackValidationError(f"{origin}: prompt_stages must be an object")
    allowed_seams = PRODUCTION_SEAM_ALLOWLIST | extra_seams
    for seam, value in stages.items():
        if seam not in allowed_seams:
            raise UnknownSeamError(
                f"{origin}: unknown seam {seam!r}; "
                f"allowed: {sorted(allowed_seams)}"
            )
        if not isinstance(value, str) or not value.strip():
            raise StoryPackValidationError(
                f"{origin}: seam {seam!r} must be a non-empty string"
            )

    # Optional/inert fields: type-checked only, never consumed in Stage 1.
    for key in ("examples", "tone_guardrails", "forbidden_plot_patterns",
                "forbidden_leakage_terms", "source_requirements",
                "ledger_validation_notes"):
        if key in data and not isinstance(data[key], list):
            raise StoryPackValidationError(f"{origin}: {key} must be a list")
    for key in ("label", "status"):
        if key in data and not isinstance(data[key], str):
            raise StoryPackValidationError(f"{origin}: {key} must be a string")

    return StoryPack(**data)


def _read_pack_data(p: Path) -> dict:
    """Read + JSON-parse a pack file fail-loud (shared by both loaders)."""
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise StoryPackNotFoundError(f"story pack not found: {p}") from exc
    except OSError as exc:
        raise StoryPackNotFoundError(f"cannot read story pack {p}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise StoryPackParseError(f"story pack {p} is not valid UTF-8: {exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_dup_keys)
    except json.JSONDecodeError as exc:
        raise StoryPackParseError(f"malformed JSON in story pack {p}: {exc}") from exc


def load_pack(path) -> StoryPack:
    """Load + validate a story pack, fail-loud, cached by absolute path.

    STRICT: prompt_stages keys must be production-allowlisted. A pack carrying
    pipeline-declared seams (e.g. simple_4) RAISES here by design -- routing
    goes through load_pack_with_seams."""
    p = Path(path)
    key = str(p.resolve())
    cached = _PACK_CACHE.get(key)
    if cached is not None:
        return cached
    pack = _validate(_read_pack_data(p), str(p))
    _PACK_CACHE[key] = pack
    return pack


def load_pack_with_seams(path, extra_seams: frozenset) -> StoryPack:
    """Stage 2 routing loader: validate prompt_stages against
    PRODUCTION_SEAM_ALLOWLIST | extra_seams (the pack's pipeline-declared
    seams). SEPARATE cache keyed (path, extra_seams) so the strict load_pack
    cache is never poisoned by a permissive load. Sole sanctioned production
    caller: nodes/_otr_story_routing.py (test-pinned)."""
    if not isinstance(extra_seams, frozenset):
        raise StoryPackValidationError(
            f"extra_seams must be a frozenset, got {type(extra_seams).__name__}"
        )
    p = Path(path)
    key = (str(p.resolve()), extra_seams)
    cached = _PACK_CACHE_WITH_SEAMS.get(key)
    if cached is not None:
        return cached
    pack = _validate(_read_pack_data(p), str(p), extra_seams=extra_seams)
    _PACK_CACHE_WITH_SEAMS[key] = pack
    return pack


def _check_seam(seam: str) -> None:
    if seam not in PRODUCTION_SEAM_ALLOWLIST:
        raise UnknownSeamError(
            f"unknown seam {seam!r}; allowed: {sorted(PRODUCTION_SEAM_ALLOWLIST)}"
        )


def get_pack_prompt_or_none(pack: StoryPack, seam: str):
    """Pre-migration passthrough: pack value if present+non-empty, else None
    (caller keeps its Python constant). Unknown seam is still a hard error."""
    _check_seam(seam)
    raw = pack.prompt_stages.get(seam, "")
    return raw if raw.strip() else None


def get_pack_prompt(pack: StoryPack, seam: str) -> str:
    """Migrated-seam accessor: RAISE if the seam is absent/empty. No fallback."""
    _check_seam(seam)
    value = pack.prompt_stages.get(seam, "")
    if not value.strip():
        raise StoryPackValidationError(
            f"migrated seam {seam!r} missing/empty in pack "
            f"{pack.source_bank_id}/{pack.story_model_id} (no fallback)"
        )
    return value
