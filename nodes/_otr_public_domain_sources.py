"""Public-domain source-bank helpers.

This module owns the public-domain source-bank helpers and the registered
fetcher. It is still fixture/local-file only: no network calls, no heavy
imports, no workflow changes.
"""
from __future__ import annotations

import html
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

try:
    from . import _otr_source_payload as _osp
except ImportError:  # pragma: no cover -- flat import harnesses
    import _otr_source_payload as _osp  # type: ignore

try:
    from ._otr_structured_call import StructuredCallFailedError, structured_call
except ImportError:  # pragma: no cover -- flat import harnesses
    from _otr_structured_call import StructuredCallFailedError, structured_call  # type: ignore

try:
    from ._otr_content_safety import find_text_hits
except ImportError:  # pragma: no cover -- flat import harnesses
    from _otr_content_safety import find_text_hits  # type: ignore

MANIFEST_SCHEMA_VERSION = "v1"
PROMPT_VERSION = "public_domain_interpreter_v2"
SCHEMA_VERSION = "public_domain_briefs_v1"


_LICENSE_STATUSES = frozenset({"public_domain_us", "cc0", "research_only"})
_ADAPTER_TYPES = frozenset({
    "project_gutenberg_text",
    "standard_ebooks_epub",
    "local_text_fixture",
})
_TOP_KEYS = frozenset({"schema_version", "sources"})
_SOURCE_KEYS = frozenset({
    "source_id",
    "title",
    "author",
    "year",
    "license_status",
    "license_url",
    "source_url",
    "source_label",
    "adapter_type",
    "search_tags",
    "recommended_word_budget",
    "cast_hints",
    "units",
})
_UNIT_KEYS = frozenset({"unit_id", "label", "synopsis", "text_path"})

_WS_RE = re.compile(r"\s+")
_PG_HEADER_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_PG_FOOTER_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)


class PublicDomainSourceError(RuntimeError):
    """Base class for public-domain source-bank skeleton failures."""


class PublicDomainManifestError(PublicDomainSourceError):
    """A manifest file violates the source-bank-v2 manifest contract."""


class PublicDomainSourceRefError(PublicDomainSourceError):
    """A source_ref did not resolve to a manifest unit."""


class PublicDomainInterpreterError(PublicDomainSourceError):
    """Raised when the public-domain source brain cannot produce briefs."""

    def __init__(self, *, attempts: int, reason: str) -> None:
        self.attempts = attempts
        self.reason = reason
        super().__init__(
            f"public-domain interpreter failed after {attempts} attempt(s): "
            f"{reason}"
        )


@dataclass(frozen=True)
class PublicDomainUnit:
    """Resolved manifest source + unit pair."""

    source: dict[str, Any]
    unit: dict[str, Any]

    @property
    def source_ref(self) -> str:
        return f"{self.source['source_id']}:{self.unit['unit_id']}"


def _check_unknown_keys(obj: dict[str, Any], allowed: frozenset[str], origin: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise PublicDomainManifestError(
            f"{origin}: unknown key(s) {unknown}; known: {sorted(allowed)}"
        )


def _require_str(obj: dict[str, Any], key: str, origin: str, *, allow_empty: bool = False) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or (not allow_empty and not val.strip()):
        tail = "a string" if allow_empty else "a non-empty string"
        raise PublicDomainManifestError(f"{origin}: {key} must be {tail}")
    return val


def _require_str_list(obj: dict[str, Any], key: str, origin: str) -> list[str]:
    val = obj.get(key)
    if not isinstance(val, list) or any(not isinstance(v, str) or not v.strip() for v in val):
        raise PublicDomainManifestError(
            f"{origin}: {key} must be a list of non-empty strings"
        )
    return list(val)


def _require_int(obj: dict[str, Any], key: str, origin: str, *, min_value: int, max_value: int | None = None) -> int:
    val = obj.get(key)
    if not isinstance(val, int) or isinstance(val, bool):
        raise PublicDomainManifestError(f"{origin}: {key} must be an integer")
    if val < min_value:
        raise PublicDomainManifestError(
            f"{origin}: {key} must be at least {min_value}"
        )
    if max_value is not None and val > max_value:
        raise PublicDomainManifestError(
            f"{origin}: {key} must be between {min_value} and {max_value}"
        )
    return val


def _validate_unit(raw: Any, origin: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PublicDomainManifestError(f"{origin}: unit must be an object")
    _check_unknown_keys(raw, _UNIT_KEYS, origin)
    unit = {
        "unit_id": _require_str(raw, "unit_id", origin),
        "label": _require_str(raw, "label", origin),
        "synopsis": _require_str(raw, "synopsis", origin),
        "text_path": _require_str(raw, "text_path", origin),
    }
    text_path = unit["text_path"]
    if Path(text_path).is_absolute() or ".." in Path(text_path).parts:
        raise PublicDomainManifestError(
            f"{origin}: text_path must be a relative manifest-local path"
        )
    return unit


def _validate_source(raw: Any, origin: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PublicDomainManifestError(f"{origin}: source must be an object")
    _check_unknown_keys(raw, _SOURCE_KEYS, origin)
    source = {
        "source_id": _require_str(raw, "source_id", origin),
        "title": _require_str(raw, "title", origin),
        "author": _require_str(raw, "author", origin),
        "year": _require_str(raw, "year", origin, allow_empty=True),
        "license_status": _require_str(raw, "license_status", origin),
        "license_url": _require_str(raw, "license_url", origin),
        "source_url": _require_str(raw, "source_url", origin),
        "source_label": _require_str(raw, "source_label", origin),
        "adapter_type": _require_str(raw, "adapter_type", origin),
        "search_tags": _require_str_list(raw, "search_tags", origin),
        "recommended_word_budget": _require_int(
            # No upper bound: this is a RECOMMENDATION the operator may
            # exceed, and the word target is a request rather than a gate.
            # A 320-word ceiling pinned these episodes at roughly two
            # minutes, too short for the scenes worth vendoring. The real
            # limit is structural, not declarative -- the beat topology
            # tops out near 1,520 spoken words -- and a request beyond it
            # simply delivers the closest performable episode.
            raw, "recommended_word_budget", origin, min_value=30
        ),
        "cast_hints": _require_str_list(raw, "cast_hints", origin),
    }
    if source["license_status"] not in _LICENSE_STATUSES:
        raise PublicDomainManifestError(
            f"{origin}: license_status {source['license_status']!r} is not "
            f"in {sorted(_LICENSE_STATUSES)}"
        )
    if source["adapter_type"] not in _ADAPTER_TYPES:
        raise PublicDomainManifestError(
            f"{origin}: adapter_type {source['adapter_type']!r} is not "
            f"in {sorted(_ADAPTER_TYPES)}"
        )
    units_raw = raw.get("units")
    if not isinstance(units_raw, list) or not units_raw:
        raise PublicDomainManifestError(f"{origin}: units must be a non-empty list")
    units: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    for idx, unit_raw in enumerate(units_raw):
        unit = _validate_unit(unit_raw, f"{origin}.units[{idx}]")
        if unit["unit_id"] in seen_units:
            raise PublicDomainManifestError(
                f"{origin}: duplicate unit_id {unit['unit_id']!r}"
            )
        seen_units.add(unit["unit_id"])
        units.append(unit)
    source["units"] = units
    return source


def validate_public_domain_manifest(data: Any, *, origin: str = "manifest") -> dict[str, Any]:
    """Validate and normalize a public-domain source manifest."""
    if not isinstance(data, dict):
        raise PublicDomainManifestError(f"{origin}: top level must be an object")
    _check_unknown_keys(data, _TOP_KEYS, origin)
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PublicDomainManifestError(
            f"{origin}: schema_version must be {MANIFEST_SCHEMA_VERSION!r}"
        )
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PublicDomainManifestError(f"{origin}: sources must be a non-empty list")
    sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for idx, raw_source in enumerate(raw_sources):
        source = _validate_source(raw_source, f"{origin}.sources[{idx}]")
        if source["source_id"] in seen_sources:
            raise PublicDomainManifestError(
                f"{origin}: duplicate source_id {source['source_id']!r}"
            )
        seen_sources.add(source["source_id"])
        sources.append(source)
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "sources": sources}


def load_public_domain_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a manifest from disk."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PublicDomainManifestError(f"manifest not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise PublicDomainManifestError(f"manifest {p}: invalid JSON: {exc}") from exc
    return validate_public_domain_manifest(data, origin=str(p))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_relative_path(raw: str, *, key: str) -> Path:
    val = str(raw or "").strip()
    if not val:
        raise PublicDomainManifestError(
            f"public-domain bank defaults must declare {key}"
        )
    path = Path(val)
    if path.is_absolute():
        return path
    if ".." in path.parts:
        raise PublicDomainManifestError(
            f"public-domain bank default {key} must not contain '..': {val!r}"
        )
    return _repo_root() / path


def resolve_manifest_unit(manifest: dict[str, Any], source_ref: str) -> PublicDomainUnit:
    """Resolve ``source_id:unit_id`` into a manifest source/unit pair."""
    ref = str(source_ref or "").strip()
    if ":" not in ref:
        raise PublicDomainSourceRefError(
            f"source_ref must be 'source_id:unit_id', got {source_ref!r}"
        )
    source_id, unit_id = [part.strip() for part in ref.split(":", 1)]
    if not source_id or not unit_id:
        raise PublicDomainSourceRefError(
            f"source_ref must include source and unit ids, got {source_ref!r}"
        )
    checked = validate_public_domain_manifest(manifest)
    for source in checked["sources"]:
        if source["source_id"] != source_id:
            continue
        for unit in source["units"]:
            if unit["unit_id"] == unit_id:
                return PublicDomainUnit(source=source, unit=unit)
    raise PublicDomainSourceRefError(f"unknown public-domain source_ref {ref!r}")


def select_public_domain_unit_ref(
    manifest: dict[str, Any], *, rng: Any | None = None
) -> str:
    """Pick one ``source_id:unit_id`` at random from a validated manifest.

    THE LIBRARY IS A DECK, NOT A SHELF WITH ONE BOOK ON IT (2026-08-04). This
    bank had no selector at all: a blank source_ref fell straight through to
    ``defaults.source_ref``, so every episode drew the same pinned unit no
    matter how many were vendored. Mirrors the Shakespeare lane's
    ``select_shakespeare_scene_ref``, and draws over UNITS rather than sources
    so a novel contributing three chapters is three chances, not one -- the
    unit is what an episode actually adapts.
    """
    checked = validate_public_domain_manifest(manifest)
    refs = [
        f"{source['source_id']}:{unit['unit_id']}"
        for source in checked["sources"]
        for unit in source["units"]
    ]
    if not refs:
        raise PublicDomainManifestError(
            "public-domain manifest declares no units to select from"
        )
    chooser = rng if rng is not None else random.SystemRandom()
    try:
        return str(chooser.choice(refs))
    except AttributeError:  # pragma: no cover -- defensive for minimal RNG shims
        return str(refs[chooser.randrange(len(refs))])


# The interpreter reads the whole canonicalized body, not a prefix of it.
INTERPRETER_TEXT_WINDOW: int = 12000


def canonicalize_public_domain_text(text: Any, *, max_chars: int = 12000) -> str:
    """Strip common public-domain boilerplate and normalize whitespace."""
    raw = html.unescape(str(text or "")).replace("\ufeff", "")
    raw = _PG_HEADER_RE.sub(" ", raw)
    raw = _PG_FOOTER_RE.sub(" ", raw)
    cleaned = _WS_RE.sub(" ", raw).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip() or cleaned[:max_chars]
    if not cleaned:
        raise PublicDomainSourceRefError("public-domain source text is empty")
    return cleaned


def payload_from_manifest_unit(
    resolved: PublicDomainUnit,
    *,
    text: str,
    excerpt_chars: int = 1200,
) -> dict[str, str]:
    """Build the legacy source payload for a resolved manifest fixture unit."""
    source = resolved.source
    unit = resolved.unit
    full_text = canonicalize_public_domain_text(text)
    excerpt = full_text[:excerpt_chars].rsplit(" ", 1)[0].rstrip() or full_text[:excerpt_chars]
    payload = {
        "headline": f"{source['title']} - {unit['label']}",
        "summary": unit["synopsis"],
        "full_text": full_text,
        "source": source["source_label"],
        "date": source["year"],
        "link": source["source_url"],
        "seed_text": (
            f"{source['title']} by {source['author']}\n"
            f"Unit: {unit['label']}\n"
            f"Synopsis: {unit['synopsis']}\n"
            f"Excerpt: {excerpt}"
        ),
    }
    return _osp.validate_source_payload(
        payload, origin=f"public_domain {resolved.source_ref}")


def fetch_public_domain_source(
    *,
    bank: Any,
    source_ref: str = "",
) -> "_osp.SourceFetchResult":
    """Load a manifest-local public-domain unit and return payload + sidecars.

    Blank source_ref uses the explicit bank default, otherwise it fails loud.
    """
    defaults = getattr(bank, "defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise PublicDomainManifestError("public-domain bank defaults must be a dict")

    manifest_path = _resolve_repo_relative_path(
        str(defaults.get("manifest_path", "")),
        key="manifest_path",
    )
    manifest = load_public_domain_manifest(manifest_path)
    # An explicit ref always pins. A blank one consults selection_mode, exactly
    # as the Shakespeare lane does -- "random" deals from the whole library,
    # "fixed" honours defaults.source_ref. Default is random: a source BANK
    # whose every episode is the same book is the defect this closes.
    effective_ref = str(source_ref or "").strip()
    selection_mode = str(
        defaults.get("selection_mode", "random") or "random"
    ).strip().lower()
    if not effective_ref:
        if selection_mode in {"random", "random_unit", "shuffle"}:
            effective_ref = select_public_domain_unit_ref(manifest)
        elif selection_mode in {"fixed", "default", "pinned"}:
            effective_ref = str(defaults.get("source_ref", "") or "").strip()
        else:
            bank_id = getattr(bank, "source_bank_id", "public_domain")
            raise PublicDomainManifestError(
                f"source_bank {bank_id!r} has unsupported public-domain "
                f"selection_mode {selection_mode!r}; expected random or fixed"
            )
    if not effective_ref:
        bank_id = getattr(bank, "source_bank_id", "public_domain")
        raise PublicDomainSourceRefError(
            f"source_bank {bank_id!r} fixed selection requires "
            "defaults.source_ref; there is no fallback"
        )

    resolved = resolve_manifest_unit(manifest, effective_ref)
    text_path = manifest_path.parent / resolved.unit["text_path"]
    try:
        text = text_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PublicDomainSourceRefError(
            f"public-domain source text not found for {resolved.source_ref}: "
            f"{text_path}"
        ) from exc

    return _osp.SourceFetchResult(
        payload=payload_from_manifest_unit(resolved, text=text),
        source_meta=source_meta_from_unit(resolved),
        source_rights=source_rights_from_unit(resolved),
    )


def source_rights_from_unit(resolved: PublicDomainUnit) -> dict[str, str]:
    """Small rights sidecar for future meta stamping."""
    source = resolved.source
    return {
        "license_status": source["license_status"],
        "license_url": source["license_url"],
        "source_url": source["source_url"],
        "source_label": source["source_label"],
    }


def source_meta_from_unit(resolved: PublicDomainUnit) -> dict[str, Any]:
    """Small metadata sidecar for future meta stamping."""
    source = resolved.source
    unit = resolved.unit
    return {
        "source_ref": resolved.source_ref,
        "source_id": source["source_id"],
        "unit_id": unit["unit_id"],
        "title": source["title"],
        "author": source["author"],
        "year": source["year"],
        "unit_label": unit["label"],
        "recommended_word_budget": source["recommended_word_budget"],
        "cast_hints": list(source["cast_hints"]),
    }


class PublicDomainBriefs(BaseModel):
    """Briefs contract consumed by ``_otr_source_payload`` and the writer."""

    casting_brief: str
    script_brief: str
    news_close_brief: str
    key_terms: list[str] = Field(default_factory=list)
    # The SOURCE's real named characters (Ebenezer Scrooge, Bob Cratchit), so the
    # adaptation cast preserves the story's people instead of rolling random pool
    # names. OPTIONAL + default empty: a source with no named characters, or an
    # older fixture, still validates; the writer then falls back to the pool. Public
    # -domain prose has no speaker markup, so unlike shakespeare (manifest cast_hints)
    # these must be LLM-extracted here.
    character_names: list[str] = Field(default_factory=list, max_length=6)

    source_hash: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    attempts: int = 0

    @field_validator("character_names", mode="before")
    @classmethod
    def _clean_character_names(cls, value):
        """Strip/dedupe the source names; cap at 6. Never raises -- an absent or
        malformed list simply yields an empty list (writer falls back to pool)."""
        if not isinstance(value, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for nm in value:
            clean = " ".join(str(nm or "").split()).strip()
            key = clean.upper()
            if clean and key not in seen:
                seen.add(key)
                out.append(clean)
            if len(out) >= 6:
                break
        return out



def _source_hash(payload: dict[str, str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in ("headline", "summary", "full_text", "source", "date", "link"):
        h.update(str(payload.get(key, "")).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _build_interpreter_prompt(payload: dict[str, str]) -> list[dict[str, str]]:
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
            f"Synopsis: {summary}",
            "Source text:",
        # The interpreter is the ONE pass that reads the source, and it is
        # instructed to preserve the ending -- so a window shorter than the
        # canonicalized body means it is asked to preserve an ending it
        # cannot see. The Wells arrival unit is 11,410 bytes against a
        # 5,000-char window: the press ridicule, "Story be damned!" and the
        # closing image of listeners' faces in the dark all fell outside it.
        # canonicalize_* already caps the body at INTERPRETER_TEXT_WINDOW,
        # so this reads whatever survived that cap and truncates nothing further.
            full_text[:INTERPRETER_TEXT_WINDOW],
        )
        if part
    )
    instruction = (
        "You are the source brain for a compact old-time-radio adaptation of "
        "public-domain literature.\n\n"
        "Turn the source material below into JSON briefs for a faithful radio "
        "adaptation. Preserve the source's named characters, central conflict, "
        "major turns, and ending. Compression is allowed; replacement is not. "
        "Do not invent an unrelated framing story, a new protagonist, or a "
        "different ending.\n\n"
        "Keep it SFW: no profanity, explicit guns/knives/weapons, or "
        "explicit sexual/nudity language in the adapted premise.\n\n"
        "Return ONE JSON object only with exactly these keys:\n"
        "{\n"
        "  \"casting_brief\": \"source-grounded roles and voices\",\n"
        "  \"script_brief\": \"faithful radio adaptation brief\",\n"
        "  \"news_close_brief\": \"source attribution note\",\n"
        "  \"key_terms\": [\"optional concise source/adaptation terms\"],\n"
        "  \"character_names\": [\"REQUIRED, 2-6. The story's OWN cast, taken "
        "straight from the source text, in order of appearance. Use each "
        "character exactly as the text refers to them -- a proper name when the "
        "source gives one (Ebenezer Scrooge, Bob Cratchit), OR the source's own "
        "role-designation when it does not (the Time Traveller, the Medical Man, "
        "Filby). List the real people from THIS source only; never invent a "
        "name, never use a generic placeholder like 'Witness 1'\"]\n"
        "}\n\n"
        f"{source_block}"
    )
    return [{"role": "user", "content": instruction}]


def build_public_domain_briefs(
    *,
    technical_fn: Callable[..., str],
    payload: dict[str, str],
    model_id: str = "",
    max_attempts: int = 3,
    base_temperature: float = 0.35,
    max_new_tokens: int = 520,
) -> PublicDomainBriefs:
    """Run the public-domain source brain through the structured JSON ladder."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    slot_calls = 0

    def _counting_slot_fn(msgs, *, temperature, max_new_tokens):
        nonlocal slot_calls
        slot_calls += 1
        # LLM slot: technical -- structured source-brief extraction for the
        # public-domain lane, routed through the writer's technical slot.
        return technical_fn(
            msgs, temperature=temperature, max_new_tokens=max_new_tokens,
        )

    def _content_validator(brief: PublicDomainBriefs) -> str | None:
        # Runtime cast extraction is the ONLY faithful casting source for the
        # public-domain lane (the manifest carries only generic role hints like
        # "traveler"). So the cast is a REQUIRED generation input, not a
        # best-effort side field: the source's real people, pulled from its own
        # text. An empty list means the source brain skipped the cast -- retry
        # until it returns one, or fail loud (a PD adaptation with no cast is a
        # broken render, not a simplification). This is a generation floor like
        # the key_terms one above, NOT a story-quality audit.
        if len(brief.character_names) < 1:
            return ("public-domain briefs require at least one source character "
                    "name (the story's own cast, from its text)")
        hay = " ".join(
            [brief.casting_brief, brief.script_brief, brief.news_close_brief]
            + list(brief.key_terms)
        ).casefold()
        safety_hits = find_text_hits(hay)
        if safety_hits:
            category, term = safety_hits[0]
            return (
                "public-domain brief contains explicit safety term "
                f"{category}={term!r}"
            )
        return None

    try:
        # LLM slot: technical -- public-domain source-brief structured pass.
        brief = structured_call(
            prompt=_build_interpreter_prompt(payload),
            schema=PublicDomainBriefs,
            slot_fn=_counting_slot_fn,
            base_temperature=float(base_temperature),
            structural_retry_temperature=float(base_temperature) / 2.0,
            post_validator=_content_validator,
            max_new_tokens=int(max_new_tokens),
            max_attempts=int(max_attempts),
            helper_name="build_public_domain_briefs",
        )
    except StructuredCallFailedError as exc:
        raise PublicDomainInterpreterError(
            attempts=exc.attempts,
            reason=(
                f"{type(exc.last_error).__name__}: {exc.last_error}"
                if exc.last_error is not None
                else "no error captured"
            ),
        ) from exc

    brief.source_hash = _source_hash(payload)
    brief.model_id = model_id
    brief.attempts = slot_calls
    return brief


def source_bank_cache_root() -> Path:
    """Resolve the public source-bank cache root without creating it."""
    env = os.environ.get("OTR_SOURCE_BANK_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    try:
        from ._otr_paths import otr_shared_cache_dir
    except ImportError:  # pragma: no cover -- flat import harnesses
        from nodes._otr_paths import otr_shared_cache_dir  # type: ignore
    return otr_shared_cache_dir() / "source_banks"


def atomic_write_json(path: str | os.PathLike[str], data: Any) -> Path:
    """Atomically write JSON to ``path`` using a same-directory temp file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return p


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "PublicDomainManifestError",
    "PublicDomainBriefs",
    "PublicDomainInterpreterError",
    "PublicDomainSourceError",
    "PublicDomainSourceRefError",
    "PublicDomainUnit",
    "SCHEMA_VERSION",
    "atomic_write_json",
    "build_public_domain_briefs",
    "canonicalize_public_domain_text",
    "fetch_public_domain_source",
    "load_public_domain_manifest",
    "payload_from_manifest_unit",
    "resolve_manifest_unit",
    "source_bank_cache_root",
    "source_meta_from_unit",
    "source_rights_from_unit",
    "validate_public_domain_manifest",
]
