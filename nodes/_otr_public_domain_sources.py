"""Public-domain source-bank helpers.

This module owns the public-domain source-bank helpers and the registered
fetcher. It is still fixture/local-file only: no network calls, no heavy
imports, no workflow changes.
"""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._otr_source_payload import SourceFetchResult, validate_source_payload

MANIFEST_SCHEMA_VERSION = "v1"

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
    "visual_style_policy",
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


def _require_int(obj: dict[str, Any], key: str, origin: str, *, min_value: int, max_value: int) -> int:
    val = obj.get(key)
    if not isinstance(val, int) or isinstance(val, bool):
        raise PublicDomainManifestError(f"{origin}: {key} must be an integer")
    if val < min_value or val > max_value:
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
            raw, "recommended_word_budget", origin, min_value=30, max_value=320
        ),
        "cast_hints": _require_str_list(raw, "cast_hints", origin),
        "visual_style_policy": _require_str(raw, "visual_style_policy", origin),
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
    return validate_source_payload(payload, origin=f"public_domain {resolved.source_ref}")


def fetch_public_domain_source(
    *,
    bank: Any,
    source_ref: str = "",
) -> SourceFetchResult:
    """Load a manifest-local public-domain unit and return payload + sidecars.

    The bank is intentionally allowed to stay non-runnable while this fetcher
    exists; the interpreter/runnable flip is a later chunk. Blank source_ref
    uses the explicit bank default, otherwise it fails loud.
    """
    defaults = getattr(bank, "defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise PublicDomainManifestError("public-domain bank defaults must be a dict")

    manifest_path = _resolve_repo_relative_path(
        str(defaults.get("manifest_path", "")),
        key="manifest_path",
    )
    effective_ref = str(source_ref or defaults.get("source_ref", "") or "").strip()
    if not effective_ref:
        bank_id = getattr(bank, "source_bank_id", "public_domain_story")
        raise PublicDomainSourceRefError(
            f"source_bank {bank_id!r} requires source_ref or "
            "defaults.source_ref; there is no fallback"
        )

    manifest = load_public_domain_manifest(manifest_path)
    resolved = resolve_manifest_unit(manifest, effective_ref)
    text_path = manifest_path.parent / resolved.unit["text_path"]
    try:
        text = text_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PublicDomainSourceRefError(
            f"public-domain source text not found for {resolved.source_ref}: "
            f"{text_path}"
        ) from exc

    return SourceFetchResult(
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
        "visual_style_policy": source["visual_style_policy"],
    }


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
    "PublicDomainManifestError",
    "PublicDomainSourceError",
    "PublicDomainSourceRefError",
    "PublicDomainUnit",
    "atomic_write_json",
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
