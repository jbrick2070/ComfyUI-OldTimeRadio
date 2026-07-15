"""Bake-off source-snapshot replay layer (pure, stdlib-only leaf module).

The bake-off's experimental control (F2, roundtable pass01 Sec C): for a base
bank, freeze ONE resolved source -- the seven-key source payload plus its
``source_meta`` / ``source_rights`` sidecars and a seed receipt -- then REPLAY
that exact source across the base / ``_v2`` / ``_v3`` triplet, so the ONLY
variable under test is the story pack, never a fresh RSS draw or a random spark.

``OTR_LedgerScriptWriter._resolve_inputs`` calls :func:`load_snapshot_for_bank`
IMMEDIATELY after bank resolution and BEFORE its three source branches (entropy /
custom-premise / external fetch). A returned snapshot bypasses all three; ``None``
means "no snapshot requested for this bank" and the live path runs unchanged.

The manifest is keyed by BASE bank id (:func:`_otr_bank_variants.base_source_bank_id`),
so one frozen source serves an entire triplet. An envelope whose declared base
does not match ``base_source_bank_id(selected)`` is REJECTED loudly (r3 ruling
B7): a requested replay that cannot be honored is a hard failure, never a silent
slide back to live sourcing.

Kept dependency-free (stdlib + the leaf ``_otr_bank_variants``) so it never sits
on an import cycle -- it is imported at module scope by the writer. Payload SHAPE
validation stays in ``_otr_source_payload.validate_source_payload``; this module
validates only the ENVELOPE (manifest wiring, base match, seed receipt, sidecar
types) and returns the raw payload for the writer to validate uniformly with the
live branches.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from . import _otr_bank_variants as _otr_bank_variants

# Env var naming the process-wide snapshot manifest JSON. Unset => no replay.
SNAPSHOT_MANIFEST_ENV = "OTR_SOURCE_SNAPSHOT_MANIFEST"

# Accepted manifest schema version. Bump only on a breaking change.
SNAPSHOT_SCHEMA_VERSION = 1

# The exact seven-key source payload contract (mirrors
# _otr_source_payload.SOURCE_PAYLOAD_KEYS -- kept as a local tuple so this leaf
# never imports source_payload). Presence is pre-checked here for a
# snapshot-attributed error; the writer still runs the authoritative
# validate_source_payload after load.
_PAYLOAD_KEYS = (
    "headline", "summary", "full_text", "source", "date", "link", "seed_text",
)


class SourceSnapshotError(Exception):
    """A requested source-snapshot replay could not be honored (loud)."""


@dataclass(frozen=True)
class SourceSnapshot:
    """A validated frozen source, ready to replay through ``_resolve_inputs``."""

    base_source_bank_id: str
    payload: dict
    seed_text: str
    seed_source: str
    source_meta: dict
    source_rights: dict
    payload_sha256: str
    seed_receipt: dict
    captured_from_bank_id: str


def snapshot_payload_sha256(payload) -> str:
    """Stable SHA-256 over a source payload (canonical JSON, sorted keys)."""
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _clear_cache() -> None:
    """Retained test hook. The manifest is re-read on every call (it is read at
    most once per episode), so there is no parsed-manifest cache to drop; a live
    re-pin of the manifest file is always seen immediately."""
    return None


def _manifest_path() -> str:
    """The configured manifest path (stripped), or '' when unset."""
    return os.environ.get(SNAPSHOT_MANIFEST_ENV, "").strip()


def _load_manifest():
    """Return ``(manifest_dict, abspath)`` or ``None`` when unconfigured.

    Re-read on every call: ``_resolve_inputs`` runs once per episode, so the
    parse cost is negligible and a live re-pin is always honored without a
    stale-cache hazard (Windows file-mtime resolution is too coarse to key a
    cache on safely). A configured-but-unreadable manifest is a misconfiguration
    and fails loud.
    """
    path = _manifest_path()
    if not path:
        return None
    abspath = os.path.abspath(path)
    try:
        with open(abspath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SourceSnapshotError(
            f"source-snapshot manifest {abspath!r} (from "
            f"{SNAPSHOT_MANIFEST_ENV}) is not readable/valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise SourceSnapshotError(
            f"source-snapshot manifest {abspath!r} must be a JSON object, got "
            f"{type(data).__name__}"
        )
    version = data.get("schema_version", SNAPSHOT_SCHEMA_VERSION)
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise SourceSnapshotError(
            f"source-snapshot manifest {abspath!r} schema_version {version!r} "
            f"!= supported {SNAPSHOT_SCHEMA_VERSION}"
        )
    return data, abspath


def _resolve_entry(entry, *, base: str, manifest_dir: str) -> dict:
    """An entry is an inline envelope dict or a ``{"path": ...}`` pointer."""
    if isinstance(entry, dict) and "path" in entry and "payload" not in entry:
        pointer = str(entry["path"]).strip()
        if not os.path.isabs(pointer):
            pointer = os.path.join(manifest_dir, pointer)
        try:
            with open(pointer, "r", encoding="utf-8") as handle:
                entry = json.load(handle)
        except (OSError, ValueError) as exc:
            raise SourceSnapshotError(
                f"source-snapshot envelope for base {base!r} at {pointer!r} "
                f"could not be read: {exc}"
            ) from exc
    if not isinstance(entry, dict):
        raise SourceSnapshotError(
            f"source-snapshot envelope for base {base!r} must be a JSON "
            f"object, got {type(entry).__name__}"
        )
    return entry


def load_snapshot_for_bank(selected_bank_id: str):
    """Return a validated :class:`SourceSnapshot` for the selected bank, or None.

    ``None`` means no manifest is configured, or the manifest has no entry for
    this bank's base -- the live source path runs unchanged. A present-but-invalid
    entry (bad JSON, base mismatch, missing ``seed_source``, altered payload)
    raises :class:`SourceSnapshotError`: a requested replay that cannot be
    honored is a hard failure, never a silent fall-through to live RSS/random.
    """
    loaded = _load_manifest()
    if loaded is None:
        return None
    manifest, abspath = loaded

    base = _otr_bank_variants.base_source_bank_id(str(selected_bank_id or ""))
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, dict):
        raise SourceSnapshotError(
            f"source-snapshot manifest {abspath!r} is missing a 'snapshots' "
            f"object (got {type(snapshots).__name__})"
        )
    if base not in snapshots:
        return None

    envelope = _resolve_entry(
        snapshots[base], base=base, manifest_dir=os.path.dirname(abspath),
    )

    declared = envelope.get("base_source_bank_id")
    if declared != base:
        raise SourceSnapshotError(
            f"source-snapshot for selected bank {selected_bank_id!r} declares "
            f"base {declared!r} but base_source_bank_id(selected) is {base!r} "
            f"-- refusing to replay a mismatched source (B7)"
        )

    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise SourceSnapshotError(
            f"source-snapshot for base {base!r}: 'payload' must be an object, "
            f"got {type(payload).__name__}"
        )
    missing = [key for key in _PAYLOAD_KEYS if key not in payload]
    if missing:
        raise SourceSnapshotError(
            f"source-snapshot for base {base!r}: payload missing key(s) "
            f"{sorted(missing)}"
        )

    seed_source = envelope.get("seed_source")
    if not isinstance(seed_source, str) or not seed_source.strip():
        raise SourceSnapshotError(
            f"source-snapshot for base {base!r}: 'seed_source' must be a "
            f"non-empty string"
        )

    source_meta = envelope.get("source_meta") or {}
    source_rights = envelope.get("source_rights") or {}
    if not isinstance(source_meta, dict) or not isinstance(source_rights, dict):
        raise SourceSnapshotError(
            f"source-snapshot for base {base!r}: 'source_meta' / "
            f"'source_rights' sidecars must be objects"
        )
    source_meta = dict(source_meta)

    # cast_hints may ride as a top-level convenience (B7 lists it beside the
    # sidecars); merge it into source_meta where the writer's adaptation
    # consumer reads it, without clobbering an already-captured value.
    top_hints = envelope.get("cast_hints")
    if top_hints and "cast_hints" not in source_meta:
        if not isinstance(top_hints, list):
            raise SourceSnapshotError(
                f"source-snapshot for base {base!r}: 'cast_hints' must be a list"
            )
        source_meta["cast_hints"] = list(top_hints)

    seed_text = envelope.get("seed_text")
    if not isinstance(seed_text, str) or not seed_text.strip():
        seed_text = str(payload.get("seed_text") or "")

    seed_receipt = envelope.get("seed_receipt") or {}
    if not isinstance(seed_receipt, dict):
        raise SourceSnapshotError(
            f"source-snapshot for base {base!r}: 'seed_receipt' must be an object"
        )

    computed_sha = snapshot_payload_sha256(payload)
    declared_sha = str(seed_receipt.get("payload_sha256") or "").strip()
    if declared_sha and declared_sha != computed_sha:
        raise SourceSnapshotError(
            f"source-snapshot for base {base!r}: payload_sha256 receipt "
            f"{declared_sha!r} != actual {computed_sha!r} -- payload was "
            f"altered after capture"
        )

    return SourceSnapshot(
        base_source_bank_id=base,
        payload=dict(payload),
        seed_text=seed_text,
        seed_source=seed_source,
        source_meta=source_meta,
        source_rights=dict(source_rights),
        payload_sha256=computed_sha,
        seed_receipt=dict(seed_receipt),
        captured_from_bank_id=str(envelope.get("captured_from_bank_id") or ""),
    )


__all__ = [
    "SNAPSHOT_MANIFEST_ENV",
    "SNAPSHOT_SCHEMA_VERSION",
    "SourceSnapshot",
    "SourceSnapshotError",
    "load_snapshot_for_bank",
    "snapshot_payload_sha256",
]
