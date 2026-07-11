"""nodes/_otr_cue_manifest.py -- shared music cue-manifest schema + helpers.

720-bakeoff C3. The music theme node (OTR_StableAudioTheme) renders every cue
into ONE padded AUDIO batch and emits this manifest alongside it. The manifest
is the ONLY authority that maps a batch row back to a cue: SceneSequencer and
EpisodeAssembler slice each cue out of the padded batch by its recorded
``sample_count`` (never by the padded tail) and resolve boundaries by
``anchor_line_id`` / ``cue_id`` -- keying is (cue_id, batch_index), NEVER
positional guesswork.

ONE parse + fail-loud validate path is shared by the writer (StableAudioTheme),
both consumers, and the tests, so a malformed / stale manifest is caught the
same way everywhere. PURE: json + hashlib only -- no torch, no I/O, no engine
imports. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

#: Manifest wire-format version. Bump only on an incompatible schema change.
MANIFEST_VERSION = 1

#: Canonical cue placements. Every row's ``placement`` maps to exactly one of
#: these (the composer / duration tables are keyed on them).
PLACEMENTS = ("opening", "closing", "interstitial")

#: Required per-cue row fields. ``output_path`` may be "" when no episode audio
#: dir was resolvable (headless / test), but the KEY is always present.
#: ``cue_spec_sha256`` / ``anchor_line_id`` may be None (legacy synth cues), but
#: the KEYs are always present.
_REQUIRED_ROW_FIELDS = (
    "cue_id", "batch_index", "sample_count", "sample_rate", "prompt",
    "prompt_sha256", "cue_spec_sha256", "placement", "anchor_line_id",
    "seed", "requested_duration_s", "actual_duration_s", "output_path",
)


class CueManifestError(ValueError):
    """Raised on any malformed / inconsistent cue manifest. Fail loud -- a bad
    manifest means the music bus and the audio batch disagree, which would glue
    the wrong (or padded) audio into the master mix."""


def prompt_sha256(prompt: str) -> str:
    """SHA256 hex of the cue prompt text (cheap identity for the manifest row)."""
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()


def build_manifest(sample_rate: int, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble + validate a manifest from finished per-cue rows. Raises
    CueManifestError if the rows are inconsistent (dup cue_id / batch_index,
    non-contiguous batch indices, bad sample_count, placement, or rate)."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "sample_rate": int(sample_rate),
        "cues": [dict(r) for r in rows],
    }
    validate_manifest(manifest, batch_size=len(rows))
    return manifest


def dumps(manifest: Dict[str, Any]) -> str:
    """Deterministic JSON encoding (sorted keys) so byte-parity holds across
    runs for an identical cue set."""
    return json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def parse_manifest(
    manifest_json: Any, *, batch_size: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Parse + validate a manifest. An empty / blank string (or None) returns
    None -- i.e. "no music bus is wired here" (the consumer default). A
    non-empty but malformed manifest raises CueManifestError. Pass
    ``batch_size`` to assert the cue count equals the AUDIO batch row count."""
    if manifest_json is None:
        return None
    if isinstance(manifest_json, dict):
        manifest = manifest_json
    else:
        text = str(manifest_json).strip()
        if not text:
            return None
        try:
            manifest = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise CueManifestError(f"cue manifest is not valid JSON: {exc}") from exc
    validate_manifest(manifest, batch_size=batch_size)
    return manifest


def validate_manifest(
    manifest: Dict[str, Any], *, batch_size: Optional[int] = None
) -> None:
    """Fail-loud structural validation. Shared by writer, consumers, tests."""
    if not isinstance(manifest, dict):
        raise CueManifestError("cue manifest must be a JSON object")
    ver = manifest.get("manifest_version")
    if ver != MANIFEST_VERSION:
        raise CueManifestError(
            f"cue manifest_version {ver!r} != supported {MANIFEST_VERSION}"
        )
    sr = manifest.get("sample_rate")
    if not isinstance(sr, int) or sr <= 0:
        raise CueManifestError(
            f"cue manifest sample_rate {sr!r} must be a positive int"
        )
    cues = manifest.get("cues")
    if not isinstance(cues, list):
        raise CueManifestError("cue manifest 'cues' must be a list")

    seen_cue_ids: set = set()
    seen_batch: set = set()
    for i, row in enumerate(cues):
        if not isinstance(row, dict):
            raise CueManifestError(f"cue row {i} must be a JSON object")
        for key in _REQUIRED_ROW_FIELDS:
            if key not in row:
                raise CueManifestError(
                    f"cue row {i} missing required field {key!r}"
                )
        cue_id = row["cue_id"]
        if not isinstance(cue_id, str) or not cue_id:
            raise CueManifestError(
                f"cue row {i} cue_id must be a non-empty string"
            )
        if cue_id in seen_cue_ids:
            raise CueManifestError(f"duplicate cue_id {cue_id!r} in manifest")
        seen_cue_ids.add(cue_id)

        batch_index = row["batch_index"]
        if not isinstance(batch_index, int) or batch_index < 0:
            raise CueManifestError(
                f"cue {cue_id!r} batch_index {batch_index!r} must be >= 0"
            )
        if batch_index in seen_batch:
            raise CueManifestError(
                f"duplicate batch_index {batch_index} in manifest"
            )
        seen_batch.add(batch_index)

        sample_count = row["sample_count"]
        if not isinstance(sample_count, int) or sample_count <= 0:
            raise CueManifestError(
                f"cue {cue_id!r} sample_count {sample_count!r} must be > 0"
            )
        row_sr = row["sample_rate"]
        if row_sr != sr:
            raise CueManifestError(
                f"cue {cue_id!r} sample_rate {row_sr!r} != manifest {sr}"
            )
        placement = row["placement"]
        if placement not in PLACEMENTS:
            raise CueManifestError(
                f"cue {cue_id!r} placement {placement!r} not in {PLACEMENTS}"
            )

    n = len(cues)
    # batch_index must be a contiguous 0..n-1 set: the manifest row count MUST
    # equal the AUDIO batch row count, one row per batch index.
    if seen_batch and seen_batch != set(range(n)):
        raise CueManifestError(
            f"cue batch_index set {sorted(seen_batch)} is not contiguous 0..{n - 1}"
        )
    if batch_size is not None and n != batch_size:
        raise CueManifestError(
            f"manifest cue count {n} != audio batch row count {batch_size}"
        )


def index_by_cue_id(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """cue_id -> row. Assumes a validated manifest (unique cue_ids)."""
    return {r["cue_id"]: r for r in (manifest.get("cues") or [])}


def index_by_anchor_line_id(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """anchor_line_id -> row, skipping rows with no anchor (legacy synth cues).
    Used by SceneSequencer to resolve a music-boundary sentinel line to its
    cue (fable2 sentinels carry no cue_id -- the anchor is the only link)."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in (manifest.get("cues") or []):
        anchor = row.get("anchor_line_id")
        if anchor:
            out[str(anchor)] = row
    return out
