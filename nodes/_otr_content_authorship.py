"""Generic accepted-artifact authorship proof for content-owned story lanes."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1


class ContentAuthorshipError(RuntimeError):
    """The accepted artifact no longer proves the canonical ledger text."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _voiced_rows(ledger_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = ledger_data.get("lines")
    if not isinstance(rows, list):
        raise ContentAuthorshipError("ledger lines must be a list")
    return [
        row for row in rows
        if isinstance(row, Mapping)
        and not bool(row.get("skip_tts"))
        and str(row.get("text") or "")
    ]


def build_receipt(
    ledger_data: Mapping[str, Any], *, owner_bank: str,
    accepted_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the receipt from final accepted artifacts and canonical rows."""
    owner = str(owner_bank or "").strip()
    if not owner:
        raise ContentAuthorshipError("owner_bank is required")
    source_bank = str((ledger_data.get("meta") or {}).get("source_bank") or "")
    if source_bank != owner:
        raise ContentAuthorshipError(
            f"owner_bank {owner!r} does not match meta.source_bank {source_bank!r}"
        )
    if not accepted_artifacts:
        raise ContentAuthorshipError("at least one accepted artifact is required")
    artifact_rows: list[dict[str, str]] = []
    for artifact_id, artifact in accepted_artifacts.items():
        aid = str(artifact_id or "").strip()
        if not aid:
            raise ContentAuthorshipError("accepted artifact id is empty")
        artifact_rows.append({
            "artifact_id": aid,
            "sha256": _sha256_bytes(_artifact_bytes(artifact)),
        })
    if len({row["artifact_id"] for row in artifact_rows}) != len(artifact_rows):
        raise ContentAuthorshipError("accepted artifact ids must be unique")
    line_rows: list[dict[str, str]] = []
    for row in _voiced_rows(ledger_data):
        line_id = str(row.get("line_id") or "").strip()
        if not line_id:
            raise ContentAuthorshipError("voiced ledger line has no line_id")
        text = str(row.get("text") or "")
        line_rows.append({
            "line_id": line_id,
            "text_sha256": _sha256_bytes(text.encode("utf-8")),
        })
    ids = [row["line_id"] for row in line_rows]
    if len(set(ids)) != len(ids):
        raise ContentAuthorshipError("voiced ledger line_ids must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "owner_bank": owner,
        "accepted_artifacts": artifact_rows,
        "line_proofs": line_rows,
        "coverage": {
            "voiced_line_count": len(line_rows),
            "proved_line_count": len(line_rows),
            "complete": True,
        },
    }


def stamp_receipt(
    ledger_data: dict[str, Any], *, owner_bank: str,
    accepted_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    # VOICE COVERAGE GATES THE MINT (PBUG-20260802-02). Both content-owned
    # lanes pass through here -- the ONE shared pre-proof boundary -- so this
    # is where "every cast member gets a voice" is enforced, by SAYABLE text
    # rather than raw text. A row that passes this gate is a row the
    # writer-tail cleanup will never empty, which is what makes the proofs
    # minted below stable instead of "a proof of nothing".
    from ._otr_cast_voice_coverage import require_voice_coverage
    require_voice_coverage(ledger_data, owner_bank=owner_bank)
    receipt = build_receipt(
        ledger_data, owner_bank=owner_bank,
        accepted_artifacts=accepted_artifacts,
    )
    ledger_data.setdefault("meta", {})["content_authorship"] = receipt
    validate_receipt(ledger_data)
    return receipt


def validate_receipt(ledger_data: Mapping[str, Any]) -> dict[str, Any]:
    meta = ledger_data.get("meta")
    if not isinstance(meta, Mapping):
        raise ContentAuthorshipError("ledger meta must be an object")
    receipt = meta.get("content_authorship")
    if not isinstance(receipt, Mapping):
        raise ContentAuthorshipError("meta.content_authorship is required")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ContentAuthorshipError("unsupported content_authorship schema_version")
    owner = str(receipt.get("owner_bank") or "")
    if owner != str(meta.get("source_bank") or ""):
        raise ContentAuthorshipError("content_authorship owner/source-bank mismatch")
    artifacts = receipt.get("accepted_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ContentAuthorshipError("accepted_artifacts must be a non-empty list")
    artifact_ids = [str((row or {}).get("artifact_id") or "") for row in artifacts]
    if any(not item for item in artifact_ids) or len(set(artifact_ids)) != len(artifact_ids):
        raise ContentAuthorshipError("accepted artifact ids are empty or duplicated")
    for row in artifacts:
        digest = str((row or {}).get("sha256") or "")
        if len(digest) != 64:
            raise ContentAuthorshipError("accepted artifact sha256 is malformed")
    proofs = receipt.get("line_proofs")
    if not isinstance(proofs, list):
        raise ContentAuthorshipError("line_proofs must be a list")
    proof_by_id: dict[str, Mapping[str, Any]] = {}
    for proof in proofs:
        if not isinstance(proof, Mapping):
            raise ContentAuthorshipError("line proof must be an object")
        line_id = str(proof.get("line_id") or "")
        if not line_id or line_id in proof_by_id:
            raise ContentAuthorshipError("line proof ids are empty or duplicated")
        proof_by_id[line_id] = proof
    live = _voiced_rows(ledger_data)
    live_by_id = {str(row.get("line_id") or ""): row for row in live}
    if set(proof_by_id) != set(live_by_id):
        missing = sorted(set(live_by_id) - set(proof_by_id))
        extra = sorted(set(proof_by_id) - set(live_by_id))
        raise ContentAuthorshipError(
            f"line proof coverage mismatch: missing={missing} extra={extra}"
        )
    for line_id, row in live_by_id.items():
        digest = _sha256_bytes(str(row.get("text") or "").encode("utf-8"))
        if proof_by_id[line_id].get("text_sha256") != digest:
            raise ContentAuthorshipError(f"canonical text hash mismatch for {line_id!r}")
    coverage = receipt.get("coverage")
    expected_count = len(live_by_id)
    if not isinstance(coverage, Mapping) or coverage != {
        "voiced_line_count": expected_count,
        "proved_line_count": expected_count,
        "complete": True,
    }:
        raise ContentAuthorshipError("content_authorship coverage summary is false")
    return dict(receipt)


def receipt_sha256(ledger_data: Mapping[str, Any]) -> str:
    meta = ledger_data.get("meta")
    receipt = meta.get("content_authorship") if isinstance(meta, Mapping) else None
    if not isinstance(receipt, Mapping):
        raise ContentAuthorshipError("meta.content_authorship is required")
    # Fingerprinting is deliberately independent of semantic validation so the
    # cascade can record entry/exit hashes even for a malformed/tampered receipt;
    # `_readonly_structural_validation` is the fail-closed semantic gate.
    return _sha256_bytes(_artifact_bytes(dict(receipt)))


__all__ = [
    "ContentAuthorshipError", "SCHEMA_VERSION", "build_receipt",
    "receipt_sha256", "stamp_receipt", "validate_receipt",
]
