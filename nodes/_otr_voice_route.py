"""Semantic validation for a QUALIFIED voice route (plan section 5.1).

WHY THIS EXISTS, AND WHY THE OLD HELPER IS NOT ENOUGH. `is_qualified_route`
(`config/cast_pools.py`) checks that receipt fields are present and non-blank.
That was the right first step -- it killed a route whose entire qualification was
the bare string ``"canonical_bark_preset_v1"`` -- but "the fields are filled in"
is not "the claim is true". A receipt can name a reference file that does not
exist, a hash that does not match the bytes on disk, an engine that disagrees
with the one actually rendering, or a rights approval that expired last month,
and every field would still be non-blank.

So the legacy helper REMAINS as a cheap compatibility check and **may never
authorize a selected route**. Authorization goes through here.

THE DESIGN RULE IS FAIL-CLOSED, EVERYWHERE. An unknown status value, an
unparseable timestamp, a missing file, a hash mismatch, an unreadable byte -- all
of them REJECT. A route that cannot prove itself has not proved itself, and the
whole point of this module is that the proof is checked rather than asserted.

WHY IT RETURNS REASONS INSTEAD OF A BOOL. A bare ``False`` on a route the
operator spent a listening session qualifying is a support ticket. Every
rejection names what failed, so the person holding the receipt can fix the right
field.

Stdlib only -- no torch, no engine imports -- so CastLock and the voice node can
both call it without dragging a model into a cold path.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

#: The only legal values. Anything else is a REJECT, not a warning: a status
#: nobody defined is a status nobody can reason about.
ROUTE_STATUSES = ("candidate", "evidenced", "technical_pass", "qualified",
                  "rejected", "revoked")
TECHNICAL_VERDICTS = ("not_run", "pass", "fail")
RIGHTS_STATUSES = ("pending", "approved", "denied", "revoked")

#: Selection requires EXACTLY this triple. Not "at least" -- exactly.
SELECTABLE_ROUTE_STATUS = "qualified"
SELECTABLE_TECHNICAL_VERDICT = "pass"
SELECTABLE_RIGHTS_STATUS = "approved"

#: A route contract version this code understands. A future version must be
#: rejected rather than guessed at -- a validator that silently accepts a schema
#: it was not written for is the same lie in a longer form.
SUPPORTED_ROUTE_CONTRACT_VERSIONS = (1,)

REFERENCE_KINDS = ("local_wav", "provider_voice")

_SHA256_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")


@dataclass(frozen=True)
class RouteValidation:
    """The verdict, plus every reason it failed. ``reasons`` is empty iff ok."""

    ok: bool
    reasons: tuple

    def __bool__(self) -> bool:          # so callers may `if not validation:`
        return self.ok

    @property
    def summary(self) -> str:
        return "route OK" if self.ok else "; ".join(self.reasons)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def parse_utc(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 UTC stamp, or return None if it is not one.

    Per the plan: accept the trailing ``Z`` spelling and compare only AWARE UTC
    datetimes -- a naive datetime compared against an aware one raises, and a
    validator that raises on a well-formed receipt is its own outage.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None                       # naive: refuse to guess a zone
    return parsed.astimezone(timezone.utc)


def sha256_of_file(path: str) -> Optional[str]:
    """Hash a file, or None if it cannot be read. Never raises: an unreadable
    reference is a validation failure, not a crash in the caller."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:                     # noqa: BLE001 -- see docstring
        return None


def validate_qualified_voice_route(
    record: Any,
    now_utc: datetime,
    *,
    active_engine: Optional[str] = None,
    bank_lookup: Optional[Callable[[str], Optional[dict]]] = None,
    repo_root: Optional[str] = None,
    require_local_bytes: bool = True,
) -> RouteValidation:
    """Decide whether ``record`` authorizes rendering a character on a route.

    ``record``        the route dict (``approved_native_routes[engine]`` shape).
    ``now_utc``       aware UTC 'now', injected so tests are not time-bombs.
    ``active_engine`` the scalar engine actually about to render, when known.
                      The three-way agreement (route == active == bank entry) is
                      the check that stops a policy approving one engine while
                      another renders.
    ``bank_lookup``   ``voice_ref_id -> bank row`` when the bank is available.
    ``repo_root``     for resolving a relative ``ref_path``.
    ``require_local_bytes`` hashing a local reference is the expensive check;
                      it stays ON by default and is only relaxed deliberately.
    """
    bad = []

    def fail(reason: str) -> None:
        bad.append(reason)

    if not isinstance(record, dict):
        return RouteValidation(False, ("route record is not a dict",))

    # --- the envelope -------------------------------------------------------
    if not str(record.get("route_id") or "").strip():
        fail("route_id is missing or blank")

    version = record.get("route_contract_version")
    if version not in SUPPORTED_ROUTE_CONTRACT_VERSIONS:
        fail("route_contract_version %r is not supported by this validator "
             "(known: %r) -- refusing to guess at an unknown schema"
             % (version, list(SUPPORTED_ROUTE_CONTRACT_VERSIONS)))

    qual = record.get("qualification_record")
    if not isinstance(qual, dict):
        return RouteValidation(False, tuple(bad + ["qualification_record is missing"]))

    # --- the three statuses that must ALL be exactly right ------------------
    status = qual.get("status")
    if status not in ROUTE_STATUSES:
        fail("route status %r is not a defined status" % (status,))
    elif status != SELECTABLE_ROUTE_STATUS:
        fail("route status is %r, not %r" % (status, SELECTABLE_ROUTE_STATUS))

    verdict = qual.get("technical_verdict")
    if verdict not in TECHNICAL_VERDICTS:
        fail("technical_verdict %r is not a defined verdict" % (verdict,))
    elif verdict != SELECTABLE_TECHNICAL_VERDICT:
        fail("technical_verdict is %r, not %r"
             % (verdict, SELECTABLE_TECHNICAL_VERDICT))

    rights = qual.get("rights")
    if not isinstance(rights, dict):
        fail("rights block is missing -- permission is never inferred from silence")
        rights = {}
    else:
        r_status = rights.get("status")
        if r_status not in RIGHTS_STATUSES:
            fail("rights.status %r is not a defined status" % (r_status,))
        elif r_status != SELECTABLE_RIGHTS_STATUS:
            fail("rights.status is %r, not %r" % (r_status, SELECTABLE_RIGHTS_STATUS))

        if rights.get("revoked_at") is not None:
            fail("rights were REVOKED at %r" % (rights.get("revoked_at"),))

        expires = rights.get("expires_at")
        if expires is not None:
            when = parse_utc(expires)
            if when is None:
                fail("rights.expires_at %r is not an ISO-8601 UTC timestamp"
                     % (expires,))
            elif when <= now_utc:
                fail("rights EXPIRED at %s" % (when.isoformat(),))

        for field in ("source", "terms_snapshot_ref", "terms_snapshot_date",
                      "scope", "decided_at"):
            if not str(rights.get(field) or "").strip():
                fail("rights.%s is missing -- the approval is not auditable "
                     "without it" % field)
        if rights.get("decided_at") and parse_utc(rights["decided_at"]) is None:
            fail("rights.decided_at %r is not an ISO-8601 UTC timestamp"
                 % (rights["decided_at"],))

    # --- identity + engine agreement ---------------------------------------
    engine = str(qual.get("engine") or "").strip()
    if not engine:
        fail("engine is missing")
    if active_engine is not None and engine and engine != active_engine:
        fail("ENGINE DISAGREEMENT: route says %r but the active scalar engine "
             "is %r -- a policy must never approve one engine while another "
             "renders" % (engine, active_engine))

    voice_ref_id = str(qual.get("voice_ref_id") or "").strip()
    if not voice_ref_id:
        fail("voice_ref_id is missing")

    bank_row = None
    if bank_lookup is not None and voice_ref_id:
        try:
            bank_row = bank_lookup(voice_ref_id)
        except Exception as exc:          # noqa: BLE001
            fail("bank lookup for %r raised %s" % (voice_ref_id, type(exc).__name__))
        if bank_row is None:
            fail("voice_ref_id %r is not present in the voice bank" % (voice_ref_id,))
        elif isinstance(bank_row, dict):
            bank_engine = str(bank_row.get("engine") or "").strip()
            if engine and bank_engine and bank_engine != engine:
                fail("ENGINE DISAGREEMENT: bank entry %r is engine %r but the "
                     "route claims %r" % (voice_ref_id, bank_engine, engine))

    # --- runtime identity ---------------------------------------------------
    runtime = qual.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime block is missing -- a receipt without model identity "
             "cannot be reproduced")
    else:
        for field in ("engine_impl_version", "model_id", "weight_revision"):
            if not str(runtime.get(field) or "").strip():
                fail("runtime.%s is missing" % field)

    # --- the reference itself ----------------------------------------------
    ref = qual.get("reference")
    if not isinstance(ref, dict):
        fail("reference block is missing")
    else:
        kind = ref.get("kind")
        if kind not in REFERENCE_KINDS:
            fail("reference.kind %r is not a defined kind" % (kind,))
        elif kind == "local_wav":
            src_sha = ref.get("source_ref_sha256")
            if not _is_sha256(src_sha):
                fail("reference.source_ref_sha256 is not 64 hex characters")
            bank_sha = ref.get("bank_ref_sha256")
            if bank_sha is not None and not _is_sha256(bank_sha):
                fail("reference.bank_ref_sha256 is not 64 hex characters")

            path = str(ref.get("absolute_path") or "").strip()
            if not path:
                fail("reference.absolute_path is missing for a local_wav route")
            elif require_local_bytes:
                full = path
                if not os.path.isabs(full) and repo_root:
                    full = os.path.join(repo_root, path)
                if not os.path.isfile(full):
                    fail("reference file does not exist: %s" % (full,))
                elif _is_sha256(src_sha):
                    actual = sha256_of_file(full)
                    if actual is None:
                        fail("reference file could not be read: %s" % (full,))
                    elif actual.lower() != str(src_sha).lower():
                        fail("REFERENCE BYTES DO NOT MATCH the receipt: %s "
                             "hashes to %s but the route claims %s"
                             % (full, actual, str(src_sha).lower()))
        elif kind == "provider_voice":
            # A cloud route has no local bytes to hash. Demanding a file here is
            # exactly the confusion that made gt_algenib look usable as a local
            # IndexTTS2 reference when its ref_sha256 is the literal "cloud".
            for field in ("provider", "provider_voice_id"):
                if not str(ref.get(field) or "").strip():
                    fail("reference.%s is missing for a provider_voice route"
                         % field)

    # --- the audition evidence ---------------------------------------------
    manifest = qual.get("audition_manifest")
    if not isinstance(manifest, dict):
        fail("audition_manifest is missing -- there is no evidence packet")
    else:
        if not str(manifest.get("path") or "").strip():
            fail("audition_manifest.path is missing")
        if not _is_sha256(manifest.get("sha256")):
            fail("audition_manifest.sha256 is not 64 hex characters")

    return RouteValidation(not bad, tuple(bad))


__all__ = [
    "ROUTE_STATUSES", "TECHNICAL_VERDICTS", "RIGHTS_STATUSES",
    "SELECTABLE_ROUTE_STATUS", "SELECTABLE_TECHNICAL_VERDICT",
    "SELECTABLE_RIGHTS_STATUS", "SUPPORTED_ROUTE_CONTRACT_VERSIONS",
    "REFERENCE_KINDS", "RouteValidation", "parse_utc", "sha256_of_file",
    "validate_qualified_voice_route",
]
