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


# --------------------------------------------------------------------------- #
# Plan 5.2 -- the EXPLICIT RE-PIN.
#
# WHAT THIS IS NOT. It is not a change to the generic seeded selector, and it
# must never become one. Lemmy was never redrawn per episode: 33 of the 35
# reference-carrying LEMMY ledger rows name the SAME reference, because every one of
# them had `meta.episode_seed=None` and the selector therefore derived an
# identical seed. He was ACCIDENTALLY PINNED. The defect is that the incumbent
# cannot PROVE the configured floor -- not that the selector misbehaves. So the
# repair is a narrow, qualified, provable claim on ONE named row, and every
# unclaimed row keeps drawing exactly as it does today.
# --------------------------------------------------------------------------- #


class VoiceRouteError(RuntimeError):
    """A SELECTED policy route could not be honoured.

    Deliberately its own type so no caller can fold it into a generic
    ``VoiceCastingError`` rescue and quietly cast someone else. A route that was
    selected and then failed is a STOP, never a fallback -- the whole point of
    qualifying a voice is that an unqualified one does not silently take its
    place.
    """


def policy_character_key(policy: Any) -> str:
    """The normalized cast key this policy claims, or ``""`` if it claims none."""
    if not isinstance(policy, dict):
        return ""
    return str(policy.get("character_key") or "").strip().lower()


def cast_row_matches_policy(entry: Any, character_key: str) -> bool:
    """True when this cast row is the one the policy claims.

    Matches the row NAME or its char_id, both normalized. char_id is positional
    (`c02` today), so name is usually what actually hits -- but a ledger that
    does spell the char_id `lemmy` must not slip through either.
    """
    if not character_key or not isinstance(entry, dict):
        return False
    key = character_key.strip().lower()
    for field in ("name", "char_id"):
        if str(entry.get(field) or "").strip().lower() == key:
            return True
    return False


def select_policy_route(policy: Any, active_engine: Optional[str]) -> Optional[dict]:
    """The approved route this policy selects for ``active_engine``, or None.

    Three outcomes, and the difference between them is the whole contract:

    * **No approved routes at all** -> ``None``. The policy is dormant. This is
      the state shipping today (``approved_native_routes`` is ``{}``), so the
      re-pin is inert until an operator audition fills it in.
    * **Routes exist, but none for the engine actually rendering** -> ``None``.
      Nothing was selected, so nothing failed. Qualifying Lemmy on IndexTTS2 must
      not break every bark render.
    * **Routes exist and the active engine is unknown** -> ``VoiceRouteError``.
      This is the case worth being loud about: we cannot prove agreement, and
      silently skipping a qualified route is the very floor-evidence failure this
      module exists to end.
    """
    if not isinstance(policy, dict):
        return None
    routes = policy.get("approved_native_routes")
    if not isinstance(routes, dict) or not routes:
        return None

    engine = str(active_engine or "").strip()
    if not engine:
        raise VoiceRouteError(
            "voice policy %r approves %d native route(s) but the active "
            "character voice engine could not be resolved -- refusing to skip a "
            "qualified route without proving it does not apply"
            % (policy.get("policy_version"), len(routes)))

    record = routes.get(engine)
    return record if isinstance(record, dict) else None


@dataclass(frozen=True)
class PolicyRouteClaim:
    """A proven claim on one cast row: which row, which reference, what receipt."""

    character_key: str
    engine: str
    voice_ref_id: str
    bank_entry: Any                       # VoiceBankEntry -- injected, not imported
    voice_route: dict                     # the immutable identity stamped on the row


def _route_payload(record: dict, engine: str, voice_ref_id: str) -> dict:
    """The immutable route identity stored on the cast row (plan 5.2)."""
    qual = record.get("qualification_record") or {}
    ref = qual.get("reference") or {}
    runtime = qual.get("runtime") or {}
    return {
        "route_id": str(record.get("route_id") or ""),
        "route_contract_version": record.get("route_contract_version"),
        "status": qual.get("status"),
        "engine": engine,
        "voice_ref_id": voice_ref_id,
        "reference_kind": ref.get("kind"),
        "ref_path": str(ref.get("absolute_path") or ""),
        "source_ref_sha256": str(ref.get("source_ref_sha256") or ""),
        "qualification_record_id": str(qual.get("record_id") or ""),
        "runtime": {
            "model_id": str(runtime.get("model_id") or ""),
            "engine_impl_version": str(runtime.get("engine_impl_version") or ""),
            "weight_revision": str(runtime.get("weight_revision") or ""),
        },
    }


def resolve_policy_route_claim(
    policy: Any,
    active_engine: Optional[str],
    now_utc: datetime,
    *,
    bank_entries: Sequence,
    repo_root: Optional[str] = None,
) -> Optional[PolicyRouteClaim]:
    """Prove the selected route, or raise. ``None`` means nothing was selected.

    ``bank_entries`` is the loaded voice bank, injected so this module stays
    stdlib-only and CastLock's cold path never drags a model in.
    """
    record = select_policy_route(policy, active_engine)
    if record is None:
        return None

    engine = str(active_engine or "").strip()
    character_key = policy_character_key(policy)
    if not character_key:
        raise VoiceRouteError(
            "voice policy %r selected an approved route on engine %r but names "
            "no character_key -- a route that claims nobody cannot be applied"
            % (policy.get("policy_version"), engine))

    rows = [e for e in (bank_entries or [])
            if str(getattr(e, "engine", "")) == engine]

    def _hits(voice_ref_id: str) -> list:
        return [e for e in rows
                if str(getattr(e, "voice_ref_id", "")) == voice_ref_id]

    route_id = str(record.get("route_id") or "<unnamed route>")
    qual = record.get("qualification_record")
    claimed_ref_id = str((qual or {}).get("voice_ref_id") or "").strip() \
        if isinstance(qual, dict) else ""

    # UNIQUENESS IS CHECKED FIRST, and separately, on purpose. The validator
    # wraps bank_lookup in a broad except and would fold this into a generic
    # "bank lookup raised" reason -- but ambiguity is a REJECT with a specific
    # cause, not a coin flip: two rows sharing one id on one engine means the
    # bank cannot say which bytes were auditioned.
    if claimed_ref_id:
        found = _hits(claimed_ref_id)
        if len(found) > 1:
            raise VoiceRouteError(
                "SELECTED voice route %r: the voice bank has %d entries for "
                "voice_ref_id %r on engine %r -- a qualified route needs "
                "exactly one, and there is no fallback"
                % (route_id, len(found), claimed_ref_id, engine))

    def _lookup(voice_ref_id: str):
        found = _hits(voice_ref_id)
        return found[0] if found else None

    validation = validate_qualified_voice_route(
        record, now_utc,
        active_engine=engine,
        bank_lookup=_lookup,
        repo_root=repo_root,
        require_local_bytes=True,
    )
    if not validation.ok:
        raise VoiceRouteError(
            "SELECTED voice route %r for %r on engine %r FAILED qualification "
            "and there is no fallback: %s"
            % (route_id, character_key, engine, validation.summary))

    voice_ref_id = claimed_ref_id
    bank_entry = _lookup(voice_ref_id)
    if bank_entry is None:
        # validate_qualified_voice_route already checks this; belt and braces,
        # because the next line dereferences it.
        raise VoiceRouteError(
            "SELECTED voice route %r names voice_ref_id %r, which has no entry "
            "on engine %r in the active voice bank"
            % (route_id, voice_ref_id, engine))

    return PolicyRouteClaim(
        character_key=character_key,
        engine=engine,
        voice_ref_id=voice_ref_id,
        bank_entry=bank_entry,
        voice_route=_route_payload(record, engine, voice_ref_id),
    )


# --------------------------------------------------------------------------- #
# Plan 5.1/5.3 -- what the VOICE NODE resolves, per cast row, before it builds
# either request.
# --------------------------------------------------------------------------- #

#: A row with no ``voice_route``. Every field empty/zero, which is exactly what
#: keeps a legacy line's cache_key byte-identical after the schema grew.
LEGACY_REFERENCE_IDENTITY = {
    "route_id": "",
    "route_contract_version": 0,
    "qualification_record_id": "",
    "weight_revision": "",
    "source_ref_sha256": "",
}


@dataclass(frozen=True)
class ResolvedReference:
    """What the request builders need from a cast row's route, if it has one."""

    is_policy_route: bool
    route_id: str = ""
    route_contract_version: int = 0
    qualification_record_id: str = ""
    weight_revision: str = ""
    source_ref_sha256: str = ""
    ref_path: str = ""
    reference_kind: str = ""

    def request_fields(self) -> dict:
        return {
            "route_id": self.route_id,
            "route_contract_version": self.route_contract_version,
            "qualification_record_id": self.qualification_record_id,
            "weight_revision": self.weight_revision,
            "source_ref_sha256": self.source_ref_sha256,
        }


LEGACY_REFERENCE = ResolvedReference(is_policy_route=False)


def resolve_and_verify_reference(
    cast_row: Any,
    active_engine: Optional[str],
    *,
    bank_lookup: Optional[Callable[[str], Optional[dict]]] = None,
    repo_root: Optional[str] = None,
    verify_bytes: bool = True,
) -> ResolvedReference:
    """Resolve a cast row's reference identity, proving a route if it has one.

    THREE ROW SHAPES, and the plan is emphatic that they stay distinguishable:

    * **legacy, no route** -> ``LEGACY_REFERENCE``. Existing resolver behaviour
      is preserved and can never be confused with a newly qualified route. The
      pair ``(voice_engine, voice_ref_id)`` on such a row is a declared BANK
      REFERENCE, not a claim that a renderer ever ran.
    * **local_wav route** -> the selected route is re-proved here, at the point
      of use, and its ``source_ref_sha256`` is supplied to the request. The
      bytes are re-hashed because a receipt proved at cast time says nothing
      about the file five minutes later.
    * **provider_voice route** -> validated on route/provider/model/voice fields
      WITHOUT pretending a cloud URI is a local file. This is the confusion that
      made ``gt_algenib`` look usable as a local IndexTTS2 reference when its
      ``ref_sha256`` is the literal string ``cloud``.

    ``verify_bytes`` is only ever relaxed by a caller that already hashed the
    same file for the same route in this render -- never to skip the check.
    """
    if not isinstance(cast_row, dict):
        return LEGACY_REFERENCE
    route = cast_row.get("voice_route")
    if not isinstance(route, dict) or not route:
        return LEGACY_REFERENCE

    route_id = str(route.get("route_id") or "<unnamed route>")
    engine = str(route.get("engine") or "").strip()
    active = str(active_engine or "").strip()

    # THE TRIPLE, first: route == active scalar == bank entry. A row carrying a
    # route for an engine that is not the one about to render is not a reason to
    # quietly render anyway -- it means the ledger and the graph disagree about
    # who is speaking.
    if not active:
        raise VoiceRouteError(
            "cast row %r carries voice_route %r but the active voice engine is "
            "unknown -- a route cannot be proved against nothing"
            % (cast_row.get("char_id") or cast_row.get("name"), route_id))
    if engine != active:
        raise VoiceRouteError(
            "ENGINE DISAGREEMENT on cast row %r: voice_route %r is for engine "
            "%r but %r is rendering"
            % (cast_row.get("char_id") or cast_row.get("name"), route_id,
               engine, active))

    version = route.get("route_contract_version")
    if version not in SUPPORTED_ROUTE_CONTRACT_VERSIONS:
        raise VoiceRouteError(
            "voice_route %r declares route_contract_version %r, which this "
            "build does not understand (known: %r)"
            % (route_id, version, list(SUPPORTED_ROUTE_CONTRACT_VERSIONS)))

    if route.get("status") != SELECTABLE_ROUTE_STATUS:
        raise VoiceRouteError(
            "voice_route %r has status %r, not %r -- it may not render"
            % (route_id, route.get("status"), SELECTABLE_ROUTE_STATUS))

    voice_ref_id = str(route.get("voice_ref_id") or "").strip()
    if not voice_ref_id:
        raise VoiceRouteError("voice_route %r names no voice_ref_id" % (route_id,))

    if bank_lookup is not None:
        try:
            bank_row = bank_lookup(voice_ref_id)
        except VoiceRouteError:
            raise
        except Exception as exc:              # noqa: BLE001
            raise VoiceRouteError(
                "voice_route %r: bank lookup for %r raised %s"
                % (route_id, voice_ref_id, type(exc).__name__))
        if bank_row is None:
            raise VoiceRouteError(
                "voice_route %r names voice_ref_id %r, which is not in the "
                "active voice bank" % (route_id, voice_ref_id))
        bank_engine = str(
            (bank_row.get("engine") if isinstance(bank_row, dict)
             else getattr(bank_row, "engine", "")) or "").strip()
        if bank_engine and bank_engine != engine:
            raise VoiceRouteError(
                "ENGINE DISAGREEMENT: bank entry %r is engine %r but "
                "voice_route %r claims %r"
                % (voice_ref_id, bank_engine, route_id, engine))

    kind = route.get("reference_kind")
    if kind not in REFERENCE_KINDS:
        raise VoiceRouteError(
            "voice_route %r has reference_kind %r, which is not a defined kind"
            % (route_id, kind))

    source_sha = str(route.get("source_ref_sha256") or "")
    ref_path = str(route.get("ref_path") or "")

    if kind == "local_wav":
        if not _is_sha256(source_sha):
            raise VoiceRouteError(
                "voice_route %r: source_ref_sha256 is not 64 hex characters"
                % (route_id,))
        if not ref_path:
            raise VoiceRouteError(
                "voice_route %r: ref_path is missing for a local_wav route"
                % (route_id,))
        if verify_bytes:
            full = ref_path
            if not os.path.isabs(full) and repo_root:
                full = os.path.join(repo_root, ref_path)
            if not os.path.isfile(full):
                raise VoiceRouteError(
                    "voice_route %r: reference file does not exist: %s"
                    % (route_id, full))
            actual = sha256_of_file(full)
            if actual is None:
                raise VoiceRouteError(
                    "voice_route %r: reference file could not be read: %s"
                    % (route_id, full))
            if actual.lower() != source_sha.lower():
                raise VoiceRouteError(
                    "voice_route %r: REFERENCE BYTES DO NOT MATCH the receipt "
                    "-- %s hashes to %s but the route claims %s"
                    % (route_id, full, actual, source_sha.lower()))
    else:
        # provider_voice: there are no local bytes, and demanding some is the
        # exact category error this branch exists to refuse.
        source_sha = ""
        ref_path = ""

    runtime = route.get("runtime") or {}
    return ResolvedReference(
        is_policy_route=True,
        route_id=str(route.get("route_id") or ""),
        route_contract_version=int(version),
        qualification_record_id=str(route.get("qualification_record_id") or ""),
        weight_revision=str(runtime.get("weight_revision") or ""),
        source_ref_sha256=source_sha,
        ref_path=ref_path,
        reference_kind=str(kind),
    )


__all__ = [
    "ROUTE_STATUSES", "TECHNICAL_VERDICTS", "RIGHTS_STATUSES",
    "SELECTABLE_ROUTE_STATUS", "SELECTABLE_TECHNICAL_VERDICT",
    "SELECTABLE_RIGHTS_STATUS", "SUPPORTED_ROUTE_CONTRACT_VERSIONS",
    "REFERENCE_KINDS", "RouteValidation", "parse_utc", "sha256_of_file",
    "validate_qualified_voice_route",
    "VoiceRouteError", "PolicyRouteClaim", "policy_character_key",
    "cast_row_matches_policy", "select_policy_route", "resolve_policy_route_claim",
    "ResolvedReference", "LEGACY_REFERENCE", "LEGACY_REFERENCE_IDENTITY",
    "resolve_and_verify_reference",
]
