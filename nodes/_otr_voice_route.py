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
import logging
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


# --------------------------------------------------------------------------- #
# LIVE RUNTIME FINGERPRINT (voice-identity fix 2026-08-18, PBUG-20260817-09,
# [QA-7])
#
# THE GAP THIS CLOSES. A qualification record stores the runtime it was proved
# under -- `qualification_record.runtime.engine_impl_version`, documented in
# `config/cast_pools.py` as "the sha256 of the adapter plus its worker script",
# the whole point being that CHANGING THE RENDERING CODE CHANGES IT. Nothing
# ever computed the live value, so nothing ever compared them: the field was
# stored, described, and never read. A route qualified by ear in August stayed
# "qualified" through every subsequent edit to the code that produced the sound
# the operator actually approved.
#
# That is not a hypothetical. The voice-identity fix changes the IndexTTS2
# adapter's seed handling and its emotion blend -- exactly the two things a
# listener judges -- so the audition that qualified Lemmy no longer describes
# what this engine does. His record is PRESERVED, unedited; it simply stops
# being SELECTED until a new audition re-qualifies it against the new code.
#
# A MISMATCH DEMOTES, IT NEVER RAISES. THE LAW: an audit may never fail an
# episode; a render degrades. A route that is not selected is the module's own
# established "nothing was selected, so nothing failed" outcome -- the cast row
# takes the ordinary draw and the episode still publishes to `otr/obs/`. Making
# this a validation FAILURE instead would have raised `VoiceRouteError` out of
# CastLock and killed every episode that casts the character.
# --------------------------------------------------------------------------- #

#: Engine -> the repo-relative source files that DEFINE how it renders. Order is
#: part of the recipe. An engine absent from this map has no live fingerprint,
#: so its routes are never demoted on this ground -- silence, not a guess.
#: THE SEED PATH IS PART OF THE RENDERING CODE. The first cut hashed only the
#: adapter and its worker, which left the file that DECIDES WHICH SEED a
#: character draws outside the net -- so a future change to
#: `_resolve_engine_seed` (or to the `_seed_to_int64` reduction it calls) would
#: shift the rendered voice without moving the fingerprint, and a route
#: re-approved today would keep validating through it. That is the exact drift
#: this mechanism exists to catch, so both files are in.
#:
#: THE COST WAS ACCEPTED BEFORE IT WAS MEASURED, AND THE MEASUREMENT REVERSED
#: THE CALL (2026-08-19). The paragraph below is kept because its PRINCIPLE
#: still governs -- "a fingerprint that under-reports is a false claim of
#: proof; one that over-reports is an inconvenience" -- but the shared
#: dispatch file it was written to justify is no longer in the recipe:
#:
#:   ORIGINAL RULING (kept for the record): "THE COST IS REAL AND IT IS
#:   ACCEPTED. The dispatch is a shared file that changes for unrelated
#:   reasons, so a qualified route will be demoted more often than the
#:   strictly-audible minimum. The failure mode is safe and loud -- a warning
#:   plus the ordinary draw, never a raise and never a lost episode -- and
#:   re-qualification is one re-audition."
#:
#: WHAT THE MEASUREMENT FOUND. `_otr_voice_node_common.py` took **19 commits
#: in 60 days** (union across all four files: 22; without it: 8). Of those 19,
#: exactly **ONE** touched the seed path this recipe named as its reason --
#: `62fb6a1f`, the voice-identity fix. So the whole-file hash produced
#: **18 false demotions and 1 true one**, and "more often than the
#: strictly-audible minimum" turned out to mean 19x.
#:
#: AND THE ONE TRUE DEMOTION IS STILL CAUGHT. `62fb6a1f` also edited
#: `eng_indextts2.py`, which remains in the recipe -- so narrowing loses
#: nothing on the only real event in 60 days of history. That is the
#: measurement that decided this, not a preference for convenience.
#:
#: THE RESIDUAL RISK, STATED PLAINLY RATHER THAN MINIMISED: a change to
#: `_resolve_engine_seed` (or the `seed_reduce` it calls) that touches NO
#: engine-specific file would now shift the rendered voice without moving this
#: fingerprint. That is possible and it is exactly what the original ruling
#: feared. It did not occur once in 60 days, because seed work IS voice work
#: and voice work touches the adapter. `weight_revision` and
#: `reference.source_ref_sha256` still gate independently, so a swapped
#: checkpoint or reference wav is caught regardless of this recipe.
#:
#: The cost of the old setting was not theoretical: it de-qualified the shipped
#: Lemmy route when a COMMENT was added, and a legitimate logging fix was
#: reverted rather than pay a GPU re-audition to keep it.
#:
#: KNOWN BLIND SPOT: ``OTR_INDEXTTS2_WORKER`` can point the adapter at a worker
#: script this map does not name, in which case the gate hashes the wrong file
#: and fails OPEN. Nothing in production sets it.
RUNTIME_FINGERPRINT_SOURCES = {
    "indextts2": (
        "nodes/_otr_audio_engines/eng_indextts2.py",
        "scripts/_otr_indextts2_worker.py",
        # nodes/_otr_voice_node_common.py -- REMOVED 2026-08-19, see above.
        # 19 commits/60d, 18 of them false demotions. Its seed path is the
        # stated reason it was here; the one commit that changed that path
        # also touched the adapter above, so the net is still caught.
        "nodes/_otr_resolved_request.py",
    ),
}

#: Hex characters kept from the reduction. Matches the width of the values
#: already stored in the qualification records.
_FINGERPRINT_WIDTH = 16

_LIVE_FINGERPRINT_CACHE: dict = {}


def _repo_root_for_routes() -> str:
    """This repo's root, derived from this module's own location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def live_engine_impl_version(engine: str,
                             repo_root: Optional[str] = None) -> str:
    """Fingerprint of the code that would render ``engine`` RIGHT NOW.

    THE RECIPE, stated so a re-qualification can reproduce it: for each source
    file in :data:`RUNTIME_FINGERPRINT_SOURCES` order, read the bytes, normalize
    CRLF and lone CR to LF, and sha256 them; then sha256 the joined
    ``<repo-relative path>:<digest>`` lines and keep the first
    :data:`_FINGERPRINT_WIDTH` hex characters.

    LINE ENDINGS ARE NORMALIZED ON PURPOSE. Git can hand two clones the same
    source with different newlines, and a fingerprint that moved on checkout
    would demote a perfectly good route for a reason nobody could hear.

    Returns ``""`` when the engine has no recipe or a source file cannot be
    read -- an unknown fingerprint is not evidence of a changed one, so the
    caller declines to judge rather than guessing.
    """
    key = (str(engine or ""), str(repo_root or ""))
    if key in _LIVE_FINGERPRINT_CACHE:
        return _LIVE_FINGERPRINT_CACHE[key]

    paths = RUNTIME_FINGERPRINT_SOURCES.get(str(engine or ""))
    result = ""
    if paths:
        root = repo_root or _repo_root_for_routes()
        lines = []
        for rel in paths:
            try:
                with open(os.path.join(root, rel), "rb") as fh:
                    raw = fh.read()
            except Exception:             # noqa: BLE001 -- see docstring
                lines = []
                break
            normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            lines.append("%s:%s" % (rel, hashlib.sha256(normalized).hexdigest()))
        if lines:
            joined = "\n".join(lines).encode("utf-8")
            result = hashlib.sha256(joined).hexdigest()[:_FINGERPRINT_WIDTH]

    _LIVE_FINGERPRINT_CACHE[key] = result
    return result


def stale_runtime_fingerprint(record: Any, engine: str,
                              repo_root: Optional[str] = None):
    """``(stored, live)`` when a route's runtime no longer matches the code.

    ``None`` means there is nothing to object to: no live recipe for this
    engine, no stored claim, or the two agree. Only a record that CLAIMS a
    runtime fingerprint can contradict one.
    """
    if not isinstance(record, dict):
        return None
    qual = record.get("qualification_record")
    if not isinstance(qual, dict):
        return None
    runtime = qual.get("runtime")
    if not isinstance(runtime, dict):
        return None
    stored = str(runtime.get("engine_impl_version") or "").strip()
    if not stored:
        return None
    live = live_engine_impl_version(engine, repo_root)
    if not live or stored.lower() == live.lower():
        return None
    return (stored, live)


def _default_path_resolver(ref_path: str, repo_root: Optional[str]) -> str:
    """Naive fallback: join a relative path onto ``repo_root``.

    Correct only when references live inside the repo, which on a real install
    they do NOT -- see ``path_resolver`` below.
    """
    if os.path.isabs(ref_path) or not repo_root:
        return ref_path
    return os.path.join(repo_root, ref_path)


def validate_qualified_voice_route(
    record: Any,
    now_utc: datetime,
    *,
    active_engine: Optional[str] = None,
    bank_lookup: Optional[Callable[[str], Optional[dict]]] = None,
    repo_root: Optional[str] = None,
    require_local_bytes: bool = True,
    path_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> RouteValidation:
    """Decide whether ``record`` authorizes rendering a character on a route.

    ``record``        the route dict (``approved_native_routes[engine]`` shape).
    ``now_utc``       aware UTC 'now', injected so tests are not time-bombs.
    ``active_engine`` the scalar engine actually about to render, when known.
                      The three-way agreement (route == active == bank entry) is
                      the check that stops a policy approving one engine while
                      another renders.
    ``bank_lookup``   ``voice_ref_id -> bank row`` when the bank is available.
    ``repo_root``     fallback root for a relative ``ref_path``.
    ``require_local_bytes`` hashing a local reference is the expensive check;
                      it stays ON by default and is only relaxed deliberately.
    ``path_resolver`` maps a bank-relative ``ref_path`` to a real file.

    WHY THE RESOLVER IS INJECTED, AND WHY IT MATTERS. Bank ref paths are written
    ``models/TTS/refs/<engine>/<id>.wav``, which is relative to ComfyUI's MODELS
    ROOT, not to this repo -- on this box that lands in
    ``C:\\ComfyUI-Models\\TTS\\refs\\...``. Joining it onto the repo root instead
    produces a path that has never existed, so a perfectly good route fails with
    "reference file does not exist". That is exactly what happened the first time
    a real qualified route was installed, and nothing before that could have
    caught it: with ``approved_native_routes`` empty, this branch had never once
    run against a live route. The naive join survives as a fallback; callers that
    can resolve properly pass ``_resolve_ref_to_disk``, the SAME resolver the
    render path uses, so validation and rendering cannot disagree about which
    file they are discussing.
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
                full = (path_resolver(path) if path_resolver
                        else _default_path_resolver(path, repo_root))
                full = full or path
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

    * **No approved routes at all** -> ``None``. The policy is dormant. This was
      the shipping state until 2026-08-10, when the G1 audition qualified
      IndexTTS2 and filled the dict; it is still the state on a build whose pack
      cannot be imported. (Comments elsewhere that call the dict empty predate
      that audition -- do not trust them, read the pack.)
    * **Routes exist, but none for the engine actually rendering** -> ``None``.
      Nothing was selected, so nothing failed. Qualifying Lemmy on IndexTTS2 must
      not break every bark render.
    * **Routes exist and the active engine is unknown** -> ``VoiceRouteError``.
      This is the case worth being loud about: we cannot prove agreement, and
      silently skipping a qualified route is the very floor-evidence failure this
      module exists to end.
    * **A route exists for the engine but its stored runtime fingerprint no
      longer matches the live code** -> ``None``, loudly logged [QA-7]. The
      audition proved a sound the current adapter no longer makes, so the record
      is not evidence about this build. It is DEMOTED, not failed: the row takes
      the ordinary draw and the episode still renders and publishes. Restore it
      by re-auditioning and writing a NEW qualification whose runtime matches
      :func:`live_engine_impl_version`.
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
    if not isinstance(record, dict):
        return None

    stale = stale_runtime_fingerprint(record, engine)
    if stale is not None:
        stored, live = stale
        logging.getLogger("OTR").warning(
            "[OTR voice route] route %r on engine %r is NOT SELECTED: it was "
            "qualified against adapter/worker fingerprint %s and this build "
            "renders %s. The record is preserved; re-audition and write a new "
            "qualification whose runtime.engine_impl_version is %s. The cast "
            "row takes the ordinary draw -- the episode still renders.",
            record.get("route_id") or "<unnamed route>", engine, stored, live,
            live)
        return None
    return record


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
    path_resolver: Optional[Callable[[str], Optional[str]]] = None,
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
        path_resolver=path_resolver,
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
# THE PROVISIONAL TIER -- a SIBLING resolver, deliberately not a parameter.
#
# WHY A SIBLING AND NOT A `tier=` ARGUMENT ON THE FUNCTIONS ABOVE. It was
# proposed three times across the campaign and rejected three times for one
# concrete reason: `cast_lock.py` does `dict(policy_claim.voice_route)`
# UNCONDITIONALLY in both stamp branches. Any object of type `PolicyRouteClaim`
# carrying a provisional (routeless) payload is a `TypeError` -- a ComfyUI server
# crash -- one default-argument mistake away. `ProvisionalPolicyClaim` HAS NO
# `voice_route` ATTRIBUTE AT ALL, so it cannot reach those lines even by accident.
# The absence of that field is the safety property, not an omission.
#
# The second reason is the fail-closed path. Threading a tier through
# `select_policy_route` / `resolve_policy_route_claim` puts unaudited rows inside
# the machinery whose entire job is to refuse anything that cannot prove itself.
# Those two functions are byte-unchanged and stay qualified-only.
#
# IT DEGRADES; IT NEVER RAISES. The qualified path is deliberately brutal --
# "there is no fallback" -- because a proven route silently not applying is the
# defect it was built to end. A provisional route inherits none of that: killing a
# render over an unauditioned convenience row inverts the risk this tier exists to
# reduce, and a render must not die. Everything that can go wrong here comes back
# as a CLOSED reason code, the row takes today's ordinary draw, and the ledger
# records `unrouted` plus the reason.
# --------------------------------------------------------------------------- #

#: The three tier values a cast row may carry. `unrouted` is not a failure state
#: in production -- it is the honest name for "the ordinary seeded draw picked
#: this voice", which is what every unclaimed row in the tree does today.
ROUTE_TIER_QUALIFIED = "qualified"
ROUTE_TIER_PROVISIONAL = "provisional"
ROUTE_TIER_UNROUTED = "unrouted"
ROUTE_TIERS = (ROUTE_TIER_QUALIFIED, ROUTE_TIER_PROVISIONAL, ROUTE_TIER_UNROUTED)

#: The cast-row fields this tier owns. Named as constants because two modules
#: write them and three read them, and a string literal in five places is a typo
#: waiting to become a silent no-op.
CAST_ROW_TIER_FIELD = "lemmy_route_tier"
CAST_ROW_ROUTE_ID_FIELD = "lemmy_route_id"
CAST_ROW_REASON_FIELD = "lemmy_route_reason_code"

#: CLOSED set. A degradation reason that is not on this list is a bug in this
#: module, not a new kind of failure -- the ledger consumer downstream reads these
#: as an enumeration and a free-text reason would make it unreadable.
PROVISIONAL_REASON_CODES = (
    "no_policy",                  # no policy dict at all
    "no_provisional_routes",      # the tier is empty -- the shipping state today
    "engine_unresolved",          # no character engine resolved to check against
    "no_route_for_engine",        # the tier exists but names a different engine
    "record_malformed",           # the route record failed its structural checks
    "bank_row_missing",           # the identity names a bank row that is not there
    "bank_row_ambiguous",         # two bank rows share the id on this engine
    "reference_file_missing",     # a local identity's file is not on disk
    "reference_bytes_mismatch",   # it is on disk and it is not the file we meant
    "provider_identity_mismatch",  # bank and receipt disagree about the voice id
    "resolver_error",             # anything unforeseen; never a raise
)

#: Bank hash sentinels. `pending` means nobody hashed it yet, `cloud` means there
#: are no local bytes to hash. Neither is a hash, and treating either as one would
#: fail every kokoro and cloud row on a comparison that was never meaningful.
_NON_HASH_BANK_SENTINELS = frozenset({"", "pending", "cloud", "none", "n/a"})


@dataclass(frozen=True)
class ProvisionalPolicyClaim:
    """A deliberate, unauditioned identity for one cast row on one engine.

    NOTE WHAT IS NOT HERE: there is no ``voice_route`` field, and there never may
    be. ``voice_route`` means "a qualified route was proved" everywhere in this
    tree -- ``resolve_and_verify_reference`` raises on any non-empty one whose
    status is not exactly ``qualified`` -- so a provisional claim that carried one
    would kill every render on these engines. The tier is legible instead through
    ``lemmy_route_tier`` / ``lemmy_route_id`` on the cast row itself, which is what
    persists to the ledger.
    """

    character_key: str
    engine: str
    route_id: str
    identity_kind: str
    identity_id: str
    voice_ref_id: str
    bank_entry: Any                       # VoiceBankEntry -- injected, not imported
    tier: str = ROUTE_TIER_PROVISIONAL


@dataclass(frozen=True)
class ProvisionalRouteDegradation:
    """Why no provisional route applied. Always a code, never an exception."""

    reason_code: str
    detail: str = ""
    engine: str = ""
    route_id: str = ""

    def __bool__(self) -> bool:           # so `if claim:` is False for a failure
        return False


def _lemmy_pools():
    """The cast-pools module, or None. Fail-soft: an unimportable pack means the
    tier is dormant, exactly as it was before this tier existed."""
    try:
        from ..config import cast_pools as _POOLS  # type: ignore
        return _POOLS
    except ImportError:
        try:
            from config import cast_pools as _POOLS  # type: ignore
            return _POOLS
        except ImportError:
            return None


def select_provisional_route(policy: Any, active_engine: Optional[str]):
    """The provisional record for ``active_engine``, or a degradation reason.

    Unlike ``select_policy_route`` this NEVER raises on an unresolved engine. The
    qualified selector is loud there because skipping a proven route silently is
    the failure it exists to prevent; there is no such stake here, and a raise
    would take down a render over a convenience row.

    QUALIFIED WINS. An engine present in BOTH dicts resolves qualified and never
    reaches this function -- the caller consults the tiers in order. This is
    checked here as well, belt and braces, because "the caller will do it" is how
    a second caller gets it wrong.
    """
    if not isinstance(policy, dict):
        return ProvisionalRouteDegradation("no_policy", "policy is not a dict")

    routes = policy.get("provisional_native_routes")
    if not isinstance(routes, dict) or not routes:
        return ProvisionalRouteDegradation(
            "no_provisional_routes", "the provisional tier is empty")

    engine = str(active_engine or "").strip()
    if not engine:
        return ProvisionalRouteDegradation(
            "engine_unresolved",
            "no character voice engine resolved, so no route can be matched")

    qualified = policy.get("approved_native_routes")
    if isinstance(qualified, dict) and engine in qualified:
        return ProvisionalRouteDegradation(
            "no_route_for_engine",
            "engine %r has a QUALIFIED route -- the provisional tier is not "
            "consulted for it" % (engine,), engine=engine)

    record = routes.get(engine)
    if not isinstance(record, dict):
        return ProvisionalRouteDegradation(
            "no_route_for_engine",
            "the provisional tier names no route for engine %r" % (engine,),
            engine=engine)
    return record


def resolve_provisional_route_claim(
    policy: Any,
    active_engine: Optional[str],
    *,
    bank_entries: Sequence,
    repo_root: Optional[str] = None,
    path_resolver: Optional[Callable[[str], Optional[str]]] = None,
    require_local_bytes: bool = True,
):
    """Resolve the provisional identity for ``active_engine``.

    Returns a ``ProvisionalPolicyClaim`` or a ``ProvisionalRouteDegradation``.
    It does not raise: every failure mode is a reason code, and the caller falls
    through to the ordinary draw.

    Three identity kinds, each checked for what it actually claims:

    * ``local_wav`` -- a clone reference. The bank row must exist, be unique on
      this engine, and its file must be on disk with matching bytes when the bank
      declares a real hash. Restricted by an explicit engine allowlist, because a
      reference approved for LOCAL clone use must never be handed to a provider.
    * ``bank_voice_id`` -- kokoro's ``.pt`` voice. Same bank checks; the file must
      exist, and its bytes are compared only when the bank has a real hash for
      them (``bm_george`` is recorded ``pending``).
    * ``provider_voice`` -- a cloud voice id. No local file exists, and demanding
      one is the category error that once made ``gt_algenib`` look like a usable
      local reference. The bank row's provider id must agree with the receipt's.
    """
    record = select_provisional_route(policy, active_engine)
    if isinstance(record, ProvisionalRouteDegradation):
        return record

    engine = str(active_engine or "").strip()
    route_id = str(record.get("route_id") or "<unnamed provisional route>")

    character_key = policy_character_key(policy)
    if not character_key:
        return ProvisionalRouteDegradation(
            "record_malformed",
            "the policy names no character_key, so a route claims nobody",
            engine=engine, route_id=route_id)

    pools = _lemmy_pools()
    if pools is None or not hasattr(pools, "provisional_route_problems"):
        return ProvisionalRouteDegradation(
            "resolver_error",
            "the cast-pools pack could not be imported, so the record cannot be "
            "structurally checked", engine=engine, route_id=route_id)
    problems = pools.provisional_route_problems(record, engine=engine)
    if problems:
        return ProvisionalRouteDegradation(
            "record_malformed", "; ".join(problems),
            engine=engine, route_id=route_id)

    receipt = record.get("provisional_receipt") or {}
    identity_kind = str(receipt.get("identity_kind") or "")
    identity_id = str(receipt.get("identity_id") or "")
    voice_ref_id = str(record.get("voice_ref_id") or identity_id).strip()

    rows = [e for e in (bank_entries or [])
            if str(getattr(e, "engine", "")) == engine
            and str(getattr(e, "voice_ref_id", "")) == voice_ref_id]
    if len(rows) > 1:
        return ProvisionalRouteDegradation(
            "bank_row_ambiguous",
            "the voice bank has %d entries for %r on engine %r, so it cannot say "
            "which one this route means" % (len(rows), voice_ref_id, engine),
            engine=engine, route_id=route_id)
    if not rows:
        return ProvisionalRouteDegradation(
            "bank_row_missing",
            "voice_ref_id %r has no entry on engine %r in the active voice bank"
            % (voice_ref_id, engine), engine=engine, route_id=route_id)
    bank_entry = rows[0]

    if identity_kind == "provider_voice":
        want = str(receipt.get("provider_voice_id") or "").strip()
        have = str(getattr(bank_entry, "provider_voice_id", "") or "").strip()
        if want and have and want != have:
            return ProvisionalRouteDegradation(
                "provider_identity_mismatch",
                "the route claims provider voice %r but bank row %r carries %r"
                % (want, voice_ref_id, have), engine=engine, route_id=route_id)
        if want and not have:
            return ProvisionalRouteDegradation(
                "provider_identity_mismatch",
                "the route claims provider voice %r but bank row %r carries none"
                % (want, voice_ref_id), engine=engine, route_id=route_id)
    else:
        ref_path = str(getattr(bank_entry, "ref_path", "") or "")
        if require_local_bytes:
            full = (path_resolver(ref_path) if path_resolver
                    else _default_path_resolver(ref_path, repo_root)) or ref_path
            if not full or not os.path.isfile(full):
                return ProvisionalRouteDegradation(
                    "reference_file_missing",
                    "bank row %r points at %s, which is not a file on this box"
                    % (voice_ref_id, full or ref_path),
                    engine=engine, route_id=route_id)
            claimed = str(getattr(bank_entry, "ref_sha256", "") or "").strip()
            if claimed.lower() not in _NON_HASH_BANK_SENTINELS and _is_sha256(claimed):
                actual = sha256_of_file(full)
                if actual is None:
                    return ProvisionalRouteDegradation(
                        "reference_file_missing",
                        "bank row %r names %s, which could not be read"
                        % (voice_ref_id, full),
                        engine=engine, route_id=route_id)
                if actual.lower() != claimed.lower():
                    return ProvisionalRouteDegradation(
                        "reference_bytes_mismatch",
                        "%s hashes to %s but bank row %r claims %s"
                        % (full, actual, voice_ref_id, claimed.lower()),
                        engine=engine, route_id=route_id)

    return ProvisionalPolicyClaim(
        character_key=character_key,
        engine=engine,
        route_id=str(record.get("route_id") or ""),
        identity_kind=identity_kind,
        identity_id=identity_id,
        voice_ref_id=voice_ref_id,
        bank_entry=bank_entry,
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
    path_resolver: Optional[Callable[[str], Optional[str]]] = None,
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
            full = (path_resolver(ref_path) if path_resolver
                    else _default_path_resolver(ref_path, repo_root)) or ref_path
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

    # LAST GATE: A LEDGER LOCKED UNDER THE OLD CODE MUST NOT RE-ASSERT A
    # WITHDRAWN CLAIM. `select_policy_route` only runs at CastLock time, so a
    # ledger frozen while the route was still qualified carries the route dict
    # on its own cast row, and this path used to trust it outright. Re-rendering
    # that ledger under changed engine code would stamp `voice_route_id` and the
    # old `qualification_record_id` into per-line receipts describing audio the
    # audition never heard -- the evidence-shaped field this contract exists to
    # refuse, re-asserted automatically.
    #
    # DELIBERATELY LAST, AFTER EVERY STRUCTURAL PROOF ABOVE. A MALFORMED route
    # must still RAISE: "an unproved route raises rather than rendering" is the
    # existing contract, and a broken route dict is a real ledger defect worth
    # being loud about. Only a route that proved everything it can, and is
    # merely no longer CURRENT, degrades here.
    #
    # Degrade, never raise: the row keeps its declared bank reference and
    # renders normally -- it simply stops claiming a proof. THE LAW.
    stale = stale_runtime_fingerprint(
        {"qualification_record": {"runtime": runtime}},
        engine or active, repo_root)
    if stale is not None:
        stored, live = stale
        logging.getLogger("OTR").warning(
            "[OTR voice route] cast row %r carries voice_route %r qualified "
            "against runtime %s, but this build renders %s -- rendering as an "
            "ordinary bank reference and stamping NO route receipt. Re-audition "
            "to restore the claim.",
            cast_row.get("char_id") or cast_row.get("name"), route_id,
            stored, live)
        return LEGACY_REFERENCE

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
    "RUNTIME_FINGERPRINT_SOURCES", "live_engine_impl_version",
    "stale_runtime_fingerprint",
    "validate_qualified_voice_route",
    "VoiceRouteError", "PolicyRouteClaim", "policy_character_key",
    "cast_row_matches_policy", "select_policy_route", "resolve_policy_route_claim",
    "ResolvedReference", "LEGACY_REFERENCE", "LEGACY_REFERENCE_IDENTITY",
    "resolve_and_verify_reference", "_default_path_resolver",
    "ROUTE_TIER_QUALIFIED", "ROUTE_TIER_PROVISIONAL", "ROUTE_TIER_UNROUTED",
    "ROUTE_TIERS", "CAST_ROW_TIER_FIELD", "CAST_ROW_ROUTE_ID_FIELD",
    "CAST_ROW_REASON_FIELD", "PROVISIONAL_REASON_CODES",
    "ProvisionalPolicyClaim", "ProvisionalRouteDegradation",
    "select_provisional_route", "resolve_provisional_route_claim",
]
