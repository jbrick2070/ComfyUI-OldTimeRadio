"""Cloud media control plane -- S0 of the cloud engine lanes.

Build plan (4-round roundtable, converged 2026-07-02, operator amendment:
audio reactivity DEFAULT-ON for all video roles).

This module is the ONE allowed process singleton for cloud media state
(a lock-guarded session table). "No module globals" in the plan means:
no credential/budget state in ADAPTER modules -- adapters fetch the
session from here. The frozen AudioEngine protocol is never touched.

Cold-import-clean: stdlib only. No torch / comfy imports at module
scope. No network code lives here -- invocation arrives with the
partner_nodes.yaml pinning chunk and resolves its session through this
table.

Env surface (all read per session-create, never mutated mid-run):
  (no credential env)             auth is ONLY the api_key_comfy_org hidden
                                  input ComfyUI injects into the OTR host
                                  node (app sign-in, or a headless
                                  submitter's extra_data). Rip 2026-09-19.
  OTR_CLOUD_MEDIA_BUDGET_USD      optional per-run USD ceiling. UNSET =
                                  no local ceiling -- the wallet 402 is
                                  the stop (operator 2026-09-16: do not
                                  hole a published episode with a fake
                                  cap). An EXPLICIT 0 = every reserve
                                  fails closed with `budget` (spend-off).

NOTE (operator directive 2026-07-02 evening): the OTR_ENABLE_COMFY_CLOUD_MEDIA
opt-in flag was REMOVED -- same clean break as the OpenRouter lane's C6
(OTR_ENABLE_OPENROUTER removal). Cloud rows run iff the user PICKS a
"Comfy Cloud" entry in the video/image/TTS dropdowns; picking one without
credentials fails LOUD at auth resolution naming all three sources.
  OTR_CLOUD_MEDIA_CACHE_DIR       cache root override.
  OTR_CLOUD_MAX_CONCURRENCY_<ID>  per-provider semaphore size.
  OTR_VIDEO_MUTE_OK_ROLES         comma list of roles allowed mute video
                                  (operator amendment: default EMPTY).
"""
from __future__ import annotations

import enum
import json
import re
import threading
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

#: This module predates a logger and used only the print-based
#: ``_LEAK_LOG`` below, which is named for leak reporting specifically.
#: The billing-ledger migration needs ordinary INFO/WARNING, so it gets
#: the pack's normal logger rather than borrowing one named for
#: something else.
log = logging.getLogger("OTR.cloud_media")
from typing import Callable, Optional

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import env as otr_env  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import env as otr_env  # type: ignore  # _otr_shared/ on sys.path

__all__ = [
    "CloudErrorCode",
    "CloudMediaError",
    "CloudAuth",
    "CostQuote",
    "ReservationState",
    "CloudMediaSession",
    "mute_ok_roles",
    "resolve_auth",
    "NO_CREDENTIAL_HINT",
    "stash_prompt_api_key",
    "get_or_create_session",
    "peek_session",
    "teardown_session",
    "provider_semaphore_size",
    "normalize_provider_id",
    "resolve_cache_root",
    "SESSION_SWEEP_MAX_AGE_S",
    "is_cloud_budget_error",
    "is_wallet_empty_message",
    "is_auth_failure_message",
    "is_content_policy_message",
    "is_content_refusal",
    "cloud_job_failure_code",
    "JOB_SCOPED_CODES",
    "RUN_SCOPED_CODES",
]

# ---------------------------------------------------------------------------
# Canonical error taxonomy (pass04 sec 1 -- ONE spelling everywhere)
# ---------------------------------------------------------------------------


class CloudErrorCode(str, enum.Enum):
    MALFORMED_CONFIG = "malformed_config"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    INCOMPATIBLE_PROFILE = "incompatible_profile"
    GATED_BY_FLAG = "gated_by_flag"
    AUTH = "auth"
    BUDGET = "budget"
    RETRYABLE_TRANSPORT = "retryable_transport"
    PROVIDER_REJECTED = "provider_rejected"
    CONTENT_REFUSED = "content_refused"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    CORRUPT_OUTPUT = "corrupt_output"
    ORPHANED_JOB = "orphaned_job"


class CloudMediaError(RuntimeError):
    """Fail-closed named error. Carries the canonical code; message names
    the missing piece and, where an operator knob exists, the knob."""

    def __init__(self, code: CloudErrorCode | str, detail: str = ""):
        self.code = CloudErrorCode(code)
        msg = f"cloud media: {self.code.value}"
        if detail:
            msg += f" -- {detail}"
        super().__init__(msg)


def is_wallet_empty_message(text: str) -> bool:
    """True when the provider refused because the Comfy account is empty.

    Live 2026-09-16: Credits returned ``HTTP 402 Payment Required``. The
    partner LTX path used to map unknown HTTP errors to
    ``PROVIDER_REJECTED``, so fan-out halt has to recognize the
    wallet-empty text too. ``402`` is matched as a whole token so a job
    id containing 1402 does not trip this.
    """
    blob = str(text or "").lower()
    if "payment required" in blob or "insufficient credit" in blob:
        return True
    if re.search(r"\b402\b", blob):
        return True
    return False


def is_auth_failure_message(text: str) -> bool:
    """True when the provider refused the key, not the job.

    ``401`` is a whole token so a job id containing 1401 does not map
    AUTH. Wallet-empty (402) is a different matcher and must win first
    at the call site -- a 402 body that also says unauthorized is still
    an empty account.
    """
    blob = str(text or "").lower()
    if "unauthorized" in blob or "forbidden" in blob:
        return True
    if re.search(r"\b401\b", blob):
        return True
    return False


#: Provider language that means "the policy gate declined this prompt".
#: Compared against an ALPHANUMERIC-ONLY normalization of both sides, because
#: five providers spell one verdict five ways and the separators are the only
#: thing that differs: LTX ``content_filtered_error`` (proven live 2026-09-16),
#: BFL ``Content Moderated`` / ``Request Moderated``, Gemini
#: ``IMAGE_PROHIBITED_CONTENT``, ByteDance
#: ``OutputAudioSensitiveContentDetected``, Comfy's own
#: ``image_content_policy_violation``. Matching the normalized form covers all
#: of them with one needle each instead of four spellings each.
#:
#: EVERY NEEDLE HERE IS SELF-ANCHORED -- it names content AND a verdict in the
#: same token. That rule is what keeps ordinary faults out. Four needles were
#: cut on 2026-09-16 review for failing it, and the reasons are worth keeping:
#: ``policy restriction`` matches a corporate proxy's "blocked by policy
#: restriction", which is systemic and would floor EVERY beat; ``safety
#: system`` matches "our safety system is temporarily unavailable", a
#: retryable outage; ``safety filter`` matches a FileNotFoundError for a model
#: file named ``safety_filter_v2.pth``; ``sensitive content`` is ordinary
#: English that a prompt echoed into an error body can carry. A needle that
#: can match a crash is worse than a needle that misses a refusal -- a miss
#: fails LOUD, which is merely the old behavior, while a false match launders
#: a broken render into a publishable "degraded" episode.
_CONTENT_POLICY_NEEDLES = (
    "contentfiltered",
    "contentpolicy",
    "contentmoderat",        # ...moderation / ...moderated
    "requestmoderated",
    "prohibitedcontent",
    "sensitivecontentdetected",
)


def _alnum(text) -> str:
    """Lowercase, with every separator removed -- see the needle note."""
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def is_content_policy_message(text: str) -> bool:
    """True when the provider refused the PROMPT, not the request.

    Distinct from :func:`is_auth_failure_message` (refused the key) and
    :func:`is_wallet_empty_message` (refused the charge). Those two say
    nothing about the beat; this one says this beat, as written, will never
    render on this provider, so a retry is money spent on the same verdict.
    """
    blob = _alnum(text)
    return any(needle in blob for needle in _CONTENT_POLICY_NEEDLES)


def _cause_chain(exc: BaseException | None):
    """Yield ``exc`` and everything it was raised from, once each."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        cur = nxt if isinstance(nxt, BaseException) else None


def is_content_refusal(exc: BaseException | None) -> bool:
    """True when this failure is a provider content-policy refusal.

    TWO PASSES, AND THE ORDER IS THE WHOLE CONTRACT. The still funnel's
    sanctioned gap (2026-08-28) is minted from a RECORDED FACT -- a receipt row
    written where the refusal happened -- and ``otr_video_render_batch`` states
    the standard as "AN ABSENCE IS NOT A SANCTION". A substring match on an
    exception string is an absence being read as a sanction, so prose may only
    speak where no verdict was recorded at all.

    PASS 1 walks the chain for a stamped :class:`CloudErrorCode`.
    ``CONTENT_REFUSED`` anywhere in the chain is the answer. Any OTHER stamped
    code VETOES -- the invoke boundary already adjudicated that failure with
    the provider's own words in hand, and prose sniffing one layer up must not
    overturn it. This is not hypothetical: ``render_shot`` wraps every failure
    in a ``RenderError`` whose message CONCATENATES the whole inner chain, so
    without the veto a ``RETRYABLE_TRANSPORT`` or ``TIMEOUT`` carrying refusal
    words in its quoted body would floor a beat the boundary called retryable.

    PASS 2 runs only when nothing in the chain carries a code -- a direct BYO
    engine, or a partner path that raised before ``_map_exception`` could
    stamp it. Prose is the last resort, never the first reading.
    """
    stamped: str | None = None
    for cur in _cause_chain(exc):
        code = getattr(cur, "code", None)
        value = getattr(code, "value", None)
        if value is None:
            continue
        if value == CloudErrorCode.CONTENT_REFUSED.value:
            return True
        if stamped is None:
            stamped = value
    if stamped is not None:
        return False
    return any(is_content_policy_message(str(cur)) for cur in _cause_chain(exc))


#: Cloud failures that are ABOUT ONE JOB. The provider answered (or failed to)
#: for THIS request; the next request is unaffected, so the beat can be floored
#: and the episode can go on. Operator directive 2026-09-16: "any of them could
#: easily get a failed output; a failed output on a cloud video or still should
#: not break the system."
JOB_SCOPED_CODES = frozenset({
    CloudErrorCode.CONTENT_REFUSED,
    CloudErrorCode.PROVIDER_REJECTED,
    CloudErrorCode.TIMEOUT,
    CloudErrorCode.RETRYABLE_TRANSPORT,
    CloudErrorCode.CORRUPT_OUTPUT,
    CloudErrorCode.ORPHANED_JOB,
})

#: Cloud failures that are ABOUT THE RUN. Flooring these would floor EVERY
#: beat and publish an empty episode as "degraded", which is the laundering
#: the sanctioned-gap channel exists to prevent -- so they stay LOUD.
#: ``BUDGET`` is loud in its own way (it halts and floors what it already
#: paid for); ``AUTH`` and the four config codes mean nothing will ever
#: render and the operator must fix the key or the config.
#: ``INTERRUPTED`` is deliberately in NEITHER set: a cancel is the operator
#: saying stop, and turning that into 40 floored beats would be obscene.
RUN_SCOPED_CODES = frozenset({
    CloudErrorCode.AUTH,
    CloudErrorCode.BUDGET,
    CloudErrorCode.MALFORMED_CONFIG,
    CloudErrorCode.UNSUPPORTED_SCHEMA,
    CloudErrorCode.INCOMPATIBLE_PROFILE,
    CloudErrorCode.GATED_BY_FLAG,
})


def cloud_job_failure_code(exc: BaseException | None):
    """The JOB-SCOPED :class:`CloudErrorCode` in this chain, or ``None``.

    ``None`` for a run-scoped code, for an interrupt, and -- crucially -- for
    anything that never reached the cloud invoke boundary at all. A local
    engine fault, a torch error or a driver assertion carries no code and gets
    no floor: those still fail LOUD under NO FALLBACKS. Read the stamped code
    and ONLY the stamped code here; unlike :func:`is_content_refusal` there is
    no prose fallback, because "some exception happened during a cloud beat"
    is far too wide a net to floor a beat on.
    """
    for cur in _cause_chain(exc):
        code = getattr(cur, "code", None)
        if not isinstance(code, CloudErrorCode):
            continue
        return code if code in JOB_SCOPED_CODES else None
    return None


def is_cloud_budget_error(exc: BaseException | None) -> bool:
    """True when this exception (or its cause chain) is a spend-cap refusal.

    Live 2026-09-16: a local ``OTR_CLOUD_MEDIA_BUDGET_USD`` ceiling and a
    partner HTTP 402 both refuse a further reserve. ``render_shot`` wraps
    the ``CloudMediaError`` in ``RenderError``. Callers that need to stop
    submitting more partner jobs have to see through that wrap. Unset
    env means no local ceiling; 402 is then the only spend halt.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        code = getattr(cur, "code", None)
        if code is CloudErrorCode.BUDGET or getattr(code, "value", None) == "budget":
            return True
        blob = str(cur)
        if "cloud media: budget" in blob or is_wallet_empty_message(blob):
            return True
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        cur = nxt if isinstance(nxt, BaseException) else None
    return False


# ---------------------------------------------------------------------------
# Flags + operator knobs
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}

def _budget_ceiling_from_env():
    """None = unlimited. Explicit 0 = spend-off. A number = a ceiling."""
    raw = str(otr_env.get("OTR_CLOUD_MEDIA_BUDGET_USD", "") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"OTR_CLOUD_MEDIA_BUDGET_USD={raw!r} is not a number",
        )


def mute_ok_roles() -> frozenset:
    """Roles the operator EXPLICITLY opted down to mute video.
    Operator amendment 2026-07-02: default EMPTY -- audio reactivity is
    default-on for every video role; ledger stamps MUTE_OPT_DOWN."""
    raw = otr_env.get("OTR_VIDEO_MUTE_OK_ROLES", "")
    return frozenset(r.strip() for r in raw.split(",") if r.strip())


# ---------------------------------------------------------------------------
# Auth broker (precedence concrete per pass04 sec 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudAuth:
    kind: str  # always "api_key_hidden" since the 2026-09-19 rip
    value: str

    def __repr__(self) -> str:  # never leak the secret into logs/ledger
        return f"CloudAuth(kind={self.kind!r}, value=***)"


NO_CREDENTIAL_HINT = (
    "no Comfy API key on this queue (hidden input api_key_comfy_org is "
    "empty). Sign into Comfy in the app WITH A COMFY API KEY (a plain "
    "email/Google login injects nothing), or submit headless through "
    "scripts/otr_api.py with OTR_COMFY_API_KEY in the SUBMITTER's "
    "environment (sent as extra_data.api_key_comfy_org). No request was sent."
)


def resolve_auth(hidden_api_key: Optional[str] = None) -> CloudAuth:
    """The ONE credential: the Comfy API key ComfyUI injected into the OTR
    host node's `api_key_comfy_org` hidden input -- the signed-in app
    session, or `extra_data.api_key_comfy_org` on a headless POST /prompt
    (scripts/otr_api.py sends it). No env var, no pack key file, no
    session bearer (rip 2026-09-19: three sources in two orders was the
    defect). Missing = fail closed, and the hint names both real paths."""
    if isinstance(hidden_api_key, str) and hidden_api_key.strip():
        return CloudAuth("api_key_hidden", hidden_api_key.strip())
    raise CloudMediaError(CloudErrorCode.AUTH, NO_CREDENTIAL_HINT)


# ---------------------------------------------------------------------------
# Provider concurrency
# ---------------------------------------------------------------------------

_PROVIDER_ID_RE = re.compile(r"^[A-Z0-9_]+$")
_DEFAULT_CONCURRENCY = 8
#: This semaphore is HTTP overlap, not OTR_CLOUD_FANOUT (submit width,
#: unset default 4). VIDU stays 8 so a later fan-out bump is not
#: silently provider-clamped.
_PROVIDER_DEFAULTS = {"VIDU": 8}


def normalize_provider_id(provider_id: str) -> str:
    pid = provider_id.strip().upper().replace("-", "_")
    if not _PROVIDER_ID_RE.match(pid):
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"provider id {provider_id!r} does not normalize to [A-Z0-9_]+",
        )
    return pid


def provider_semaphore_size(provider_id: str) -> int:
    pid = normalize_provider_id(provider_id)
    env_name = f"OTR_CLOUD_MAX_CONCURRENCY_{pid}"
    raw = otr_env.get(env_name, "").strip()
    if raw:
        try:
            size = int(raw)
        except ValueError:
            raise CloudMediaError(
                CloudErrorCode.MALFORMED_CONFIG,
                f"{env_name}={raw!r} is not an integer",
            )
        if size < 1:
            raise CloudMediaError(
                CloudErrorCode.MALFORMED_CONFIG, f"{env_name} must be >= 1"
            )
        return size
    return _PROVIDER_DEFAULTS.get(pid, _DEFAULT_CONCURRENCY)


# ---------------------------------------------------------------------------
# Cache root (S0: env override > repo-relative default; S1 wires the
# ComfyUI output-base helper -- same resolution as otr\episodes\)
# ---------------------------------------------------------------------------


def _legacy_pack_cloud_media_cache_root() -> Path:
    """Former default inside the installed pack; still read, never migrated."""
    return Path(__file__).resolve().parents[2] / "otr" / "cache" / "cloud_media"


def resolve_cache_root() -> Path:
    override = otr_env.get("OTR_CLOUD_MEDIA_CACHE_DIR", "").strip()
    if override:
        return Path(override)
    try:
        from .._otr_paths import otr_shared_cache_dir
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from _otr_paths import otr_shared_cache_dir  # type: ignore
    return Path(otr_shared_cache_dir()) / "cloud_media"


# ---------------------------------------------------------------------------
# Budget state machine (session-owned; pass04 sec 2)
# ---------------------------------------------------------------------------


class ReservationState(str, enum.Enum):
    RESERVED = "RESERVED"
    SUBMITTED = "SUBMITTED"
    BILLED_ESTIMATE = "BILLED_ESTIMATE"
    BILLED_ACTUAL = "BILLED_ACTUAL"
    RELEASED = "RELEASED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class CostQuote:
    provider: str
    row_id: str
    unit: str
    unit_price_usd: float
    quantity: float
    estimated_usd: float
    max_usd: float
    pricing_source_version: str


@dataclass
class _Reservation:
    rid: str
    estimated_usd: float
    state: ReservationState
    job_id: Optional[str] = None
    actual_usd: Optional[float] = None
    created_at: float = field(default_factory=time.time)


class CloudMediaSession:
    """Per-run cloud state. Owned by the module session table; obtained
    via get_or_create_session -- adapters never construct one."""

    def __init__(self, prompt_id: str, auth: CloudAuth, *,
                 budget_ceiling_usd: Optional[float] = None,
                 cache_root: Optional[Path] = None):
        self.prompt_id = prompt_id
        self.auth = auth
        self.episode_id: Optional[str] = None  # metadata, never the key
        self.created_at = time.time()
        if budget_ceiling_usd is not None:
            self.budget_ceiling_usd = float(budget_ceiling_usd)
        else:
            self.budget_ceiling_usd = _budget_ceiling_from_env()
        self.cache_root = cache_root or resolve_cache_root()
        self._lock = threading.Lock()
        self._reservations: dict = {}
        self._spent_usd = 0.0
        self._semaphores: dict = {}
        self._ledger_lock = threading.Lock()
        self._ledger_path: Optional[Path] = None

    # -- budget state machine (all transitions under the lock) ----------

    def _open_exposure_locked(self) -> float:
        open_states = (ReservationState.RESERVED, ReservationState.SUBMITTED)
        return sum(r.estimated_usd for r in self._reservations.values()
                   if r.state in open_states)

    def reserve(self, estimated_usd: float) -> str:
        if estimated_usd < 0:
            raise CloudMediaError(CloudErrorCode.MALFORMED_CONFIG,
                                  "negative cost estimate")
        with self._lock:
            projected = self._spent_usd + self._open_exposure_locked() + estimated_usd
            ceiling = self.budget_ceiling_usd
            if ceiling is not None and projected > ceiling:
                raise CloudMediaError(
                    CloudErrorCode.BUDGET,
                    f"reserve ${estimated_usd:.4f} would take projected spend "
                    f"to ${projected:.4f} > ceiling "
                    f"${ceiling:.4f} "
                    f"(OTR_CLOUD_MEDIA_BUDGET_USD); spent="
                    f"${self._spent_usd:.4f}",
                )
            rid = uuid.uuid4().hex[:12]
            self._reservations[rid] = _Reservation(
                rid=rid, estimated_usd=estimated_usd,
                state=ReservationState.RESERVED)
            return rid

    def _transition(self, rid: str, allowed_from: tuple,
                    to_state: ReservationState) -> _Reservation:
        res = self._reservations.get(rid)
        if res is None:
            raise CloudMediaError(CloudErrorCode.MALFORMED_CONFIG,
                                  f"unknown reservation {rid!r}")
        if res.state not in allowed_from:
            raise CloudMediaError(
                CloudErrorCode.MALFORMED_CONFIG,
                f"reservation {rid} is {res.state.value}, cannot -> "
                f"{to_state.value}",
            )
        res.state = to_state
        return res

    def submit(self, rid: str, job_id: Optional[str]) -> None:
        with self._lock:
            res = self._transition(rid, (ReservationState.RESERVED,),
                                   ReservationState.SUBMITTED)
            res.job_id = job_id

    def bill(self, rid: str, actual_usd: Optional[float] = None) -> None:
        with self._lock:
            res = self._transition(
                rid, (ReservationState.SUBMITTED,),
                ReservationState.BILLED_ACTUAL if actual_usd is not None
                else ReservationState.BILLED_ESTIMATE)
            res.actual_usd = actual_usd
            self._spent_usd += (actual_usd if actual_usd is not None
                                else res.estimated_usd)

    def release(self, rid: str) -> None:
        """No-submit / provider-rejected: the reservation never bills."""
        with self._lock:
            self._transition(
                rid,
                (ReservationState.RESERVED, ReservationState.SUBMITTED),
                ReservationState.RELEASED)


    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    def open_reservations(self) -> list:
        with self._lock:
            return [r for r in self._reservations.values()
                    if r.state in (ReservationState.RESERVED,
                                   ReservationState.SUBMITTED)]

    # -- provider semaphores (held submit -> terminal per attempt) ------

    def provider_semaphore(self, provider_id: str) -> threading.BoundedSemaphore:
        pid = normalize_provider_id(provider_id)
        with self._lock:
            sem = self._semaphores.get(pid)
            if sem is None:
                sem = threading.BoundedSemaphore(provider_semaphore_size(pid))
                self._semaphores[pid] = sem
            return sem

    # -- billing ledger (append-only JSONL, single writer) --------------

    def ledger_path(self) -> Path:
        """The cloud-media billing ledger -- an append-only record of real money.

        MOVED OUT OF THE PACK DIRECTORY 2026-09-11. It used to live under
        ``cache_root``, which resolves to ``<repo>/otr/cache/cloud_media`` --
        INSIDE the installed pack. A registry update or a reinstall wipes that
        tree, and this file is the ONLY copy of that spend history anywhere on
        disk (verified: nothing reads it back, so nothing could rebuild it).

        It now lives under ``otr_state_dir()``, which is the tier that already
        exists for exactly this -- durable per-machine runtime state under the
        user's output tree, never swept by the janitor (which is scoped to
        ``_shared/tmp`` alone) and never treated as disposable. The cache tier
        would have been the wrong home for the opposite reason: its own
        contract says "a cache entry is NEVER the only copy", and this is.

        ``cache_root`` is unchanged and still owns ``partner_tmp/`` -- genuinely
        transient bytes, written and consumed inside one node execution.

        Existing ledgers are COPIED FORWARD once, not abandoned and not moved:
        a copy leaves the old file intact, so a half-finished migration cannot
        lose spend history. ``OTR_CLOUD_MEDIA_CACHE_DIR`` still overrides
        ``cache_root``; a box that sets it keeps its ledger beside that override
        rather than being silently relocated.
        """
        if self._ledger_path is not None:
            return self._ledger_path

        override = otr_env.get("OTR_CLOUD_MEDIA_CACHE_DIR", "").strip()
        if override:
            # An explicit override names the whole tier; respect it verbatim.
            self.cache_root.mkdir(parents=True, exist_ok=True)
            self._ledger_path = self.cache_root / "billing_ledger.jsonl"
            return self._ledger_path

        try:
            from .._otr_paths import otr_state_dir
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_paths import otr_state_dir  # type: ignore
        dest_dir = Path(otr_state_dir()) / "cloud_media"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "billing_ledger.jsonl"

        legacy_sources = []
        cache_legacy = self.cache_root / "billing_ledger.jsonl"
        if cache_legacy not in legacy_sources:
            legacy_sources.append(cache_legacy)
        if not override:
            try:
                at_default_cache = (
                    Path(self.cache_root).resolve()
                    == resolve_cache_root().resolve()
                )
            except OSError:
                at_default_cache = False
            if at_default_cache:
                pack_legacy = (
                    _legacy_pack_cloud_media_cache_root() / "billing_ledger.jsonl"
                )
                if pack_legacy not in legacy_sources:
                    legacy_sources.insert(0, pack_legacy)
        for legacy in legacy_sources:
            if not legacy.is_file() or dest.exists():
                continue
            try:
                shutil.copy2(str(legacy), str(dest))
                log.info(
                    "[cloud_media] copied the billing ledger forward out of a "
                    "legacy location: %s -> %s (the original is left in place)",
                    legacy, dest)
                break
            except OSError as exc:  # noqa: BLE001
                log.warning(
                    "[cloud_media] could not copy the billing ledger forward "
                    "from %s (%s); trying next legacy source", legacy, exc)

        self._ledger_path = dest
        return self._ledger_path

    def ledger_append(self, record: dict) -> None:
        base = {
            "ts": time.time(),
            "prompt_id": self.prompt_id,
            "episode_id": self.episode_id,
        }
        base.update(record)
        line = json.dumps(base, ensure_ascii=True, sort_keys=True)
        with self._ledger_lock:
            path = self.ledger_path()
            with open(path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Session table -- THE allowed singleton (lock-guarded, keyed by prompt_id)
# ---------------------------------------------------------------------------

SESSION_SWEEP_MAX_AGE_S = 6 * 3600

_TABLE_LOCK = threading.Lock()
_SESSIONS: dict = {}
# prompt_id -> (api_key, stashed_at). Every OTR host node that can run a
# partner engine stashes its api_key_comfy_org hidden input here at the top
# of its execute (cloud_media_invoke.stash_comfy_api_key); the first partner
# call under that prompt opens the session with it. Swept with the sessions.
_PROMPT_API_KEYS: dict = {}
_LEAK_LOG: Callable[[str], None] = lambda msg: print(f"[cloud_media] {msg}")


def stash_prompt_api_key(prompt_id: str, api_key: Optional[str]) -> bool:
    """Remember the Comfy API key ComfyUI injected into a host node so the
    partner calls under this prompt can open their session with it. Empty
    or non-string values store nothing (a local-only graph never has one)
    and the lane then fails closed at resolve_auth, never here."""
    if not prompt_id or not isinstance(api_key, str) or not api_key.strip():
        return False
    now = time.time()
    with _TABLE_LOCK:
        _sweep_locked(now)
        _PROMPT_API_KEYS[prompt_id] = (api_key.strip(), now)
    return True


def _sweep_locked(now: float) -> None:
    stale_keys = [pid for pid, (_key, at) in _PROMPT_API_KEYS.items()
                  if now - at > SESSION_SWEEP_MAX_AGE_S]
    for pid in stale_keys:
        _PROMPT_API_KEYS.pop(pid, None)
    stale = [pid for pid, s in _SESSIONS.items()
             if now - s.created_at > SESSION_SWEEP_MAX_AGE_S]
    for pid in stale:
        sess = _SESSIONS.pop(pid)
        open_res = sess.open_reservations()
        _LEAK_LOG(
            f"LEAKED_SESSION prompt_id={pid} age_s="
            f"{now - sess.created_at:.0f} unreleased_reservations="
            f"{[(r.rid, r.state.value, r.estimated_usd) for r in open_res]}"
        )


def get_or_create_session(
    prompt_id: str,
    hidden_api_key: Optional[str] = None,
    **session_kwargs,
) -> CloudMediaSession:
    """Open (or reuse) the prompt's session. The credential is the explicit
    ``hidden_api_key`` when a caller has it in hand, else the key the host
    node stashed for this prompt_id; nothing else is consulted."""
    if not prompt_id:
        raise CloudMediaError(CloudErrorCode.MALFORMED_CONFIG,
                              "empty prompt_id")
    with _TABLE_LOCK:
        _sweep_locked(time.time())
        sess = _SESSIONS.get(prompt_id)
        if sess is None:
            stashed = _PROMPT_API_KEYS.get(prompt_id)
            auth = resolve_auth(hidden_api_key or (stashed[0] if stashed else None))
            sess = CloudMediaSession(prompt_id, auth, **session_kwargs)
            _SESSIONS[prompt_id] = sess
        return sess


def peek_session(prompt_id: str) -> Optional[CloudMediaSession]:
    with _TABLE_LOCK:
        return _SESSIONS.get(prompt_id)


def teardown_session(prompt_id: str) -> None:
    """Called on assembler done / prompt completion. Logs (never raises)
    if open reservations remain -- those are ORPHANED_JOB territory."""
    with _TABLE_LOCK:
        sess = _SESSIONS.pop(prompt_id, None)
        _PROMPT_API_KEYS.pop(prompt_id, None)
    if sess is not None:
        for r in sess.open_reservations():
            _LEAK_LOG(
                f"ORPHANED_JOB prompt_id={prompt_id} rid={r.rid} "
                f"state={r.state.value} job_id={r.job_id} "
                f"estimated_usd={r.estimated_usd:.4f}"
            )
