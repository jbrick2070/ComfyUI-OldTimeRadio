"""Cloud media control plane -- S0 of the cloud engine lanes.

Build doc: docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md
(4-round roundtable, converged 2026-07-02, operator amendment: audio
reactivity DEFAULT-ON for all video roles).

This module is the ONE allowed process singleton for cloud media state
(a lock-guarded session table). "No module globals" in the plan means:
no credential/budget state in ADAPTER modules -- adapters fetch the
session from here. The frozen AudioEngine protocol is never touched.

Cold-import-clean: stdlib only. No torch / comfy imports at module
scope. No network code lives here -- invocation arrives with the
partner_nodes.yaml pinning chunk and resolves its session through this
table.

Env surface (all read per session-create, never mutated mid-run):
  OTR_COMFY_API_KEY               auth precedence #1 (headless).
  OTR_CLOUD_MEDIA_BUDGET_USD      per-run USD ceiling. UNSET = the
                                  DEFAULT_BUDGET_USD safety cap (operator
                                  directive 2026-07-02: the dropdown pick
                                  IS the enable -- no hidden switch, so
                                  the cap must not act as one). An
                                  EXPLICIT 0 = every reserve fails closed
                                  with `budget` (a deliberate spend-off).

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
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
    "DEFAULT_BUDGET_USD",
    "mute_ok_roles",
    "resolve_auth",
    "get_or_create_session",
    "peek_session",
    "teardown_session",
    "provider_semaphore_size",
    "normalize_provider_id",
    "resolve_cache_root",
    "SESSION_SWEEP_MAX_AGE_S",
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


# ---------------------------------------------------------------------------
# Flags + operator knobs
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}

#: Per-run USD safety cap when OTR_CLOUD_MEDIA_BUDGET_USD is unset.
#: PRICING.md 2026-07-02: episode envelope ~$1.55 (Kling lipsync-heavy);
#: $10 covers a heavy episode with headroom while still bounding a runaway.
DEFAULT_BUDGET_USD = 10.0


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
    kind: str  # "api_key_env" | "api_key_hidden" | "bearer_hidden"
    value: str

    def __repr__(self) -> str:  # never leak the secret into logs/ledger
        return f"CloudAuth(kind={self.kind!r}, value=***)"


def resolve_auth(
    hidden_api_key: Optional[str] = None,
    hidden_auth_token: Optional[str] = None,
) -> CloudAuth:
    """OTR_COMFY_API_KEY env > hidden api_key_comfy_org > hidden
    auth_token_comfy_org. Missing everything = fail closed, naming all
    three sources."""
    env_key = otr_env.get("OTR_COMFY_API_KEY", "").strip()
    if env_key:
        return CloudAuth("api_key_env", env_key)
    if hidden_api_key and hidden_api_key.strip():
        return CloudAuth("api_key_hidden", hidden_api_key.strip())
    if hidden_auth_token and hidden_auth_token.strip():
        return CloudAuth("bearer_hidden", hidden_auth_token.strip())
    raise CloudMediaError(
        CloudErrorCode.AUTH,
        "no credentials: set OTR_COMFY_API_KEY, or run with a logged-in "
        "Comfy account (hidden inputs api_key_comfy_org / "
        "auth_token_comfy_org)",
    )


# ---------------------------------------------------------------------------
# Provider concurrency
# ---------------------------------------------------------------------------

_PROVIDER_ID_RE = re.compile(r"^[A-Z0-9_]+$")
_DEFAULT_CONCURRENCY = 2
_PROVIDER_DEFAULTS = {"KLING": 1}


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


def resolve_cache_root() -> Path:
    override = otr_env.get("OTR_CLOUD_MEDIA_CACHE_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "otr" / "cache" / "cloud_media"


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
        raw = otr_env.get("OTR_CLOUD_MEDIA_BUDGET_USD", "").strip()
        if budget_ceiling_usd is not None:
            self.budget_ceiling_usd = float(budget_ceiling_usd)
        else:
            try:
                # UNSET -> DEFAULT_BUDGET_USD safety cap (the dropdown pick is
                # the enable; the cap must not be a hidden switch). EXPLICIT
                # "0" -> 0.0 -> every reserve fails closed (deliberate
                # spend-off remains available).
                self.budget_ceiling_usd = (
                    float(raw) if raw else DEFAULT_BUDGET_USD)
            except ValueError:
                raise CloudMediaError(
                    CloudErrorCode.MALFORMED_CONFIG,
                    f"OTR_CLOUD_MEDIA_BUDGET_USD={raw!r} is not a number",
                )
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
            if projected > self.budget_ceiling_usd:
                raise CloudMediaError(
                    CloudErrorCode.BUDGET,
                    f"reserve ${estimated_usd:.4f} would take projected spend "
                    f"to ${projected:.4f} > ceiling "
                    f"${self.budget_ceiling_usd:.4f} "
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

    def abort(self, rid: str) -> None:
        with self._lock:
            self._transition(
                rid,
                (ReservationState.RESERVED, ReservationState.SUBMITTED),
                ReservationState.ABORTED)

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
        if self._ledger_path is None:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            self._ledger_path = self.cache_root / "billing_ledger.jsonl"
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
_LEAK_LOG: Callable[[str], None] = lambda msg: print(f"[cloud_media] {msg}")


def _sweep_locked(now: float) -> None:
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
    hidden_auth_token: Optional[str] = None,
    **session_kwargs,
) -> CloudMediaSession:
    if not prompt_id:
        raise CloudMediaError(CloudErrorCode.MALFORMED_CONFIG,
                              "empty prompt_id")
    with _TABLE_LOCK:
        _sweep_locked(time.time())
        sess = _SESSIONS.get(prompt_id)
        if sess is None:
            auth = resolve_auth(hidden_api_key, hidden_auth_token)
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
    if sess is not None:
        for r in sess.open_reservations():
            _LEAK_LOG(
                f"ORPHANED_JOB prompt_id={prompt_id} rid={r.rid} "
                f"state={r.state.value} job_id={r.job_id} "
                f"estimated_usd={r.estimated_usd:.4f}"
            )
