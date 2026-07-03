"""Cloud partner-node invocation bridge -- S0 (pass04 sec 3).

    invoke_partner_node(node_key, inputs, *, timeout_s) -> PartnerResult

- ONE backend-owned asyncio event-loop thread; partner-node coroutines
  run there while the CALLING (ComfyUI executor) thread blocks in a
  watchdog loop.
- The session is resolved INTERNALLY from the current prompt context
  (comfy_execution.utils.get_executing_context) -- never a parameter.
  Headless smokes/tests bind one explicitly via bind_prompt_id().
- Watchdog (S0 promotion requirement): while blocked, every tick the
  caller checks the ComfyUI interrupt flag (-> cancel + INTERRUPTED)
  and emits a ProgressBar heartbeat at least every 30s so the queue
  shows life during long provider polls.
- Provider outputs are normalized to a streamed temp file under the
  session cache root (whole media is never held in memory here); the
  per-modality canonicalizers (S1-S3) take it from there.
- Every failure maps onto the canonical CloudErrorCode taxonomy and
  settles the budget reservation honestly: provider-said-no releases,
  ambiguous outcomes (timeout / interrupt / corrupt download) bill the
  estimate.

Cold-import-clean: stdlib only at module scope. comfy / torch / yaml /
provider imports are all lazy, inside the code paths that need them.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import importlib
import inspect
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .cloud_media_backend import (
    CloudErrorCode,
    CloudMediaError,
    get_or_create_session,
)
from .cloud_media_canonical import PartnerResult, validate_partner_result

__all__ = [
    "invoke_partner_node",
    "bind_prompt_id",
    "current_prompt_id",
    "partner_rows",
    "PARTNER_NODES_YAML",
    "WATCHDOG_TICK_S",
    "HEARTBEAT_EVERY_S",
]

PARTNER_NODES_YAML = Path(__file__).resolve().parent / "partner_nodes.yaml"

#: how often the blocked caller wakes to check interrupt/timeout.
WATCHDOG_TICK_S = 5.0
#: heartbeat cadence; the S0 requirement is "every tick (<=30s)".
HEARTBEAT_EVERY_S = 20.0
#: grace wait for the coroutine to unwind after a cancel.
_CANCEL_UNWIND_S = 5.0

_USAGE_SOURCE = "ComfyUI-OldTimeRadio"

_CONTENT_TYPE_BY_RETURN = {
    "VIDEO": ("video/mp4", ".mp4"),
    "AUDIO": ("audio/wav", ".wav"),
    "IMAGE": ("image/png", ".png"),
    "STRING": ("text/plain", ".txt"),
}
_CONTENT_TYPE_BY_EXT = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".webp": "image/webp",
}

# ---------------------------------------------------------------------------
# Prompt context (pass04: "session resolved internally, never a parameter")
# ---------------------------------------------------------------------------

_PROMPT_ID_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar(
    "otr_cloud_prompt_id", default=None)


@contextlib.contextmanager
def bind_prompt_id(prompt_id: str):
    """Explicit prompt binding for headless smokes and tests. Inside a
    real ComfyUI execution the executing context supplies it instead."""
    token = _PROMPT_ID_OVERRIDE.set(prompt_id)
    try:
        yield
    finally:
        _PROMPT_ID_OVERRIDE.reset(token)


def current_prompt_id() -> str:
    bound = _PROMPT_ID_OVERRIDE.get()
    if bound:
        return bound
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
    except Exception:
        ctx = None
    if ctx is not None and getattr(ctx, "prompt_id", None):
        return ctx.prompt_id
    raise CloudMediaError(
        CloudErrorCode.MALFORMED_CONFIG,
        "no prompt context: invoke_partner_node must run inside a ComfyUI "
        "execution, or under bind_prompt_id(...) for headless smokes",
    )


# ---------------------------------------------------------------------------
# Partner-node pin table (partner_nodes.yaml, chunk 2 of S0)
# ---------------------------------------------------------------------------

_ROWS_LOCK = threading.Lock()
_ROWS_CACHE: Optional[dict] = None


def partner_rows() -> dict:
    """Rows keyed by node_key; schema pin-2026-07-02 (class_name,
    function, import_path, inputs.{hidden,required,optional},
    provider_id, return_types, selected_output, status)."""
    global _ROWS_CACHE
    with _ROWS_LOCK:
        if _ROWS_CACHE is None:
            try:
                import yaml  # lazy: module stays stdlib-clean at import
                with open(PARTNER_NODES_YAML, "r", encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh)
                rows = doc["rows"]
            except CloudMediaError:
                raise
            except Exception as exc:
                raise CloudMediaError(
                    CloudErrorCode.MALFORMED_CONFIG,
                    f"cannot load partner pin table {PARTNER_NODES_YAML}: "
                    f"{exc}",
                )
            if not isinstance(rows, dict) or not rows:
                raise CloudMediaError(
                    CloudErrorCode.MALFORMED_CONFIG,
                    f"partner pin table has no rows: {PARTNER_NODES_YAML}",
                )
            _ROWS_CACHE = rows
        return _ROWS_CACHE


def _set_rows_for_tests(rows: Optional[dict]) -> None:
    global _ROWS_CACHE
    with _ROWS_LOCK:
        _ROWS_CACHE = rows


def _row_for(node_key: str) -> dict:
    rows = partner_rows()
    row = rows.get(node_key)
    if row is None:
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"unknown partner node_key {node_key!r} (pin table has "
            f"{len(rows)} rows; regenerate via "
            f"scripts/otr_pin_partner_nodes.py if the menu moved)",
        )
    status = str(row.get("status", "")).upper()
    if status != "OK":
        raise CloudMediaError(
            CloudErrorCode.UNSUPPORTED_SCHEMA,
            f"partner row {node_key!r} pinned status={status!r} (not OK); "
            f"re-pin against the live comfy_api_nodes before invoking",
        )
    return row


def _resolve_node_class(row: dict):
    """Import the pinned partner class directly via its import_path --
    no dependency on the full NODE_CLASS_MAPPINGS registry, so headless
    smokes work with just the ComfyUI repo on sys.path."""
    import_path = row.get("import_path", "")
    class_name = row.get("class_name", "")
    try:
        mod = importlib.import_module(import_path)
        return getattr(mod, class_name)
    except Exception as exc:
        raise CloudMediaError(
            CloudErrorCode.UNSUPPORTED_SCHEMA,
            f"cannot resolve pinned partner class {import_path}."
            f"{class_name}: {exc}",
        )


# ---------------------------------------------------------------------------
# Backend-owned event loop (ONE daemon thread for all partner calls)
# ---------------------------------------------------------------------------

_LOOP_LOCK = threading.Lock()
_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="otr-cloud-media-loop",
                daemon=True,
            )
            thread.start()
            _LOOP = loop
        return _LOOP


# ---------------------------------------------------------------------------
# Watchdog seams (monkeypatchable offline; lazy comfy imports live here)
# ---------------------------------------------------------------------------


def _interrupt_requested() -> bool:
    try:
        from comfy import model_management
        return bool(model_management.processing_interrupted())
    except Exception:
        return False


_HEARTBEAT_WARNED = False


def _emit_heartbeat(node_key: str, elapsed_s: float, timeout_s: float) -> None:
    """Progress keep-alive via the executor progress API. Telemetry
    only -- a broken heartbeat logs LOUDLY once and never kills the
    provider call."""
    global _HEARTBEAT_WARNED
    try:
        from comfy.utils import ProgressBar
        total = max(int(timeout_s), 1)
        bar = ProgressBar(total)
        bar.update_absolute(min(int(elapsed_s), total - 1), total)
    except Exception as exc:
        if not _HEARTBEAT_WARNED:
            _HEARTBEAT_WARNED = True
            print(f"[cloud_media] WARNING heartbeat unavailable for "
                  f"{node_key}: {exc} (provider call continues)")


def _cancel_and_unwind(fut: concurrent.futures.Future) -> None:
    fut.cancel()
    with contextlib.suppress(Exception):
        fut.result(timeout=_CANCEL_UNWIND_S)


def _run_with_watchdog(coro, *, timeout_s: float, node_key: str) -> Any:
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    start = time.monotonic()
    last_beat = start
    while True:
        remaining = timeout_s - (time.monotonic() - start)
        if remaining <= 0:
            _cancel_and_unwind(fut)
            raise CloudMediaError(
                CloudErrorCode.TIMEOUT,
                f"{node_key} exceeded timeout_s={timeout_s:.0f}",
            )
        try:
            return fut.result(timeout=min(WATCHDOG_TICK_S, remaining))
        except concurrent.futures.TimeoutError:
            pass
        except concurrent.futures.CancelledError:
            raise CloudMediaError(
                CloudErrorCode.INTERRUPTED,
                f"{node_key} coroutine cancelled",
            )
        if _interrupt_requested():
            _cancel_and_unwind(fut)
            raise CloudMediaError(
                CloudErrorCode.INTERRUPTED,
                f"{node_key} cancelled by user interrupt",
            )
        now = time.monotonic()
        if now - last_beat >= HEARTBEAT_EVERY_S:
            _emit_heartbeat(node_key, now - start, timeout_s)
            last_beat = now


# ---------------------------------------------------------------------------
# The partner call itself (runs on the loop thread)
# ---------------------------------------------------------------------------


async def _call_partner(row: dict, kwargs: dict) -> Any:
    node_cls = _resolve_node_class(row)
    fn_name = row.get("function", "")
    target = getattr(node_cls, fn_name, None)
    if target is None:
        raise CloudMediaError(
            CloudErrorCode.UNSUPPORTED_SCHEMA,
            f"pinned function {fn_name!r} missing on "
            f"{row.get('class_name')!r}; re-pin the partner table",
        )
    bound = None
    try:
        bound = getattr(node_cls(), fn_name)
    except Exception:
        bound = target  # classmethod/static or non-instantiable V3 class
    result = bound(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _inject_hidden_inputs(row: dict, inputs: dict, session) -> dict:
    """Merge caller inputs with the hidden auth/usage inputs the row
    declares. Caller-provided keys are never overridden."""
    kwargs = dict(inputs)
    hidden = (row.get("inputs") or {}).get("hidden") or {}
    auth = session.auth
    if auth.kind in ("api_key_env", "api_key_hidden"):
        auth_name = "api_key_comfy_org"
    else:
        auth_name = "auth_token_comfy_org"
    if auth_name not in hidden:
        raise CloudMediaError(
            CloudErrorCode.AUTH,
            f"row declares no hidden input {auth_name!r} for auth kind "
            f"{auth.kind!r} (hidden inputs: {sorted(hidden)})",
        )
    kwargs.setdefault(auth_name, auth.value)
    if "unique_id" in hidden:
        kwargs.setdefault("unique_id", f"otr-cloud-{uuid.uuid4().hex[:8]}")
    if "comfy_usage_source" in hidden:
        kwargs.setdefault("comfy_usage_source", _USAGE_SOURCE)
    return kwargs


# ---------------------------------------------------------------------------
# Output normalization -> PartnerResult (streamed temp file)
# ---------------------------------------------------------------------------


def _temp_dest(session, node_key: str, ext: str) -> Path:
    root = Path(session.cache_root) / "partner_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{node_key}-{uuid.uuid4().hex[:8]}{ext}"


def _stream_to_temp(url: str, dest: Path) -> None:
    """Chunked download; never buffers the whole body."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": _USAGE_SOURCE})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, \
                open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except CloudMediaError:
        raise
    except Exception as exc:
        with contextlib.suppress(OSError):
            dest.unlink()
        raise CloudMediaError(
            CloudErrorCode.RETRYABLE_TRANSPORT,
            f"download failed from {url!r}: {exc}",
        )


def _unwrap_outputs(raw: Any) -> tuple:
    args = getattr(raw, "args", None)  # comfy_api V3 NodeOutput
    if args is not None:
        return tuple(args)
    if isinstance(raw, (tuple, list)):
        return tuple(raw)
    return (raw,)


def _materialize(value: Any, row: dict, session, node_key: str,
                 declared_ext: str) -> Path:
    """Turn ONE provider output into a real file on disk."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            dest = _temp_dest(session, node_key, declared_ext)
            _stream_to_temp(value, dest)
            return dest
        p = Path(value)
        if p.is_file():
            return p
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"{node_key} returned a string that is neither a URL nor an "
            f"existing file: {value[:200]!r}",
        )
    save_to = getattr(value, "save_to", None)
    if callable(save_to):  # comfy_api VideoInput and friends
        dest = _temp_dest(session, node_key, declared_ext or ".mp4")
        save_to(str(dest))
        return dest
    if isinstance(value, dict) and "waveform" in value and "sample_rate" in value:
        dest = _temp_dest(session, node_key, ".wav")
        try:
            import soundfile as sf  # torchaudio save is torchcodec-broken
            wave = value["waveform"]
            if hasattr(wave, "detach"):
                wave = wave.detach().cpu().numpy()
            while getattr(wave, "ndim", 0) > 2:
                wave = wave[0]
            sf.write(str(dest), wave.T, int(value["sample_rate"]))
        except CloudMediaError:
            raise
        except Exception as exc:
            raise CloudMediaError(
                CloudErrorCode.CORRUPT_OUTPUT,
                f"{node_key} audio dict could not be written: {exc}",
            )
        return dest
    if hasattr(value, "ndim") and getattr(value, "ndim", 0) == 4:
        dest = _temp_dest(session, node_key, ".png")
        try:
            import numpy as np
            from PIL import Image
            frame = value[0]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            arr = np.clip(np.asarray(frame) * 255.0, 0, 255).astype("uint8")
            Image.fromarray(arr).save(str(dest))
        except CloudMediaError:
            raise
        except Exception as exc:
            raise CloudMediaError(
                CloudErrorCode.CORRUPT_OUTPUT,
                f"{node_key} image tensor could not be written: {exc}",
            )
        return dest
    raise CloudMediaError(
        CloudErrorCode.CORRUPT_OUTPUT,
        f"{node_key} returned an unrecognized output type "
        f"{type(value).__name__}",
    )


def _normalize_result(raw: Any, row: dict, session,
                      node_key: str) -> PartnerResult:
    outputs = _unwrap_outputs(raw)
    sel = int(row.get("selected_output", 0) or 0)
    if sel >= len(outputs):
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"{node_key} returned {len(outputs)} outputs but the pin "
            f"selects index {sel}",
        )
    return_types = list(row.get("return_types") or [])
    declared = return_types[sel] if sel < len(return_types) else ""
    content_type, ext = _CONTENT_TYPE_BY_RETURN.get(declared, ("", ""))
    path = _materialize(outputs[sel], row, session, node_key, ext)
    if not content_type:
        content_type = _CONTENT_TYPE_BY_EXT.get(
            path.suffix.lower(), "application/octet-stream")
    elif path.suffix.lower() in _CONTENT_TYPE_BY_EXT:
        content_type = _CONTENT_TYPE_BY_EXT[path.suffix.lower()]
    # Side-channel string outputs (e.g. Kling video_id) -> meta/job id.
    provider_job_id = None
    extra: dict = {}
    for idx, val in enumerate(outputs):
        if idx == sel or not isinstance(val, str) or not val.strip():
            continue
        if len(val) <= 200:
            extra[f"output_{idx}"] = val
            if provider_job_id is None:
                provider_job_id = val
    return validate_partner_result({
        "path": str(path),
        "content_type": content_type,
        "duration_s": None,  # canonicalizers (S1-S3) measure for real
        "provider_job_id": provider_job_id,
        "raw_meta": {
            "node_key": node_key,
            "class_name": row.get("class_name"),
            "provider_id": row.get("provider_id"),
            "return_types": return_types,
            "selected_output": sel,
            "extra_outputs": extra,
        },
    })


# ---------------------------------------------------------------------------
# Error taxonomy mapping + honest budget settlement
# ---------------------------------------------------------------------------

#: provider said no / never charged -> the reservation releases.
_RELEASE_CODES = frozenset({
    CloudErrorCode.AUTH,
    CloudErrorCode.BUDGET,
    CloudErrorCode.PROVIDER_REJECTED,
    CloudErrorCode.RETRYABLE_TRANSPORT,
    CloudErrorCode.UNSUPPORTED_SCHEMA,
    CloudErrorCode.MALFORMED_CONFIG,
    CloudErrorCode.INCOMPATIBLE_PROFILE,
    CloudErrorCode.GATED_BY_FLAG,
})


def _map_exception(exc: BaseException, node_key: str) -> CloudMediaError:
    if isinstance(exc, CloudMediaError):
        return exc
    if isinstance(exc, (asyncio.CancelledError,
                        concurrent.futures.CancelledError)):
        return CloudMediaError(CloudErrorCode.INTERRUPTED,
                               f"{node_key} cancelled")
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return CloudMediaError(CloudErrorCode.TIMEOUT, f"{node_key}: {exc}")
    if isinstance(exc, (ConnectionError, OSError)):
        return CloudMediaError(CloudErrorCode.RETRYABLE_TRANSPORT,
                               f"{node_key}: {exc}")
    text = str(exc).lower()
    if "unauthorized" in text or "401" in text or "forbidden" in text:
        return CloudMediaError(CloudErrorCode.AUTH, f"{node_key}: {exc}")
    return CloudMediaError(CloudErrorCode.PROVIDER_REJECTED,
                           f"{node_key}: {exc}")


def _settle_failure(session, rid: str, err: CloudMediaError,
                    node_key: str) -> None:
    """Release when the provider clearly never charged; bill the
    estimate on ambiguous outcomes (timeout/interrupt/corrupt) so the
    budget ceiling stays honest."""
    try:
        if err.code in _RELEASE_CODES:
            session.release(rid)
            settlement = "released"
        else:
            session.bill(rid)  # BILLED_ESTIMATE
            settlement = "billed_estimate"
        session.ledger_append({
            "event": "invoke_failed", "node_key": node_key, "rid": rid,
            "code": err.code.value, "settlement": settlement,
        })
    except Exception as settle_exc:  # never mask the real failure
        print(f"[cloud_media] WARNING settlement for {rid} failed: "
              f"{settle_exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def invoke_partner_node(node_key: str, inputs: dict, *,
                        timeout_s: float,
                        estimated_usd: float = 0.0) -> PartnerResult:
    """Synchronous bridge the adapters call from the executor thread.

    Resolves the session from the current prompt context, injects the
    row's hidden auth inputs, reserves budget, runs the pinned partner
    coroutine on the backend loop under the interrupt/heartbeat
    watchdog, and normalizes the output to a validated PartnerResult.
    """
    # Operator directive 2026-07-02: NO hidden enable switch -- the user's
    # dropdown pick is the enable. Auth resolves fail-closed downstream
    # (resolve_auth names all three credential sources when missing).
    if timeout_s is None or timeout_s <= 0:
        raise CloudMediaError(CloudErrorCode.MALFORMED_CONFIG,
                              f"timeout_s must be > 0, got {timeout_s!r}")
    row = _row_for(node_key)
    session = get_or_create_session(current_prompt_id())
    kwargs = _inject_hidden_inputs(row, inputs or {}, session)
    sem = session.provider_semaphore(str(row.get("provider_id", "")))
    rid = session.reserve(float(estimated_usd))
    try:
        with sem:
            session.submit(rid, None)
            raw = _run_with_watchdog(
                _call_partner(row, kwargs),
                timeout_s=float(timeout_s), node_key=node_key)
        result = _normalize_result(raw, row, session, node_key)
    except BaseException as exc:
        err = _map_exception(exc, node_key)
        _settle_failure(session, rid, err, node_key)
        raise err from exc
    session.bill(rid)  # estimate; provider actuals land with S1+ metering
    session.ledger_append({
        "event": "invoke_ok", "node_key": node_key, "rid": rid,
        "provider_id": row.get("provider_id"),
        "estimated_usd": float(estimated_usd),
        "path": result["path"], "content_type": result["content_type"],
    })
    return result
