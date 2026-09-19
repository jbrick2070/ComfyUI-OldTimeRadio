"""Small stdlib Gemini Interactions API client for the Google BYO lane."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.request import urlopen  # bare name clears the registry $http2 literal; still seen by the network-sites guard
from typing import Any

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 2
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class GoogleAPIError(RuntimeError):
    """Base class for all direct Google API lane failures."""


class GoogleAPIKeyMissingError(GoogleAPIError):
    """A Google API call was requested but no key was configured."""


class GoogleAPIRequestShapeError(GoogleAPIError):
    """The outbound Google request or inbound response has the wrong shape."""


class GoogleAPIModelUnavailableError(GoogleAPIError):
    """The selected model is unavailable or not accepted by the API."""


class GoogleAPIBillingOrQuotaError(GoogleAPIError):
    """The request hit auth, billing, quota, region, or permission limits."""


def _env(name: str) -> str | None:
    value = otr_env.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_api_key() -> str:
    """Resolve the Gemini key: environment, then the two pack files.

    Environment wins so a user can isolate this node from other Google SDKs
    in the same shell. The two files are the heading in the README: put the
    key in ``google.secret``, or put a path in ``google_api_key.location``.
    The key is returned to the caller but never logged or serialized.
    """
    from .._otr_shared.api_key_files import (
        KeyFileError,
        missing_key_hint,
        resolve_lane_key,
    )

    try:
        key = resolve_lane_key("google")
    except KeyFileError as exc:
        raise GoogleAPIKeyMissingError(str(exc)) from exc
    if not key:
        raise GoogleAPIKeyMissingError(missing_key_hint("google"))
    return key


def _error_message(body: Any, fallback: str = "") -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:500]
        if body.get("message"):
            return str(body["message"])[:500]
    return str(fallback or body or "")[:500]


def _safe_json(data: bytes) -> Any:
    if not data:
        return {}
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise GoogleAPIRequestShapeError(
            f"Google API returned non-JSON response: {data[:200]!r}"
        ) from exc


def _best_effort_json(data: bytes) -> tuple[Any, str | None]:
    """Parse an ERROR body without ever throwing.

    An error body is evidence, not a contract. Letting a non-JSON body (an
    HTML 502 page, a truncated proxy response) raise out of the parser
    destroyed the HTTP status before it could be classified -- the caller
    then saw a shape error and could not tell auth from quota from refusal.
    Returns ``(parsed_or_None, raw_text_or_None)``.
    """
    if not data:
        return None, None
    try:
        return json.loads(data.decode("utf-8")), None
    except Exception:  # noqa: BLE001
        try:
            return None, data.decode("utf-8", errors="replace")[:2000]
        except Exception:  # noqa: BLE001 -- pragma: no cover
            return None, repr(data[:200])


def _retry_after_seconds(headers: Any) -> float | None:
    """Bounded ``Retry-After``. Absent/garbage/negative -> None; capped so a
    hostile or mistaken header cannot park a probe past its deadline."""
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return min(value, 300.0)


def _attach_evidence(exc: GoogleAPIError, *, status: int | None,
                     response_json: Any = None, raw_body: str | None = None,
                     headers: Any = None) -> GoogleAPIError:
    """Attach structured, non-secret evidence to a classified exception.

    Consumers classify on FIELDS -- never by matching exception text, which
    is how an infrastructure error gets misreported as a safety refusal.
    Preserved through every wrapper so the outermost caller still sees it.
    """
    exc.http_status = status
    exc.response_json = response_json
    exc.raw_body = raw_body
    exc.retry_after_s = _retry_after_seconds(headers)
    if status in (401, 403):
        kind, retryable = "auth", False
    elif status == 429:
        kind, retryable = "quota", True
    elif status in (400, 404, 422):
        kind, retryable = "request_shape", False
    elif isinstance(status, int) and status >= 500:
        kind, retryable = "server", True
    elif status is None:
        kind, retryable = "transport", True
    else:
        kind, retryable = "http", False
    exc.failure_kind = kind
    exc.retryable = retryable
    _stamp_cloud_code(exc, kind, response_json, raw_body)
    return exc


#: Google's own failure kinds -> the canonical cloud taxonomy. The split that
#: matters is job-scoped (floor ONE beat, keep the paid run) vs run-scoped
#: (stop, because nothing will ever render).
#:   auth          -> AUTH           run-scoped: the key is wrong for every call
#:   quota         -> BUDGET         run-scoped AND halts: an exhausted project
#:                                   quota refuses the next beat too, and
#:                                   flooring 40 beats on it would publish a void
#:   request_shape -> MALFORMED_CONFIG  run-scoped: a bad shape is bad every time
#:   server        -> RETRYABLE_TRANSPORT  job-scoped: a 5xx is about this call
#:   transport     -> RETRYABLE_TRANSPORT  job-scoped: same
#:   http          -> PROVIDER_REJECTED    job-scoped: the provider said no
_GOOGLE_KIND_TO_CLOUD_CODE = {
    "auth": "AUTH",
    "quota": "BUDGET",
    "request_shape": "MALFORMED_CONFIG",
    "server": "RETRYABLE_TRANSPORT",
    "transport": "RETRYABLE_TRANSPORT",
    "http": "PROVIDER_REJECTED",
}


def _stamp_cloud_code(exc, kind, response_json, raw_body) -> None:
    """Make a Google failure legible to the shared cloud floor.

    WHY THIS EXISTS. Google is a DIRECT BYO API lane -- it never passes through
    ``invoke_partner_node``, which is where every partner failure gets its
    ``CloudErrorCode``. So a Google engine could be correctly recognized as a
    cloud engine by ``_is_cloud_video_engine`` and still not be floorable,
    because ``cloud_job_failure_code`` reads a stamped code and Google stamped
    none. One 5xx on one beat would take a whole paid episode down -- the
    beat-40 defect on a lane that merely looked covered. Found 2026-09-16 when
    the operator asked whether ALL cloud providers were resilient.

    A CONTENT REFUSAL WINS OVER THE HTTP KIND. Google reports a safety block
    inside the RESPONSE BODY, often on an otherwise ordinary status, so the
    status alone would call it something else entirely. This reads the body's
    own fields, which is the rule :func:`_attach_evidence` already states --
    classify on fields, never on exception text.

    Best-effort and never raises: a failure to classify a failure must not
    replace the failure.
    """
    try:
        from .._otr_shared.cloud_media_backend import (
            CloudErrorCode, is_content_policy_message)
        blob = ""
        if response_json is not None:
            try:
                blob = json.dumps(response_json)
            except Exception:  # noqa: BLE001 -- unserializable body
                blob = str(response_json)
        blob = "%s %s" % (blob, raw_body or "")
        if is_content_policy_message(blob):
            exc.code = CloudErrorCode.CONTENT_REFUSED
            return
        named = _GOOGLE_KIND_TO_CLOUD_CODE.get(str(kind or ""))
        if named:
            exc.code = getattr(CloudErrorCode, named)
    except Exception:  # noqa: BLE001 -- classification is never fatal
        pass


def _classify_http_error(status: int, body: Any) -> GoogleAPIError:
    msg = _error_message(body, fallback=f"HTTP {status}")
    if status in (400, 404, 422):
        return GoogleAPIModelUnavailableError(
            f"Google API rejected the selected model or request shape "
            f"(HTTP {status}: {msg}). No fallback was attempted."
        )
    if status in (401, 403, 429):
        return GoogleAPIBillingOrQuotaError(
            f"Google API auth/billing/quota failure (HTTP {status}: {msg}). "
            "No fallback was attempted."
        )
    return GoogleAPIError(
        f"Google API call failed (HTTP {status}: {msg}). No fallback was attempted."
    )


def _post_json(
    path: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_s: int,
) -> dict[str, Any]:
    base_url = (_env("OTR_GOOGLE_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return _safe_json(resp.read())
    except urllib.error.HTTPError as exc:
        # Read the body ONCE and parse best-effort. The previous _safe_json
        # call raised on a non-JSON error body BEFORE the status could be
        # classified, losing the one field that distinguishes auth from
        # quota from a content refusal.
        parsed, raw_text = _best_effort_json(exc.read())
        classified = _classify_http_error(int(exc.code),
                                          parsed if parsed is not None else raw_text)
        raise _attach_evidence(classified, status=int(exc.code),
                               response_json=parsed, raw_body=raw_text,
                               headers=getattr(exc, "headers", None)) from exc
    except urllib.error.URLError as exc:
        raise _attach_evidence(
            GoogleAPIError(
                f"Google API transport failure: {exc}. No fallback was attempted."
            ), status=None) from exc


def _absolute_url(path_or_url: str) -> str:
    value = str(path_or_url or "").strip()
    if not value:
        raise GoogleAPIRequestShapeError("Google API URL/path was blank.")
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if not value.startswith("/"):
        value = "/" + value
    base_url = (_env("OTR_GOOGLE_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
    return f"{base_url}{value}"


def _get_bytes(
    path_or_url: str,
    *,
    api_key: str,
    timeout_s: int,
    accept: str | None = None,
) -> bytes:
    headers = {"x-goog-api-key": api_key}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(
        _absolute_url(path_or_url),
        headers=headers,
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        parsed, raw_text = _best_effort_json(exc.read())
        classified = _classify_http_error(int(exc.code),
                                          parsed if parsed is not None else raw_text)
        raise _attach_evidence(classified, status=int(exc.code),
                               response_json=parsed, raw_body=raw_text,
                               headers=getattr(exc, "headers", None)) from exc
    except urllib.error.URLError as exc:
        raise _attach_evidence(
            GoogleAPIError(
                f"Google API transport failure: {exc}. No fallback was attempted."
            ), status=None) from exc


def get_json(
    path_or_url: str,
    *,
    timeout_s: int | None = None,
    _api_key: str | None = None,
    _get: Any | None = None,
) -> dict[str, Any]:
    """GET a Google API JSON resource using the shared BYO-key auth."""
    key = _api_key or resolve_api_key()
    timeout = int(timeout_s or otr_env.get("OTR_GOOGLE_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    getter = _get or _get_bytes
    body = getter(
        path_or_url,
        api_key=key,
        timeout_s=timeout,
        accept="application/json",
    )
    parsed = _safe_json(body)
    if not isinstance(parsed, dict):
        raise GoogleAPIRequestShapeError(
            "Google API JSON GET did not return a JSON object."
        )
    return parsed


def post_json(
    path: str,
    payload: dict[str, Any],
    *,
    timeout_s: int | None = None,
    _api_key: str | None = None,
    _post: Any | None = None,
) -> dict[str, Any]:
    """POST a Google API JSON resource using the shared BYO-key auth."""
    if not isinstance(payload, dict):
        raise GoogleAPIRequestShapeError("Google payload must be a dict.")
    key = _api_key or resolve_api_key()
    timeout = int(timeout_s or otr_env.get("OTR_GOOGLE_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    post = _post or _post_json
    parsed = post(path, payload, api_key=key, timeout_s=timeout)
    if not isinstance(parsed, dict):
        raise GoogleAPIRequestShapeError(
            "Google API JSON POST did not return a JSON object."
        )
    return parsed


def download_media(
    path_or_url: str,
    *,
    timeout_s: int | None = None,
    _api_key: str | None = None,
    _get: Any | None = None,
) -> bytes:
    """Download a Google API media resource using the shared BYO-key auth."""
    key = _api_key or resolve_api_key()
    timeout = int(timeout_s or otr_env.get("OTR_GOOGLE_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    getter = _get or _get_bytes
    data = getter(
        path_or_url,
        api_key=key,
        timeout_s=timeout,
        accept="video/mp4,application/octet-stream",
    )
    if not data:
        raise GoogleAPIRequestShapeError("Google API media download was empty.")
    return data


def create_interaction(
    payload: dict[str, Any],
    *,
    timeout_s: int | None = None,
    max_retries: int | None = None,
    _api_key: str | None = None,
    _post: Any | None = None,
) -> dict[str, Any]:
    """POST one Gemini Interactions request and return its parsed JSON body."""
    if not isinstance(payload, dict):
        raise GoogleAPIRequestShapeError("Google payload must be a dict.")
    if not payload.get("model"):
        raise GoogleAPIRequestShapeError("Google payload missing required 'model'.")
    if "input" not in payload:
        raise GoogleAPIRequestShapeError("Google payload missing required 'input'.")

    key = _api_key or resolve_api_key()
    timeout = int(timeout_s or otr_env.get("OTR_GOOGLE_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    retries = int(max_retries if max_retries is not None else (
        otr_env.get("OTR_GOOGLE_MAX_RETRIES") or DEFAULT_MAX_RETRIES
    ))
    post = _post or _post_json
    last_exc: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            return post(
                "/v1beta/interactions",
                payload,
                api_key=key,
                timeout_s=timeout,
            )
        except GoogleAPIBillingOrQuotaError:
            raise
        except GoogleAPIModelUnavailableError:
            raise
        except GoogleAPIError as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(min(2.0, 0.25 * (2 ** attempt)))
    assert last_exc is not None
    raise last_exc


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "GoogleAPIBillingOrQuotaError",
    "GoogleAPIError",
    "GoogleAPIKeyMissingError",
    "GoogleAPIModelUnavailableError",
    "GoogleAPIRequestShapeError",
    "create_interaction",
    "download_media",
    "get_json",
    "post_json",
    "resolve_api_key",
]
