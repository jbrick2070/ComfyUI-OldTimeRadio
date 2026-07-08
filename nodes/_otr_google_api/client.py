"""Small stdlib Gemini Interactions API client for the Google BYO lane."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

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
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_api_key() -> str:
    """Resolve the Gemini key from environment only.

    OTR gives its explicit variable highest precedence so a user can isolate
    this node from other Google SDKs in the same shell. The key is returned to
    the caller but never logged or serialized.
    """
    key = _env("OTR_GOOGLE_API_KEY") or _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    if not key:
        raise GoogleAPIKeyMissingError(
            "Google API LLM selected but no API key is configured. Set "
            "OTR_GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY. No request "
            "was sent."
        )
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
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return _safe_json(resp.read())
    except urllib.error.HTTPError as exc:
        body = _safe_json(exc.read())
        raise _classify_http_error(int(exc.code), body) from exc
    except urllib.error.URLError as exc:
        raise GoogleAPIError(
            f"Google API transport failure: {exc}. No fallback was attempted."
        ) from exc


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
    timeout = int(timeout_s or os.environ.get("OTR_GOOGLE_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    retries = int(max_retries if max_retries is not None else (
        os.environ.get("OTR_GOOGLE_MAX_RETRIES") or DEFAULT_MAX_RETRIES
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
    "resolve_api_key",
]
