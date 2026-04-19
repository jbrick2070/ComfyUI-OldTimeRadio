"""
_hf_token.py  --  User-scope HF_TOKEN resolver
=================================================================
Reads HF_TOKEN from the Windows user environment (HKCU\\Environment)
because the running ComfyUI process often does not inherit it.

Order of precedence:
    1. os.environ["HF_TOKEN"] (already set in this process)
    2. winreg HKEY_CURRENT_USER\\Environment\\HF_TOKEN
    3. None  (public repos still work; gated models will 401)

Design intent:
    - Pure stdlib, no heavy imports.
    - Never raises -- missing token is a valid state.
    - Exports the resolved token back into os.environ so downstream
      huggingface_hub / transformers calls pick it up automatically
      without having to pass token= everywhere.
    - Windows-only lookup (winreg) is guarded so the module is
      importable on Linux/macOS test environments without errors.

Usage:
    from otr_v2.visual._hf_token import ensure_hf_token
    token = ensure_hf_token()  # idempotent, safe to call repeatedly
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("OTR.visual._hf_token")

_HF_TOKEN_CACHE: dict[str, str | None] = {"value": None, "resolved": False}
_REG_KEY = "Environment"
_REG_VALUE = "HF_TOKEN"


def _read_from_winreg() -> str | None:
    """Look up HF_TOKEN in HKEY_CURRENT_USER\\Environment.

    Returns the string value or None when the key is absent, the
    platform is not Windows, or the lookup raises for any reason.
    """
    try:
        import winreg  # Windows-only stdlib module
    except ImportError:
        return None

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_KEY,
            0,
            winreg.KEY_READ,
        ) as key:
            value, _type = winreg.QueryValueEx(key, _REG_VALUE)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except FileNotFoundError:
        # Key/value not present -- expected for users who never set
        # the token.  Not an error.
        return None
    except OSError as exc:
        log.debug("[hf_token] winreg lookup failed: %s", exc)
        return None
    except Exception as exc:  # defensive -- never crash a node load
        log.debug("[hf_token] unexpected winreg error: %s", exc)
        return None

    return None


def ensure_hf_token() -> str | None:
    """Resolve HF_TOKEN from env or HKCU and ensure it is in os.environ.

    Idempotent.  Returns the resolved token string, or None when no
    token is available anywhere (public repos only).
    """
    if _HF_TOKEN_CACHE["resolved"]:
        return _HF_TOKEN_CACHE["value"]

    token = os.environ.get("HF_TOKEN") or None
    source = "os.environ" if token else None

    if not token:
        token = _read_from_winreg()
        if token:
            source = "HKCU\\Environment"
            # Export so transformers + huggingface_hub pick it up.
            os.environ["HF_TOKEN"] = token
            # Some HF libs still look at HUGGING_FACE_HUB_TOKEN.
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    _HF_TOKEN_CACHE["value"] = token
    _HF_TOKEN_CACHE["resolved"] = True

    if token:
        # Log presence + length, never the token itself.
        log.info(
            "[hf_token] HF_TOKEN resolved from %s (len=%d)",
            source,
            len(token),
        )
    else:
        log.info("[hf_token] No HF_TOKEN found in env or HKCU -- public repos only")

    return token


def reset_cache() -> None:
    """Clear the in-module cache (test hook)."""
    _HF_TOKEN_CACHE["value"] = None
    _HF_TOKEN_CACHE["resolved"] = False


__all__ = ["ensure_hf_token", "reset_cache"]
