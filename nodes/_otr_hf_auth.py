"""HuggingFace token resolution -- cross-platform safe.

S30 B1a2. Used by GatedModelError pre-flight checks and
auto_download_if_missing's token forwarding.

Resolution order:
    1. os.environ.get("HF_TOKEN")
    2. (Windows only) HKCU\\Environment via winreg
    3. None

The winreg branch is gated on os.name == "nt" so macOS/Linux callers
never hit `import winreg` (which raises ImportError off-Windows).

Every token-presence check in the codebase MUST go through this helper.
Never read os.environ["HF_TOKEN"] directly -- the HKCU fallback is the
common case for ComfyUI Desktop on Windows since that process doesn't
inherit user-scope env vars without an explicit bake-in.
"""

from __future__ import annotations

import os


def resolve_hf_token() -> str | None:
    """Return the user's HF_TOKEN if available, else None.

    Reads from process env first (works everywhere), then HKCU on
    Windows (catches the ComfyUI Desktop env-inheritance gap), then
    gives up cleanly.
    """
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    if os.name == "nt":
        try:
            import winreg  # type: ignore

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, "HF_TOKEN")
                return value or None
        except (ImportError, FileNotFoundError, OSError):
            return None
    return None


__all__ = ["resolve_hf_token"]
