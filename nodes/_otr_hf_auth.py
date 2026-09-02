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


def resolve_hf_token_runtime() -> str | None:
    """Token resolution for the EXECUTION path -- env, HKCU, then token FILES.

    ``resolve_hf_token`` above stays pure stdlib and is what import time uses:
    node registration must not import the Hub client or touch credential files.
    This one runs only when a gated download or generate is actually about to
    happen, so it may do the fuller job.

    WHY IT EXISTS. `hf auth login` -- the documented, recommended way to
    authenticate -- writes a plain text file containing the token and nothing
    else. OTR could not see it: the resolver read only the environment and
    HKCU, so the standard login was invisible and the README could not honestly
    recommend it (PBUG-20260829-10).

    AND THE FILE MOVES, which is the subtle half. `_otr_hf_env` relocates
    ``HF_HOME`` to the canonical models root, and the Hub derives its token
    path FROM ``HF_HOME``. So a login performed in an ordinary shell lands in
    the user's default cache while OTR-inside-ComfyUI looks somewhere else --
    same machine, same user, same token, two paths. Both are checked here, in
    that order, so a login works whether it was done before or after OTR moved
    the cache.

    Order, and each step is a superset of the last:
      1. ``resolve_hf_token()`` -- process env, then HKCU on Windows.
      2. ``huggingface_hub.get_token()`` -- honours ``HF_TOKEN_PATH``, the
         cached login, and the Hub's own alias/precedence rules.
      3. the DEFAULT ``~/.cache/huggingface/token``, which step 2 stops seeing
         once ``HF_HOME`` is relocated.

    Never raises, and never logs the value. Absence is a valid answer: every
    ungated model must keep working with no token at all.
    """
    token = resolve_hf_token()
    if token:
        return token

    try:  # imported HERE, never at module scope -- import time stays pure
        from huggingface_hub import get_token  # type: ignore

        token = get_token()
        if token:
            return str(token).strip() or None
    except Exception:  # noqa: BLE001 -- an absent or old hub must not break a public load
        pass

    try:
        default_token_file = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "token")
        if os.path.isfile(default_token_file):
            # utf-8-sig swallows a BOM that a Windows editor may have stamped
            # on the token file; a non-UTF-8 file raises UnicodeDecodeError
            # (a ValueError, not an OSError), which used to escape this
            # best-effort read (2026-09-01 ship audit, encoding-os-03).
            with open(default_token_file, "r", encoding="utf-8-sig") as fh:
                token = fh.read().strip()
            if token:
                return token
    except (OSError, ValueError):
        pass

    return None


__all__ = ["resolve_hf_token", "resolve_hf_token_runtime"]
