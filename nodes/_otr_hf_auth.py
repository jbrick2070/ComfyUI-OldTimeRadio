"""HuggingFace token resolution -- cross-platform safe.

Used by the gated-model pre-flight checks and by the token forwarding of a
model download the user asked for.

Resolution order:
    1. ``HF_TOKEN`` in the environment that launched ComfyUI
    2. (run time only) the Hub's own login -- ``hf auth login`` -- see
       :func:`resolve_hf_token_runtime`
    3. None

NO REGISTRY READ. Until 2.3.64 this also read ``HF_TOKEN`` out of the Windows
user registry, and the pack's ``__init__`` exported what it found into the
whole ComfyUI process at every start, for gated models the pack no longer
ships. A credential is now read only from where the user put it for this
process, only when a download needs it, and it is handed to that one download.

Every token-presence check in the codebase goes through this helper.
"""

from __future__ import annotations

import os

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

def resolve_hf_token() -> str | None:
    """Return ``HF_TOKEN`` from the process environment, else None.

    Pure stdlib, and what import time uses. The login file is read by
    :func:`resolve_hf_token_runtime`, on the execution path.
    """
    return otr_env.get("HF_TOKEN") or None


def resolve_hf_token_runtime() -> str | None:
    """Token resolution for the EXECUTION path -- env, then token FILES.

    ``resolve_hf_token`` above stays pure stdlib and is what import time uses:
    node registration must not import the Hub client or touch credential files.
    This one runs only when a gated download or generate is actually about to
    happen, so it may do the fuller job.

    WHY IT EXISTS. `hf auth login` -- the documented, recommended way to
    authenticate -- writes a plain text file containing the token and nothing
    else. OTR could not see it: the resolver did not read the login file, so
    the standard login was invisible and the README could not honestly
    recommend it (PBUG-20260829-10).

    AND THE FILE MOVES, which is the subtle half. `_otr_hf_env` relocates
    ``HF_HOME`` to the canonical models root, and the Hub derives its token
    path FROM ``HF_HOME``. So a login performed in an ordinary shell lands in
    the user's default cache while OTR-inside-ComfyUI looks somewhere else --
    same machine, same user, same token, two paths. Both are checked here, in
    that order, so a login works whether it was done before or after OTR moved
    the cache.

    Order, and each step is a superset of the last:
      1. ``resolve_hf_token()`` -- the process environment.
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
