"""Shared error classes + helpers for LLM model_id inputs.

The S30 two-model-selector centralizes LLM model picks on two writer
widgets (creative_writing_model + technical_model). Every other
LLM-consuming node receives its model id via a STRING input socket
wired from the writer's broadcast outputs. This module provides the
canonical exception types raised at the socket boundary and a shared
resolver helper.

B1a (offline) defines:
    MissingModelInputError -- consumer socket was unwired at run time.
    UnknownModelError      -- model id failed validation (structural
                              reject, not in any admit-path).
    require_model(...)     -- shared helper that fails loud if the
                              socket arrived with a falsy value.

B1c will add VRAMFitFailedError + ContextCapUnknownError.
"""

from __future__ import annotations


class MissingModelInputError(RuntimeError):
    """Raised when a consumer node's STRING model-id input arrives
    None / empty. Signals the workflow wire is not connected to the
    writer's broadcast output. The recovery is graph-level (wire
    the socket), not a code path the consumer can fix itself."""


class UnknownModelError(RuntimeError):
    """Raised by validate_model_id when the supplied id fails
    structural rejection (path traversal, drive letters, unsafe
    formats) or does not match any of the three admit-paths
    (curated / locally-scanned / valid org/name when auto-download
    enabled). Carries an actionable recovery hint in the message."""


class GatedModelError(RuntimeError):
    """Raised by auto_download_if_missing's pre-flight check when the
    requested repo_id is in GATED_CURATED_MODELS and resolve_hf_token()
    returns None. The user gets a dual-path recovery message: HF account
    setup vs. pick a different model.

    Critical: this is raised BEFORE any snapshot_download attempt so
    the user sees the helpful message instead of a raw 401 buried in
    the HF stack trace."""


class InsufficientDiskSpaceError(RuntimeError):
    """Raised by auto_download_if_missing's pre-flight disk-space
    check when (free_bytes - download_bytes - 5GB_margin) < 0.
    Refuses to attempt the download because partial-download cleanup
    on a near-full disk is brittle."""


class VRAMFitFailedError(RuntimeError):
    """Raised by request_slot when check_vram_fit returns a FAIL verdict
    (e.g. 70B-on-16GB). The model would OOM at load; the user gets a
    pre-load failure with estimated-vs-ceiling reason instead of an
    opaque CUDA error at first generation token.

    Carries verdict + reason; callers can str() it for the recovery
    message or inspect .estimated_gb / .ceiling_gb attributes if needed.
    """

    def __init__(self, message: str, *, estimated_gb: float = 0.0, ceiling_gb: float = 0.0):
        super().__init__(message)
        self.estimated_gb = estimated_gb
        self.ceiling_gb = ceiling_gb


def require_model(model_id: str | None, *, slot: str) -> str:
    """Resolve / fail-loud helper used by every consumer socket.

    `slot` is "creative" or "technical" -- used in the error message
    so a user can immediately see which writer output they forgot to
    wire.
    """
    if not model_id or not isinstance(model_id, str) or not model_id.strip():
        raise MissingModelInputError(
            f"LLM consumer received empty {slot}_model input. "
            f"Wire the {slot}_writing_model or {slot}_model output "
            f"from OTR_LedgerScriptWriter to this socket."
        )
    return model_id.strip()
