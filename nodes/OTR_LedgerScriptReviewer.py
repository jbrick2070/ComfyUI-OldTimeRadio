"""nodes/OTR_LedgerScriptReviewer.py -- back-compat shim.

Renamed to OTR_LedgerFreezeCascade in LFC sprint commit 2 of 14
(2026-05-11). The renamed class lives in
`nodes/OTR_LedgerFreezeCascade.py`. This module re-exports it under
the old name so existing workflow JSONs that still reference
"OTR_LedgerScriptReviewer" (class name, file path, or import path)
continue to load without re-wiring.

`__init__.py` registers both `OTR_LedgerFreezeCascade` (canonical)
and `OTR_LedgerScriptReviewer` (alias) in NODE_CLASS_MAPPINGS so
either name resolves to the same class. The first commit to break
the alias would also need to migrate every workflow JSON that
references the old name. Until then this shim is the safety net.

ADR: docs/2026-05-11-multi-turn-polish-adr.md (section 9).
"""
from __future__ import annotations

from .OTR_LedgerFreezeCascade import (  # noqa: F401
    OTR_LedgerFreezeCascade,
    DEFAULT_MODEL_ID,
    _no_ledger_error_json,
)


# Class alias preserves the legacy import surface
# (`from nodes.OTR_LedgerScriptReviewer import OTR_LedgerScriptReviewer`).
OTR_LedgerScriptReviewer = OTR_LedgerFreezeCascade


__all__ = [
    "OTR_LedgerScriptReviewer",
    "OTR_LedgerFreezeCascade",
    "DEFAULT_MODEL_ID",
]
