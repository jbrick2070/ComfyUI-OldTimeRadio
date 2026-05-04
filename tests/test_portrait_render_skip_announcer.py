"""BUG-LOCAL-094: OTR_BatchFluxPortraitRender must skip cast members
whose only role is announcer (or music/sfx).

Pre-094 the skip check looked at ``cast.speaker_role`` / ``cast.role``,
which the ledger cast block doesn't carry (speaker_role lives per-line).
The check always returned an empty string and the skip_announcer guard
never fired. ANNOUNCER cast members got a wasted ~30s FLUX portrait
that HuMo never used (announcer beats route to LTX via BUG-129b).

Post-094 the dispatch walks ``ledger.lines[]`` to derive a
``char_id_has_character_line`` map; cast members with zero
character-role lines are skipped. Falls back to a name-match
("ANNOUNCER") when the lines block is missing, for legacy ledgers.

These tests pin the SKIP DECISION LOGIC at source-code level. Full
end-to-end (FLUX render) needs ComfyUI runtime + GPU and is exercised
via Jeffrey's live workflow, not here.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = _REPO_ROOT / "visual" / "batch_flux_portrait_render.py"


def test_skip_announcer_widget_default_true():
    """skip_announcer widget defaults True so the skip is the
    out-of-the-box behaviour. Set to False only for debug runs that
    want to see what FLUX produces for the ANNOUNCER prompt."""
    src = NODE_PATH.read_text(encoding="utf-8")
    # The widget block: "skip_announcer": ("BOOLEAN", {... "default": True ...})
    pat = re.compile(
        r'"skip_announcer"\s*:\s*\(\s*"BOOLEAN"\s*,\s*\{[^}]*"default"\s*:\s*True',
        re.DOTALL,
    )
    assert pat.search(src), (
        "skip_announcer widget must default to True (BUG-LOCAL-094)"
    )


def test_dispatch_uses_lines_block_not_cast_speaker_role():
    """Pre-094 the check was ``c.get("speaker_role") or c.get("role")``
    on the cast dict, which always returns empty string because cast
    entries don't carry that field. Post-094 the dispatch builds a
    ``char_id_has_character_line`` map by walking ledger.lines[]. Pin
    that the source no longer relies on cast.speaker_role for the
    skip decision."""
    src = NODE_PATH.read_text(encoding="utf-8")
    # The new path must reference char_id_has_character_line.
    assert "char_id_has_character_line" in src, (
        "BUG-LOCAL-094 dispatch must build a char_id_has_character_line "
        "map from ledger.lines[]"
    )
    # The new path must reference resolve_speaker_role (canonical helper).
    assert "resolve_speaker_role" in src, (
        "BUG-LOCAL-094 must use resolve_speaker_role from _otr_speaker_role"
    )


def test_dispatch_has_legacy_name_match_fallback():
    """When the lines block is missing/empty (degraded ledger), the
    dispatch falls back to a literal name match against 'ANNOUNCER'
    so legacy ledgers without a lines[] block still skip the wasted
    portrait. Pin the fallback string."""
    src = NODE_PATH.read_text(encoding="utf-8")
    assert 'speaker.upper().strip() == "ANNOUNCER"' in src, (
        "BUG-LOCAL-094 must keep the legacy name-match fallback for "
        "ledgers without a lines block"
    )


def test_no_pre094_speaker_role_check_remaining():
    """Pin that the broken pre-094 check is gone. A future refactor
    that re-introduces ``c.get("speaker_role") or c.get("role")`` as
    the skip predicate will fail this test before reaching a live
    render."""
    src = NODE_PATH.read_text(encoding="utf-8")
    pat = re.compile(
        r'role\s*=\s*\(\s*c\.get\(\s*["\']speaker_role["\']\s*\)\s*or\s*c\.get\(\s*["\']role["\']\s*\)',
    )
    assert not pat.search(src), (
        "BUG-LOCAL-094: pre-094 skip predicate "
        "`c.get('speaker_role') or c.get('role')` must not return; "
        "use char_id_has_character_line derived from ledger.lines[] "
        "instead."
    )


def test_skip_logs_bug_094_reference():
    """The skip log line should reference BUG-LOCAL-094 so a future
    Jeffrey scan of the runtime log can correlate the skip back to
    this fix."""
    src = NODE_PATH.read_text(encoding="utf-8")
    assert "BUG-LOCAL-094" in src, (
        "skip-path log lines should reference BUG-LOCAL-094 for "
        "audit traceability"
    )
