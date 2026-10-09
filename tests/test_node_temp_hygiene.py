"""tests/test_node_temp_hygiene.py -- system-temp hygiene ratchet (2026-06-30).

The retired soak hygiene gate failed a leg if a
new ``otr*``-named entry appears in the system temp dir. On 2026-06-30 the overnight
combo soak scored 11 cleanly-rendered legs SOAK_FAIL because a top-level node
wrote an intermediate to the ambient system TEMP and never deleted it -- so on a
server NOT booted via the soak launcher (TEMP unrepointed) it orphaned in
%LOCALAPPDATA%\\Temp.

The existing ``test_engine_tmp_in_tree`` scans only ``nodes/_otr_video_engines/*.py``,
so it missed the top-level node. This test scans ALL workflow nodes and RATCHETS:
no NEW ``otr*`` system-temp writer may appear outside a small, documented
allowlist of known failure-edge / test-only paths.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_NODES = _REPO / "nodes"

# The PRECISE gate-relevant persistent-leak class: a direct
# ``os.path.join(tempfile.gettempdir(), "otr...")`` -- an otr_*-named file written
# straight into the ambient system temp dir, which (unlike a self-cleaning
# mkdtemp that rmtree's in a finally) persists and trips the soak hygiene gate.
# (mkdtemp(prefix="otr_...") callers that clean up in a finally never persist ->
# deliberately NOT flagged here; that would false-fail otr_silent_composite,
# which self-deletes its assemble workdir. rtx_upscale was retired in queue
# item 8 -- 2026-08-08.)
_JOIN_GETTEMP_OTR = re.compile(
    r"join\(\s*(?:_?tempfile\.)?gettempdir\(\)\s*,\s*[frbu]*['\"]otr")


def _offending_files() -> dict:
    out = {}
    for p in _NODES.rglob("*.py"):
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _JOIN_GETTEMP_OTR.search(src):
            out[p.relative_to(_REPO).as_posix()] = ["gettempdir()+otr-join"]
    return out


# Known failure-edge otr_* system-temp writer (documented debt, NOT an always-persist
# leak). video_engine writes otr_video_audio.wav and removes it after encode but
# NOT in a finally, so it leaks only on an ffmpeg failure; it did NOT fire in the
# 2026-06-30 soak. TODO: route it through nodes/_otr_paths.otr_shared_tmp_dir() + a
# finally in a follow-up. The ratchet blocks any NEW persistent gettempdir+otr leak.
_ALLOWLIST = {
    "nodes/video_engine.py",   # otr_video_audio.wav; removed post-encode (not in a finally)
}


def test_no_new_otr_system_temp_writers():
    """RATCHET: no node may create an otr_* path under the ambient system temp dir
    except the documented failure-edge allowlist. A NEW offender fails here."""
    offenders = _offending_files()
    new = {f: tags for f, tags in offenders.items() if f not in _ALLOWLIST}
    assert not new, (
        "new otr_* system-temp writer(s) -- route through "
        "_otr_paths.otr_shared_tmp_dir(): %r" % new)


def test_allowlist_entries_still_exist():
    """Keep the allowlist honest: if an allowlisted file no longer offends (it got
    fixed), drop it from the allowlist so the ratchet stays tight."""
    offenders = set(_offending_files())
    stale = [f for f in _ALLOWLIST if f not in offenders]
    assert not stale, (
        "allowlist entries no longer offend (remove them from _ALLOWLIST): %r" % stale)
