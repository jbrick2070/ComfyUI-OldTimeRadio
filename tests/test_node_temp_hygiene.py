"""tests/test_node_temp_hygiene.py -- system-temp hygiene ratchet (2026-06-30).

The retired soak hygiene gate failed a leg if a
new ``otr*``-named entry appears in the system temp dir. On 2026-06-30 the overnight
combo soak scored 11 cleanly-rendered legs SOAK_FAIL because
``otr_scene_aware_scopes`` wrote its scopes intermediate to the ambient
system TEMP and never deleted it -- so on a server NOT booted via the soak
launcher (TEMP unrepointed) it orphaned in %LOCALAPPDATA%\\Temp.

The existing ``test_engine_tmp_in_tree`` scans only ``nodes/_otr_video_engines/*.py``,
so it missed the top-level node. This test scans ALL workflow nodes and RATCHETS:
the SceneAwareScopes leak is fixed (must not recur) and no NEW ``otr*`` system-temp
writer may appear outside a small, documented allowlist of known failure-edge /
test-only paths.

SUPERSEDED IN PART, 2026-09-11 (PBUG-20260911-03). The 06-30 repair moved the
scopes MP4 out of the ambient temp dir and into ``episodes/_shared/tmp`` -- the
janitor-swept SCRATCH tier -- and this file then REQUIRED it to stay there. That
was the wrong contract: the scopes video is a RETAINED deliverable consumed by
OTR_PostUpscaleProcgenBlend, so parking it in a sweepable tier stranded five
files from five episodes and left the asset one janitor pass from vanishing
mid-run. It now writes through ``_otr_paths.otr_composited_dir(episode_id)``,
the validated per-episode authority, with NO fallback. The scratch-tier
requirement below is replaced by an asset-owner assertion; the system-temp ban
is KEPT and STRENGTHENED, because the regex alone never actually matched this
node's two-step fallback (a ``_tmp_root`` assignment, then a join on it) and so
never guarded the bug it was named for.
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


# Known failure-edge otr_* system-temp writer (documented debt, NOT the always-persist
# scopes bug). video_engine writes otr_video_audio.wav and removes it after encode but
# NOT in a finally, so it leaks only on an ffmpeg failure; it did NOT fire in the
# 2026-06-30 soak. TODO: route it through nodes/_otr_paths.otr_shared_tmp_dir() + a
# finally in a follow-up. The ratchet blocks any NEW persistent gettempdir+otr leak.
_ALLOWLIST = {
    "nodes/video_engine.py",   # otr_video_audio.wav; removed post-encode (not in a finally)
}


def test_scene_aware_scopes_writes_under_its_owning_episode():
    """PBUG-20260911-03: the scopes MP4 is a DURABLE EPISODE ASSET, not scratch.

    It must be placed through the validated per-episode authority using the
    manifest's own episode_id, with no scratch tier and no ambient-temp fallback
    underneath it. Where the file actually LANDS is proven by the real-producer
    tests in tests/test_video_scene_aware_scopes.py; this is the cheap
    source-level ratchet that stops the two owners it must never use again from
    creeping back in.
    """
    src = (_NODES / "otr_scene_aware_scopes.py").read_text(encoding="utf-8")
    assert "otr_composited_dir" in src, (
        "otr_scene_aware_scopes must place the scopes MP4 through "
        "_otr_paths.otr_composited_dir(episode_id) -- the validated per-episode "
        "authority that raises on an empty, reserved or traversing identity")
    assert "otr_shared_tmp_dir" not in src, (
        "otr_scene_aware_scopes is back in the janitor-swept _shared/tmp scratch "
        "tier -- the scopes video is a retained deliverable and belongs under its "
        "own episode (PBUG-20260911-03)")
    # The tree-wide regex below is kept, but THIS node gets a stricter structural
    # ban: no reference to the ambient temp resolver at all. The regex only ever
    # matched a one-line join(); this node's fallback was a two-step assignment it
    # could not see, which is exactly how the leak survived a repair named for it.
    assert not _JOIN_GETTEMP_OTR.search(src)
    assert "gettempdir" not in src, (
        "otr_scene_aware_scopes references the ambient system temp dir again -- "
        "an unplaceable scopes render must RAISE, never divert (that diversion is "
        "what PBUG-20260911-03 records)")


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
