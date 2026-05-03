"""
tests/test_filename_pattern_audit.py -- Phase C (BUG-LOCAL-016) regression

Codifies the rule from the QA pass (docs/2026-05-02-rtx-upscale-qa-pass.md):

    Do NOT compute on-disk paths from `ep_id` slugs in any DISCOVERY or
    DESTRUCTIVE code path. The on-disk filename for cache files (wavs,
    sidecar txt) is determined by the producer (sha + ts), not by the
    episode_id slug. Discovery must go through audio_dir.glob(...).

What's allowed (canonical writer/reader pairs):
  * `out_dir / f"{episode_id}.mp4"` -- writing/reading the canonical
    composited mp4 by name. The reader (RTXUpscale OBS guard) and the
    writer (VideoComposite) share the contract, so reconstruction is
    safe.
  * `_slugify(self.episode_id, limit=120)_ledger.json` inside the
    Ledger class -- the Ledger IS the authoritative producer of that
    filename.

What's banned in this audit:
  * Audio-cache discovery via slug:
      `audio_dir / f"opening_{ep_id}.wav"`
      `audio_dir / f"sfx_{ep_id}_..."`
    These files do not exist by that name on disk -- they're written
    as `<role>_<sha>_<ts>.wav` by musicgen_theme / batch_audiogen_generator.
    Discovery must use `audio_dir.glob(f"{role}_*.wav")` or similar.
  * Treatment / sidecar discovery via slug-then-exists:
      `if (audio_dir / f"{ep_id}_treatment.txt").exists()`
    The treatment is written by video_engine using the canonical
    `<out_path>_treatment.txt` pattern; readers should glob.

This test is a static analyzer over `nodes/`. It greps for banned
patterns. False-positive-prone -- if a legit case shows up, allowlist
it explicitly here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

NODES_DIR = Path(__file__).resolve().parent.parent / "nodes"

# Files we audit. story_orchestrator + production_ledger + video_composite
# + episode_assembler are the QA pass hit-list. We add audio nodes too
# since they handle cache files.
AUDIT_GLOB = ("nodes/*.py",)

# Patterns that indicate slug-reconstruction-for-discovery on audio cache
# files. These should NEVER appear in a disk-touching path.
BANNED_PATTERNS = [
    (
        r'audio_dir\s*[/+]\s*f["\']opening_\{[^}]*(ep_id|episode_id|slug)[^}]*\}',
        "opening_<ep_id>.wav reconstruction (use glob 'opening_*.wav')",
    ),
    (
        r'audio_dir\s*[/+]\s*f["\']closing_\{[^}]*(ep_id|episode_id|slug)[^}]*\}',
        "closing_<ep_id>.wav reconstruction (use glob 'closing_*.wav')",
    ),
    (
        r'audio_dir\s*[/+]\s*f["\']interstitial_\{[^}]*(ep_id|episode_id|slug)[^}]*\}',
        "interstitial_<ep_id>.wav reconstruction "
        "(use glob 'interstitial_*.wav')",
    ),
    (
        r'audio_dir\s*[/+]\s*f["\']sfx_\{[^}]*(ep_id|episode_id|slug)[^}]*\}',
        "sfx_<ep_id>_*.wav reconstruction (use glob 'sfx_*.wav')",
    ),
]

# Allowlist: legitimate canonical writer/reader pairs that share a
# convention. Format: (file relative to repo root, line snippet substring).
# When extending this allowlist, add a comment explaining the contract.
ALLOWLIST = [
    # OBS final mp4 -- the writer (RTXUpscale.execute) lands the file at
    # otr/obs/<episode_id>.mp4, the spacesaver-existence guard reads
    # back at the same path. Both use raw episode_id; they share the
    # contract by construction.
    ("nodes/rtx_upscale.py", 'obs" / f"{ep_id}.mp4"'),
    # Composited mp4 -- VideoComposite writes otr/episodes/<ep>/composited/
    # <episode_id>.mp4 by canonical convention. RTXUpscale picks it up
    # via the src socket, not by reconstruction.
    ("nodes/video_composite.py", 'out_dir / f"{episode_id}.mp4"'),
    # Procgen mp4 (signal_lost_<safe_title>_<ts>.mp4) -- write path
    # only. Treatment write below derives from this canonical name via
    # str.replace, not via slug reconstruction.
    ("nodes/video_engine.py", 'f"signal_lost_{safe_title}_{ts}.mp4"'),
    # Treatment write -- derives from out_path (canonical mp4), not
    # from ep_id slug. Write-side, not discovery-side.
    ("nodes/video_engine.py", 'out_path.replace(".mp4", "_treatment.txt")'),
    # Ledger filename in the Ledger class itself -- the class IS the
    # authoritative writer of <ep>_ledger.json. Readers must glob for
    # *_ledger.json (which Phase A spacesaver and Phase B sidecar
    # rename now do).
    (
        "nodes/production_ledger.py",
        '_slugify(self.episode_id, limit=120)}_ledger.json',
    ),
    (
        "nodes/production_ledger.py",
        '_slugify(old, limit=120)}_ledger.json',
    ),
]


def _file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _is_allowlisted(rel_path: str, line: str) -> bool:
    for allow_rel, allow_substr in ALLOWLIST:
        # Normalize separators for cross-platform comparison
        if rel_path.replace("\\", "/").endswith(
            allow_rel.replace("\\", "/").split("nodes/", 1)[-1]
        ) and allow_substr in line:
            return True
    return False


def _scan_for_banned(node_path: Path):
    """Yield (line_num, line_text, reason) for each banned pattern hit."""
    text = _file_text(node_path)
    rel = "nodes/" + node_path.name
    for line_num, line in enumerate(text.splitlines(), start=1):
        for pattern, reason in BANNED_PATTERNS:
            if re.search(pattern, line):
                if _is_allowlisted(rel, line):
                    continue
                yield (line_num, line.strip(), reason)


def test_no_audio_cache_slug_reconstruction():
    """No code in nodes/ reconstructs an audio cache filename from an
    episode_id slug. Cache files use <role>_<sha>_<ts>.wav format and
    must be discovered via glob.
    """
    violations = []
    for node_path in sorted(NODES_DIR.glob("*.py")):
        for line_num, line, reason in _scan_for_banned(node_path):
            violations.append(
                f"  {node_path.name}:{line_num}  {reason}\n    -> {line}"
            )
    if violations:
        pytest.fail(
            "Slug-reconstruction violations found in nodes/:\n"
            + "\n".join(violations)
            + "\n\nIf this is a legitimate canonical writer/reader pair, "
            "add it to ALLOWLIST in this test file with a comment "
            "explaining the contract."
        )


def test_allowlist_entries_still_present():
    """Sanity check: every ALLOWLIST entry should still resolve to a
    real line in its file. If an entry is stale (the source line was
    refactored or removed), surface it so we can prune the allowlist
    rather than have it shield future drift."""
    stale = []
    for allow_rel, allow_substr in ALLOWLIST:
        path = (
            Path(__file__).resolve().parent.parent / allow_rel.replace("\\", "/")
        )
        if not path.exists():
            stale.append(f"  {allow_rel}: file missing")
            continue
        text = _file_text(path)
        if allow_substr not in text:
            stale.append(
                f"  {allow_rel}: substring not found: {allow_substr!r}"
            )
    if stale:
        pytest.fail(
            "ALLOWLIST entries are stale (source line moved or removed):\n"
            + "\n".join(stale)
            + "\n\nPrune the stale entry from ALLOWLIST in this file."
        )


def test_destructive_paths_use_glob_not_reconstruction():
    """The two destructive code paths (rtx_upscale spacesaver, ledger
    rename sidecar loop) must use glob discovery, not slug reconstruction.
    This is a positive assertion: the substring patterns we expect
    must be present.
    """
    rtx = (NODES_DIR / "rtx_upscale.py").read_text(encoding="utf-8")
    ledger = (NODES_DIR / "production_ledger.py").read_text(encoding="utf-8")

    assert 'audio_dir.glob("*_treatment.txt")' in rtx, (
        "rtx_upscale spacesaver lost its glob-based treatment discovery"
    )
    assert 'audio_dir.glob("*_ledger.json")' in rtx, (
        "rtx_upscale spacesaver lost its glob-based ledger discovery"
    )
    assert 'sidecar_dir.glob(f"{old_prefix}*.txt")' in ledger, (
        "production_ledger sidecar rename lost its old-prefix glob"
    )
