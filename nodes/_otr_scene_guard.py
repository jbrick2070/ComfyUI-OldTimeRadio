"""Header<->scene structural coherence guard (v4 campaign, P1(vi)).

Scoreboard defect (unlogged): a "header/script scene mismatch" -- the scene
records (the script's structural header) and the lines assigned to scenes (the
body) disagreed. A SEMANTIC match (does scene ``env`` text describe the beats?)
would require an LLM verdict and is forbidden as a ship gate (THE LAW: only a
deterministic validator ends an episode). So this guard is STRUCTURAL only:

  * every declared ``scene_id`` is unique, and
  * every NON-music line that names a ``scene_id`` references a scene that
    actually exists.

A line pointing at a scene that is not in the ``scenes`` table is an
unambiguous header<->body break. Exact ``scene.line_count`` matching is
deliberately NOT enforced -- what that field counts (voiced vs all rows) is
lane-dependent, so an equality gate would risk a unit false-fail; the
referential check has no such ambiguity.

Opt-in per bank via ``defaults.scene_coherence_check`` (default False -> INERT
for every current bank; a v4 bank flips it True). Gracefully skips when no
``scenes`` are declared (a lane that does not use scenes is not penalised).
Terminal = ``_otr_ledger_freeze._check_g15_scene_coherence`` (in
``run_gap_audit`` -> the one path every family crosses). Self-contained; the
writer's stage-direction and text_for_tts passes are untouched. UTF-8 no BOM, SFW.

THE JOIN IS beat_id -> beats[].scene_id, NOT lines[].scene_id (fixed
2026-08-28, kibitz r2 codex-reviewed, PBUG candidate: 55 published ledgers
carried real scene_id data on their BEATS -- scifi_news writes it there -- and
this check passed all 55 anyway, because no writer has ever put scene_id on a
LINE row. ``lines[]`` carries ``beat_id``; ``beats[]`` carries ``scene_id``;
that is the only join the schema has. See
``docs/2026-08-28-scene-coherence-vacuity/CODING_PLAN.md``.
"""
from __future__ import annotations

from typing import Any, List

__all__ = ["find_scene_coherence_issues"]


def find_scene_coherence_issues(
        ledger_data: Any) -> "tuple[List[str], int]":
    """Return ``(issues, checked)``. Read-only; never raises.

    ``issues`` is empty when clean, when no ``scenes`` are declared, or on
    malformed input. ``checked`` is the count of non-music LINE rows that
    resolved, via ``beat_id``, to a beat carrying a non-empty ``scene_id`` --
    per line, so several lines sharing one scene-bearing beat each count
    individually. The caller uses ``checked`` to detect VACUITY: an armed
    gate that examined zero real linkages despite scenes being declared
    cannot vouch for coherence it never looked at (see
    ``_otr_ledger_freeze._check_g15_scene_coherence``, which is the only
    place that distinction matters -- this function stays a plain read).
    """
    if not isinstance(ledger_data, dict):
        return [], 0
    scenes = ledger_data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return [], 0  # no header declared -> nothing to reconcile
    ids: "List[str]" = []
    for s in scenes:
        if isinstance(s, dict):
            sid = str(s.get("scene_id") or "").strip()
            if sid:
                ids.append(sid)
    id_set = set(ids)
    issues: "List[str]" = []
    for sid in sorted({s for s in ids if ids.count(s) > 1}):
        issues.append(f"duplicate scene_id {sid!r}")

    # beat_id -> scene_id, THE join the schema actually has. Guarded the same
    # way `_check_per_line_invariants` guards a line's own beat_id
    # (`_otr_ledger_freeze.py`) -- a non-string/empty id is skipped, never
    # raised on, so a malformed or adversarial ledger cannot break this
    # "never raises" read. Last-write-wins on a duplicate beat_id: this map
    # is a private two-field lookup, not a second duplicate-detector: the
    # scenes[] table above already has one, and a beat_id collision is a
    # different malformation than this fix exists to catch.
    beats = ledger_data.get("beats")
    beat_scene: "dict[str, str]" = {}
    if isinstance(beats, list):
        for b in beats:
            if not isinstance(b, dict):
                continue
            bid = b.get("beat_id")
            if not isinstance(bid, str) or not bid:
                continue
            sid = str(b.get("scene_id") or "").strip()
            if sid:
                beat_scene[bid] = sid

    lines = ledger_data.get("lines")
    checked = 0
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, dict):
                continue
            if str(line.get("speaker_role") or "").strip().lower().startswith("music"):
                continue
            bid = line.get("beat_id")
            if not isinstance(bid, str) or not bid:
                continue
            sid = beat_scene.get(bid)
            if sid is None:
                continue  # this beat declares no scene -- not an error
            checked += 1
            if sid not in id_set:
                lid = line.get("line_id") or "<no line_id>"
                issues.append(
                    f"line {lid} (beat {bid!r}) references unknown scene "
                    f"{sid!r}")
    return issues, checked
