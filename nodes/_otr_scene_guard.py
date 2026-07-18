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
"""
from __future__ import annotations

from typing import Any, List

__all__ = ["find_scene_coherence_issues"]


def find_scene_coherence_issues(ledger_data: Any) -> "List[str]":
    """Return structural scene<->line issues (empty when clean or no scenes
    declared). Read-only; never raises."""
    if not isinstance(ledger_data, dict):
        return []
    scenes = ledger_data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return []  # no header declared -> nothing to reconcile
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
    lines = ledger_data.get("lines")
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, dict):
                continue
            if str(line.get("speaker_role") or "").strip().lower().startswith("music"):
                continue
            sid = str(line.get("scene_id") or "").strip()
            if sid and sid not in id_set:
                lid = line.get("line_id") or "<no line_id>"
                issues.append(f"line {lid} references unknown scene {sid!r}")
    return issues
