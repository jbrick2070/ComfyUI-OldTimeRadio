"""nodes/_otr_render_plan.py -- Sprint 6 legacy critic render-plan receipt.

Sprint 5B's story critic emits per-line dramatic verdicts -- a
`render_priority` ordering (dramatic peaks first), a `flat_lines` list,
and a whole-episode `arc_verdict`. Sprint 5C's reroll loop acts on
`reroll_targets` and stamps `cycle_count` + `reroll_history`. This
module reads those signals plus 4 operator widgets and computes a
`render_plan` receipt: which line_ids the story critic would prioritize, in
what order, capped at how many, and whether the whole render would have been
BLOCKED because the arc verdict never cleared.

The plan rides on `meta.render_plan` as passive story-QA telemetry. The real
episode renderer deliberately ignores it as a delivery filter and renders every
ShotLock `video.shots` row, so a story-QA receipt can never remove a voiced
beat's visible coverage. When all four widgets are at their defaults AND the
arc-verdict gate does not fire, the receipt simply lists every character
line_id in ledger order.

WIDGETS (live on `OTR_LedgerFreezeCascade`'s node surface).
  render_selection     "all" | "dramatic_peaks_only" -- default "all".
                       "dramatic_peaks_only" reorders the plan to the
                       critic's `render_priority` (peaks first).
  render_max_n         INT, default 6. Cap on plan length. 0 = no cap.
  protagonist_only     BOOL, default False. When True, the plan is
                       restricted to the protagonist's lines (the
                       character with the most CHARACTER-role beats).
  manual_line_ids      STRING, default "". Comma-separated explicit
                       override -- when non-empty, supersedes
                       render_selection / flat_lines / arc_verdict
                       gating. The operator's hand wins.

ARC-VERDICT GATE.
  When `cycle_count >= MAX_REROLL_CYCLES` (i.e. the reroll loop already
  ran its cap) AND `arc_verdict in ("mid_collapse", "flat")`, the plan
  is stamped `blocked=True` with `line_ids=[]`. This is retained as a
  story-QA receipt; it does not suppress real episode rendering.

FLAT-LINE EXCLUSION.
  Lines the critic flagged in `flat_lines` are dropped from the plan
  UNLESS they appear in `meta.reroll_history` with `status="rerolled"`
  (a rerolled flat line earned its place back -- it is no longer the
  original draft).

PRIME DIRECTIVES.
  * PD1 (audio is king): `build_render_plan` NEVER raises. Any internal
    failure degrades to "stamp nothing" (return None).
  * PD3 (JSON-wiring): the 4 widgets live on `OTR_LedgerFreezeCascade`;
    the workflow JSON's `widgets_values` array carries their defaults.
  * PD6: this module makes NO LLM calls -- it is pure dict / list
    arithmetic over the already-stamped critic report.

UTF-8 no BOM. No em-dashes. 4-space indentation.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("OTR.render_plan")


__all__ = [
    "RENDER_SELECTION_CHOICES",
    "DEFAULT_RENDER_MAX_N",
    "build_render_plan",
]


# Mirrors the constant the Sprint 5C reroll loop uses for its own cap;
# the arc-verdict gate fires once the reroll has run its full cap and
# the critic STILL hates the script.
try:
    from ._otr_reroll import MAX_REROLL_CYCLES
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_reroll import MAX_REROLL_CYCLES  # type: ignore


RENDER_SELECTION_CHOICES = ("all", "dramatic_peaks_only")
DEFAULT_RENDER_MAX_N = 6

# Whole-episode arc verdicts that block the render once the reroll cap
# has been reached. `uneven` is acceptable (the arc works but has soft
# patches); only the genuinely broken arcs block.
_BLOCKING_ARC_VERDICTS: frozenset[str] = frozenset({"mid_collapse", "flat"})


def build_render_plan(
    ledger_data: dict,
    *,
    render_selection: str = "all",
    render_max_n: int = DEFAULT_RENDER_MAX_N,
    protagonist_only: bool = False,
    manual_line_ids: str = "",
) -> Optional[dict]:
    """Compute the Sprint 6 render plan from the critic report + widgets.

    Returns the plan dict to stamp on `meta.render_plan`, or `None` on
    an internal failure (PD1: HuMo falls back to legacy render-all on
    a missing stamp).

    Plan shape:
      {
        "line_ids":                [...ordered list of line_ids...],
        "selection_mode":          "manual" | "protagonist_only" |
                                   "dramatic_peaks_only" | "all",
        "blocked":                 bool,
        "blocked_reason":          str (empty unless blocked),
        "applied_max_n":           int (0 = no cap was applied),
        "source_cycle_count":      int,
        "source_arc_verdict":      str,
        "candidate_pool_size":     int (size of the all-character pool),
        "excluded_flat_lines":     [...line_ids dropped by the flat-lines
                                       exclusion (sans rerolled ones)...],
      }
    """
    try:
        return _build_render_plan_inner(
            ledger_data,
            render_selection=render_selection,
            render_max_n=render_max_n,
            protagonist_only=protagonist_only,
            manual_line_ids=manual_line_ids,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_RenderPlan] build_render_plan degraded (%s: %s); "
            "stamping no plan -- HuMo falls back to render-all (PD1)",
            type(exc).__name__, exc,
        )
        return None


def _build_render_plan_inner(
    ledger_data: dict,
    *,
    render_selection: str,
    render_max_n: int,
    protagonist_only: bool,
    manual_line_ids: str,
) -> dict:
    meta = ledger_data.get("meta", {}) or {}
    lines = ledger_data.get("lines", []) or []
    cast_rows = ledger_data.get("cast", []) or []

    # Pool: every CHARACTER line_id in ledger order. Announcer / music
    # beats are visual coverage Sprint 6 does not gate -- they
    # follow their existing render rules in HuMo + VideoComposite.
    candidate_pool: list[str] = [
        str(ln.get("line_id"))
        for ln in lines
        if isinstance(ln, dict)
        and str(ln.get("speaker_role") or "") == "character"
        and ln.get("line_id")
    ]
    candidate_set = set(candidate_pool)

    # Critic + reroll signals (always present in production; the
    # `_coerce` helpers tolerate absences for tests / early-stage runs).
    critic_report = meta.get("story_critic_report") or {}
    if not isinstance(critic_report, dict):
        critic_report = {}
    arc_verdict = str(critic_report.get("arc_verdict") or "strong")
    render_priority_raw = critic_report.get("render_priority") or []
    render_priority = [
        str(x) for x in render_priority_raw if str(x)
    ]
    flat_lines_raw = critic_report.get("flat_lines") or []
    flat_line_ids = {
        str(fl.get("line_id"))
        for fl in flat_lines_raw
        if isinstance(fl, dict) and fl.get("line_id")
    }

    cycle_count = _coerce_int(meta.get("cycle_count"))
    rerolled_line_ids = _rerolled_line_ids(meta.get("reroll_history"))

    # ---- Arc-verdict gate -- runs FIRST so manual_line_ids cannot
    # accidentally ship a render the operator does not realise the
    # critic flagged. The operator override path below LIFTS the gate
    # explicitly. ---------------------------------------------------
    arc_blocked = (
        cycle_count >= MAX_REROLL_CYCLES
        and arc_verdict in _BLOCKING_ARC_VERDICTS
    )

    # ---- manual_line_ids -- highest-priority override. Lifts the
    # arc-verdict gate because an operator typing line_ids by hand is
    # deliberately bypassing critic gating. -----------------------------
    manual_list = _parse_manual_line_ids(manual_line_ids)
    if manual_list:
        # Preserve operator-supplied order, drop unknown ids, dedupe.
        ordered: list[str] = []
        seen: set = set()
        for lid in manual_list:
            if lid in candidate_set and lid not in seen:
                ordered.append(lid)
                seen.add(lid)
        applied_n = _apply_cap(ordered, render_max_n)
        return _stamp(
            line_ids=applied_n,
            selection_mode="manual",
            blocked=False,
            blocked_reason="",
            applied_max_n=(
                render_max_n if (render_max_n and len(applied_n) >= render_max_n)
                else 0
            ),
            cycle_count=cycle_count,
            arc_verdict=arc_verdict,
            candidate_pool_size=len(candidate_pool),
            excluded_flat_lines=[],
        )

    # ---- Honour the arc-verdict gate before any further selection. -----
    if arc_blocked:
        return _stamp(
            line_ids=[],
            selection_mode=(
                "dramatic_peaks_only" if render_selection == "dramatic_peaks_only"
                else ("protagonist_only" if protagonist_only else "all")
            ),
            blocked=True,
            blocked_reason=(
                f"arc_verdict={arc_verdict!r} after "
                f"{cycle_count} reroll cycle(s); render withheld"
            ),
            applied_max_n=0,
            cycle_count=cycle_count,
            arc_verdict=arc_verdict,
            candidate_pool_size=len(candidate_pool),
            excluded_flat_lines=[],
        )

    # ---- Protagonist filter -- before flat-line exclusion + priority. --
    if protagonist_only:
        protagonist_char_id = _identify_protagonist(lines, cast_rows)
        if protagonist_char_id:
            candidate_pool = [
                str(ln.get("line_id"))
                for ln in lines
                if isinstance(ln, dict)
                and str(ln.get("speaker_role") or "") == "character"
                and str(ln.get("char_id") or "") == protagonist_char_id
                and ln.get("line_id")
            ]
            candidate_set = set(candidate_pool)

    # ---- dramatic_peaks_only -- reorder candidate_pool to follow the
    # critic's render_priority (peaks first), keeping only entries that
    # are in both the pool and the priority list. Lines NOT named in
    # render_priority fall to the end in ledger order so the cap-then-N
    # path stays deterministic. --------------------------------------
    if render_selection == "dramatic_peaks_only" and render_priority:
        primary = [lid for lid in render_priority if lid in candidate_set]
        tail = [
            lid for lid in candidate_pool
            if lid not in set(primary)
        ]
        ordered = primary + tail
    else:
        ordered = list(candidate_pool)

    # ---- Flat-line exclusion -- a flat_line drops UNLESS it was
    # rerolled (status="rerolled" in reroll_history). A rerolled flat
    # line is a fresh draft and earned its place back. ----------------
    excluded: list[str] = []
    if flat_line_ids:
        keep: list[str] = []
        for lid in ordered:
            if lid in flat_line_ids and lid not in rerolled_line_ids:
                excluded.append(lid)
                continue
            keep.append(lid)
        ordered = keep

    # ---- Apply render_max_n. -----------------------------------------
    applied = _apply_cap(ordered, render_max_n)

    selection_mode = (
        "protagonist_only" if protagonist_only
        else ("dramatic_peaks_only" if render_selection == "dramatic_peaks_only"
              else "all")
    )
    applied_max_n = (
        render_max_n if (render_max_n and len(applied) >= render_max_n)
        else 0
    )
    return _stamp(
        line_ids=applied,
        selection_mode=selection_mode,
        blocked=False,
        blocked_reason="",
        applied_max_n=applied_max_n,
        cycle_count=cycle_count,
        arc_verdict=arc_verdict,
        candidate_pool_size=len(candidate_pool),
        excluded_flat_lines=excluded,
    )


def _stamp(*, line_ids, selection_mode, blocked, blocked_reason,
           applied_max_n, cycle_count, arc_verdict, candidate_pool_size,
           excluded_flat_lines) -> dict:
    return {
        "line_ids": list(line_ids),
        "selection_mode": selection_mode,
        "blocked": bool(blocked),
        "blocked_reason": str(blocked_reason),
        "applied_max_n": int(applied_max_n),
        "source_cycle_count": int(cycle_count),
        "source_arc_verdict": str(arc_verdict),
        "candidate_pool_size": int(candidate_pool_size),
        "excluded_flat_lines": list(excluded_flat_lines),
    }


def _apply_cap(ordered: list, cap: int) -> list:
    """Trim `ordered` to `cap` entries; cap <= 0 disables the cap."""
    try:
        n = int(cap)
    except (TypeError, ValueError):
        return list(ordered)
    if n <= 0:
        return list(ordered)
    return list(ordered)[:n]


def _parse_manual_line_ids(raw: str) -> list:
    """Parse the operator's comma-separated manual_line_ids widget."""
    if not raw:
        return []
    out: list = []
    for chunk in str(raw).split(","):
        s = chunk.strip()
        if s:
            out.append(s)
    return out


def _identify_protagonist(lines: list, cast_rows: list):
    """Return the char_id of the character with the most CHARACTER-role
    beats. Ties broken by the order the char appears in the cast roster
    (stable). Returns None if the cast carries no character rows or no
    character has any lines.
    """
    counts: dict = {}
    for ln in lines or []:
        if not isinstance(ln, dict):
            continue
        if str(ln.get("speaker_role") or "") != "character":
            continue
        cid = str(ln.get("char_id") or "").strip()
        if cid and cid != "announcer":
            counts[cid] = counts.get(cid, 0) + 1
    if not counts:
        return None
    # Highest-count first; ties broken by cast-roster order for
    # determinism (so a tie does not flip protagonist between runs).
    cast_order = []
    for row in cast_rows or []:
        if isinstance(row, dict):
            cid = str(row.get("char_id") or "").strip()
            if cid:
                cast_order.append(cid)
    # Score: -count first (so max sorts to front), then position in
    # cast_order (lower index first), then char_id alphabetical as a
    # final tiebreak.
    def _key(cid: str):
        try:
            pos = cast_order.index(cid)
        except ValueError:
            pos = len(cast_order)
        return (-counts[cid], pos, cid)
    return sorted(counts.keys(), key=_key)[0]


def _rerolled_line_ids(history) -> set:
    """Return the set of line_ids the reroll loop successfully rerolled.

    `meta.reroll_history` is the audit list `_otr_reroll` stamps; only
    entries with status == "rerolled" count -- a `compose_failed` or
    `line_not_found` entry did NOT improve the line.
    """
    if not isinstance(history, list):
        return set()
    out: set = set()
    for entry in history:
        if isinstance(entry, dict) and entry.get("status") == "rerolled":
            lid = entry.get("line_id")
            if lid:
                out.add(str(lid))
    return out


def _coerce_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
