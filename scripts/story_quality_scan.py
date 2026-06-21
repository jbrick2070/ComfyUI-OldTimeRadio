r"""scripts/story_quality_scan.py -- story-engine v1 measurement harness.

Reads a set of episode LEDGER json files (the `*_ledger.json` the freeze
cascade writes under `episodes/<ep>/audio/`) and reports the story-quality
metrics the v1 sprint is graded on. READ-ONLY: never mutates a ledger.

Metrics (SPRINT_READY_PLAN measurement contract):
  * length_ratio          -- ALL VOICED words (character + announcer; EXCLUDE
                             music) / target_words.
  * length_pass_fired     -- did the post-script length normalizer activate
                             (meta.length_pass_report).
  * episode_valid         -- freeze CRITICAL pass AND dramatic-contract pass
                             (meta.slot_drama_contracts_audit).
  * outro_hedge_vs_resolved -- the outro contains a HEDGE_LIST phrase AND the
                             ending_change is_resolved (a contradiction to fix).
  * narration_self_address_lines -- count flagged by the F7 detector.

The HEDGE_LIST + is_resolved_ending_change live in nodes/_otr_dramatic_state.py
so the scan and the F3 outro repair share ONE source of truth. The narration
detector is imported from nodes/_otr_line_hygiene when present (F7); a vendored
fallback keeps the scan runnable on pre-F7 code for the baseline.

Usage:
  python scripts/story_quality_scan.py --ledgers "<glob>" [--target-words 864]
      [--json-out out.json] [--md-out SPRINT_BASELINE_FRAGMENT.md] [--label baseline]

PURE-where-possible: the metric functions are importable + unit tested.
UTF-8, no BOM, ASCII-only source. Deterministic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Shared resolved/hedge helper -- ONE source of truth with the F3 repair.
try:
    from nodes._otr_dramatic_state import (  # type: ignore
        HEDGE_LIST, is_resolved_ending_change,
    )
except Exception:  # noqa: BLE001 -- keep the scan runnable in isolation
    HEDGE_LIST = (
        "remains to be seen", "only tomorrow will tell", "open question",
        "remains unknown", "time will tell", "yet to be seen",
    )

    def is_resolved_ending_change(ending_change) -> bool:  # type: ignore
        s = " ".join(str(ending_change or "").lower().split())
        if len(s) < 4:
            return False
        return not any(m in s for m in (
            "remains to be seen", "remains unknown", "yet to be seen",
            "open question", "unresolved", "time will tell", "uncertain",
            "may never", "might never", "left unanswered", "unanswered",
        ))

# F7 narration detector -- prefer the engine's (shared); fall back to a
# vendored copy so the baseline runs before F7 lands. T2.3 mirrors this exactly.
try:
    from nodes._otr_line_hygiene import detect_narration_self_address  # type: ignore
    _HAS_ENGINE_DETECTOR = True
except Exception:  # noqa: BLE001
    _HAS_ENGINE_DETECTOR = False

    _NARRATION_VERBS = frozenset({
        "paces", "pacing", "stops", "stopping", "gazes", "gazing", "stares",
        "staring", "contemplates", "contemplating", "sighs", "sighing",
        "nods", "nodding", "shrugs", "shrugging", "turns", "turning", "walks",
        "walking", "leans", "leaning", "frowns", "frowning", "smiles",
        "smiling", "glances", "glancing", "reaches", "reaching", "stands",
        "standing", "sits", "sitting", "watches", "watching", "moves",
        "moving", "steps", "stepping", "looks", "looking",
    })
    _THIRD_PERSON_LEAD = re.compile(r"^\s*(he|she|they)\b", re.IGNORECASE)

    def detect_narration_self_address(text: Any, speaker_name: Any = "") -> bool:  # type: ignore
        """Vendored fallback (mirrors the future _otr_line_hygiene detector).

        Fires only when a line narrates the SPEAKER's own physical action in
        third person, or opens with the speaker's own name + a narration verb.
        Excludes first-person and legitimate 3rd-person references to OTHERS.
        """
        try:
            s = " ".join(str(text or "").split())
            if not s:
                return False
            low = s.lower()
            words = re.findall(r"[a-z']+", low)
            if not words:
                return False
            # 3rd-person pronoun lead + a narration verb anywhere = self-narration
            if _THIRD_PERSON_LEAD.match(s):
                if any(w in _NARRATION_VERBS for w in words[:4]):
                    return True
            # speaker's own name as a 3rd-person subject + a narration verb
            name = str(speaker_name or "").strip().lower()
            first = name.split()[0] if name.split() else ""
            if first and len(first) > 1 and words and words[0] == first:
                if any(w in _NARRATION_VERBS for w in words[1:4]):
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False


_VOICED_ROLES = frozenset({"character", "announcer"})
_MUSIC_ROLES = frozenset({"music_open", "music_close", "music_inter"})
_WORD = re.compile(r"[A-Za-z0-9']+")


def _wc(text: Any) -> int:
    return len(_WORD.findall(str(text or "")))


def _meta(ledger: Dict[str, Any]) -> Dict[str, Any]:
    m = ledger.get("meta")
    return m if isinstance(m, dict) else {}


def _lines(ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    ls = ledger.get("lines")
    return [l for l in ls if isinstance(l, dict)] if isinstance(ls, list) else []


def resolve_target_words(ledger: Dict[str, Any], override: Optional[int]) -> int:
    if override:
        return int(override)
    m = _meta(ledger)
    for key in ("target_words",):
        v = m.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    gp = m.get("gen_params_initial")
    if isinstance(gp, dict) and isinstance(gp.get("target_words"), (int, float)):
        return int(gp["target_words"])
    return 0


def voiced_word_total(ledger: Dict[str, Any]) -> int:
    """ALL voiced words (character + announcer), excluding music lines.

    Prefers the freeze-stamped meta counts when present (authoritative); else
    sums line text for voiced roles.
    """
    m = _meta(ledger)
    cw = m.get("character_word_count")
    aw = m.get("announcer_word_count")
    if isinstance(cw, (int, float)) and isinstance(aw, (int, float)):
        return int(cw) + int(aw)
    total = 0
    for ln in _lines(ledger):
        role = str(ln.get("speaker_role") or "").strip()
        if role in _VOICED_ROLES:
            total += _wc(ln.get("text"))
    return total


def length_ratio(ledger: Dict[str, Any], target_override: Optional[int]) -> Optional[float]:
    target = resolve_target_words(ledger, target_override)
    if target <= 0:
        return None
    return round(voiced_word_total(ledger) / target, 4)


def length_pass_fired(ledger: Dict[str, Any]) -> Optional[bool]:
    """Did the post-script length normalizer activate (meta.length_pass_report)."""
    m = _meta(ledger)
    rep = m.get("length_pass_report")
    if not isinstance(rep, dict):
        return None
    for key in ("fired", "activated", "applied", "changed", "expanded"):
        if isinstance(rep.get(key), bool):
            return bool(rep.get(key))
    for key in ("lines_changed", "words_added", "n_changed", "adjustments"):
        v = rep.get(key)
        if isinstance(v, (int, float)):
            return v > 0
    return None


def episode_valid(ledger: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """(episode_valid, freeze_ok, dramatic_ok).

    freeze_ok  = freeze_verdict indicates a frozen episode (CRITICAL pass).
    dramatic_ok = slot_drama_contracts_audit.ok (the dramatic-contract pass);
                  absent -> treat as the freeze result (no separate signal).
    """
    m = _meta(ledger)
    verdict = str(m.get("freeze_verdict") or "").strip().lower()
    freeze_ok = verdict.startswith("frozen")
    audit = m.get("slot_drama_contracts_audit")
    if isinstance(audit, dict) and isinstance(audit.get("ok"), bool):
        dramatic_ok = bool(audit["ok"])
    elif isinstance(audit, dict) and "valid" in audit and isinstance(audit["valid"], bool):
        dramatic_ok = bool(audit["valid"])
    else:
        dramatic_ok = freeze_ok
    return (freeze_ok and dramatic_ok, freeze_ok, dramatic_ok)


def find_outro_text(ledger: Dict[str, Any]) -> str:
    """The episode's closing announcer line (last announcer-role line)."""
    outro = ""
    for ln in _lines(ledger):
        if str(ln.get("speaker_role") or "").strip() == "announcer":
            t = str(ln.get("text") or "").strip()
            if t:
                outro = t
    return outro


def outro_hedge_vs_resolved(ledger: Dict[str, Any]) -> bool:
    """True when the outro hedges WHILE the ending is resolved (a contradiction)."""
    m = _meta(ledger)
    ds = m.get("dramatic_state")
    ending = ds.get("ending_change") if isinstance(ds, dict) else ""
    if not is_resolved_ending_change(ending):
        return False
    outro = find_outro_text(ledger).lower()
    return any(phrase in outro for phrase in HEDGE_LIST)


def narration_self_address_lines(ledger: Dict[str, Any]) -> int:
    """Count of CHARACTER lines flagged by the F7 narration detector."""
    n = 0
    for ln in _lines(ledger):
        if str(ln.get("speaker_role") or "").strip() != "character":
            continue
        if detect_narration_self_address(ln.get("text"), ln.get("char_name") or ln.get("name") or ""):
            n += 1
    return n


def scan_ledger(path: str, target_override: Optional[int]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    if not isinstance(ledger, dict):
        raise ValueError("ledger top level is not a dict: %s" % path)
    valid, freeze_ok, dramatic_ok = episode_valid(ledger)
    m = _meta(ledger)
    return {
        "ledger": os.path.basename(path),
        "episode_id": str(ledger.get("episode_id") or ""),
        "target_words": resolve_target_words(ledger, target_override),
        "voiced_words": voiced_word_total(ledger),
        "length_ratio": length_ratio(ledger, target_override),
        "length_pass_fired": length_pass_fired(ledger),
        "episode_valid": valid,
        "freeze_ok": freeze_ok,
        "dramatic_ok": dramatic_ok,
        "freeze_verdict": str(m.get("freeze_verdict") or ""),
        "dramatic_state_source": str(m.get("dramatic_state_source") or ""),
        "arc_shape": str(m.get("arc_shape") or ""),
        "outro_hedge_vs_resolved": outro_hedge_vs_resolved(ledger),
        "narration_self_address_lines": narration_self_address_lines(ledger),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ratios = [r["length_ratio"] for r in rows if isinstance(r["length_ratio"], (int, float))]
    return {
        "n_legs": n,
        "length_ratio_mean": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "length_pass_fired_count": sum(1 for r in rows if r["length_pass_fired"] is True),
        "episode_valid_count": sum(1 for r in rows if r["episode_valid"]),
        "outro_hedge_vs_resolved_count": sum(1 for r in rows if r["outro_hedge_vs_resolved"]),
        "narration_self_address_total": sum(int(r["narration_self_address_lines"]) for r in rows),
        "arc_shapes": sorted({r["arc_shape"] for r in rows if r["arc_shape"]}),
        "engine_detector": _HAS_ENGINE_DETECTOR,
    }


def render_markdown(agg: Dict[str, Any], rows: List[Dict[str, Any]], label: str) -> str:
    n = agg["n_legs"]
    out = ["## story_quality_scan -- %s (%d legs)" % (label, n), ""]
    out.append("| metric | value | v1 target |")
    out.append("|---|---|---|")
    lr = agg["length_ratio_mean"]
    out.append("| length_ratio mean | %s | >= 0.85 |" % (lr if lr is not None else "n/a"))
    out.append("| length_pass_fired | %d/%d | <= 2/12 |" % (agg["length_pass_fired_count"], n))
    out.append("| episode_valid | %d/%d | >= 11/12 |" % (agg["episode_valid_count"], n))
    out.append("| outro_hedge_vs_resolved | %d/%d | 0/12 |" % (agg["outro_hedge_vs_resolved_count"], n))
    out.append("| narration_self_address | %d | 0 |" % agg["narration_self_address_total"])
    out.append("| arc_shapes seen | %s | not single-valued |" % (", ".join(agg["arc_shapes"]) or "(none)"))
    out.append("")
    out.append("| leg | ratio | valid | hedge | narr | arc_shape | ds_source |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append("| %s | %s | %s | %s | %d | %s | %s |" % (
            r["ledger"], r["length_ratio"], r["episode_valid"],
            r["outro_hedge_vs_resolved"], r["narration_self_address_lines"],
            r["arc_shape"] or "-", r["dramatic_state_source"] or "-",
        ))
    out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="OTR story-quality scan")
    ap.add_argument("--ledgers", required=True,
                    help="glob for *_ledger.json (quote it)")
    ap.add_argument("--target-words", type=int, default=0,
                    help="override target_words (e.g. 864 for the fixed smoke)")
    ap.add_argument("--label", default="scan")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.ledgers))
    if not paths:
        print("[story_quality_scan] no ledgers matched: %s" % args.ledgers,
              file=sys.stderr)
        return 2
    override = args.target_words or None
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows.append(scan_ledger(p, override))
        except Exception as exc:  # noqa: BLE001
            print("[story_quality_scan] skip %s: %r" % (p, exc), file=sys.stderr)
    agg = aggregate(rows)
    md = render_markdown(agg, rows, args.label)
    print(md)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "aggregate": agg, "legs": rows},
                      f, ensure_ascii=False, indent=1)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
