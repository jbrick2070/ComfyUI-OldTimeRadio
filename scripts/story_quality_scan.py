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
# R2 craft-lever helpers (S2/S3/C0/C1/C2/C5). Guarded so the baseline still runs
# if the helpers are absent (they no-op to "clean").
try:
    from nodes._otr_line_hygiene import (  # type: ignore
        detect_leading_stage_business, flag_cliche, flag_on_the_nose,
        flag_stage_business, flag_thesis_close,
        # 3.x (2026-06-27) gate-seam detectors -- re-run on final text by the scan.
        detect_mojibake, extract_specificity_anchors_from_header,
        flag_anchor_stuffing, flag_one_breath, flag_personal_cost_boilerplate,
        is_whole_line_stage_action,
        # G1 (story-quality v2, 2026-06-28) -- budget-derived one-breath cap.
        derive_one_breath_cap,
    )
    from nodes._otr_dramatic_state import wants_are_default  # type: ignore
    from nodes._otr_casting import (  # type: ignore
        _SIG_NEAR_DUP_THRESHOLD, speech_signature_overlap,
    )
    _HAS_R2_HELPERS = True
except Exception:  # noqa: BLE001
    _HAS_R2_HELPERS = False

    def _r2_false(*_a, **_k):  # type: ignore
        return (False, "")

    flag_cliche = flag_on_the_nose = flag_stage_business = flag_thesis_close = _r2_false  # type: ignore
    flag_anchor_stuffing = flag_one_breath = flag_personal_cost_boilerplate = _r2_false  # type: ignore

    def derive_one_breath_cap(_r):  # type: ignore  # G1 fallback: legacy 28 cap
        return 28

    def detect_leading_stage_business(_t):  # type: ignore
        return (False, "")

    def wants_are_default(_s):  # type: ignore
        return False

    def extract_specificity_anchors_from_header(_h):  # type: ignore
        return []

    def is_whole_line_stage_action(_t, **_k):  # type: ignore
        return False

    def detect_mojibake(_t):  # type: ignore
        return False

    def speech_signature_overlap(_a, _b):  # type: ignore
        return 0.0

    _SIG_NEAR_DUP_THRESHOLD = 0.5  # type: ignore

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


def r2_lever_metrics(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Story-Quality R2 craft-lever counts over a frozen ledger (read-only).

    Per-line flag counts (the weak end should DROP after the levers ship) + the
    structural signals: a thesis close, default boilerplate wants, whether the
    specificity anchors / central object were derived, and voice distinctness.
    """
    m = _meta(ledger)
    char = [ln for ln in _lines(ledger)
            if str(ln.get("speaker_role") or "").strip() == "character"]
    cliche = sum(1 for ln in char if flag_cliche(ln.get("text"))[0])
    nose = sum(1 for ln in char if flag_on_the_nose(ln.get("text"))[0])
    biz = sum(1 for ln in char if flag_stage_business(ln.get("text"))[0])
    lead_sd = sum(1 for ln in char if detect_leading_stage_business(ln.get("text"))[0])

    # default-wants classifier over meta.dramatic_state (shim a state object)
    ds = m.get("dramatic_state")
    wants_default = None
    if isinstance(ds, dict):
        from types import SimpleNamespace
        wants_default = bool(wants_are_default(SimpleNamespace(
            character_a_wants=ds.get("character_a_wants", ""),
            character_b_wants=ds.get("character_b_wants", ""))))

    # voice distinctness: distinct speech registers / number of character voices
    sigs = []
    for c in (ledger.get("cast") or []):
        if isinstance(c, dict) and str(c.get("name") or "").upper() != "ANNOUNCER":
            sigs.append(" ".join(str(c.get("speech_signature") or "").lower().split()))
    sigs = [s for s in sigs if s]
    voice_distinct = (round(len(set(sigs)) / len(sigs), 3) if sigs else None)

    anchors = m.get("specificity_anchors")
    return {
        "thesis_close": bool(flag_thesis_close(find_outro_text(ledger))[0]),
        "cliche_lines": cliche,
        "on_the_nose_lines": nose,
        "stage_business_lines": biz,
        "leading_stage_dir_lines": lead_sd,
        "wants_default": wants_default,
        "has_specificity_anchors": bool(anchors),
        "n_specificity_anchors": len(anchors) if isinstance(anchors, list) else 0,
        "has_central_object": bool(m.get("central_object")),
        "voice_distinct_ratio": voice_distinct,
    }


# ---------------------------------------------------------------------------
# Story-Quality 3.x gate-seam cluster (2026-06-27) -- read-only counters.
# ---------------------------------------------------------------------------
#: Low-effort generic close openers (the close should land the real news image).
_GENERIC_BRIDGE_OPENERS = (
    "the real story", "the true account", "and now, the real world",
    "meanwhile, in the real world", "but in the real world",
    "now, the real world",
)
#: 3.1 ownership-template surface markers (over a people-class central object).
_OWNERSHIP_TEMPLATE_MARKERS = (
    "take sole credit for", "control what becomes of", "control of",
    "sole credit", "ownership of", "passes to whoever",
)
_PEOPLE_HEAD_NOUNS_SCAN = frozenset({
    "people", "person", "persons", "men", "women", "children", "kids",
    "residents", "patients", "workers", "citizens", "population", "populations",
    "community", "communities", "humanity", "victims", "families", "individuals",
    "group", "groups",
})


def _compose_flags(ln: Dict[str, Any]) -> List[str]:
    cf = ln.get("compose_flags")
    return [str(x) for x in cf] if isinstance(cf, list) else []


def _is_people_class_object_scan(term: Any) -> bool:
    words = re.findall(r"[A-Za-z]+", str(term or "").casefold())
    if not words:
        return False
    head = words[-1]
    return head in _PEOPLE_HEAD_NOUNS_SCAN or (
        head.endswith("s") and head[:-1] in _PEOPLE_HEAD_NOUNS_SCAN)


def r3_quality_metrics(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Story-Quality 3.x gate-seam counts over a frozen ledger (read-only).

    Re-runs the v2 detectors on FINAL text (anchor-stuffing, one-breath,
    whole-line stage-action, personal-cost boilerplate) and reads the engine
    -stamped quality_* / news_coda_* / dramatic_state_* breadcrumbs. mojibake +
    generic-bridge + register-overlap are scan-derived. Never raises -> zeros."""
    m = _meta(ledger)
    lines = _lines(ledger)
    char = [ln for ln in lines
            if str(ln.get("speaker_role") or "").strip() == "character"]
    anchors = m.get("specificity_anchors")
    anchors = anchors if isinstance(anchors, list) else []

    anchor_stuffing = sum(
        1 for ln in char if flag_anchor_stuffing(ln.get("text"), anchors)[0])
    # G1 (2026-06-28): score one-breath on the SAME budget-derived cap + relaxed
    # clause threshold the composer gate + reroll used (one derive_one_breath_cap).
    # Absent words_per_beat_range (v2-OFF ledger) => 28/3 => legacy-identical count.
    _ob_cap = derive_one_breath_cap(m.get("words_per_beat_range"))
    _ob_clause = max(3, _ob_cap // 8)
    one_breath = sum(
        1 for ln in char
        if flag_one_breath(ln.get("text"), max_words=_ob_cap,
                           max_clause_markers=_ob_clause)[0])
    stage_action_leak = sum(
        1 for ln in char if is_whole_line_stage_action(ln.get("text")))
    personal_cost = sum(
        1 for ln in char if flag_personal_cost_boilerplate(ln.get("text"))[0])

    quality_retry = sum(
        1 for ln in char if any(f.endswith("_retry") for f in _compose_flags(ln)))
    quality_residual = sum(
        1 for ln in char
        if any(f.startswith("quality_residual:") for f in _compose_flags(ln)))
    quality_degraded = sum(
        1 for ln in char if "quality_reroll_degraded" in _compose_flags(ln))

    coda_trunc = sum(1 for ln in lines if "news_coda_truncated" in _compose_flags(ln))
    coda_fallback = sum(1 for ln in lines if "news_coda_fallback" in _compose_flags(ln))
    outro = find_outro_text(ledger)
    coda_mojibake = 1 if detect_mojibake(outro) else 0
    low_outro = outro.strip().lower()
    coda_generic = 1 if any(
        low_outro.startswith(p) for p in _GENERIC_BRIDGE_OPENERS) else 0

    ds_source = str(m.get("dramatic_state_source") or "")
    ds_fallback = 1 if ds_source == "fallback" else 0
    ds_replaced = 1 if m.get("dramatic_state_fallback_term_replaced") else 0
    central = m.get("central_object")
    ownable_people = 1 if _is_people_class_object_scan(central) else 0
    ownership_on_nonownable = 0
    if ownable_people:
        ds = m.get("dramatic_state")
        blob = ""
        if isinstance(ds, dict):
            blob = " ".join(str(ds.get(k) or "") for k in (
                "character_a_wants", "character_b_wants", "ending_change"))
        if any(mk in blob.lower() for mk in _OWNERSHIP_TEMPLATE_MARKERS):
            ownership_on_nonownable = 1

    sigs = []
    for c in (ledger.get("cast") or []):
        if isinstance(c, dict) and str(c.get("name") or "").upper() != "ANNOUNCER":
            s = str(c.get("speech_signature") or "")
            if s.strip():
                sigs.append(s)
    near_dup = 0
    max_overlap = 0.0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            ov = speech_signature_overlap(sigs[i], sigs[j])
            max_overlap = max(max_overlap, ov)
            if ov >= _SIG_NEAR_DUP_THRESHOLD:
                near_dup += 1

    return {
        "anchor_stuffing_lines": anchor_stuffing,
        "one_breath_violation_lines": one_breath,
        "stage_action_leak_lines": stage_action_leak,
        "personal_cost_boilerplate_lines": personal_cost,
        "quality_retry_lines": quality_retry,
        "quality_residual_lines": quality_residual,
        "quality_reroll_degraded_lines": quality_degraded,
        "news_coda_truncated_count": coda_trunc,
        "news_coda_fallback_count": coda_fallback,
        "news_coda_mojibake_count": coda_mojibake,
        "news_coda_generic_bridge_count": coda_generic,
        "dramatic_state_fallback_count": ds_fallback,
        "dramatic_state_fallback_replaced_count": ds_replaced,
        "ownable_people_object_count": ownable_people,
        "ownership_template_on_nonownable_count": ownership_on_nonownable,
        "speech_signature_near_duplicate_count": near_dup,
        "register_overlap_ratio": round(max_overlap, 3),
        "r2_helpers_loaded": bool(_HAS_R2_HELPERS),
    }


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
        **r2_lever_metrics(ledger),
        **r3_quality_metrics(ledger),
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
        # Story-Quality 3.x gate-seam totals (2026-06-27).
        "anchor_stuffing_total": sum(int(r.get("anchor_stuffing_lines", 0)) for r in rows),
        "one_breath_violation_total": sum(int(r.get("one_breath_violation_lines", 0)) for r in rows),
        "stage_action_leak_total": sum(int(r.get("stage_action_leak_lines", 0)) for r in rows),
        "personal_cost_boilerplate_total": sum(int(r.get("personal_cost_boilerplate_lines", 0)) for r in rows),
        "quality_retry_total": sum(int(r.get("quality_retry_lines", 0)) for r in rows),
        "quality_residual_total": sum(int(r.get("quality_residual_lines", 0)) for r in rows),
        "quality_reroll_degraded_total": sum(int(r.get("quality_reroll_degraded_lines", 0)) for r in rows),
        "news_coda_truncated_total": sum(int(r.get("news_coda_truncated_count", 0)) for r in rows),
        "news_coda_fallback_total": sum(int(r.get("news_coda_fallback_count", 0)) for r in rows),
        "news_coda_mojibake_total": sum(int(r.get("news_coda_mojibake_count", 0)) for r in rows),
        "news_coda_generic_bridge_total": sum(int(r.get("news_coda_generic_bridge_count", 0)) for r in rows),
        "dramatic_state_fallback_total": sum(int(r.get("dramatic_state_fallback_count", 0)) for r in rows),
        "dramatic_state_fallback_replaced_total": sum(int(r.get("dramatic_state_fallback_replaced_count", 0)) for r in rows),
        "ownable_people_object_total": sum(int(r.get("ownable_people_object_count", 0)) for r in rows),
        "ownership_template_on_nonownable_total": sum(int(r.get("ownership_template_on_nonownable_count", 0)) for r in rows),
        "speech_signature_near_duplicate_total": sum(int(r.get("speech_signature_near_duplicate_count", 0)) for r in rows),
        "r2_helpers_loaded": bool(_HAS_R2_HELPERS),
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
