"""Audit published ledgers for cast rows whose gender, voice and portrait disagree.

The operator's rule is CONSISTENCY BEATS ACCURACY: a female Scrooge with a female
voice and a female portrait is coherent but unfaithful; a male Scrooge with a
female voice is broken. This checks the coherence half. It does not care whether
the gender is the historically correct one -- that is a different item.

ENGINE-AWARE BY CONSTRUCTION. Each engine declares which identity field it
actually speaks (``_otr_voice_node_common``), so checking every field on every
row would both miss real defects and invent fake ones. The ACTIVE field -- the
one the row's engine consumes -- is a violation when it disagrees. Every other
populated identity field is reported separately as DORMANT: worth knowing, since
21 modules read ``voice_preset`` and a bark leg would speak it, but not the same
claim as "the audience heard the wrong gender".

Read-only. Never writes to a ledger.

Exit codes:
  0  every included ledger is consistent
  1  at least one ACTIVE-field violation on a policy-era ledger
  2  the scan is INCOMPLETE and its verdict cannot be trusted -- an unreadable
     ledger, an unreadable authority, a bad root, or zero ledgers found

The 2 is deliberate and is not what ``audit_spoken_citations.py`` does: that one
reports an operational error only when it has no findings, so a partial scan can
masquerade as a complete clean report. A scan that did not finish is not a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_roster_gender import (  # noqa: E402
    VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY,
    VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION,
    get_presentation_gender,
    normalize_gender,
)


class AuthorityError(RuntimeError):
    """A resolver input could not be loaded, so the scan is not trustworthy."""


# --------------------------------------------------------------------------- #
# Authorities


def load_authorities() -> Tuple[Dict[str, str], Dict[str, str], str, str]:
    """(ref_gender_by_id, preset_gender_by_id, bank_sha, preset_digest).

    Raises AuthorityError rather than degrading: an audit that silently loses
    its gender authority would report every row as unresolvable and call it
    clean.
    """
    try:
        from nodes._otr_voice_bank import load_voice_bank
        bank, bank_sha = load_voice_bank()
    except Exception as exc:  # noqa: BLE001
        raise AuthorityError(f"voice bank unavailable: {exc!r}") from exc
    if not bank:
        raise AuthorityError("voice bank loaded but is empty")
    ref_gender = {
        e.voice_ref_id: str(getattr(e, "gender", "") or "").strip().lower()
        for e in bank
    }
    try:
        from config import cast_pools
        profiles = list(getattr(cast_pools, "VOICE_PROFILES", ()))
    except Exception as exc:  # noqa: BLE001
        raise AuthorityError(f"cast_pools unavailable: {exc!r}") from exc
    if not profiles:
        raise AuthorityError("cast_pools.VOICE_PROFILES is empty")
    preset_gender = {
        str(p): str(g).strip().lower() for p, g, _lang, _tags in profiles
    }
    digest = hashlib.sha1(
        "|".join(f"{p}:{g}" for p, g in sorted(preset_gender.items()))
        .encode("utf-8")
    ).hexdigest()[:12]
    return ref_gender, preset_gender, str(bank_sha or "")[:12], digest


def load_workflow_engines(path: str) -> Dict[str, Any]:
    """Engine widget values declared in the canonical workflow.

    Nodes 81/82 dispatch their OWN engine widget, independently of the engine
    CastLock stamps into meta. That disagreement cannot be seen from a ledger at
    all -- it exists only in the graph -- so without this the audit can certify a
    stamped voice while a different engine actually speaks.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            graph = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise AuthorityError(f"workflow unreadable at {path}: {exc!r}") from exc
    sha = hashlib.sha256(
        json.dumps(graph, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    wanted = ("OTR_BatchCharacterVoices", "OTR_AnnouncerVoice", "OTR_CastLock")
    found: Dict[str, List[Any]] = {k: [] for k in wanted}
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        ntype = str(node.get("type") or "")
        if ntype in found:
            found[ntype].append(node.get("widgets_values") or [])
    ambiguous = [k for k, v in found.items() if len(v) != 1]
    return {"sha": sha, "nodes": found, "ambiguous": ambiguous}


# --------------------------------------------------------------------------- #
# Per-row resolution


#: Which cast-row field each engine actually speaks. The default is the clone
#: path, which resolves a reference id to a file. Mirrors
#: `_otr_voice_node_common`'s `voice_ref_field` dispatch -- kept as data here so
#: the audit states its assumption instead of importing render machinery.
_ACTIVE_FIELD_BY_ENGINE = {
    "bark": "voice_preset",
    "kokoro": "voice_ref_id",
    "google_tts": "provider_voice_id",
    "elevenlabs": "provider_voice_id",
    "cloud_elevenlabs": "provider_voice_id",
}
_DEFAULT_ACTIVE_FIELD = "voice_ref_id"   # indextts2 / chatterbox / dia


def active_field_for(engine: str) -> str:
    return _ACTIVE_FIELD_BY_ENGINE.get(
        str(engine or "").strip().lower(), _DEFAULT_ACTIVE_FIELD)


def is_announcer(row: Dict[str, Any]) -> bool:
    return "ANNOUNCER" in str(row.get("name") or "").upper()


def field_gender(field: str, row: Dict[str, Any], ref_gender, preset_gender) -> str:
    """The gender a populated identity field implies, or '' when it is absent
    or unresolvable. The two are reported differently and never as agreement."""
    value = str(row.get(field) or "").strip()
    if not value:
        return ""
    if field == "voice_preset":
        return preset_gender.get(value) or ref_gender.get(value) or "?"
    if field == "voice_ref_id":
        return ref_gender.get(value) or "?"
    return "?"     # provider ids carry no gender we can resolve offline


def audit_row(row, engine, ref_gender, preset_gender) -> Dict[str, Any]:
    """One cast row -> {expected, active, active_gender, verdict, dormant[]}."""
    expected = get_presentation_gender(row)
    active = active_field_for(engine)
    active_gender = field_gender(active, row, ref_gender, preset_gender)

    if not active_gender:
        verdict = "active_absent"
    elif active_gender == "?":
        verdict = "active_unresolved"
    elif active_gender == expected:
        verdict = "agree"
    else:
        verdict = "DISAGREE"

    dormant = []
    for field in ("voice_preset", "voice_ref_id"):
        if field == active:
            continue
        g = field_gender(field, row, ref_gender, preset_gender)
        if g and g != "?" and g != expected:
            dormant.append({"field": field, "implies": g})
    return {
        "char": str(row.get("name") or row.get("char_id") or "?"),
        "label_gender": normalize_gender(row.get("gender")),
        "expected": expected,
        "active_field": active,
        "active_gender": active_gender,
        "verdict": verdict,
        "dormant": dormant,
    }


def iter_ledgers(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith("_ledger.json"):
                yield os.path.join(dirpath, name)


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit cast rows for gender/voice/portrait consistency.")
    ap.add_argument("--root", required=True,
                    help="Directory scanned recursively for *_ledger.json.")
    ap.add_argument("--workflow",
                    default=os.path.join(_REPO, "workflows", "otr_canonical.json"),
                    help="Canonical workflow, for the intended-vs-delivered "
                         "engine check.")
    ap.add_argument("--fail-on-legacy", action="store_true",
                    help="Also fail on pre-policy ledgers (default: report "
                         "them, do not fail -- they were frozen before the "
                         "contract existed).")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"OPERATIONAL: --root is not a directory: {args.root}")
        return 2
    try:
        ref_gender, preset_gender, bank_sha, preset_digest = load_authorities()
        workflow = load_workflow_engines(args.workflow)
    except AuthorityError as exc:
        print(f"OPERATIONAL: {exc}")
        return 2

    print("== voice/gender consistency audit ==")
    print(f"root              : {args.root}")
    print(f"policy revision   : {VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION} "
          f"(key {VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY})")
    print(f"voice bank sha    : {bank_sha}")
    print(f"preset table sha  : {preset_digest}")
    print(f"workflow sha      : {workflow['sha']}")
    print("inclusion         : every *_ledger.json under root")
    print("announcer         : audited; its reference is episode-seeded by "
          "design, so its presentation_gender is compared, never its label")
    print("active field      : the one the row's engine speaks; other populated "
          "identity fields are reported as DORMANT, not as violations")
    if workflow["ambiguous"]:
        print(f"OPERATIONAL: workflow node(s) missing or duplicated: "
              f"{workflow['ambiguous']}")
        return 2

    scanned = unreadable = 0
    policy_ledgers = legacy_ledgers = 0
    violations: List[Dict[str, Any]] = []
    legacy_findings = 0
    dormant_rows = 0
    absent_rows = unresolved_rows = 0

    for path in iter_ledgers(args.root):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                led = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            unreadable += 1
            print(f"UNREADABLE {path}: {exc!r}")
            continue
        if not isinstance(led, dict):
            unreadable += 1
            print(f"UNREADABLE {path}: not a JSON object")
            continue
        scanned += 1
        meta = led.get("meta") if isinstance(led.get("meta"), dict) else {}
        cast = led.get("cast")
        if not isinstance(cast, list):
            continue
        revision = meta.get(VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY)
        is_policy = isinstance(revision, int) and revision >= 1
        if is_policy:
            policy_ledgers += 1
        else:
            legacy_ledgers += 1

        char_engine = str(meta.get("char_voice_engine") or "")
        ann_engine = str(meta.get("announcer_voice_engine") or "")

        for row in cast:
            if not isinstance(row, dict):
                continue
            engine = ann_engine if is_announcer(row) else char_engine
            res = audit_row(row, engine, ref_gender, preset_gender)
            if res["dormant"]:
                dormant_rows += 1
            if res["verdict"] == "active_absent":
                absent_rows += 1
            elif res["verdict"] == "active_unresolved":
                unresolved_rows += 1
            elif res["verdict"] == "DISAGREE":
                record = dict(res)
                record["ledger"] = path
                record["policy"] = is_policy
                if is_policy:
                    violations.append(record)
                else:
                    legacy_findings += 1
                    if legacy_findings <= 10:
                        print(f"LEGACY  {os.path.basename(os.path.dirname(path))} "
                              f"{res['char']}: expected {res['expected']}, "
                              f"{res['active_field']} implies "
                              f"{res['active_gender']}")

    print()
    print(f"scanned           : {scanned} ledger(s) "
          f"({policy_ledgers} policy, {legacy_ledgers} legacy)")
    print(f"unreadable        : {unreadable}")
    print(f"active absent     : {absent_rows} row(s)")
    print(f"active unresolved : {unresolved_rows} row(s)")
    print(f"dormant mismatch  : {dormant_rows} row(s) "
          f"(inaudible today, still a ledger that disagrees with itself)")
    print(f"legacy findings   : {legacy_findings} (pre-policy, not failed "
          f"unless --fail-on-legacy)")
    print(f"VIOLATIONS        : {len(violations)}")
    for v in violations[:40]:
        print(f"  {os.path.basename(os.path.dirname(v['ledger']))} "
              f"{v['char']}: expected {v['expected']}, {v['active_field']} "
              f"implies {v['active_gender']}")

    if unreadable or scanned == 0:
        print("OPERATIONAL: the scan is incomplete; its verdict is not a pass.")
        return 2
    if violations:
        return 1
    if args.fail_on_legacy and legacy_findings:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
