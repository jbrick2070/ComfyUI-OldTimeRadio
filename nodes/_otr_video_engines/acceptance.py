"""WIRE-W5 -- THE ACCEPTANCE GRADER. Did the episode render what it froze?

r4/A6, verbatim: *"Per shot require ``video.shots[].engine_id ==
video.roles_effective[shot.role]``, then require every delivered clip-manifest
row's ``engine_id`` to match that frozen expected value. CUT aggregate engine
histograms as acceptance evidence -- they cannot detect two shots EXCHANGING
engines while totals stay identical. Grade OBS publication and canonical
artifacts separately. Never query live routing state."*

THREE THINGS THIS MODULE REFUSES TO DO, each because a panel round said so:

* **It never queries live routing state.** The director freezes the route at
  plan time and ShotLock validates it there; asking ``route_freeze`` again at
  grading time is a CLOCK-DOMAIN MISMATCH -- the environment has moved on, so a
  disagreement would report the grader's clock rather than the episode's. This
  module takes the two DOCUMENTS and nothing else, and it imports nothing that
  could reach an environment (a test asserts the import list).
* **It never uses the engine histogram.** ``manifest["engine_histogram"]`` is
  right there and it is useless as evidence: two shots that EXCHANGE engines
  leave every total identical. Acceptance is per-shot or it is decorative.
* **It never grades a composited frame.** kibitz r1 (2026-07-29) proved the
  trap with a shipped test: ``test_credits_roll_spec.py:446-470`` scrolls text
  over a DELIBERATELY CONSTANT backdrop, so "did the frame change" goes green
  on a frozen background because the overlay moves. Grade the SOURCE
  COMPONENTS -- the per-clip receipts below -- before anything is overlaid.

Pure and stdlib-only. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations


#: One acceptance failure. Plain dicts rather than a class so the findings
#: survive JSON round-trip into a receipt without a codec.
def _finding(rule, shot_id, detail):
    return {"rule": str(rule), "shot_id": str(shot_id or ""),
            "detail": str(detail)}


#: The route the ledger FROZE disagrees with the shot it stamped.
RULE_FROZEN_ROUTE = "frozen_route"

#: The clip that was DELIVERED came from a different engine than the frozen one.
RULE_DELIVERED_ENGINE = "delivered_engine"

#: A beat the plan says needs several real clips delivered a padded one.
RULE_MULTICLIP_HONESTY = "multiclip_honesty"

#: A beat that planned coverage delivered no clip at all.
RULE_MISSING_CLIP = "missing_clip"


def frozen_route(ledger):
    """``{role: engine_id}`` as the ledger FROZE it, or ``{}``.

    ``video.roles_effective`` is stamped by ShotLock from the policy's
    ``effective_video_models``. Absent on a legacy ledger, which is reported by
    the caller rather than guessed at here -- an empty map must not read as
    "every shot agrees"."""
    return dict(((ledger or {}).get("video") or {}).get("roles_effective") or {})


def _shots(ledger):
    return list(((ledger or {}).get("video") or {}).get("shots") or ())


def grade_frozen_route(ledger):
    """Every shot renders the engine its ROLE was frozen to.

    The first half of A6, and it catches the case a per-role check cannot: a
    shot whose ``engine_id`` was rewritten after the freeze. A role absent from
    the frozen map is itself a finding -- an unfrozen role is a role whose
    delivery cannot be judged at all."""
    frozen = frozen_route(ledger)
    findings = []
    for shot in _shots(ledger):
        shot_id = shot.get("shot_id")
        role = str(shot.get("role") or "")
        actual = str(shot.get("engine_id") or "")
        if role not in frozen:
            findings.append(_finding(
                RULE_FROZEN_ROUTE, shot_id,
                "role %r is not in the ledger's frozen route %s, so what this "
                "shot was SUPPOSED to render is unknowable"
                % (role, sorted(frozen))))
            continue
        expected = str(frozen.get(role) or "")
        if actual != expected:
            findings.append(_finding(
                RULE_FROZEN_ROUTE, shot_id,
                "role %r froze to %r but the shot row stamps %r"
                % (role, expected, actual)))
    return findings


def _plan_segment_count(shot):
    plan = shot.get("coverage_plan") or {}
    return len(plan.get("segments") or ())


def grade_delivered(ledger, manifest):
    """Every DELIVERED clip came from the engine its role froze to.

    The second half of A6. Judged per shot against the frozen expectation --
    never against the shot row, because a rewritten shot row would then agree
    with its own rewrite, and never in aggregate, because two shots exchanging
    engines leave every total identical.

    A planned beat with no delivered row is a separate, named finding: a
    missing clip and a wrong clip are different failures and an operator fixes
    them differently."""
    frozen = frozen_route(ledger)
    rows = {str(r.get("shot_id") or ""): r
            for r in ((manifest or {}).get("clips") or ())}
    findings = []
    for shot in _shots(ledger):
        shot_id = str(shot.get("shot_id") or "")
        if int(shot.get("target_frame_count") or 0) <= 0:
            continue                       # renders nothing; owes nothing
        row = rows.get(shot_id)
        if row is None or not row.get("exists"):
            findings.append(_finding(
                RULE_MISSING_CLIP, shot_id,
                "the plan gave this beat %d frame(s) and the manifest carries "
                "%s" % (int(shot.get("target_frame_count") or 0),
                        "no row for it" if row is None
                        else "no clip on disk")))
            continue
        expected = str(frozen.get(str(shot.get("role") or "")) or "")
        delivered = str(row.get("engine_id") or "")
        if expected and delivered != expected:
            findings.append(_finding(
                RULE_DELIVERED_ENGINE, shot_id,
                "role %r froze to %r but %r delivered the clip"
                % (shot.get("role"), expected, delivered)))
    return findings


def grade_multiclip_honesty(ledger, manifest):
    """A beat the plan splits into real clips may not deliver a PADDED one.

    ``frame_count`` cannot answer this: a ping-pong-extended clip carries
    exactly the number a real one does, which is what makes the pad forgeable.
    ``extension_mode`` and ``native_frame_count`` are the receipts WIRE-W3b put
    on every WAN clip for precisely this check.

    SILENCE IS NOT A PASS. A multi-clip beat whose row carries NO extension
    receipt at all is reported too -- an engine that never declares how its
    frames got there cannot be graded, and "no receipt" is exactly what a lane
    that pads without saying so looks like."""
    rows = {str(r.get("shot_id") or ""): r
            for r in ((manifest or {}).get("clips") or ())}
    findings = []
    for shot in _shots(ledger):
        if _plan_segment_count(shot) <= 1:
            continue
        shot_id = str(shot.get("shot_id") or "")
        row = rows.get(shot_id)
        if row is None or not row.get("exists"):
            continue                       # already reported by grade_delivered
        mode = row.get("extension_mode")
        if mode is None:
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat is planned across %d clips but %r declares no "
                "extension_mode, so nothing says whether its frames were "
                "rendered or mirrored"
                % (_plan_segment_count(shot), row.get("engine_id"))))
            continue
        if str(mode) != "none":
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat is planned across %d clips and %r delivered it with "
                "extension_mode=%r -- a lane claiming real multi-clip coverage "
                "may not pad" % (_plan_segment_count(shot),
                                 row.get("engine_id"), mode)))
            continue
        native = row.get("native_frame_count")
        if native is not None and int(native) != int(row.get("frame_count") or 0):
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat delivered %s frame(s) of which only %s were "
                "rendered, while declaring extension_mode='none'"
                % (row.get("frame_count"), native)))
    return findings


def grade_episode(ledger, manifest):
    """Every per-shot rule, in one pass. Returns a list of findings; EMPTY
    means the episode delivered the route it froze.

    Deliberately returns rather than raises: the caller decides whether a
    finding blocks (the durable script exits non-zero) or is recorded, and a
    grader that raised on the first problem would hide the rest."""
    return (grade_frozen_route(ledger)
            + grade_delivered(ledger, manifest)
            + grade_multiclip_honesty(ledger, manifest))


def format_findings(findings):
    """One line per finding, stable order, for a log or a console."""
    return "\n".join(
        "%-20s %-24s %s" % (f.get("rule"), f.get("shot_id"), f.get("detail"))
        for f in (findings or ()))


__all__ = [
    "RULE_FROZEN_ROUTE",
    "RULE_DELIVERED_ENGINE",
    "RULE_MULTICLIP_HONESTY",
    "RULE_MISSING_CLIP",
    "frozen_route",
    "grade_frozen_route",
    "grade_delivered",
    "grade_multiclip_honesty",
    "grade_episode",
    "format_findings",
]
