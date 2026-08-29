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

#: A delivered clip declares that some of its frames were MANUFACTURED.
#: Separate from :data:`RULE_MULTICLIP_HONESTY` on purpose: that rule asks
#: whether a multi-clip beat's arithmetic adds up, and this one asks the flat
#: policy question of EVERY beat, single- or multi-clip, with no counts and no
#: projection involved. A padded single-render clip was invisible to the
#: honesty rule -- it skips plans of one segment -- so the mirror ban needed a
#: rule that never skips anything.
RULE_NO_MIRROR = "no_mirror"

#: The manifest is not shaped like a manifest. Reported instead of crashing, and
#: reported BEFORE the semantic rules read it, so one malformed row produces one
#: named finding rather than a cascade of nonsense from every other rule.
RULE_MANIFEST_SHAPE = "manifest_shape"

#: The ledger's own shot rows are not shaped like shot rows. The manifest half
#: of this was scheduled; the LEDGER half was scheduled nowhere, and
#: ``grade_frozen_route`` -- kept precisely because it is manifest-independent --
#: still crashed on a malformed shot row.
RULE_LEDGER_SHAPE = "ledger_shape"


def frozen_route(ledger):
    """``{role: engine_id}`` as the ledger FROZE it, or ``{}``.

    ``video.roles_effective`` is stamped by ShotLock from the policy's
    ``effective_video_models``. Absent on a legacy ledger, which is reported by
    the caller rather than guessed at here -- an empty map must not read as
    "every shot agrees"."""
    if not isinstance(ledger, dict):
        return {}
    video = ledger.get("video")
    if not isinstance(video, dict):
        return {}
    frozen = video.get("roles_effective")
    return dict(frozen) if isinstance(frozen, dict) else {}


def _shots(ledger):
    """The ledger's shot rows, RECORDS ONLY. TOTAL -- never raises.

    Non-record rows are FILTERED rather than tolerated, and that is deliberate:
    every rule below does ``shot.get(...)``, so one stray string or list in
    ``video.shots`` raised ``AttributeError`` out of whichever rule reached it
    first. Through ``scripts/grade_episode.py`` that escaped as a traceback with
    exit code 1 -- the code reserved for "graded, findings found" -- so
    automation keying on exit codes read a CRASHED grader as a graded episode.

    The rows that were dropped here are not swallowed: :func:`grade_ledger_shape`
    reports each one by position. This function makes the rules SAFE; that one
    makes the defect VISIBLE. Filtering without reporting would be the worse
    half of the pair.
    """
    if not isinstance(ledger, dict):
        return []
    video = ledger.get("video")
    if not isinstance(video, dict):
        return []
    shots = video.get("shots")
    if not isinstance(shots, (list, tuple)):
        return []
    return [shot for shot in shots if isinstance(shot, dict)]


def manifest_clips(manifest):
    """The manifest's delivered rows as a LIST. TOTAL -- never raises.

    ``(manifest or {}).get("clips")`` was written out at each call site, and it
    is wrong in two ways that only a malformed document reveals. A manifest that
    is not a mapping at all raises ``AttributeError`` out of the grader; and a
    ``clips`` value that is a STRING satisfies ``or ()`` and then iterates into
    its CHARACTERS, so every "row" is a one-character string and every rule
    reports nonsense instead of a shape problem.

    The container rule is deliberately the same one :func:`beat_frame_accounting`
    applies to its segments: list or tuple, nothing else. Two functions judging
    what counts as a sequence of rows must not disagree about it.
    """
    if not isinstance(manifest, dict):
        return []
    clips = manifest.get("clips")
    if not isinstance(clips, (list, tuple)):
        return []
    return list(clips)


def clip_rows_by_shot(manifest):
    """``{shot_id: row}`` over the delivered rows. TOTAL -- never raises.

    THE ONE INDEX. Both per-shot rules built this map with their own dict
    comprehension, and both inherited the same two faults.

    **It overwrote.** ``{str(r.get("shot_id") or ""): r for r in ...}`` keeps the
    LAST row for a repeated ``shot_id`` and silently discards the earlier one, so
    a manifest carrying the same beat twice was graded as though it carried it
    once -- and which of the two got graded depended on file order. Here the
    FIRST occurrence wins and the duplicate is left for ``RULE_MANIFEST_SHAPE``
    to report: a duplicate is a defect to name, never a conflict to resolve
    quietly. That rule HAS landed -- ``grade_manifest_shape()`` detects
    duplicate shot ids and ``grade_episode()`` calls it -- so the duplicate is
    reported rather than silently tolerated. (This paragraph said "NOT YET
    LANDED" until 2026-08-28.) The change here is only WHICH of the two rows
    gets graded, and both regimes are audit-only.

    **It crashed on a non-record.** ``r.get`` on a list, a string or a number
    inside ``clips`` raised ``AttributeError`` straight out of the grader, past
    the durable script's exit-code contract -- so automation keying on exit
    codes read a CRASHED grader as a graded episode. Non-records are skipped
    here and reported there.

    An EMPTY ``shot_id`` is skipped rather than indexed under ``""``. The old
    comprehension gave every anonymous row the same key, so a ledger shot that
    was itself missing a ``shot_id`` would JOIN to an unrelated manifest row
    purely because both were blank. Two things with no identity are not the same
    thing.
    """
    rows = {}
    for row in manifest_clips(manifest):
        if not isinstance(row, dict):
            continue
        shot_id = str(row.get("shot_id") or "")
        if not shot_id or shot_id in rows:
            continue
        rows[shot_id] = row
    return rows


def grade_frozen_route(ledger):
    """Every shot renders the engine its ROLE was frozen to.

    The first half of A6, and it catches the case a per-role check cannot: a
    shot whose ``engine_id`` was rewritten after the freeze. A role absent from
    the frozen map is itself a finding -- an unfrozen role is a role whose
    delivery cannot be judged at all."""
    # STAND DOWN WHEN THE FROZEN MAP ITSELF IS THE DEFECT. If
    # ``roles_effective`` is present but not a map, ``grade_ledger_shape``
    # already says so once -- and every shot would then report its role as
    # "not in the frozen route", turning one defect into N+1 findings that all
    # describe the same broken map. One cause, one finding.
    video = ledger.get("video") if isinstance(ledger, dict) else None
    if isinstance(video, dict):
        declared = video.get("roles_effective")
        if declared is not None and not isinstance(declared, dict):
            return []
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
                # ``map(str, ...)`` because a native (non-JSON) ledger can carry
                # mixed-type keys, and ``sorted`` on those raises TypeError --
                # out of a function whose contract says it never does.
                % (role, sorted(map(str, frozen)))))
            continue
        expected = str(frozen.get(role) or "")
        if actual != expected:
            findings.append(_finding(
                RULE_FROZEN_ROUTE, shot_id,
                "role %r froze to %r but the shot row stamps %r"
                % (role, expected, actual)))
    return findings


def _plan_segments(shot):
    """The segments the ledger FROZE for this beat, in plan order.

    Callers want the rows themselves as often as the count -- the honesty rule
    cross-checks the delivered receipt against them -- so this returns the list
    and lets ``len()`` answer the other question.
    """
    if not isinstance(shot, dict):
        return []
    plan = shot.get("coverage_plan")
    # A ``coverage_plan`` that is a LIST reached ``.get`` and raised. The plan is
    # a record or it is not a plan; anything else is reported by
    # :func:`grade_ledger_shape` rather than interpreted here.
    if not isinstance(plan, dict):
        return []
    segments = plan.get("segments")
    if not isinstance(segments, (list, tuple)):
        return []
    return list(segments)


#: Every ``extension_mode`` an engine is allowed to declare. An unknown string
#: INVALIDATES the receipt rather than riding along as an unrecognised value --
#: a mode nobody can interpret answers the honesty question no better than no
#: mode at all.
EXTENSION_MODES = ("none", "ping_pong")

#: What the grader can PARSE. Identical to :data:`EXTENSION_MODES`, and named
#: separately because the two answer different questions and conflating them is
#: how a rejected mode becomes an unreadable one. ``"ping_pong"`` remains
#: parseable forever: legacy receipts on disk carry it, and a receipt the grader
#: cannot read is reported as malformed accounting rather than as the policy
#: violation it actually is.
KNOWN_EXTENSION_MODES = ("none", "ping_pong")

#: What may SHIP. Operator ruling 2026-08-06: "there is no mirror or ping pong
#: unless for credits" -- and the one sanctioned reuse is the closing-theme
#: BACKDROP, which is a composite decision about a tail, never a clip receipt.
#: So no delivered clip may declare anything but ``"none"``.
DELIVERABLE_EXTENSION_MODES = ("none",)


#: The exact keys of the durable per-segment projection. Enumerated here rather
#: than described in prose so the producer and the grader cannot drift, and
#: deliberately WITHOUT ``path`` (segment scratch files are never persisted --
#: ``persist_episode_clips`` moves only the assembled beat) and without
#: ``visible_frames`` (exactly derivable from the other three, so persisting it
#: would only add a second thing that can disagree).
#:
#: THE PRODUCER CONTRACT THIS RECEIPT ASSUMES, stated here because it is
#: load-bearing and a count alone cannot reveal its violation: **manufactured
#: frames are APPENDED AT THE TAIL**, so a segment's real frames are the prefix
#: ``[0, native_frame_count)``. Every shipped producer honours it -- WAN pads at
#: the tail and now refuses to pad a coverage-planned segment at all, and
#: ``ltx_8gb`` delivers only rendered frames in order. An engine that padded at
#: the HEAD would satisfy this arithmetic while lying, because the frames the
#: seam drops would be counted as the manufactured ones. Any new engine that
#: extends a clip must extend it at the end, or must not stamp these fields.
SEGMENT_RECEIPT_KEYS = ("segment_id", "segment_index", "render_frames",
                        "drop_head", "trim_tail", "native_frame_count",
                        "extension_mode")


def frame_count(value, *, maximum=None):
    """A frame count as a NON-NEGATIVE int, or ``None``. NEVER raises.

    Every count on this path arrives from a JSON document written by another
    process, so "unreadable" is a real state and it must be a FINDING rather
    than a traceback. Before 2026-08-06 both operands were coerced with a bare
    ``int()``: a manifest carrying ``"not-a-number"`` raised ``ValueError``
    straight out of the grader, past a durable script that guards only document
    LOADING, so an unreadable receipt read as a crash instead of a verdict.

    ``bool`` is rejected explicitly, and that is not pedantry. ``bool`` is an
    ``int`` subclass, so ``True`` silently coerced to 1 and produced the
    receipt "this beat delivered 50 frame(s) of which only True were rendered"
    -- a nonsense number that passed as a real one.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is not paranoia: JSON has no infinity literal, but
        # ``json.load`` parses the bare token ``Infinity`` by default, and
        # ``int(float("inf"))`` raises OverflowError rather than ValueError --
        # so a receipt reading Infinity would have gone straight through a
        # two-exception guard and out of the grader.
        return None
    if count < 0 or (maximum is not None and count > maximum):
        return None
    return count


#: What an unusable set of segment receipts aggregates to. Every field is None,
#: never a partial answer: filtering the holes out and agreeing on the remainder
#: would synthesize a clean "none" from a beat that has one, which is precisely
#: the evidence these receipts exist to make unforgeable.
_UNKNOWN_ACCOUNTING = {"native_frame_count": None,
                       "delivered_native_frame_count": None,
                       "extension_mode": None}


def beat_frame_accounting(segments):
    """PURE: the beat-scope frame accounting for one assembled beat.

    THE ONE OWNER of this arithmetic. ``render_beat_coverage`` calls it to MINT
    the beat receipt and the grader calls it to RECOMPUTE that receipt from the
    durable projection -- so a receipt that disagrees with its own evidence is
    detectable, which a number nobody can re-derive never is. Written here, in
    the module that imports nothing, so both callers can reach it without the
    grader gaining an import (a test pins this module's import list to exactly
    ``["__future__"]``).

    Returns ``rendered`` and ``delivered`` native totals plus the derived beat
    mode. The two counts are kept DISTINCT because they answer different
    questions and one field cannot answer both -- which is the whole defect
    this repairs. A chained beat of three 81-frame renders does 243 frames of
    WORK and delivers 241, because each chained successor opens on a duplicate
    of its predecessor's last frame and that frame is dropped at the seam. 243
    is the honest answer to "what did this beat render" and 241 to "what did it
    deliver"; grading the first against the beat's length is what reported
    every honest multi-segment beat as padded.

    ``PRODUCTION_SPRINT_LESSONS.md`` lesson 33 already required exactly this --
    "preserve requested, rendered, visible, and overlap-trimmed counts so
    intentional edits do not masquerade as engine underruns". An intentional
    seam drop masquerading as a pad is that sentence, verbatim.

    NEVER RAISES. Any segment whose receipt is missing, unreadable or
    impossible makes the WHOLE beat unknown, because a beat is only as provable
    as its least provable segment.
    """
    # A LIST, specifically. ``list(segments or ())`` would raise on an int and
    # would happily iterate a STRING into its characters -- both of which a
    # hand-edited manifest can hold, and neither of which is a set of segments.
    if not isinstance(segments, (list, tuple)) or not segments:
        return dict(_UNKNOWN_ACCOUNTING)
    rows = list(segments)
    rendered = 0
    delivered = 0
    # The modes of the segments that actually put a non-native frame INTO the
    # delivered beat. Padding a lane trimmed entirely away is not padding the
    # viewer sees, so it must not condemn the beat; padding that survives must.
    surviving_modes = set()
    for row in rows:
        if not isinstance(row, dict):
            return dict(_UNKNOWN_ACCOUNTING)
        render_frames = frame_count(row.get("render_frames"))
        drop_head = frame_count(row.get("drop_head"))
        trim_tail = frame_count(row.get("trim_tail"))
        if not render_frames or drop_head is None or trim_tail is None:
            return dict(_UNKNOWN_ACCOUNTING)
        # A native count ABOVE the segment's own render length is not a large
        # number, it is an impossible one: the count is emitted-scope, so it
        # cannot exceed what the segment emitted. Clamping it with min() would
        # launder a broken receipt into a pass.
        native = frame_count(row.get("native_frame_count"), maximum=render_frames)
        mode = row.get("extension_mode")
        if native is None or mode not in EXTENSION_MODES:
            return dict(_UNKNOWN_ACCOUNTING)
        keep_start = drop_head
        keep_end = render_frames - trim_tail
        if keep_end < keep_start:
            return dict(_UNKNOWN_ACCOUNTING)
        # Manufactured frames are appended at the TAIL, so the native frames are
        # the prefix [0, native). Intersect that prefix with the window this
        # segment actually contributes, [keep_start, keep_end).
        segment_delivered_native = max(0, min(native, keep_end) - keep_start)
        rendered += native
        delivered += segment_delivered_native
        if (keep_end - keep_start) > segment_delivered_native:
            surviving_modes.add(mode)
    if not surviving_modes:
        beat_mode = "none"
    elif len(surviving_modes) == 1:
        beat_mode = surviving_modes.pop()
    else:
        # Segments that disagree about HOW they padded leave no single honest
        # answer, so the receipt fails closed rather than inventing a "mixed"
        # value nothing else in the pipeline knows how to read.
        return dict(_UNKNOWN_ACCOUNTING)
    return {"native_frame_count": rendered,
            "delivered_native_frame_count": delivered,
            "extension_mode": beat_mode}


def plan_visible_frames(plan_segments):
    """How many frames the FROZEN plan says this beat delivers, or ``None``.

    ``sum(render_frames - drop_head - trim_tail)``, which
    ``coverage_plan.partition_beat`` guarantees equals the beat's target. The
    grader re-derives it so that ``frame_count`` -- the length the row claims
    it delivered -- is checked against the plan rather than believed.

    Without this, ``frame_count`` was the one beat-scope integer nothing
    re-derived, and everything else was weighed AGAINST it. Two coordinated
    lies (a short ``frame_count`` plus segment counts that agree with it) could
    therefore launder a beat carrying surviving padded frames, even though
    either lie alone is caught. The plan is the authority on how long the beat
    should be; nothing the render says about itself can move it.

    Lenient toward an unreadable PLAN value -- returns ``None`` rather than a
    verdict, because a corrupt frozen document is a different failure from a
    dishonest render and is not this rule's to report.
    """
    total = 0
    for planned in (plan_segments or ()):
        if not isinstance(planned, dict):
            return None
        render_frames = frame_count(planned.get("render_frames"))
        drop_head = frame_count(planned.get("drop_head")) or 0
        trim_tail = frame_count(planned.get("trim_tail")) or 0
        if not render_frames:
            return None
        visible = render_frames - drop_head - trim_tail
        if visible < 0:
            return None
        total += visible
    return total or None


def _projection_fault(rows, plan_segments):
    """Why this beat's per-segment projection cannot be TRUSTED, or ``""``.

    The projection is EVIDENCE, and evidence nobody checks is decoration. A
    beat-level count on its own is the easiest thing for a padded beat to carry
    -- it is one integer -- so the count is only believed when the per-segment
    rows behind it exist, match the plan the ledger froze, and re-derive it.

    Total and non-throwing, like everything the grader touches: a malformed
    projection is a finding, never an exception.
    """
    # The SAME container rule ``beat_frame_accounting`` applies, deliberately:
    # two functions judging one value must not disagree about what that value
    # is allowed to be.
    if not isinstance(rows, (list, tuple)) or not rows:
        return ("carries no per-segment receipt, so its frame accounting "
                "cannot be checked against the plan")
    if len(rows) != len(plan_segments):
        return ("carries %d per-segment receipt(s) for a %d-segment plan"
                % (len(rows), len(plan_segments)))
    seen_ids = set()
    by_index = {}
    for row in rows:
        if not isinstance(row, dict):
            return "carries a per-segment receipt that is not a record"
        segment_id = str(row.get("segment_id") or "")
        if not segment_id:
            return "carries a per-segment receipt with no segment_id"
        if segment_id in seen_ids:
            return "names segment_id %r twice" % (segment_id,)
        seen_ids.add(segment_id)
        index = frame_count(row.get("segment_index"))
        if index is None or index in by_index:
            return ("carries a per-segment receipt whose segment_index is "
                    "missing or repeated (%r)" % (row.get("segment_index"),))
        by_index[index] = row
    for position, planned in enumerate(plan_segments):
        row = by_index.get(position)
        if row is None:
            return ("has no per-segment receipt for position %d, so its "
                    "segments are not the plan's" % (position,))
        if not isinstance(planned, dict):
            continue
        # The ledger plan spells this key ``index``; the render receipt spells
        # it ``segment_index``. They are the same position and the mapping is
        # load-bearing -- comparing them by name would fail every honest row.
        for key in ("render_frames", "drop_head", "trim_tail"):
            want = frame_count(planned.get(key))
            got = frame_count(row.get(key))
            if want is None:
                continue
            if got != want:
                return ("segment %s reports %s=%r where the frozen plan says "
                        "%d" % (row.get("segment_id"), key, row.get(key), want))
    return ""


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
    rows = clip_rows_by_shot(manifest)
    findings = []
    for shot in _shots(ledger):
        shot_id = str(shot.get("shot_id") or "")
        # THROUGH ``frame_count``, like every other count in this module. These
        # two sites kept a bare ``int()`` when the rest of the file was hardened
        # on 2026-08-06, so a ledger stamping ``target_frame_count`` as anything
        # unparseable raised straight out of the grader and past the durable
        # script's exit-code contract -- the exact failure ``frame_count``
        # exists to prevent, surviving in the one rule nobody re-read.
        # A shot that cannot say how many frames it wanted owes nothing that can
        # be judged, so it is skipped rather than reported: a missing CLIP and
        # an unreadable PLAN are different failures, and this rule owns the
        # first one only.
        target = frame_count(shot.get("target_frame_count"))
        if not target:
            continue                       # renders nothing; owes nothing
        row = rows.get(shot_id)
        if row is not None and str(row.get("status") or "") == "sanctioned_gap":
            # C6 (2026-08-28): a SANCTIONED gap is not a missing clip.
            #
            # The status string is compared LITERALLY rather than through
            # ``_otr_shared.still_receipt``, because this module is required to
            # import nothing but ``__future__`` -- a ratified rule with its own
            # structural test, so that the grader CANNOT consult live routing
            # state. Reading a field off the manifest row it was handed is
            # exactly what it is allowed to do; importing the constant is not.
            # The image model refused this beat's required still and the
            # operator's ruling is that the episode continues without it. A
            # checker that reports the sanctioned outcome as a defect is a
            # checker that gets ignored -- and an ignored grader catches
            # nothing at all. The UNsanctioned absence below still reports.
            continue
        if row is None or not row.get("exists"):
            findings.append(_finding(
                RULE_MISSING_CLIP, shot_id,
                "the plan gave this beat %d frame(s) and the manifest carries "
                "%s" % (target,
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
    ``extension_mode`` and the native counts are the receipts WIRE-W3b put on
    every WAN clip for precisely this check.

    SILENCE IS NOT A PASS. A multi-clip beat whose row carries NO extension
    receipt at all is reported too -- an engine that never declares how its
    frames got there cannot be graded, and "no receipt" is exactly what a lane
    that pads without saying so looks like.

    THE QUANTITY IS DELIVERED-NATIVE, NOT NATIVE (2026-08-06). This rule used to
    compare ``native_frame_count`` against ``frame_count``, and those two
    numbers were produced at different SCOPES -- the first for one SEGMENT, the
    second for the whole assembled BEAT, because ``render_beat_coverage`` built
    the beat clip by copying its LAST segment and overwriting only four keys.
    So an honest ``wan_ti2v`` beat of three real 81-frame renders was graded
    "241 delivered, of which only 81 were rendered", and **the rule fired on
    exactly the beats that proved it was satisfied.**

    Summing the segments does not fix it either: a chained beat RENDERS more
    than it DELIVERS by design, one duplicated head frame per seam, so 3x81 does
    243 frames of work to deliver 241. The honest comparison is the native
    frames that actually SURVIVED into the beat, re-derived from the per-segment
    projection rather than believed off a single integer.

    NOTHING HERE RAISES. Every value arrives from a JSON document written by
    another process; an unreadable one is a FINDING, because a grader that dies
    on a malformed receipt has not graded the episode."""
    rows = clip_rows_by_shot(manifest)
    findings = []
    for shot in _shots(ledger):
        planned = _plan_segments(shot)
        if len(planned) <= 1:
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
                % (len(planned), row.get("engine_id"))))
            continue
        if (isinstance(mode, str) and mode in KNOWN_EXTENSION_MODES
                and mode not in DELIVERABLE_EXTENSION_MODES):
            # POLICY IS NOT THIS RULE'S TO GRADE. ``grade_no_mirror`` owns the
            # flat ban and reports it for every beat; continuing here would emit
            # a SECOND, differently-worded finding for one violation, and an
            # operator reading two findings looks for two defects. This rule
            # keeps only the questions that are its own -- does the arithmetic
            # add up, does the projection match the frozen plan -- and neither
            # is answerable once the beat is already condemned.
            continue
        if str(mode) != "none":
            # An UNKNOWN mode is still reported HERE, and the distinction
            # matters: ``grade_no_mirror`` rejects it as policy, and this rule
            # records that the accounting below cannot be performed at all,
            # because a mode nobody can interpret makes the native counts
            # unreadable rather than merely disallowed.
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat is planned across %d clips and %r delivered it with "
                "extension_mode=%r, which is not a mode this grader can "
                "interpret, so its frame accounting cannot be checked"
                % (len(planned), row.get("engine_id"), mode)))
            continue
        fault = _projection_fault(row.get("segments"), planned)
        if fault:
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat is planned across %d clips and %s"
                % (len(planned), fault)))
            continue
        recomputed = beat_frame_accounting(row.get("segments"))
        delivered_native = recomputed["delivered_native_frame_count"]
        if delivered_native is None:
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat is planned across %d clips and at least one segment "
                "receipt is missing or impossible, so how many of its frames "
                "were rendered cannot be established" % (len(planned),)))
            continue
        delivered = frame_count(row.get("frame_count"))
        if delivered is None:
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat reports frame_count=%r, which is not a length, so "
                "there is nothing to weigh its rendered frames against"
                % (row.get("frame_count"),)))
            continue
        # THE LENGTH ITSELF IS RE-DERIVED, NOT BELIEVED. Everything else here is
        # weighed against ``frame_count``, so leaving it unchecked made it the
        # one number a dishonest receipt could move -- shorten the claimed
        # length and supply segment counts that agree with the short version,
        # and a beat carrying real padded frames adds up. The frozen plan says
        # how long this beat is; the render does not get a vote.
        expected = plan_visible_frames(planned)
        if expected is not None and delivered != expected:
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat reports frame_count=%d but the plan it was cut "
                "against covers %d frame(s)" % (delivered, expected)))
            continue
        # THE ROW'S OWN CLAIMS, AGAINST ITS OWN EVIDENCE. Every beat-scope value
        # is re-derived from the segments and must agree. A beat-level number
        # that disagrees with the rows behind it is the one thing a forged
        # receipt looks like and an honest one never does -- and checking only
        # SOME of them would leave the unchecked ones free to say anything.
        disagreements = []
        for field in ("native_frame_count", "delivered_native_frame_count"):
            declared = frame_count(row.get(field))
            if declared is None or declared != recomputed[field]:
                disagreements.append(
                    "%s=%r against %r" % (field, row.get(field),
                                          recomputed[field]))
        if str(row.get("extension_mode")) != str(recomputed["extension_mode"]):
            disagreements.append(
                "extension_mode=%r against %r"
                % (row.get("extension_mode"), recomputed["extension_mode"]))
        if disagreements:
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat's own per-segment receipts do not support what it "
                "declares -- %s" % ("; ".join(disagreements),)))
            continue
        if delivered_native != delivered:
            findings.append(_finding(
                RULE_MULTICLIP_HONESTY, shot_id,
                "this beat delivered %d frame(s) of which only %d were "
                "rendered, while declaring extension_mode='none'"
                % (delivered, delivered_native)))
    return findings


def grade_manifest_shape(manifest):
    """The manifest is shaped like a manifest, or it is a named finding.

    Runs BEFORE every manifest-dependent rule, because a document nobody can
    read makes all of them report nonsense. Deterministic order: rows are judged
    by position, so two runs over the same bad file produce the same findings in
    the same sequence.
    """
    findings = []
    if manifest is not None and not isinstance(manifest, dict):
        return [_finding(RULE_MANIFEST_SHAPE, "",
                         "the manifest is %s, not a record"
                         % (type(manifest).__name__,))]
    clips = (manifest or {}).get("clips")
    if clips is None:
        return []
    if not isinstance(clips, (list, tuple)):
        return [_finding(RULE_MANIFEST_SHAPE, "",
                         "manifest clips is %s, not a list of rows"
                         % (type(clips).__name__,))]
    seen = set()
    for position, row in enumerate(clips):
        if not isinstance(row, dict):
            findings.append(_finding(
                RULE_MANIFEST_SHAPE, "",
                "manifest clip %d is %s, not a record"
                % (position, type(row).__name__)))
            continue
        raw = row.get("shot_id")
        shot_id = str(raw or "")
        if not shot_id:
            findings.append(_finding(
                RULE_MANIFEST_SHAPE, "",
                "manifest clip %d declares shot_id=%r, so nothing can be "
                "matched to it" % (position, raw)))
            continue
        if shot_id in seen:
            findings.append(_finding(
                RULE_MANIFEST_SHAPE, shot_id,
                "manifest clip %d repeats shot_id %r, so which row was graded "
                "would depend on file order" % (position, shot_id)))
            continue
        seen.add(shot_id)
    return findings


def grade_ledger_shape(ledger):
    """The ledger's video section is shaped like one, or it is a named finding.

    THE HALF THAT WAS SCHEDULED NOWHERE. ``grade_frozen_route`` was kept out of
    the manifest-shape skip because it is manifest-INDEPENDENT -- and it still
    crashed on a malformed LEDGER row, which is the same defect through the
    other door.
    """
    findings = []
    if ledger is not None and not isinstance(ledger, dict):
        return [_finding(RULE_LEDGER_SHAPE, "",
                         "the ledger is %s, not a record"
                         % (type(ledger).__name__,))]
    video = (ledger or {}).get("video")
    if video is None:
        return []
    if not isinstance(video, dict):
        return [_finding(RULE_LEDGER_SHAPE, "",
                         "ledger video is %s, not a record"
                         % (type(video).__name__,))]
    frozen = video.get("roles_effective")
    if frozen is not None and not isinstance(frozen, dict):
        findings.append(_finding(
            RULE_LEDGER_SHAPE, "",
            "ledger video.roles_effective is %s, not a map, so what any shot "
            "was SUPPOSED to render is unknowable"
            % (type(frozen).__name__,)))
    shots = video.get("shots")
    if shots is None:
        return findings
    if not isinstance(shots, (list, tuple)):
        findings.append(_finding(
            RULE_LEDGER_SHAPE, "",
            "ledger video.shots is %s, not a list of rows"
            % (type(shots).__name__,)))
        return findings
    for position, shot in enumerate(shots):
        if not isinstance(shot, dict):
            findings.append(_finding(
                RULE_LEDGER_SHAPE, "",
                "ledger shot %d is %s, not a record"
                % (position, type(shot).__name__)))
            continue
        plan = shot.get("coverage_plan")
        if plan is not None and not isinstance(plan, dict):
            findings.append(_finding(
                RULE_LEDGER_SHAPE, str(shot.get("shot_id") or ""),
                "coverage_plan is %s, not a record" % (type(plan).__name__,)))
    return findings


def grade_no_mirror(ledger, manifest):
    """NO delivered clip may declare that its frames were manufactured.

    Operator ruling 2026-08-06: *"there is no mirror or ping pong unless for
    credits."* The one sanctioned reuse is the closing-theme BACKDROP, which is
    a composite decision about a TAIL and never a clip receipt -- so no clip,
    anywhere, may ship a non-``none`` mode.

    **ALL BEATS.** No counts, no projection, no segment gate, and above all no
    ``len(planned) <= 1`` skip. ``grade_multiclip_honesty`` answers a different
    question -- does this multi-clip beat's arithmetic add up -- and it
    deliberately ignores single-clip beats, which is correct for that question
    and left the flat policy unenforced on exactly the beats a single padded
    render produces. A rule that skips the simple case cannot ban anything.

    ABSENCE IS JUDGED BY THE MANIFEST'S OWN VERSION, and that is what makes a
    REGRESSION detectable. Under ``frame_receipt_version: 1`` every delivered
    video row must SAY ``"none"``; on an unversioned manifest a missing field is
    legacy silence and is allowed. Without the version, "reject an explicit
    non-none" could never catch an adapter that quietly stopped stamping --
    absence would stay legal forever, and silence is precisely what an
    unreceipted pad looks like.
    """
    # ITERATE THE DELIVERED ROWS, NOT THE LEDGER'S SHOTS. The rule's own
    # sentence is "any delivered row", and walking shots instead made a manifest
    # row whose ``shot_id`` is absent from the ledger invisible to EVERY rule --
    # every other rule walks shots too, so nothing graded it. A row declaring
    # ``ping_pong`` under a shot id the episode never planned produced
    # "ACCEPTED: 1 shot(s)" and exit 0. Reproduced end to end before this fix.
    shots_by_id = {}
    for shot in _shots(ledger):
        shot_id = str(shot.get("shot_id") or "")
        if shot_id and shot_id not in shots_by_id:
            shots_by_id[shot_id] = shot
    versioned = frame_count((manifest or {}).get("frame_receipt_version")) == 1 \
        if isinstance(manifest, dict) else False
    findings = []
    for shot_id, row in clip_rows_by_shot(manifest).items():
        shot = shots_by_id.get(shot_id)
        if not row.get("exists"):
            continue                       # grade_delivered owns a missing clip
        is_video = str(row.get("type") or "video") == "video"
        mode = row.get("extension_mode")
        if mode is None or mode == "":
            # SILENCE is only interrogable on a VIDEO row, only under v1, and
            # only on a beat ``grade_multiclip_honesty`` is not already asking.
            # That rule owns silence on a MULTI-segment plan ("declares no
            # extension_mode"), so reporting it here too produced TWO findings
            # for ONE missing field -- the boundary this pair exists to keep.
            # ``""`` is silence in both rules now; it used to be silence here
            # and "cannot interpret" there, which is one defect wearing two
            # contradictory classifications.
            if versioned and is_video and len(_plan_segments(shot)) <= 1:
                findings.append(_finding(
                    RULE_NO_MIRROR, shot_id,
                    "this manifest declares frame_receipt_version=1, in which "
                    "every delivered video row must state how its frames got "
                    "there, and %r declares no extension_mode"
                    % (row.get("engine_id"),)))
            continue
        # FROM HERE THE ROW HAS SPOKEN, AND WHAT IT SAID IS JUDGED WHATEVER ITS
        # TYPE IS. The ``type`` exemption used to sit above this, skipping the
        # ban entirely for any non-"video" string -- so a row stamped
        # ``type="still"`` with ``extension_mode="ping_pong"`` passed clean. It
        # also opened a REGRESSION: ``grade_multiclip_honesty`` now stands down
        # on a known non-deliverable mode expecting this rule to catch it, and
        # on a non-video row NEITHER rule fired, where the honesty rule alone
        # used to. A declaration of padding is a violation on its face; only the
        # questions about SILENCE need to know what kind of clip this is.
        if not isinstance(mode, str) or mode not in KNOWN_EXTENSION_MODES:
            # SAY WHAT IS TRUE. This used to deliver the "no clip may carry
            # manufactured frames" verdict for an UNINTERPRETABLE mode -- which
            # asserts a fact the grader has just admitted it cannot establish.
            # On a single-segment beat that misworded line was the ONLY finding,
            # so an operator was told the lane padded when what actually
            # happened is that nobody can tell. Still fail-closed; only honest.
            findings.append(_finding(
                RULE_NO_MIRROR, shot_id,
                "%r declares extension_mode=%r, which is not a mode this "
                "grader can interpret, so whether this clip carries "
                "manufactured frames cannot be established"
                % (row.get("engine_id"), mode)))
            continue
        if mode not in DELIVERABLE_EXTENSION_MODES:
            findings.append(_finding(
                RULE_NO_MIRROR, shot_id,
                "%r delivered this beat with extension_mode=%r -- no delivered "
                "clip may carry manufactured frames (the ONE sanctioned reuse "
                "is the closing-theme backdrop, which is a composite decision "
                "about the tail and never a clip receipt)"
                % (row.get("engine_id"), mode)))
            continue
        # THE SECOND HALF OF THE v1 CONTRACT: a BOUNDED engine also owes the
        # NUMBER. Declaring "none" costs an adapter one string; proving it costs
        # a count that must survive the arithmetic, and only an engine with a
        # ceiling a beat can overflow is ever in a position to pad. Boundedness
        # comes from the shot's own freeze-time stamp and nowhere else (spec
        # 7.3): this module may not import a registry, and an engine's contract
        # can change between the render and the grade.
        #
        # ONLY an explicit True demands it. Absent means the freeze never said,
        # which is every ledger written before the stamp existed -- indicting
        # those would report the grader's calendar, not the episode.
        if shot is not None and versioned and shot.get("frame_bounded") is True:
            declared_native = frame_count(row.get("native_frame_count"))
            if declared_native is None:
                findings.append(_finding(
                    RULE_NO_MIRROR, shot_id,
                    "%r is a BOUNDED engine declaring extension_mode='none' in "
                    "a frame_receipt_version=1 manifest, but carries no "
                    "readable native_frame_count (%r) -- the claim that every "
                    "frame was rendered is exactly the claim that owes evidence"
                    % (row.get("engine_id"), row.get("native_frame_count"))))
            elif len(_plan_segments(shot)) <= 1:
                # AND THEN WEIGH IT. Collecting the number and never comparing
                # it was the whole rule's blind spot: a single-render clip
                # stamping frame_count=50 / native_frame_count=17 /
                # extension_mode="none" CONFESSES 33 manufactured frames in its
                # own receipt, and graded ACCEPTED at exit 0. The multi-clip
                # rule skips it (one segment) and this rule believed the mode
                # STRING while ignoring the COUNT sitting beside it -- which is
                # exactly the single-render hole RULE_NO_MIRROR was minted to
                # close, left open for any pad that mislabels its mode.
                #
                # SINGLE-SEGMENT ONLY. On a multi-segment beat the same
                # arithmetic is ``grade_multiclip_honesty``'s, done properly
                # against the per-segment projection and the frozen plan; doing
                # it here too would indict one beat twice. The producer
                # contract (see SEGMENT_RECEIPT_KEYS) puts manufactured frames
                # at the TAIL, so on an honest single render every delivered
                # frame is native and ``native < delivered`` is a confession.
                delivered = frame_count(row.get("frame_count"))
                if delivered is not None and declared_native < delivered:
                    findings.append(_finding(
                        RULE_NO_MIRROR, shot_id,
                        "%r delivered %d frame(s) declaring extension_mode="
                        "'none', but its own receipt says only %d were "
                        "rendered -- %d frame(s) came from somewhere the "
                        "engine will not name"
                        % (row.get("engine_id"), delivered, declared_native,
                           delivered - declared_native)))
    return findings


def grade_episode(ledger, manifest):
    """Every per-shot rule, in one pass. Returns a list of findings; EMPTY
    means the episode delivered the route it froze.

    Deliberately returns rather than raises: the caller decides whether a
    finding blocks (the durable script exits non-zero) or is recorded, and a
    grader that raised on the first problem would hide the rest.

    SHAPE FIRST, AND A SHAPE DEFECT SUPPRESSES THE RULES THAT WOULD CASCADE.
    One malformed manifest row used to make every semantic rule report its own
    confused version of the same problem; now it produces one named
    ``manifest_shape`` finding and the manifest-dependent rules stand down. The
    LEDGER-only rule keeps running either way, because a broken manifest says
    nothing about whether the route was frozen correctly -- and it is now safe
    to run, which it was not before ``_shots`` began filtering non-records."""
    shape = grade_ledger_shape(ledger) + grade_manifest_shape(manifest)
    findings = list(shape) + grade_frozen_route(ledger)
    if shape:
        return findings
    return (findings
            + grade_delivered(ledger, manifest)
            + grade_no_mirror(ledger, manifest)
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
    "RULE_NO_MIRROR",
    "RULE_MANIFEST_SHAPE",
    "RULE_LEDGER_SHAPE",
    "grade_no_mirror",
    "grade_manifest_shape",
    "grade_ledger_shape",
    "KNOWN_EXTENSION_MODES",
    "DELIVERABLE_EXTENSION_MODES",
    "EXTENSION_MODES",
    "SEGMENT_RECEIPT_KEYS",
    "frame_count",
    "beat_frame_accounting",
    "plan_visible_frames",
    "manifest_clips",
    "clip_rows_by_shot",
    "frozen_route",
    "grade_frozen_route",
    "grade_delivered",
    "grade_multiclip_honesty",
    "grade_episode",
    "format_findings",
]
