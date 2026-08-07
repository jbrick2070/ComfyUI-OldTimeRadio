"""WIRE-W5 -- the acceptance grader: did the episode render what it FROZE?

r4/A6: *"Per shot require ``video.shots[].engine_id ==
video.roles_effective[shot.role]``, then require every delivered clip-manifest
row's ``engine_id`` to match that frozen expected value. CUT aggregate engine
histograms as acceptance evidence -- they cannot detect two shots EXCHANGING
engines while totals stay identical. Never query live routing state."*

Three refusals are as load-bearing as the checks, and each has a test:

* NEVER the live route. The director freezes at plan time and ShotLock
  validates there; asking again at grading time is a clock-domain mismatch.
* NEVER the histogram. Two shots exchanging engines leave every total
  identical -- which is a test here, on real data, not an assertion of faith.
* NEVER a composited frame. kibitz r1 proved the trap with a shipped test
  (``test_credits_roll_spec.py:446-470`` scrolls text over a deliberately
  CONSTANT backdrop), so "did the frame change" goes green on a frozen
  background because the overlay moved. Grade the SOURCE receipts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import acceptance as acc

#: Distinguishes "the caller said nothing" from "the caller said None", because
#: a beat with NO projection and a beat with an explicitly empty one are
#: different failures and both need testing.
_MISSING = object()


def _ledger(shots, frozen):
    return {"video": {"roles_effective": dict(frozen), "shots": list(shots)}}


def _shot(shot_id, role, engine_id, *, frames=50, segments=1):
    shot = {"shot_id": shot_id, "role": role, "engine_id": engine_id,
            "target_frame_count": frames}
    if segments:
        shot["coverage_plan"] = {
            "segments": [{"index": i, "render_frames": 25}
                         for i in range(segments)]}
    return shot


def _segments(shot_id, *, count=2, render_frames=25, native=None):
    """The per-segment projection an assembled beat really carries.

    Built to MATCH the plan ``_shot`` freezes, because the grader re-derives the
    beat totals from these rows and refuses a projection that disagrees with the
    plan. ``native`` is the beat's total delivered-native count, spread across
    the segments front-to-back -- so a row asking for fewer native frames than
    the beat delivers produces a SHORT tail segment, which is exactly the shape
    a lane that under-rendered leaves behind.
    """
    remaining = count * render_frames if native is None else int(native)
    rows = []
    for index in range(count):
        take = max(0, min(render_frames, remaining))
        remaining -= take
        rows.append({"segment_id": "%s_seg%02d" % (shot_id, index),
                     "segment_index": index,
                     "render_frames": render_frames,
                     "drop_head": 0, "trim_tail": 0,
                     "native_frame_count": take,
                     "extension_mode": "none"})
    return rows


def _row(shot_id, engine_id, *, frames=50, exists=True,
         extension_mode="none", native=None, segments=2, projection=_MISSING):
    row = {"shot_id": shot_id, "engine_id": engine_id, "exists": exists,
           "frame_count": frames, "extension_mode": extension_mode,
           "native_frame_count": frames if native is None else native,
           "delivered_native_frame_count": frames if native is None else native}
    # The receipt the grader re-derives its verdict from. Defaulted to a
    # projection that AGREES with the row, so a test that means to break one
    # thing does not accidentally break this too; pass ``projection=None`` to
    # test a beat that carries no evidence at all.
    row["segments"] = (_segments(shot_id, count=segments, native=native)
                       if projection is _MISSING else projection)
    return row


def _manifest(rows):
    hist = {}
    for row in rows:
        if row.get("exists"):
            hist[row["engine_id"]] = hist.get(row["engine_id"], 0) + 1
    return {"clips": list(rows), "engine_histogram": hist}


# ---------------------------------------------------------------------------
# A6 first half: the shot renders the engine its role froze to
# ---------------------------------------------------------------------------

def test_a_CLEAN_episode_produces_NO_findings():
    ledger = _ledger([_shot("shot_b0", "announcer_visual", "ltx_audio_in"),
                      _shot("shot_b1", "character_video", "humo")],
                     {"announcer_visual": "ltx_audio_in",
                      "character_video": "humo"})
    manifest = _manifest([_row("shot_b0", "ltx_audio_in"),
                          _row("shot_b1", "humo")])
    assert acc.grade_episode(ledger, manifest) == []


def test_a_SHOT_REWRITTEN_after_the_freeze_is_caught():
    """The case a per-role check alone cannot see: the frozen map still says
    what it always said, and the shot row no longer agrees with it."""
    ledger = _ledger([_shot("shot_b1", "character_video", "wan_i2v")],
                     {"character_video": "humo"})
    findings = acc.grade_frozen_route(ledger)
    assert [f["rule"] for f in findings] == [acc.RULE_FROZEN_ROUTE]
    assert "humo" in findings[0]["detail"] and "wan_i2v" in findings[0]["detail"]


def test_a_ROLE_MISSING_from_the_frozen_map_is_a_FINDING_not_a_pass():
    """An empty or partial frozen map must not read as "every shot agrees" --
    an unfrozen role is a role whose delivery cannot be judged at all."""
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")], {})
    findings = acc.grade_frozen_route(ledger)
    assert len(findings) == 1
    assert "unknowable" in findings[0]["detail"]


# ---------------------------------------------------------------------------
# A6 second half: the DELIVERED clip came from that same engine
# ---------------------------------------------------------------------------

def test_a_DELIVERED_clip_from_the_WRONG_ENGINE_is_caught():
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")],
                     {"character_video": "humo"})
    manifest = _manifest([_row("shot_b1", "still_pan")])
    findings = acc.grade_delivered(ledger, manifest)
    assert [f["rule"] for f in findings] == [acc.RULE_DELIVERED_ENGINE]
    assert "still_pan" in findings[0]["detail"]


def test_the_DELIVERY_is_judged_against_the_FROZEN_value_not_the_shot_row():
    """If it were judged against the shot row, a rewritten row would agree with
    its own rewrite and the delivery check would pass on a route nobody chose.
    Here BOTH were rewritten to the same wrong engine -- the frozen-route rule
    fires AND the delivery rule fires."""
    ledger = _ledger([_shot("shot_b1", "character_video", "still_pan")],
                     {"character_video": "humo"})
    manifest = _manifest([_row("shot_b1", "still_pan")])
    rules = {f["rule"] for f in acc.grade_episode(ledger, manifest)}
    assert rules == {acc.RULE_FROZEN_ROUTE, acc.RULE_DELIVERED_ENGINE}


def test_TWO_SHOTS_EXCHANGING_ENGINES_is_caught_and_the_HISTOGRAM_cannot():
    """r4's own argument for cutting histograms, run as an experiment rather
    than asserted: swap two shots' engines and every aggregate total is
    IDENTICAL, while the per-shot grader reports both."""
    frozen = {"announcer_visual": "ltx_audio_in", "character_video": "humo"}
    shots = [_shot("shot_b0", "announcer_visual", "ltx_audio_in"),
             _shot("shot_b1", "character_video", "humo")]
    honest = _manifest([_row("shot_b0", "ltx_audio_in"),
                        _row("shot_b1", "humo")])
    swapped = _manifest([_row("shot_b0", "humo"),
                         _row("shot_b1", "ltx_audio_in")])
    assert honest["engine_histogram"] == swapped["engine_histogram"], (
        "the premise of this test is that the totals cannot tell them apart")
    assert acc.grade_episode(_ledger(shots, frozen), honest) == []
    findings = acc.grade_episode(_ledger(shots, frozen), swapped)
    assert {f["shot_id"] for f in findings} == {"shot_b0", "shot_b1"}


def test_a_PLANNED_beat_with_NO_CLIP_is_its_OWN_named_finding():
    """A missing clip and a wrong clip are different failures and an operator
    fixes them differently, so they do not share a rule name."""
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")],
                     {"character_video": "humo"})
    findings = acc.grade_delivered(ledger, _manifest([]))
    assert [f["rule"] for f in findings] == [acc.RULE_MISSING_CLIP]
    findings = acc.grade_delivered(
        ledger, _manifest([_row("shot_b1", "humo", exists=False)]))
    assert [f["rule"] for f in findings] == [acc.RULE_MISSING_CLIP]


def test_a_beat_that_RENDERS_NOTHING_owes_nothing():
    """CONTROL. A zero-frame row is not a missing clip; demanding one would
    make every non-rendering beat a finding."""
    ledger = _ledger([_shot("shot_b0", "announcer_visual", "ltx_audio_in",
                            frames=0)],
                     {"announcer_visual": "ltx_audio_in"})
    assert acc.grade_delivered(ledger, _manifest([])) == []


# ---------------------------------------------------------------------------
# The multi-clip honesty check -- what WIRE-W3b's receipts are FOR
# ---------------------------------------------------------------------------

def test_a_PING_PONGED_clip_on_a_MULTI_CLIP_beat_is_REJECTED():
    """The whole reason native_frame_count / extension_mode exist. The clip
    carries the RIGHT frame count -- that is what makes a pad forgeable -- so
    nothing but the receipt can catch it.

    THE VERDICT MOVED, THE REJECTION DID NOT (no-mirror step 3, 2026-08-06).
    ``grade_multiclip_honesty`` used to own this finding. It now STANDS DOWN on
    a known non-deliverable mode and leaves the verdict to ``RULE_NO_MIRROR``,
    which bans the pad on EVERY beat rather than only on multi-clip ones --
    otherwise one violation produced two differently-worded findings and an
    operator went looking for two defects.

    So this asserts through ``grade_episode``: the clip is still rejected,
    exactly once, and the rule that owns it is named."""
    ledger = _ledger([_shot("shot_b1", "character_video", "wan_ti2v",
                            segments=3)],
                     {"character_video": "wan_ti2v"})
    manifest = _manifest([_row("shot_b1", "wan_ti2v",
                               extension_mode="ping_pong", native=17)])
    findings = acc.grade_episode(ledger, manifest)
    rules = [f["rule"] for f in findings]
    assert rules.count(acc.RULE_NO_MIRROR) == 1, rules
    assert acc.RULE_MULTICLIP_HONESTY not in rules, (
        "one violation must not be indicted twice")
    assert "ping_pong" in next(f["detail"] for f in findings
                               if f["rule"] == acc.RULE_NO_MIRROR)
    assert manifest["clips"][0]["frame_count"] == 50, (
        "the padded clip wears the right count, which is the point")


def test_SILENCE_is_not_a_PASS_on_a_multi_clip_beat():
    """An engine that never declares how its frames got there cannot be
    graded, and "no receipt" is exactly what a lane that pads without saying so
    looks like."""
    ledger = _ledger([_shot("shot_b1", "character_video", "humo", segments=2)],
                     {"character_video": "humo"})
    row = _row("shot_b1", "humo")
    row["extension_mode"] = None
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert [f["rule"] for f in findings] == [acc.RULE_MULTICLIP_HONESTY]
    assert "declares no extension_mode" in findings[0]["detail"]


def test_a_SINGLE_CLIP_beat_is_OUT_OF_SCOPE_for_the_honesty_rule_but_NOT_for_the_ban():
    """THE OLD CONTROL, AND ITS PREMISE IS NOW HISTORY. It read: "CONTROL, and
    it is the shipped 8 GB WAN tier: a single-clip beat renders short on purpose
    and fills the beat with a mirror (PBUG-20260723-02). If this went red the
    grader would be failing production's majority path."

    That sentence is no longer true in either half. ``eng_wan_ti2v``'s
    adapter-side ping-pong was DELETED under the operator's no-mirror ruling, so
    padding is not production's majority path -- it is not any path. And the
    ruling is flat: *"there is no mirror or ping pong unless for credits."*

    What survives is a SCOPE fact, not a permission: ``grade_multiclip_honesty``
    still ignores a one-segment plan, because its question is whether a
    multi-clip beat's arithmetic adds up. That silence was the entire hole --
    the flat ban was unenforced on exactly the beats a single padded render
    produces -- and ``RULE_NO_MIRROR`` is what closed it. Both halves are pinned
    here so the scope fact can never again be mistaken for a licence."""
    ledger = _ledger([_shot("shot_b1", "character_video", "wan_ti2v",
                            segments=1)],
                     {"character_video": "wan_ti2v"})
    manifest = _manifest([_row("shot_b1", "wan_ti2v",
                               extension_mode="ping_pong", native=17)])
    # The honesty rule does not ask about this beat...
    assert acc.grade_multiclip_honesty(ledger, manifest) == []
    # ...and the ban most certainly does.
    rules = [f["rule"] for f in acc.grade_episode(ledger, manifest)]
    assert rules.count(acc.RULE_NO_MIRROR) == 1, rules


def test_a_CLIP_claiming_NO_EXTENSION_must_have_RENDERED_every_frame():
    """The other half of the receipt: a lane can declare "none" and still hand
    back fewer real frames than it emitted."""
    ledger = _ledger([_shot("shot_b1", "character_video", "wan_ti2v",
                            segments=2)],
                     {"character_video": "wan_ti2v"})
    manifest = _manifest([_row("shot_b1", "wan_ti2v", frames=50, native=33)])
    findings = acc.grade_multiclip_honesty(ledger, manifest)
    assert [f["rule"] for f in findings] == [acc.RULE_MULTICLIP_HONESTY]
    assert "only 33" in findings[0]["detail"]


# ---------------------------------------------------------------------------
# The inversion itself: an HONEST chained beat renders MORE than it delivers
# ---------------------------------------------------------------------------

def _chain_shot(shot_id, *, segment_frames=81, count=3):
    """A real ``wan_ti2v`` chain: N native renders, one duplicated head frame
    dropped at every seam after the first."""
    return {"shot_id": shot_id, "role": "character_video",
            "engine_id": "wan_ti2v",
            "target_frame_count": count * segment_frames - (count - 1),
            "coverage_plan": {"join_mode": "chain", "segments": [
                {"index": i, "render_frames": segment_frames,
                 "drop_head": (1 if i else 0), "trim_tail": 0}
                for i in range(count)]}}


def _chain_row(shot_id, *, segment_frames=81, count=3, natives=None):
    segments = []
    for index in range(count):
        native = segment_frames if natives is None else natives[index]
        segments.append({"segment_id": "%s_seg%02d" % (shot_id, index),
                         "segment_index": index,
                         "render_frames": segment_frames,
                         "drop_head": (1 if index else 0), "trim_tail": 0,
                         "native_frame_count": native,
                         "extension_mode": "none"})
    delivered = sum(
        max(0, min(s["native_frame_count"],
                   s["render_frames"] - s["trim_tail"]) - s["drop_head"])
        for s in segments)
    return {"shot_id": shot_id, "engine_id": "wan_ti2v", "exists": True,
            "frame_count": count * segment_frames - (count - 1),
            "extension_mode": "none",
            "native_frame_count": sum(s["native_frame_count"] for s in segments),
            "delivered_native_frame_count": delivered,
            "segments": segments}


def test_an_HONEST_CHAINED_beat_that_renders_MORE_than_it_delivers_is_ACCEPTED():
    """THE DEFECT, as a test. Three real 81-frame renders do 243 frames of work
    and deliver 241, because each chained successor opens on a duplicate of its
    predecessor's last frame and that frame is dropped at the seam.

    Before 2026-08-06 this beat was graded "241 delivered, of which only 81 were
    rendered" -- the rule fired on exactly the beats that proved it was
    satisfied, because it weighed ONE SEGMENT's native count against the WHOLE
    BEAT's length."""
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1")
    assert row["native_frame_count"] == 243, "the WORK"
    assert row["delivered_native_frame_count"] == 241, "the OUTPUT"
    assert row["frame_count"] == 241
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    assert acc.grade_multiclip_honesty(ledger, _manifest([row])) == []


def test_SUMMING_the_segments_would_have_failed_this_same_beat():
    """The fix r1 killed, pinned so it cannot come back. Summing the raw native
    counts gives 243 against a 241-frame beat, so an equality test fed the SUM
    rejects the honest chain for a second, different wrong reason."""
    row = _chain_row("shot_b1")
    assert row["native_frame_count"] != row["frame_count"], (
        "243 != 241 -- summing is the right answer to 'what did this beat "
        "render' and the wrong answer to 'what did it deliver'")


def test_PADDING_that_SURVIVES_the_seam_is_still_REJECTED():
    """The rule must keep its teeth: a chain whose middle segment rendered 60
    real frames of its 81 delivers 21 frames that came from nowhere."""
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1", natives=[81, 60, 81])
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert [f["rule"] for f in findings] == [acc.RULE_MULTICLIP_HONESTY]
    assert "only 220" in findings[0]["detail"], findings[0]["detail"]


def test_PADDING_the_TRIM_removes_entirely_does_NOT_condemn_the_beat():
    """Padding the viewer never sees is not padding. A tail segment that
    rendered 40 real frames and had its last 41 trimmed away delivers only real
    frames, whatever happened inside the render."""
    shot = {"shot_id": "shot_b1", "role": "character_video",
            "engine_id": "wan_ti2v", "target_frame_count": 121,
            "coverage_plan": {"join_mode": "chain", "segments": [
                {"index": 0, "render_frames": 81, "drop_head": 0, "trim_tail": 0},
                {"index": 1, "render_frames": 81, "drop_head": 1,
                 "trim_tail": 40}]}}
    row = {"shot_id": "shot_b1", "engine_id": "wan_ti2v", "exists": True,
           "frame_count": 121, "extension_mode": "none",
           "native_frame_count": 122, "delivered_native_frame_count": 121,
           "segments": [
               {"segment_id": "b1_seg00", "segment_index": 0,
                "render_frames": 81, "drop_head": 0, "trim_tail": 0,
                "native_frame_count": 81, "extension_mode": "none"},
               {"segment_id": "b1_seg01", "segment_index": 1,
                "render_frames": 81, "drop_head": 1, "trim_tail": 40,
                "native_frame_count": 41, "extension_mode": "none"}]}
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    assert acc.grade_multiclip_honesty(ledger, _manifest([row])) == []


# ---------------------------------------------------------------------------
# FAIL CLOSED: an unprovable beat is not an accepted one, and never a crash
# ---------------------------------------------------------------------------

def test_a_MISSING_native_count_is_a_FINDING_not_a_PASS():
    """Before 2026-08-06 the check was guarded by ``native is not None``, so a
    beat declaring "none" while counting nothing sailed through -- a hole in the
    fail-closed model the receipt exists to serve."""
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1")
    row["segments"][1]["native_frame_count"] = None
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert [f["rule"] for f in findings] == [acc.RULE_MULTICLIP_HONESTY]
    assert "missing or impossible" in findings[0]["detail"]


def test_a_beat_with_NO_per_segment_receipt_is_a_FINDING():
    """A beat-level count on its own is one integer -- the easiest thing for a
    padded beat to carry. It is believed only when the segments behind it
    exist and re-derive it."""
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1")
    row["segments"] = None
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert "carries no per-segment receipt" in findings[0]["detail"]


@pytest.mark.parametrize("field,value", [
    ("delivered_native_frame_count", 246),
    ("native_frame_count", 999),
])
def test_a_beat_whose_OWN_COUNT_disagrees_with_its_SEGMENTS_is_a_FINDING(
        field, value):
    """The forgery test. An honest beat's totals are re-derivable from its
    parts; a fabricated one is not. BOTH counts are checked -- an unchecked
    beat-scope field is free to say anything."""
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1")
    row[field] = value
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert "do not support what it declares" in findings[0]["detail"]
    assert field in findings[0]["detail"], findings[0]["detail"]


def test_TWO_COORDINATED_LIES_cannot_launder_a_padded_beat():
    """The laundering the Fable arithmetic gate found, closed.

    Every other number here is weighed AGAINST ``frame_count``, so leaving that
    one unchecked made it the field a dishonest receipt could move: shorten the
    claimed length to 220 and hand over segment counts that agree with the
    short version, and a beat carrying 21 real padded frames adds up perfectly.
    Either lie alone is caught; the pair was not.

    The frozen plan is the authority on how long the beat is -- 81 + 80 + 80 --
    and the render does not get a vote.
    """
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1", natives=[81, 60, 81])
    # The pad is real: 220 native frames survive into a 241-frame beat.
    assert row["delivered_native_frame_count"] == 220
    # Now forge the length so the receipt is internally consistent at 220.
    row["frame_count"] = 220
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert [f["rule"] for f in findings] == [acc.RULE_MULTICLIP_HONESTY]
    assert "the plan it was cut against covers 241" in findings[0]["detail"], \
        findings[0]["detail"]


def test_the_PLAN_is_the_authority_on_the_beats_length():
    """``plan_visible_frames`` is the re-derivation, and it is lenient toward a
    corrupt FROZEN document -- a broken plan is a different failure from a
    dishonest render, and not this rule's to report."""
    assert acc.plan_visible_frames(
        [{"render_frames": 81, "drop_head": 0, "trim_tail": 0},
         {"render_frames": 81, "drop_head": 1, "trim_tail": 0},
         {"render_frames": 81, "drop_head": 1, "trim_tail": 0}]) == 241
    assert acc.plan_visible_frames(
        [{"render_frames": 33, "drop_head": 0, "trim_tail": 8}]) == 25
    for broken in (None, (), [{"render_frames": "x"}], [{"drop_head": 1}],
                   [{"render_frames": 10, "drop_head": 8, "trim_tail": 8}],
                   ["not-a-row"]):
        assert acc.plan_visible_frames(broken) is None, broken


def test_a_projection_that_CONTRADICTS_the_frozen_plan_is_a_FINDING():
    """The plan is what the ledger FROZE. A receipt describing different
    segments describes a different beat."""
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1")
    row["segments"][2]["render_frames"] = 49
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert "the frozen plan says 81" in findings[0]["detail"], \
        findings[0]["detail"]


def test_a_NATIVE_count_ABOVE_the_segments_own_length_is_IMPOSSIBLE():
    """Not a large number -- an impossible one. The count is emitted-scope, so
    it cannot exceed what the segment emitted, and clamping it with min() would
    launder a broken receipt into a pass."""
    shot = _chain_shot("shot_b1")
    row = _chain_row("shot_b1", natives=[81, 999, 81])
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    findings = acc.grade_multiclip_honesty(ledger, _manifest([row]))
    assert "missing or impossible" in findings[0]["detail"]


def test_DUPLICATE_segment_ids_and_indices_are_FINDINGS():
    shot = _chain_shot("shot_b1")
    dup_id = _chain_row("shot_b1")
    dup_id["segments"][1]["segment_id"] = dup_id["segments"][0]["segment_id"]
    dup_index = _chain_row("shot_b1")
    dup_index["segments"][1]["segment_index"] = 0
    ledger = _ledger([shot], {"character_video": "wan_ti2v"})
    assert "twice" in acc.grade_multiclip_honesty(
        ledger, _manifest([dup_id]))[0]["detail"]
    assert "missing or repeated" in acc.grade_multiclip_honesty(
        ledger, _manifest([dup_index]))[0]["detail"]


@pytest.mark.parametrize("bad", ["not-a-number", True, -1, None, 3.5e400])
def test_an_UNREADABLE_receipt_is_a_FINDING_and_NEVER_a_TRACEBACK(bad):
    """A grader that dies on a malformed receipt has not graded the episode.
    Both operands used to be coerced with a bare ``int()``, so a manifest
    carrying "not-a-number" raised straight out of the grader -- past a durable
    script that guards only document LOADING.

    ``True`` is in this list on purpose: ``bool`` is an ``int`` subclass, so it
    coerced to 1 and produced "of which only True were rendered" -- a nonsense
    number that passed as a real one."""
    shot = _chain_shot("shot_b1")
    for field in ("frame_count", "delivered_native_frame_count",
                  "native_frame_count"):
        row = _chain_row("shot_b1")
        row[field] = bad
        findings = acc.grade_multiclip_honesty(
            _ledger([shot], {"character_video": "wan_ti2v"}), _manifest([row]))
        assert findings, "%s=%r must be reported, not ignored" % (field, bad)
        assert all(f["rule"] == acc.RULE_MULTICLIP_HONESTY for f in findings)


@pytest.mark.parametrize("bad", ["not-a-number", True, None, 3.5e400, {}])
def test_grade_DELIVERED_never_raises_on_an_unreadable_target_count(bad):
    """The hardening that skipped one rule.

    ``frame_count`` was added so no count coerced from another process's JSON
    could raise out of the grader -- and every site in this module was moved on
    to it EXCEPT the two in ``grade_delivered``, which kept a bare ``int()``.
    A ledger stamping an unreadable ``target_frame_count`` therefore raised
    straight past ``grade_episode`` and past the durable script's documented
    0/1/2 exit contract, as an uncaught traceback.

    An unreadable PLAN and a missing CLIP are different failures, and this rule
    owns only the second -- so a shot that cannot say what it wanted is skipped,
    not reported here."""
    ledger = _ledger([{"shot_id": "shot_b1", "role": "character_video",
                       "engine_id": "humo", "target_frame_count": bad}],
                     {"character_video": "humo"})
    assert acc.grade_delivered(ledger, _manifest([])) == []
    assert acc.grade_episode(ledger, _manifest([])) is not None


def test_the_ACCOUNTING_HELPER_never_raises_on_any_garbage():
    """The one owner of the arithmetic, hit with the shapes a broken producer
    or a hand-edited manifest really emits."""
    for junk in (None, [], [None], ["x"], [{}], 7, "rows",
                 [{"render_frames": 0}],
                 [{"render_frames": 81, "drop_head": 0, "trim_tail": 0,
                   "native_frame_count": 81, "extension_mode": "sideways"}],
                 [{"render_frames": 81, "drop_head": 90, "trim_tail": 0,
                   "native_frame_count": 81, "extension_mode": "none"}]):
        out = acc.beat_frame_accounting(junk)
        assert out["delivered_native_frame_count"] is None, junk
        assert out["native_frame_count"] is None, junk
        assert out["extension_mode"] is None, junk


# ---------------------------------------------------------------------------
# The three refusals
# ---------------------------------------------------------------------------

def test_the_GRADER_IMPORTS_NOTHING_that_could_reach_the_environment():
    """"Never query live routing state" is the ratified rule, and the strongest
    form of it is a module that CANNOT: no registry, no route_freeze, no os."""
    import ast
    path = os.path.join(_REPO, "nodes", "_otr_video_engines", "acceptance.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    assert names == ["__future__"], names


def test_the_GRADER_never_reads_the_ENGINE_HISTOGRAM():
    """Structural, because the field is right there in the manifest and using
    it would look reasonable to the next reader."""
    path = os.path.join(_REPO, "nodes", "_otr_video_engines", "acceptance.py")
    src = open(path, encoding="utf-8").read()
    body = src.split('"""', 2)[-1]           # skip the module docstring
    assert "engine_histogram" not in body


def test_the_MANIFEST_carries_the_receipts_the_grader_reads():
    """A grader reading a field nobody stamps is a grader that always passes.

    BY VALUE, not by spelling (2026-08-06). This asserted that two literal
    source lines appeared in ``build_clip_manifest``, which proves the lines
    exist and nothing about what an assembled beat actually produces -- and it
    would have gone green throughout the entire scope inversion it was written
    to guard against. Build a real result and read the row.
    """
    from nodes._otr_video_engines import render_driver as rd
    beat_clip = {
        "path": "", "type": "video", "frame_count": 241, "engine_id": "wan_ti2v",
        "native_frame_count": 243, "delivered_native_frame_count": 241,
        "extension_mode": "none",
        "segments": [{"segment_id": "b001_seg00", "beat_id": "b001",
                      "segment_index": 0, "segment_count": 3,
                      "render_frames": 81, "drop_head": 0, "trim_tail": 0,
                      "visible_frames": 81, "native_frame_count": 81,
                      "extension_mode": "none", "init_source": "still",
                      "fear_cape": False, "path": "otr_seg00.mp4"}],
    }
    result = {"ledger": {"video": {"shots": [
        {"shot_id": "shot_b1", "role": "character_video",
         "engine_id": "wan_ti2v", "target_frame_count": 241}]}},
        "clips": {"shot_b1": beat_clip}}
    row = rd.build_clip_manifest(result)["clips"][0]

    assert row["native_frame_count"] == 243, "the beat's rendered-native work"
    assert row["delivered_native_frame_count"] == 241, "what survived the seams"
    assert row["extension_mode"] == "none"
    assert row["frame_count"] == 241
    # The projection travels, SANITIZED: accounting keys only.
    assert [r["segment_id"] for r in row["segments"]] == ["b001_seg00"]
    assert set(row["segments"][0]) == set(acc.SEGMENT_RECEIPT_KEYS)
    assert "path" not in row["segments"][0], (
        "a segment scratch file is never persisted, so naming one in a durable "
        "manifest points a later reader at a file that was swept")
    assert "visible_frames" not in row["segments"][0], (
        "derivable from render_frames/drop_head/trim_tail -- a second number "
        "that can disagree with the first three")


def test_a_SINGLE_RENDER_beat_carries_NO_delivered_native_count():
    """The absence is INFORMATION: it says the historical single-render path
    produced this clip rather than the coverage assembler, so a later reader
    must not "tidy up" the null."""
    from nodes._otr_video_engines import render_driver as rd
    result = {"ledger": {"video": {"shots": [
        {"shot_id": "shot_b1", "role": "character_video", "engine_id": "humo",
         "target_frame_count": 50}]}},
        "clips": {"shot_b1": {"path": "", "type": "video", "frame_count": 50,
                              "engine_id": "humo"}}}
    row = rd.build_clip_manifest(result)["clips"][0]
    assert row["delivered_native_frame_count"] is None
    assert row["segments"] is None


# ---------------------------------------------------------------------------
# The durable script -- "a grader nobody can run is an unowned ruling"
# ---------------------------------------------------------------------------

def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _run_script(ledger_path, manifest_path, *extra):
    script = os.path.join(_REPO, "scripts", "grade_episode.py")
    return subprocess.run(
        [sys.executable, script, "--ledger", ledger_path,
         "--manifest", manifest_path] + list(extra),
        capture_output=True, text=True)


def test_the_SCRIPT_exits_ZERO_on_a_clean_episode(tmp_path):
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")],
                     {"character_video": "humo"})
    manifest = _manifest([_row("shot_b1", "humo")])
    out = _run_script(_write(tmp_path, "l.json", ledger),
                      _write(tmp_path, "m.json", manifest))
    assert out.returncode == 0, out.stderr
    assert "ACCEPTED" in out.stdout


def test_the_SCRIPT_exits_ONE_and_NAMES_the_shot(tmp_path):
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")],
                     {"character_video": "humo"})
    manifest = _manifest([_row("shot_b1", "still_pan")])
    out = _run_script(_write(tmp_path, "l.json", ledger),
                      _write(tmp_path, "m.json", manifest))
    assert out.returncode == 1
    assert "shot_b1" in out.stdout and "still_pan" in out.stdout


def test_the_SCRIPT_exits_TWO_on_an_UNREADABLE_document(tmp_path):
    """A document that cannot be read is not an accepted episode and it is not
    a rejected one either -- conflating "unreadable" with "clean" is how a
    grader reports success on a run it never saw."""
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    out = _run_script(str(bad), str(bad))
    assert out.returncode == 2
    assert "cannot read" in out.stderr


def test_the_SCRIPT_grades_the_WRAPPER_the_render_batch_actually_writes(tmp_path):
    """THE VACUOUS PASS (2026-08-06). ``OTR_VideoRenderBatch`` retains its
    ledger as ``{"ledger": {...}, "master_audio_path": "..."}``, and this script
    handed that WRAPPER straight to a grader that looks for ``video.shots`` at
    the ROOT. Pointed at the real retained artifact it printed

        ACCEPTED: 0 shot(s) delivered the route this episode froze.

    and exited 0 -- success reported on an episode it never graded. Even a
    perfectly repaired honesty rule would never have fired."""
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")],
                     {"character_video": "humo"})
    wrapped = {"ledger": ledger, "master_audio_path": "master.wav"}
    manifest = _manifest([_row("shot_b1", "still_pan")])
    out = _run_script(_write(tmp_path, "wrapped.json", wrapped),
                      _write(tmp_path, "m.json", manifest))
    assert out.returncode == 1, out.stdout + out.stderr
    assert "shot_b1" in out.stdout and "still_pan" in out.stdout


def test_the_SCRIPT_grades_a_WRAPPED_and_a_DIRECT_ledger_IDENTICALLY(tmp_path):
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")],
                     {"character_video": "humo"})
    manifest = _write(tmp_path, "m.json", _manifest([_row("shot_b1", "humo")]))
    direct = _run_script(_write(tmp_path, "d.json", ledger), manifest, "--json")
    wrapped = _run_script(
        _write(tmp_path, "w.json", {"ledger": ledger, "master_audio_path": ""}),
        manifest, "--json")
    assert direct.returncode == wrapped.returncode == 0
    assert direct.stdout == wrapped.stdout


@pytest.mark.parametrize("root", ["[1, 2, 3]", '"a string"', "7", "null"])
def test_the_SCRIPT_exits_TWO_on_a_document_of_the_WRONG_SHAPE(tmp_path, root):
    """PARSEABLE IS NOT READABLE. ``json.load`` happily returns a list, a
    string or a number for a file whose root is not an object, and every reader
    downstream assumes a mapping -- so a document of the wrong shape used to
    crash with an AttributeError instead of exiting 2, which is the exact
    verdict this script promises for a document it cannot read."""
    bad = tmp_path / "bad.json"
    bad.write_text(root, encoding="utf-8")
    good = _write(tmp_path, "m.json", _manifest([]))
    for ledger_path, manifest_path in ((str(bad), good), (good, str(bad))):
        out = _run_script(ledger_path, manifest_path)
        assert out.returncode == 2, out.stdout + out.stderr
        assert "not a JSON object" in out.stderr
        assert "Traceback" not in out.stderr


def test_the_SCRIPT_exits_TWO_on_a_ledger_with_NO_SHOTS(tmp_path):
    """Every rule is per-shot, so an empty shot list makes all of them
    VACUOUSLY true. "Could not grade" belongs with the other document failures
    at exit 2 -- this script already knew unreadable is not clean, and empty had
    to learn the same lesson."""
    out = _run_script(_write(tmp_path, "l.json", _ledger([], {})),
                      _write(tmp_path, "m.json", _manifest([])))
    assert out.returncode == 2
    assert "nothing to grade" in out.stderr
    assert "ACCEPTED" not in out.stdout


def test_the_SCRIPT_can_emit_JSON_for_a_receipt(tmp_path):
    ledger = _ledger([_shot("shot_b1", "character_video", "humo")],
                     {"character_video": "humo"})
    manifest = _manifest([_row("shot_b1", "still_pan")])
    out = _run_script(_write(tmp_path, "l.json", ledger),
                      _write(tmp_path, "m.json", manifest), "--json")
    assert out.returncode == 1
    parsed = json.loads(out.stdout)
    assert parsed[0]["rule"] == acc.RULE_DELIVERED_ENGINE
    assert parsed[0]["shot_id"] == "shot_b1"
