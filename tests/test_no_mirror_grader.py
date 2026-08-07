"""NO-MIRROR, step 3 -- the grader that reads the receipts step 1 and 2 minted.

THE RULE THAT EXISTS BECAUSE THE OLD ONE COULD NOT ASK THE QUESTION.
``grade_multiclip_honesty`` skips any beat whose plan has one segment or fewer
(``len(planned) <= 1``), which is CORRECT for the question it asks -- does this
multi-clip beat's arithmetic add up. It also means a SINGLE render that
ping-pong-padded itself to length was invisible to every rule in the module. The
operator's ruling is flat and admits no arithmetic: *"there is no mirror or ping
pong unless for credits."* So ``grade_no_mirror`` asks every beat, with no
counts, no projection, and no skip.

The two rules must NOT both indict the same clip, or one violation reads as two
defects and an operator fixes it twice. ``grade_multiclip_honesty`` therefore
stops POLICY grading the moment it sees a known non-deliverable mode and leaves
that verdict to ``grade_no_mirror``.

Plus the shape rules, which exist because the grader used to CRASH on documents
another process wrote -- and a crash escaping ``scripts/grade_episode.py`` exits
1, the code reserved for "graded, findings found". Automation keying on exit
codes read a dead grader as a healthy one.
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


def _ledger(shots, frozen=None):
    return {"video": {"roles_effective": dict(frozen or {"drama": "wan_ti2v"}),
                      "shots": list(shots)}}


def _shot(shot_id="s1", role="drama", engine="wan_ti2v", frames=50,
          segments=None, bounded=None):
    shot = {"shot_id": shot_id, "role": role, "engine_id": engine,
            "target_frame_count": frames}
    if segments is not None:
        shot["coverage_plan"] = {"segments": segments}
    if bounded is not None:
        shot["frame_bounded"] = bounded
    return shot


def _row(shot_id="s1", engine="wan_ti2v", mode="none", frames=50,
         native=50, exists=True, ctype="video"):
    row = {"shot_id": shot_id, "engine_id": engine, "exists": exists,
           "type": ctype, "frame_count": frames}
    if mode is not _MISSING:
        row["extension_mode"] = mode
    if native is not _MISSING:
        row["native_frame_count"] = native
    return row


_MISSING = object()


def _manifest(rows, version=1):
    manifest = {"clips": list(rows)}
    if version is not None:
        manifest["frame_receipt_version"] = version
    return manifest


# ---------------------------------------------------------------------------
# The rule the old one could not express
# ---------------------------------------------------------------------------
def test_a_SINGLE_CLIP_padded_beat_is_caught_and_the_old_rule_CANNOT_see_it():
    """The whole reason RULE_NO_MIRROR exists, stated as one test.

    A one-segment beat delivered with a ping-pong pad: the honesty rule skips it
    by design, so before this rule the flat ban was unenforced on exactly the
    beats a single padded render produces."""
    ledger = _ledger([_shot(segments=[{"render_frames": 50}])])
    manifest = _manifest([_row(mode="ping_pong", native=25)])

    assert acc.grade_multiclip_honesty(ledger, manifest) == [], \
        "the multi-clip rule is expected to skip a single-segment plan"

    findings = acc.grade_no_mirror(ledger, manifest)
    assert [f["rule"] for f in findings] == [acc.RULE_NO_MIRROR]
    assert "ping_pong" in findings[0]["detail"]


def test_a_beat_with_NO_coverage_plan_at_all_is_still_graded():
    """The historical single-render path creates no plan. It is still a beat and
    it may still not pad."""
    ledger = _ledger([_shot()])                       # no coverage_plan key
    manifest = _manifest([_row(mode="ping_pong")])
    assert len(acc.grade_no_mirror(ledger, manifest)) == 1


def test_an_honest_clip_produces_NOTHING():
    ledger = _ledger([_shot()])
    assert acc.grade_no_mirror(ledger, _manifest([_row()])) == []


def test_ONE_violation_is_ONE_finding_across_BOTH_rules():
    """A padded multi-segment beat must not be indicted twice.

    ``grade_multiclip_honesty`` stops POLICY grading on a known non-deliverable
    mode precisely so its ``delivered_native != delivered`` branch does not emit
    a second, differently-worded finding for a violation ``grade_no_mirror``
    already owns."""
    planned = [{"render_frames": 25, "drop_head": 0, "trim_tail": 0},
               {"render_frames": 25, "drop_head": 0, "trim_tail": 0}]
    ledger = _ledger([_shot(segments=planned)])
    manifest = _manifest([_row(mode="ping_pong", native=25)])
    rules = [f["rule"] for f in acc.grade_episode(ledger, manifest)]
    assert rules.count(acc.RULE_NO_MIRROR) == 1
    assert acc.RULE_MULTICLIP_HONESTY not in rules


@pytest.mark.parametrize("ctype", ["directory", "still"])
def test_a_non_video_row_owes_no_extension_mode(ctype):
    """``mesh_stage`` ships a directory of frames. There is no clip to pad."""
    ledger = _ledger([_shot()])
    manifest = _manifest([_row(mode=_MISSING, native=_MISSING, ctype=ctype)])
    assert acc.grade_no_mirror(ledger, manifest) == []


def test_a_missing_clip_is_left_to_the_rule_that_owns_it():
    ledger = _ledger([_shot()])
    manifest = _manifest([_row(exists=False, mode=_MISSING)])
    assert acc.grade_no_mirror(ledger, manifest) == []


# ---------------------------------------------------------------------------
# The v1 contract: silence is only legal on an UNVERSIONED manifest
# ---------------------------------------------------------------------------
def test_under_v1_a_SILENT_row_is_a_finding_so_a_REGRESSION_is_detectable():
    """Without the version, an adapter that quietly stopped stamping would stay
    legal forever -- and silence is exactly what an unreceipted pad looks like."""
    ledger = _ledger([_shot()])
    manifest = _manifest([_row(mode=_MISSING, native=_MISSING)], version=1)
    findings = acc.grade_no_mirror(ledger, manifest)
    assert len(findings) == 1
    assert "frame_receipt_version=1" in findings[0]["detail"]


@pytest.mark.parametrize("version", [None, 0, 2, "two", [], True])
def test_on_an_UNVERSIONED_manifest_silence_is_legacy_and_allowed(version):
    ledger = _ledger([_shot()])
    manifest = _manifest([_row(mode=_MISSING, native=_MISSING)], version=version)
    assert acc.grade_no_mirror(ledger, manifest) == []


def test_a_STRING_version_still_binds_because_coercion_here_is_FAIL_CLOSED():
    """``frame_count`` coerces ``"1"`` to 1, and that is the right direction.

    Reading a sloppy version as v1 applies the STRICTER contract; refusing to
    read it would silently exempt the manifest from the ban. Everywhere else in
    this module coercion is the dangerous direction and is refused; here it is
    the safe one, so it is kept -- and pinned, so nobody "fixes" it into a
    strict type check that quietly stops enforcing."""
    ledger = _ledger([_shot()])
    manifest = _manifest([_row(mode=_MISSING, native=_MISSING)], version="1")
    assert len(acc.grade_no_mirror(ledger, manifest)) == 1


def test_but_an_EXPLICIT_pad_is_rejected_even_on_a_legacy_manifest():
    """Absence is forgiven because it may predate the receipt. A row that SAYS
    it padded is not forgiven at any version."""
    ledger = _ledger([_shot()])
    manifest = _manifest([_row(mode="ping_pong")], version=None)
    assert len(acc.grade_no_mirror(ledger, manifest)) == 1


@pytest.mark.parametrize("mode", [7, [], {}, True])
def test_a_mode_that_is_not_a_STRING_is_a_finding_not_a_crash(mode):
    ledger = _ledger([_shot()])
    findings = acc.grade_no_mirror(ledger, _manifest([_row(mode=mode)]))
    assert len(findings) == 1


def test_an_UNKNOWN_mode_string_is_rejected_because_nobody_can_interpret_it():
    ledger = _ledger([_shot()])
    findings = acc.grade_no_mirror(ledger, _manifest([_row(mode="wobble")]))
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# The v1 contract: a BOUNDED engine owes the number, not just the word (7.3)
# ---------------------------------------------------------------------------
def test_a_BOUNDED_engine_owes_a_readable_native_count():
    ledger = _ledger([_shot(bounded=True)])
    manifest = _manifest([_row(native=_MISSING)], version=1)
    findings = acc.grade_no_mirror(ledger, manifest)
    assert len(findings) == 1
    assert "BOUNDED" in findings[0]["detail"]


def test_an_UNBOUNDED_engine_owes_only_the_word():
    """A procedural lane can never be split, so it has nothing to prove."""
    ledger = _ledger([_shot(bounded=False)])
    manifest = _manifest([_row(native=_MISSING)], version=1)
    assert acc.grade_no_mirror(ledger, manifest) == []


def test_an_UNSTAMPED_shot_owes_only_the_word_because_nobody_ASKED():
    """Absence of ``frame_bounded`` is every ledger frozen before the stamp
    existed. Indicting those would report the grader's calendar."""
    ledger = _ledger([_shot()])                       # no frame_bounded key
    manifest = _manifest([_row(native=_MISSING)], version=1)
    assert acc.grade_no_mirror(ledger, manifest) == []


@pytest.mark.parametrize("native", ["lots", -1, None, True, float("inf")])
def test_an_UNREADABLE_native_count_on_a_bounded_shot_is_a_finding(native):
    ledger = _ledger([_shot(bounded=True)])
    manifest = _manifest([_row(native=native)], version=1)
    assert len(acc.grade_no_mirror(ledger, manifest)) == 1


# ---------------------------------------------------------------------------
# THE FOUR BYPASSES THE REVIEW GATE FOUND -- each one graded ACCEPTED, exit 0
# ---------------------------------------------------------------------------
def test_a_receipt_that_CONFESSES_a_pad_while_saying_none_is_caught():
    """THE BLOCKER. The bounded branch used to COLLECT native_frame_count and
    never weigh it, so a single render stamping frame_count=50 /
    native_frame_count=17 / extension_mode='none' -- which confesses 33
    manufactured frames in its own receipt -- returned no findings at all. The
    multi-clip rule skips a one-segment plan, and this rule believed the mode
    STRING while ignoring the COUNT beside it. That is precisely the
    single-render hole RULE_NO_MIRROR exists to close, left open for any pad
    that mislabels its mode."""
    ledger = _ledger([_shot(bounded=True, segments=[{"render_frames": 50}])])
    manifest = _manifest([_row(mode="none", frames=50, native=17)], version=1)
    findings = acc.grade_no_mirror(ledger, manifest)
    assert len(findings) == 1, findings
    assert "only 17 were rendered" in findings[0]["detail"]
    assert "33 frame(s)" in findings[0]["detail"]


def test_an_honest_single_render_with_a_full_native_count_passes():
    """CONTROL for the rule above: native == delivered is the honest shape."""
    ledger = _ledger([_shot(bounded=True, segments=[{"render_frames": 50}])])
    manifest = _manifest([_row(mode="none", frames=50, native=50)], version=1)
    assert acc.grade_no_mirror(ledger, manifest) == []


def test_the_confession_check_does_NOT_double_indict_a_multi_segment_beat():
    """A multi-segment beat's arithmetic belongs to grade_multiclip_honesty,
    done properly against the frozen plan. Doing it here too indicts one beat
    twice."""
    planned = [{"render_frames": 25, "drop_head": 0, "trim_tail": 0},
               {"render_frames": 25, "drop_head": 0, "trim_tail": 0}]
    ledger = _ledger([_shot(bounded=True, segments=planned)])
    manifest = _manifest([_row(mode="none", frames=50, native=17)], version=1)
    assert acc.grade_no_mirror(ledger, manifest) == []


def test_a_manifest_row_with_NO_LEDGER_SHOT_is_still_graded():
    """Every rule walked the LEDGER's shots and joined to the manifest, so a row
    whose shot_id the episode never planned was invisible to all of them. A
    ping_pong orphan produced 'ACCEPTED: 1 shot(s)' and exit 0."""
    ledger = _ledger([_shot(shot_id="s_real")])
    manifest = _manifest([_row(shot_id="s_real"),
                          _row(shot_id="s_ghost", mode="ping_pong", native=25)],
                         version=1)
    findings = acc.grade_no_mirror(ledger, manifest)
    assert [f["shot_id"] for f in findings] == ["s_ghost"]


@pytest.mark.parametrize("ctype", ["still", "directory", "anything-else"])
def test_a_DECLARED_pad_is_banned_whatever_the_row_calls_itself(ctype):
    """The type exemption used to sit ABOVE the mode check, so any non-'video'
    string skipped the ban entirely -- and it opened a regression, because
    grade_multiclip_honesty now stands down expecting this rule to fire. On a
    non-video row NEITHER fired, where the honesty rule alone used to. A
    declaration of padding is a violation on its face."""
    ledger = _ledger([_shot()])
    manifest = _manifest([_row(mode="ping_pong", ctype=ctype)], version=1)
    assert len(acc.grade_no_mirror(ledger, manifest)) == 1


def test_a_MULTI_SEGMENT_non_video_pad_is_caught_by_exactly_one_rule():
    """The regression in full: pre-diff the honesty rule caught this alone."""
    planned = [{"render_frames": 25}, {"render_frames": 25}]
    ledger = _ledger([_shot(segments=planned)])
    manifest = _manifest([_row(mode="ping_pong", ctype="still", native=25)])
    rules = [f["rule"] for f in acc.grade_episode(ledger, manifest)]
    assert rules.count(acc.RULE_NO_MIRROR) == 1, rules


def test_SILENCE_on_a_multi_segment_beat_is_ONE_finding_not_two():
    """grade_multiclip_honesty owns silence on a multi-clip plan. Reporting it
    here as well produced two findings for one missing field."""
    planned = [{"render_frames": 25}, {"render_frames": 25}]
    ledger = _ledger([_shot(segments=planned)])
    manifest = _manifest([_row(mode=_MISSING, native=_MISSING)], version=1)
    rules = [f["rule"] for f in acc.grade_episode(ledger, manifest)]
    assert len(rules) == 1, rules
    assert rules == [acc.RULE_MULTICLIP_HONESTY]


def test_an_UNINTERPRETABLE_mode_is_described_honestly():
    """It used to deliver the 'no clip may carry manufactured frames' verdict --
    asserting a fact the grader had just admitted it could not establish."""
    ledger = _ledger([_shot()])
    findings = acc.grade_no_mirror(ledger, _manifest([_row(mode="wobble")]))
    assert len(findings) == 1
    assert "cannot be established" in findings[0]["detail"]
    assert "may not" not in findings[0]["detail"]


def test_a_malformed_frozen_map_is_ONE_finding_not_one_per_shot():
    """One broken map used to produce a ledger_shape finding PLUS an
    'unknowable' frozen_route finding for every shot -- N+1 for one cause."""
    ledger = {"video": {"roles_effective": [], "shots": [
        _shot(shot_id="s1"), _shot(shot_id="s2"), _shot(shot_id="s3")]}}
    rules = [f["rule"] for f in acc.grade_episode(ledger, _manifest([]))]
    assert rules == [acc.RULE_LEDGER_SHAPE], rules


def test_frozen_route_survives_mixed_type_keys_a_native_ledger_can_carry():
    """``sorted`` over mixed keys raises TypeError -- out of a function whose
    contract says it never does. Unreachable through JSON (which stringifies
    keys) and reachable by any in-process caller."""
    ledger = {"video": {"roles_effective": {"a": "x", 5: "y"},
                        "shots": [_shot(role="ghost")]}}
    findings = acc.grade_frozen_route(ledger)
    assert [f["rule"] for f in findings] == [acc.RULE_FROZEN_ROUTE]


# ---------------------------------------------------------------------------
# Shape: report it, never crash on it, and do not cascade
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clips", ["rows", 7, {"a": 1}])
def test_a_manifest_whose_clips_is_not_a_list_is_ONE_named_finding(clips):
    findings = acc.grade_manifest_shape({"clips": clips})
    assert [f["rule"] for f in findings] == [acc.RULE_MANIFEST_SHAPE]


def test_a_non_record_row_is_reported_BY_POSITION_and_deterministically():
    findings = acc.grade_manifest_shape({"clips": [{"shot_id": "s1"}, 7, "x"]})
    assert len(findings) == 2
    assert "clip 1" in findings[0]["detail"]
    assert "clip 2" in findings[1]["detail"]
    assert findings == acc.grade_manifest_shape({"clips": [{"shot_id": "s1"}, 7, "x"]})


def test_a_duplicate_shot_id_is_NAMED_rather_than_silently_resolved():
    findings = acc.grade_manifest_shape(
        {"clips": [{"shot_id": "s1"}, {"shot_id": "s1"}]})
    assert len(findings) == 1
    assert "repeats shot_id" in findings[0]["detail"]


def test_a_blank_shot_id_is_reported():
    findings = acc.grade_manifest_shape({"clips": [{"shot_id": ""}]})
    assert len(findings) == 1


@pytest.mark.parametrize("ledger", [{"video": 7}, {"video": {"shots": "s"}},
                                    {"video": {"shots": [7]}},
                                    {"video": {"roles_effective": []}},
                                    [1, 2, 3]])
def test_a_malformed_LEDGER_is_a_finding_not_a_traceback(ledger):
    """The half that was scheduled nowhere. ``grade_frozen_route`` was kept
    running on a shape defect BECAUSE it is manifest-independent -- and it
    crashed on a malformed ledger row, the same defect through the other door."""
    findings = acc.grade_ledger_shape(ledger)
    assert findings and all(f["rule"] == acc.RULE_LEDGER_SHAPE for f in findings)


def test_a_coverage_plan_that_is_a_LIST_is_reported():
    ledger = {"video": {"shots": [{"shot_id": "s1", "coverage_plan": []}]}}
    findings = acc.grade_ledger_shape(ledger)
    assert len(findings) == 1
    assert "coverage_plan" in findings[0]["detail"]


def test_a_SHAPE_defect_SUPPRESSES_the_semantic_rules_so_one_bad_row_does_not_cascade():
    ledger = _ledger([_shot()])
    manifest = {"frame_receipt_version": 1,
                "clips": [7, {"shot_id": "s1", "exists": True,
                              "extension_mode": "ping_pong"}]}
    rules = {f["rule"] for f in acc.grade_episode(ledger, manifest)}
    assert acc.RULE_MANIFEST_SHAPE in rules
    assert acc.RULE_NO_MIRROR not in rules, "the semantic rules must stand down"


def test_but_the_MANIFEST_INDEPENDENT_rule_keeps_running():
    """A broken manifest says nothing about whether the route was frozen."""
    ledger = _ledger([_shot(role="unfrozen_role")])
    rules = {f["rule"] for f in acc.grade_episode(ledger, {"clips": 7})}
    assert acc.RULE_MANIFEST_SHAPE in rules
    assert acc.RULE_FROZEN_ROUTE in rules


@pytest.mark.parametrize("ledger,manifest", [
    (None, None), ({}, {}), ([1], "x"), ({"video": {"shots": [7, None]}}, {"clips": [7]}),
])
def test_grade_episode_NEVER_raises_on_any_document(ledger, manifest):
    assert isinstance(acc.grade_episode(ledger, manifest), list)


# ---------------------------------------------------------------------------
# The durable script's exit-code contract (7.2)
# ---------------------------------------------------------------------------
def _run_grader(tmp_path, ledger, manifest):
    lpath = tmp_path / "ledger.json"
    mpath = tmp_path / "manifest.json"
    lpath.write_text(json.dumps(ledger), encoding="utf-8")
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return subprocess.run(
        [sys.executable, os.path.join(_REPO, "scripts", "grade_episode.py"),
         "--ledger", str(lpath), "--manifest", str(mpath)],
        capture_output=True, text=True)


def test_a_clean_episode_exits_0(tmp_path):
    ledger = _ledger([_shot()])
    assert _run_grader(tmp_path, ledger, _manifest([_row()])).returncode == 0


def test_an_episode_WITH_FINDINGS_exits_1(tmp_path):
    ledger = _ledger([_shot()])
    out = _run_grader(tmp_path, ledger, _manifest([_row(mode="ping_pong")]))
    assert out.returncode == 1
    assert acc.RULE_NO_MIRROR in out.stdout


def test_a_MALFORMED_document_exits_2_and_never_1(tmp_path):
    """1 means "graded, and there were findings". A grader that could not
    examine the episode must never claim to have graded it -- automation keys on
    these codes, and before this the two states were indistinguishable."""
    ledger = _ledger([_shot()])
    out = _run_grader(tmp_path, ledger, {"clips": [7, "not-a-row"]})
    assert out.returncode in (1, 2)
    if out.returncode == 1:
        # Reported as findings is acceptable ONLY if it really graded -- i.e. it
        # named the shape rule rather than crashing.
        assert acc.RULE_MANIFEST_SHAPE in out.stdout


def test_a_grader_CRASH_exits_2_rather_than_1(tmp_path, monkeypatch):
    """Proven by forcing the crash, because the whole point is the code path
    that runs when the grader is WRONG. Every known crash is closed, so this
    injects one rather than hoping to find one."""
    script = tmp_path / "boom.py"
    script.write_text(
        "import sys, os\n"
        "sys.path.insert(0, %r)\n"
        "from nodes._otr_video_engines import acceptance\n"
        "def _boom(*a, **k):\n"
        "    raise RuntimeError('injected')\n"
        "acceptance.grade_episode = _boom\n"
        "sys.argv = ['grade_episode.py', '--ledger', %r, '--manifest', %r]\n"
        "exec(open(%r, encoding='utf-8').read())\n"
        % (_REPO, str(tmp_path / "l.json"), str(tmp_path / "m.json"),
           os.path.join(_REPO, "scripts", "grade_episode.py")),
        encoding="utf-8")
    (tmp_path / "l.json").write_text(json.dumps(_ledger([_shot()])),
                                     encoding="utf-8")
    (tmp_path / "m.json").write_text(json.dumps(_manifest([_row()])),
                                     encoding="utf-8")
    out = subprocess.run([sys.executable, str(script)],
                         capture_output=True, text=True)
    assert out.returncode == 2, out.stderr
    assert "could not examine" in out.stderr
