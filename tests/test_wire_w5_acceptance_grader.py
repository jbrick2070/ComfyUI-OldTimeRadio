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


def _row(shot_id, engine_id, *, frames=50, exists=True,
         extension_mode="none", native=None):
    row = {"shot_id": shot_id, "engine_id": engine_id, "exists": exists,
           "frame_count": frames, "extension_mode": extension_mode,
           "native_frame_count": frames if native is None else native}
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
    nothing but the receipt can catch it."""
    ledger = _ledger([_shot("shot_b1", "character_video", "wan_ti2v",
                            segments=3)],
                     {"character_video": "wan_ti2v"})
    manifest = _manifest([_row("shot_b1", "wan_ti2v",
                               extension_mode="ping_pong", native=17)])
    findings = acc.grade_multiclip_honesty(ledger, manifest)
    assert [f["rule"] for f in findings] == [acc.RULE_MULTICLIP_HONESTY]
    assert "ping_pong" in findings[0]["detail"]
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


def test_a_SINGLE_CLIP_beat_may_PAD_all_it_likes():
    """CONTROL, and it is the shipped 8 GB WAN tier: a single-clip beat renders
    short on purpose and fills the beat with a mirror (PBUG-20260723-02). If
    this went red the grader would be failing production's majority path."""
    ledger = _ledger([_shot("shot_b1", "character_video", "wan_ti2v",
                            segments=1)],
                     {"character_video": "wan_ti2v"})
    manifest = _manifest([_row("shot_b1", "wan_ti2v",
                               extension_mode="ping_pong", native=17)])
    assert acc.grade_multiclip_honesty(ledger, manifest) == []


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
    ``build_clip_manifest`` must put both receipts on the row."""
    import inspect
    from nodes._otr_video_engines import render_driver as rd
    src = inspect.getsource(rd.build_clip_manifest)
    assert '"native_frame_count": clip.get("native_frame_count")' in src
    assert '"extension_mode": clip.get("extension_mode")' in src


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
