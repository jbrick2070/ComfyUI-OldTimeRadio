"""B5 -- the 8 GB tier renders at the canvas IT declares, wherever it is sent.

THE DEFECT (O1). ``build_request_from_shot`` overwrites the request canvas to
the shared landscape default for every non-face family, with deliberate
per-engine branches after it for ``ltx_video`` and ``ltx_audio_in`` -- and none
for ``ltx_8gb``. So the tier that exists because 8 GB cannot afford the pixels
rendered at 1472x832: 8.3x the pixels its own profile asked for.

THE DESIGN, AND WHY IT IS NOT THE OBVIOUS ONE. The profile's 512x288 already
rode the ledger as ``video.canonical_canvas``, so reading that stamp looks like
the fix -- and the first draft of B5 did exactly that. The O1 canvas judgment
had already ruled it out: it enumerated five channels that name a render canvas,
found this one DEAD, and required the engine to declare its canvas STATICALLY,
"not an env var, not a ledger read, not a fourth inline branch." The pre-push
panel supplied the case that shows why: the coverage-matrix harness routes
``ltx_8gb`` onto the CANONICAL workflow through profile ``role_overrides`` and
copies no canvas, so a ledger-read design inherits that workflow's 26:15 canvas
and must either pillarbox or refuse an episode that renders today.

**A declaration cannot be displaced by where it is pointed.** That is the whole
property this file pins. The profile channel keeps one job -- a DRIFT GUARD, so
the number it still carries cannot silently disagree with the declaration.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd

LANE = "ltx_8gb"
DECLARED = (512, 288)
LANDSCAPE = (1472, 832)              # what every non-face engine still takes
REPO = pathlib.Path(__file__).resolve().parents[1]


def _ledger(canvas=None, fps=25, engine=LANE):
    """A ledger complete enough for the REAL request builder.

    The scene still is not scaffolding: an image_to_video engine with no
    per-beat scene still fails LOUD upstream of the canvas chain (the
    no-fallback rip), so a fixture without one never reaches the code under
    test. ``canvas``/``fps`` are here so the tests can prove the ledger is NOT
    consulted -- they are deliberately hostile in several cases below.
    """
    return {
        "episode_id": "ep_b5",
        "images": {"images": [{"beat_id": "b001", "kind": "scene_beat",
                               "path": "C:/tmp/scene_b001.png"}]},
        "video": {
            "fps": fps,
            "canonical_canvas": dict(canvas) if canvas is not None else None,
            "shots": [{"shot_id": "shot_b001", "role": "character_video",
                       "group_id": "grp_character_video", "engine_id": engine,
                       "family": "", "target_frame_count": 25}],
        },
    }


def _shot(ledger):
    return ledger["video"]["shots"][0]


def _canvas(req):
    return (req["canvas"]["w"], req["canvas"]["h"])


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

def test_the_engine_declares_its_own_render_canvas():
    """Beside the frame contract, the aspect and the fps -- the same kind of
    fact, readable without loading anything."""
    engine = vreg.get_engine(LANE)
    assert tuple(engine.render_canvas) == DECLARED
    assert rd.declared_render_canvas(LANE) == DECLARED


def test_the_declared_canvas_is_exactly_16_9_and_32_clean():
    """The two properties that made 512x288 the decision (8gb judgment section
    1): /32 for the latent grid, exactly 16:9 so it scales 3.75x to the
    1920x1080 deliverable with ZERO pad area. Asserted as arithmetic rather
    than as the literal, so the reason survives a future re-tuning to the only
    other qualifying rung.

    The deliverable check is an ASPECT identity, not a divisibility one: the
    scale is 3.75x, so 512 does not divide 1920 evenly and never needed to.
    What matters for zero pad area is that the two rectangles are the same
    shape.
    """
    width, height = DECLARED
    assert width % 32 == 0 and height % 32 == 0
    assert width * 9 == height * 16
    assert width * 1080 == height * 1920      # same shape as the deliverable


def test_engines_that_declare_NOTHING_are_left_alone():
    """The seam is per-adapter opt-in by declaration. An engine that declares
    no canvas keeps whatever the existing chain gives it -- this is not a
    global canvas rewrite wearing one engine's name."""
    # ltx_video LEFT this group deliberately on 2026-08-02 (kibitz r3): its
    # 169-frame decode floor is only true at one canvas, so an undeclared canvas
    # was a live channel for invalidating a static contract via
    # OTR_LTX_RENDER_CANVAS. It now declares (832, 480). wan_i2v left the same
    # way on 2026-08-11 (video transplant, lane 1) for the same reason -- with
    # no declaration it fell through to 1472x832, 3.07x the pixels its tier was
    # configured for, on a 14B fp8 lane at a 14.5 GiB gate. The invariant this
    # test guards is unchanged -- an engine that declares NOTHING is still
    # untouched -- and still_pan now carries the control.
    # `humo` left this list in lane 4 (2026-08-11) when the HuMo family
    # closed -- the third occupant to leave. `ltx_audio_in` left in lane 7 the
    # same day: its ia2v graph halves the canvas for stage A and doubles it
    # back with a fixed-x2 upsampler, so an undeclared canvas could produce a
    # stage-A latent LTX's /32 grid rejects. It declares (1024, 576) now.
    # `mesh_stage` left in lane 10: it had been choosing 1472x832 with an
    # inline sniff inside render_clip, which is a declaration everywhere except
    # where anything could read it, and it declares (1472, 832) properly now.
    # What remains are lanes whose own packets have not run yet.
    for other in ("still_pan", "viz_mxc_cpu"):
        assert rd.declared_render_canvas(other) is None
    assert rd.declared_render_canvas("an_engine_that_does_not_exist") is None
    assert rd.declared_render_canvas("") is None


def test_render_single_takes_the_DECLARATION_not_the_aspect_default(
        monkeypatch):
    """THE THIRD CANVAS CHANNEL, found on a live render in lane 7 (2026-08-11).

    `build_request_from_shot` applies `declared_render_canvas` LAST so nothing
    can clobber it -- but `render_single` builds its OWN request and never
    asked. It derived the canvas from `render_aspect` plus
    `OTR_VIDEO_RENDER_CANVAS`: wide -> 832x480, else the portrait default.

    EVERY solo lane smoke runs through that function, so every lane was
    validating the aspect default rather than its declaration. It stayed
    invisible through six lanes because all six declared exactly what this path
    already produced (832x480 wide, 480x832 portrait). `ltx_audio_in` was the
    first lane to declare something else, and its stage-A /32 guard failed the
    live render immediately -- which is how this was found.

    Asserted here rather than on ltx_audio_in alone: the property belongs to
    the declaration mechanism, not to one lane.
    """
    captured = {}
    monkeypatch.setattr(rd, "build_request",
                        lambda shot, assets, frames, canvas: captured.setdefault(
                            "canvas", canvas) or {"canvas": canvas})
    monkeypatch.setattr(rd, "_render_one",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setenv("OTR_VIDEO_RENDER_CANVAS", "832x480")
    rd.render_single(LANE, frame_count=9)
    assert captured["canvas"] == DECLARED, (
        "render_single handed %r to build_request while %s declares %r -- the "
        "declaration must reach every request builder, not just the shot one"
        % (captured["canvas"], LANE, DECLARED))
    # an EXPLICIT canvas still wins, so an off-declaration probe stays possible
    captured.clear()
    rd.render_single(LANE, frame_count=9, canvas=(640, 384))
    assert captured["canvas"] == (640, 384)


@pytest.mark.parametrize("bad,needle", [
    ((0, 0), "positive"),
    ((-512, -288), "positive"),
    ((500, 288), "divisible by 32"),
    ((512, 300), "divisible by 32"),
])
def test_an_UNRENDERABLE_declaration_is_refused(monkeypatch, bad, needle):
    """A declaration its own latent grid cannot accept is a bug in the adapter.
    It must not reach a render, and it must not fall through to the landscape
    default -- that silent fall-through IS the defect."""
    monkeypatch.setattr(vreg.get_engine(LANE), "render_canvas", bad,
                        raising=False)
    with pytest.raises(rd.RenderError) as excinfo:
        rd.declared_render_canvas(LANE)
    assert needle in str(excinfo.value)
    assert "NO FALLBACK" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["512x288", 512, {"w": 512}, (512,)])
def test_a_MALFORMED_declaration_is_refused(monkeypatch, bad):
    monkeypatch.setattr(vreg.get_engine(LANE), "render_canvas", bad,
                        raising=False)
    with pytest.raises(rd.RenderError, match="not a .width, height. pair"):
        rd.declared_render_canvas(LANE)


# ---------------------------------------------------------------------------
# The suppression, through the real request builder
# ---------------------------------------------------------------------------

def test_the_request_carries_the_DECLARED_canvas_not_the_landscape_default():
    """THE ASSERTION THAT BREAKS IF THE SUPPRESSION IS MISSING.

    Driven through the real ``build_request_from_shot``, which is where the
    1472x832 overwrite lives. 512 is not 1472 by a margin no rounding explains.
    """
    ledger = _ledger()
    assert _canvas(rd.build_request_from_shot(_shot(ledger), ledger)) == DECLARED


def test_a_SIBLING_lane_still_takes_the_landscape_default():
    """The differential control: a lane that declares NOTHING must be
    untouched, or this was a global canvas rewrite rather than a per-adapter
    declaration.

    The control has moved three times, which is itself the story: wan_ti2v held
    it until 2026-08-02, wan_i2v until 2026-08-11 morning, mesh_stage until
    lane 10 that same day, and every handover happened because the lane gained
    a declaration of its own. still_pan holds it now (humo cannot -- it is a
    face family and takes the portrait canvas). When still_pan declares its
    canvas in lane 16, this control moves again rather than the test being
    deleted: the invariant outlives every occupant.
    """
    ledger = _ledger(engine="still_pan")
    assert _canvas(rd.build_request_from_shot(_shot(ledger),
                                              ledger)) == LANDSCAPE


def test_ltx_video_keeps_its_own_declared_canvas():
    """ltx_video keeps ITS canvas; the ltx_8gb branch sits after it and must
    not disturb it. That is the property, and it is unchanged.

    The number moved 832x480 -> 1024x576 by operator ruling (2026-08-11):
    the HQ two-stage path halves for stage A and upsamples with a fixed-x2
    node, so both axes must be /64 or stage A is illegal. Read from the
    declaration rather than repeated here, so the next move does not make
    this test lie (lesson L10).
    """
    from nodes._otr_video_engines.eng_ltx_video import LtxVideoEngine
    ledger = _ledger(engine="ltx_video")
    got = _canvas(rd.build_request_from_shot(_shot(ledger), ledger))
    assert got == tuple(LtxVideoEngine.render_canvas)


# ---------------------------------------------------------------------------
# THE PROPERTY: the ledger cannot displace the declaration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hostile", [
    {"w": 832, "h": 480},            # the default stamp -- 26:15
    {"w": 1472, "h": 832},           # the landscape default
    {"w": 1920, "h": 1080},          # the deliverable
    None,                            # no stamp at all
])
def test_the_LEDGER_canvas_cannot_displace_the_declaration(hostile):
    """THE TEST THE FIRST DRAFT WOULD HAVE FAILED.

    A ledger-reading design returns the stamp here -- 832x480 for the canonical
    workflow, or a refusal when the stamp is missing. The declaration ignores
    all four and renders at 512x288, which is what lets an operator force this
    engine onto any workflow and still get the canvas the tier was built for.
    """
    ledger = _ledger(canvas=hostile)
    assert _canvas(rd.build_request_from_shot(_shot(ledger),
                                              ledger)) == DECLARED


def test_FORCING_this_engine_onto_a_non_16_9_workflow_still_renders():
    """The case the pre-push panel supplied, pinned by name.

    ``OTR_FORCE_ENGINE_MAP`` and the coverage matrix's profile
    ``role_overrides`` both reroute a shot to ``ltx_8gb`` without touching the
    workflow's canvas. Under the ledger-reading draft that refused the whole
    episode; under the declaration it renders at 512x288, which is the point.
    """
    ledger = _ledger(canvas={"w": 832, "h": 480}, engine="wan_ti2v")
    shot = _shot(ledger)
    shot["engine_id"] = LANE                       # what the force map does
    assert _canvas(rd.build_request_from_shot(shot, ledger)) == DECLARED


def test_the_ledger_fps_cannot_displace_it_either():
    ledger = _ledger(fps=30)
    assert _canvas(rd.build_request_from_shot(_shot(ledger),
                                              ledger)) == DECLARED


# ---------------------------------------------------------------------------
# The drift guard the judgment asked for
# ---------------------------------------------------------------------------

def test_the_profile_canvas_matches_the_declaration():
    """The DEAD channel keeps exactly one job.

    ``config/profiles/otr_8gb_ltx.json`` still carries `render.canvas_w/h`, and
    nothing on the render path consumes it any more. It is not authority -- but
    it must not silently disagree with the adapter either, or the next reader
    trusts the wrong number. This is the O1 judgment's item 6, and it is the
    reason the profile channel may stay unconsumed rather than be deleted.
    """
    profile = json.loads(
        (REPO / "config" / "profiles" / "otr_8gb_ltx.json").read_text("utf-8"))
    render = profile.get("render") or {}
    assert (int(render.get("canvas_w")), int(render.get("canvas_h"))) == DECLARED
    assert int(render.get("fps")) == 25
    assert (profile.get("slot_overrides") or {}).get(
        "video_render_engine") == LANE


def test_the_8gb_variant_workflow_agrees_with_the_declaration():
    """Same guard, one channel further out: the shipped variant's director
    widgets are what actually produce the (now unconsumed) stamp."""
    doc = json.loads((REPO / "workflows" / "variants"
                      / "otr_8gb_ltx.json").read_text("utf-8"))
    director = next(n for n in doc["nodes"]
                    if str(n.get("type") or "") == "OTR_VideoDirector")
    values = director["widgets_values"]
    assert (int(values[7]), int(values[8])) == DECLARED
    assert int(values[6]) == 25


# ---------------------------------------------------------------------------
# The ordering: judged before the weights load
# ---------------------------------------------------------------------------

def _plan_shot(ledger):
    shot = _shot(ledger)
    shot["coverage_plan"] = {
        "target_visible_frames": 129, "join_mode": "chain",
        "segments": [
            {"index": 0, "render_frames": 65, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 65, "drop_head": 1, "trim_tail": 0},
        ],
    }
    return shot


def test_a_broken_declaration_is_refused_BEFORE_the_beat_session_opens(
        monkeypatch):
    """Opening the session PREPARES -- it loads a 6.34 GiB checkpoint -- and
    the per-segment request builder runs INSIDE it. Detonating BeatSession
    proves the refusal comes first; without the pre-flight this fails with
    AssertionError instead of RenderError."""
    from nodes._otr_video_engines import beat_session as bs

    def _must_not_open(*_a, **_k):
        raise AssertionError("BeatSession opened before the canvas was judged")

    monkeypatch.setattr(bs, "BeatSession", _must_not_open)
    monkeypatch.setattr(vreg.get_engine(LANE), "render_canvas", (500, 288),
                        raising=False)
    ledger = _ledger()
    with pytest.raises(rd.RenderError, match="divisible by 32"):
        rd.render_beat_coverage(_plan_shot(ledger), ledger,
                                request_builder=rd.build_request_from_shot)


def test_a_SOUND_declaration_still_reaches_the_session(monkeypatch):
    """The control for the ordering test: the pre-flight must not refuse a
    legal declaration, or it would block every 8 GB beat rather than the broken
    ones. Reaching the sentinel IS the pass."""
    from nodes._otr_video_engines import beat_session as bs

    class _Reached(RuntimeError):
        pass

    def _sentinel(*_a, **_k):
        raise _Reached("the canvas passed and the session was reached")

    monkeypatch.setattr(bs, "BeatSession", _sentinel)
    ledger = _ledger()
    with pytest.raises(_Reached):
        rd.render_beat_coverage(_plan_shot(ledger), ledger,
                                request_builder=rd.build_request_from_shot)
