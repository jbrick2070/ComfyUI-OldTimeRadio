"""WIRE-W2 -- the typed cast-time image gap, and the swallow that is gone.

THE DEFECT THIS CLOSES. ShotLock decided whether a failed cast-time request
was "the image phase has not run yet" (defer) or "this beat is broken" (fail)
by SUBSTRING-MATCHING the exception message against four needles. The LTX-I2V
gap says "LTX-I2V requires a minted scene still for beat %s; the image phase
produced no usable path", which matches none of them -- so ShotLock re-raised,
plan-build died, and `ltx_video` came back NO_RENDER in the 2026-07-28
engine-coverage campaign. An engine was off the air because of prose.

Deferrability is a property of the RAISE SITE, so the raise site declares it.

AND THE SAME BLOCK WAS FAIL-OPEN TWICE. A non-deferrable `ValueError` and a
bare `except Exception` both logged a warning and returned, so cast-time
preflight reported success on beats it had never actually checked. Those are
gone, and the four propagation tests below are what keep them gone.
"""

from __future__ import annotations

import ast
import os

import pytest

from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import render_errors


# ---------------------------------------------------------------------------
# The type itself
# ---------------------------------------------------------------------------

def test_the_gap_type_is_a_render_error_so_the_render_path_still_fails_loud():
    """The subclass relationship is the safety property, not a detail.

    ShotLock catches the narrow type; everything else -- the render path, the
    batch node, any test -- catches ``RenderError`` and must keep catching
    these. If this ever became a sibling type instead of a subclass, every
    one of those broader handlers would start MISSING a real render failure.
    """
    assert issubclass(render_errors.DeferredImageGapError, render_errors.RenderError)
    assert issubclass(render_errors.RenderError, RuntimeError)
    # Re-exported, and the SAME object -- not a copy. A copy would make
    # `except rd.RenderError` miss a `render_errors.RenderError` and the two
    # import paths would silently disagree.
    assert rd.RenderError is render_errors.RenderError
    assert rd.DeferredImageGapError is render_errors.DeferredImageGapError


def test_the_error_leaf_imports_nothing_from_this_package():
    """It exists to be importable from both sides of a would-be cycle.

    ``retry_taxonomy`` was the obvious home and is the wrong one: render_driver
    already imports it, so a RenderError subclass there imports render_driver
    back. This module stays a LEAF, and this test is what notices if someone
    adds a convenience import to it later.
    """
    src = os.path.join(os.path.dirname(rd.__file__), "render_errors.py")
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module == "__future__", (
                "render_errors must stay a dependency leaf; it imports %r"
                % (node.module,))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in ("os", "sys"), (
                    "render_errors must stay a dependency leaf; it imports %r"
                    % (alias.name,))


# ---------------------------------------------------------------------------
# WHICH raise sites defer -- a structural roster, not a line-number list
# ---------------------------------------------------------------------------

def _raise_roster():
    """Every `raise X(...)` in render_driver, as (exc_name, first_message)."""
    src = os.path.join(os.path.dirname(rd.__file__), "render_driver.py")
    tree = ast.parse(open(src, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        call = node.exc
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        msg = ""
        if call.args:
            arg = call.args[0]
            while isinstance(arg, ast.BinOp):        # "..." % (...)
                arg = arg.left
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                msg = arg.value
            elif isinstance(arg, ast.JoinedStr):
                msg = "".join(v.value for v in arg.values
                              if isinstance(v, ast.Constant))
        out.append((call.func.id, " ".join(msg.split()), node.lineno))
    return out


#: The five CAST-TIME image gaps. Matched by a distinctive phrase from each
#: message rather than by line number, because line numbers rot and this file
#: has already had every cite in it move at least once.
DEFERRABLE = (
    "has NO scene still",                              # scene-init family
    "IA2V TALKING register:",                          # missing cast portrait
    "(MetaBrief mints it when the engine lip-syncs)",  # missing radio face
    "LTX-I2V requires a minted scene still",           # THE ltx_video NO_RENDER
    "OTR_ENABLE_HUMO_HOSTS is ON but no radio_host_portrait",
)

#: Must stay TERMINAL. The first three are POST-image still-spine validation:
#: reaching them means image generation actually ran and FAILED, so deferring
#: would defer to nobody. The last two are wrong-STYLE/ASPECT -- the still
#: exists, it is simply the wrong shape, which no later phase will fix.
TERMINAL = (
    "still-spine handoff missing materialized scene still",
    "still-spine handoff missing materialized jump-segment",
    "still-spine handoff missing materialized portrait",
    "RADIO FACE LOGIC:",
    "radio-face still %r is %dx%d (NOT wide)",
)


@pytest.mark.parametrize("phrase", DEFERRABLE)
def test_every_cast_time_image_gap_declares_itself_deferrable(phrase):
    hits = [(exc, line) for exc, msg, line in _raise_roster() if phrase in msg]
    assert hits, "no raise site carries %r any more -- re-pin this phrase" % phrase
    for exc, line in hits:
        assert exc == "DeferredImageGapError", (
            "render_driver.py:%d raises %s for a CAST-TIME image gap (%r). "
            "ShotLock defers by TYPE now, so a plain RenderError here puts the "
            "engine back on the NO_RENDER list." % (line, exc, phrase))


@pytest.mark.parametrize("phrase", TERMINAL)
def test_post_image_and_wrong_shape_failures_stay_terminal(phrase):
    hits = [(exc, line) for exc, msg, line in _raise_roster() if phrase in msg]
    assert hits, "no raise site carries %r any more -- re-pin this phrase" % phrase
    for exc, line in hits:
        assert exc == "RenderError", (
            "render_driver.py:%d raises %s, but this failure is NOT deferrable "
            "(%r). Deferring a post-image or wrong-shape failure hands it to a "
            "phase that has already run, which is how a broken beat reaches a "
            "render." % (line, exc, phrase))


def test_the_deferrable_and_terminal_sets_do_not_overlap():
    """The two ltx_audio_in bookend raises sit four lines apart and open with
    the SAME sentence -- one is a missing still (deferrable), one is a still of
    the wrong aspect (terminal). A phrase that matched both would let this
    file pass while asserting nothing, so the disjointness is checked."""
    roster = _raise_roster()
    defer = {line for exc, msg, line in roster
             if any(p in msg for p in DEFERRABLE)}
    terminal = {line for exc, msg, line in roster
                if any(p in msg for p in TERMINAL)}
    assert defer and terminal
    assert not (defer & terminal), (
        "these raise lines match BOTH sets: %r" % sorted(defer & terminal))


# ---------------------------------------------------------------------------
# ShotLock's catch: ONE type deferred, everything else propagates
# ---------------------------------------------------------------------------

def _beat():
    return {"beat_id": "b001", "role": "character_video", "char_id": "c1",
            "target_frame_count": 25, "dur_s": 1.0}


def _ledger():
    return {"video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b001", "char_id": "c1",
                       "start_s": 0.0, "dur_s": 1.0}],
            "images": {"images": []}}


def _policy():
    return {"policy_version": 2,
            "video_models": {"character_video_model": {"engine_id": "wan_i2v"}}}


def _preflight(monkeypatch, raiser):
    """Run the cast-time preflight with build_request_from_shot exploding."""
    import nodes._otr_video_engines  # noqa: F401 -- populate the registry
    from nodes import otr_shot_lock as sl

    def _boom(*_a, **_kw):
        raise raiser

    monkeypatch.setattr(rd, "build_request_from_shot", _boom)
    return sl._assert_family_inputs_satisfiable_cast_time(
        "wan_i2v", _beat(), _ledger(), _policy())


def test_a_declared_cast_time_gap_is_DEFERRED_not_raised(monkeypatch):
    """The whole point: cast time runs BEFORE the image phase, so a missing
    still here is expected and the beat proceeds on a placeholder init that
    the still spine later proves was replaced by a real image."""
    _preflight(monkeypatch, rd.DeferredImageGapError(
        "LTX-I2V requires a minted scene still for beat b001; the image "
        "phase produced no usable path."))


def test_a_plain_render_error_still_PROPAGATES(monkeypatch):
    """Deferring is narrow. A RenderError that is not a declared image gap is
    a broken beat and must take plan-build down, exactly as before."""
    with pytest.raises(rd.RenderError, match="engine exploded"):
        _preflight(monkeypatch, rd.RenderError("engine exploded"))


def test_a_value_error_PROPAGATES_it_is_no_longer_swallowed(monkeypatch):
    """REGRESSION for a FAIL-OPEN path that is now gone.

    This used to hit `except ValueError` -> log a warning -> `return`, so
    cast-time preflight reported the beat fine when it had never checked it.
    A preflight that answers 'fine' because it crashed is worse than no
    preflight: the plan is then built on its silence.
    """
    with pytest.raises(ValueError, match="malformed beat"):
        _preflight(monkeypatch, ValueError("malformed beat"))


def test_an_unexpected_exception_PROPAGATES_it_is_no_longer_swallowed(monkeypatch):
    """The second FAIL-OPEN path: a bare `except Exception` that swallowed
    anything at all -- a KeyError in the ledger, a TypeError in a helper, a
    genuine bug -- and returned success."""
    with pytest.raises(KeyError):
        _preflight(monkeypatch, KeyError("images"))
    with pytest.raises(RuntimeError, match="something else entirely"):
        _preflight(monkeypatch, RuntimeError("something else entirely"))


def test_the_ltx_i2v_wording_is_what_the_OLD_needles_missed():
    """The live defect, pinned as arithmetic rather than as a story.

    `ltx_video` came back NO_RENDER because ShotLock's four substring needles
    did not cover the LTX-I2V gap's prose. Deleting the needles fixed it; this
    records WHY, so nobody reintroduces message-matching thinking the needle
    list was merely incomplete. It was not incomplete -- it was the wrong
    mechanism, and any new raise site would have re-broken it.
    """
    old_needles = ("Missing still", "no radio_host_portrait FACE still",
                   "NO portrait in the ledger", "NO scene still in the ledger")
    ltx_i2v = ("LTX-I2V requires a minted scene still for beat b001; the image "
               "phase produced no usable path. NO FALLBACK to text-only "
               "rendering. Fix the image target receipt before video dispatch.")
    assert not any(n in ltx_i2v for n in old_needles)
    # ...and the needles are gone from the source entirely.
    from nodes import otr_shot_lock as sl
    src = open(sl.__file__, encoding="utf-8").read()
    assert "image_gap_needles" not in src
    assert "def _is_deferred_image_gap" not in src
