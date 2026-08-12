"""THE BOOMERANG TRIPWIRE -- it is gone, and nothing may bring it back.

CONVERTED, NOT DELETED (no-mirror step 5, 2026-08-06). This file used to prove
the mirror WORKED: `_boomerang_frames` semantics, `_ltx_loop_source_length`
arithmetic, and the class gate. All of that machinery is deleted, so those tests
would have gone with it -- and a deletion that takes its own tests with it goes
green while proving nothing.

What it proves now is the opposite property: that no symbol, no environment
value, and no session state can restore a mirror on this adapter.

WHY A TRIPWIRE AND NOT SILENCE. The boomerang was disarmed twice before it was
removed -- the default flipped off on 2026-08-01, the gate made unconditional on
2026-08-02 -- and it had ALREADY been re-armed once by a default flip going the
other way. A rule that survives only while nobody flips a default is not a rule.
The operator's ruling is flat: no mirror or ping-pong anywhere except the closing
theme, which is a COMPOSITE decision about a tail and never a clip receipt.

WHAT THIS TRIPWIRE DOES NOT COVER, stated so nobody mistakes it for total
(QA finding, 2026-08-06). It catches the SYMBOLS and the RECEIPT. It would NOT
catch someone re-implementing the mirror inline under a different name -- e.g.
``frames = frames + list(reversed(frames[:-1]))`` written straight into
``render_clip``. The durable defence against that is not here: it is
``acceptance.grade_no_mirror``, which asks every delivered clip whether its
frames were rendered and re-derives the count rather than believing the mode,
and ``render_driver``'s refusal of any segment whose length differs from its
plan. A behavioural test would need a GPU to be honest, so it belongs on the
live leg rather than in this file.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import eng_ltx_video as mod
from nodes._otr_video_engines.eng_ltx_video import LtxVideoEngine


#: Every symbol the mirror needed. Named individually rather than pattern-matched
#: so a reader sees exactly what was removed, and so re-adding any ONE of them
#: fails by name instead of slipping in under a partial revival.
_DELETED_SYMBOLS = (
    "_boomerang_frames",
    "_ltx_loop_source_length",
    "_LTX_LOOP_MIN_DECODE_FRAMES_DEFAULT",
)

_DELETED_METHODS = (
    "_loop_via_reverse",
    "_loop_fill_allowed",
    "_LOOP_VIA_REVERSE_DEFAULT",
)


def _code_facts(obj):
    """(string literals, identifiers) actually present in a function's CODE.

    Parsed with ``ast`` rather than scanned as text, because a tombstone that
    EXPLAINS a deletion has to quote the thing it deleted -- so a substring scan
    over raw source flags its own explanation, which is exactly how the first
    draft of this file failed. Comments and docstrings are not code and are not
    consulted; judge the code, not the prose about it.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    literals, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.add(node.value)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    # The docstring is an ast.Constant too; drop it explicitly.
    doc = ast.get_docstring(tree.body[0]) if tree.body else None
    literals.discard(doc)
    return literals, names


@pytest.mark.parametrize("name", _DELETED_SYMBOLS)
def test_the_module_level_mirror_machinery_is_GONE(name):
    assert not hasattr(mod, name), (
        "%s is back in eng_ltx_video. The boomerang mirrored a render "
        "forward-then-backward under dialogue that keeps running forward; it "
        "was deleted 2026-08-06 under the operator's no-mirror ruling." % name)


@pytest.mark.parametrize("name", _DELETED_METHODS)
def test_the_engine_gate_and_its_default_are_GONE(name):
    assert not hasattr(LtxVideoEngine, name), (
        "LtxVideoEngine.%s is back. The gate was the choke point the mirror "
        "ran through; keeping it would let a future edit re-arm the mirror by "
        "changing one return value." % name)


@pytest.mark.parametrize("value", ["on", "1", "true", "yes"])
def test_NO_ENVIRONMENT_VALUE_can_restore_the_mirror(monkeypatch, value):
    """The env hatch had to die with the machinery.

    ``OTR_LTX_LOOP_VIA_REVERSE=on`` opted back into rendering half a beat and
    mirroring it. Asserted BEHAVIOURALLY -- set the knob, then ask the surviving
    length resolver what it renders. A lexical scan of the module would fight
    the tombstone that explains the removal, and would prove spelling rather
    than policy.
    """
    monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", value)
    monkeypatch.setenv("OTR_LTX_LOOP_MIN_DECODE_FRAMES", "97")
    # The half-length the mirror used to render was 97. The surviving resolver
    # must still answer the FULL asked length whatever these knobs say.
    #
    # ASSERTED AGAINST THE ASK, NOT AGAINST THE DECODE FLOOR (lane 9,
    # 2026-08-11). This read `_ltx_frame_length(9, 25) == 169` and passed for a
    # reason that had nothing to do with mirrors: EVERY ask returned 169,
    # because the decode floor raised it. The floor moved to 9 when lane 9
    # measured the band open at 1024x576, and this test went red -- correctly,
    # and while proving nothing about mirrors either way.
    #
    # What actually distinguishes "no mirror" is that the resolver never returns
    # HALF of what was asked for a mirror to double back. So ask for lengths and
    # check the answer is the length, not the half -- an assertion that survives
    # the floor moving again.
    for ask in (169, 121, 97, 49, 9):
        assert mod._ltx_frame_length(ask, 25) == ask, (
            "ask %d came back as something else -- a halved return is the "
            "mirror's signature" % ask)
    # And there is no gate left to consult them.
    assert not hasattr(LtxVideoEngine, "_loop_via_reverse")
    assert not hasattr(LtxVideoEngine, "_loop_fill_allowed")


def test_NO_SESSION_STATE_can_restore_the_mirror():
    """The old gate keyed on ``session_ctx.multi_clip``: a SINGLE-clip beat was
    the case that still mirrored. There is no callable left to key on, for any
    session shape."""
    for prepared in ({}, None, {"session_ctx": {"multi_clip": False}},
                     {"session_ctx": {"multi_clip": True}}):
        for name in ("_loop_fill_allowed", "_loop_via_reverse"):
            assert getattr(LtxVideoEngine, name, None) is None, (
                "%s survives and would decide %r" % (name, prepared))


def test_BOTH_render_paths_declare_none_UNCONDITIONALLY():
    """The receipt is the durable half of the proof.

    While the branch existed both raw returns stamped
    ``"ping_pong" if mirrored else "none"``. With the branch gone the
    conditional must be gone too -- a receipt that can still say ``ping_pong``
    means a code path can still produce one.

    Checked on the METHOD BODIES with comments stripped, because the comment
    explaining the collapse necessarily quotes the code it replaced -- a naive
    substring scan fails on its own tombstone, which is how the first draft of
    this test broke.
    """
    for name in ("render_clip", "_render_clip_hq"):
        literals, names = _code_facts(getattr(LtxVideoEngine, name))
        assert "extension_mode" in literals, name
        assert "none" in literals, name
        assert "ping_pong" not in literals, (
            "%s can still stamp ping_pong" % name)
        assert "ltx_loop_via_reverse" not in literals, (
            "%s still carries the retired receipt field" % name)
        assert "mirrored" not in names, (
            "%s still branches on a mirror flag" % name)
        assert "_boomerang_frames" not in names, name
        assert "_loop_fill_allowed" not in names, name


def test_the_ADAPTER_still_renders_forward_only_lengths():
    """CONTROL -- the deletion must not have taken the length resolver with it.

    ``_ltx_frame_length`` is the surviving authority. It caps a long ask, raises
    an ask below the decode floor, and snaps to 8n+1; it never halves one for a
    mirror to double back. A beat this adapter cannot cover in one render is
    split by coverage planning into chained forward segments.

    THE FLOOR IS READ FROM THE MODULE, NOT WRITTEN AS 169 (lane 9, 2026-08-11).
    The old body asserted `_ltx_frame_length(9, 25) == 169`, which stopped being
    true when lane 9 measured the decode band open at the declared 1024x576 and
    moved the floor to the bottom of the ladder. Derived here so the number can
    move again -- the INVARIANT is "an ask below the floor comes back AT the
    floor", and that outlives any particular floor.
    """
    assert hasattr(mod, "_ltx_frame_length")
    floor = mod._LTX_DECODE_FLOOR_DEFAULT
    cap = mod._LTX_MAX_FRAMES_DEFAULT
    # below the floor -> raised to it (identity when the floor is the ladder's
    # own bottom, which is what an open decode band means)
    assert mod._ltx_frame_length(mod._LTX_MIN_FRAMES, 25) == max(
        floor, mod._LTX_MIN_FRAMES)
    # at the cap -> the cap, unhalved
    assert mod._ltx_frame_length(cap, 25) == cap
    # above the cap -> capped, never mirrored up to it
    assert mod._ltx_frame_length(cap + 200, 25) == cap
