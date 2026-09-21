# -*- coding: utf-8 -*-
"""The 24 GB foley lane declares 832x480, and this is the pin that says so.

G2 (canvas truth) refuses a lane that declares a ``render_canvas`` with no
test naming both the engine id and the literal canvas. The reason is drift:
a declaration applies LAST and overrules the profile, so if the number here
and the number in the graph ever part company, the graph loses silently and
an operator reading the profile is misled.

This lane inherits its canvas from ``ltx25_foley_plus`` and must keep it.
The Q5 build changes which DiT file loads and nothing else -- same recipe,
same two-stage graph, same 97-frame rung -- so a canvas that moved here
would mean the inheritance broke, not that somebody retuned the lane.
"""
from __future__ import annotations

from nodes._otr_video_engines import eng_ltx25
from nodes._otr_video_engines import registry as vreg

ENGINE_ID = "ltx25_foley_plus_24gb"
DECLARED = (832, 480)


def _engine_cls():
    for cls in (eng_ltx25.Ltx25FoleyPlus24gbEngine,):
        return cls
    raise AssertionError("the 24 GB foley engine is not importable")


def test_the_lane_is_registered_under_that_exact_id():
    assert ENGINE_ID in set(vreg.all_engine_names()), (
        "%s is not registered; G2 pins a canvas for a lane that must exist"
        % ENGINE_ID)
    assert _engine_cls().name == ENGINE_ID


def test_the_declared_canvas_is_832x480_and_matches_its_parent():
    cls = _engine_cls()
    declared = (getattr(cls, "render_canvas", None)
                or getattr(eng_ltx25.Ltx25FoleyPlusEngine, "render_canvas", None))
    assert declared is not None, (
        "%s declares no render_canvas, so G2 has nothing to pin" % ENGINE_ID)
    assert tuple(declared) == DECLARED, (
        "%s declares %r; this pin says %r. If the canvas genuinely moved, "
        "move it here in the same commit and say why -- the declaration "
        "overrules the profile, so the two must never disagree in silence"
        % (ENGINE_ID, tuple(declared), DECLARED))
    parent = tuple(eng_ltx25.Ltx25FoleyPlusEngine.render_canvas)
    assert tuple(declared) == parent, (
        "the Q5 sibling drifted off its parent's canvas (%r vs %r). It is "
        "meant to differ ONLY in which DiT file loads." % (tuple(declared), parent))


def test_the_only_difference_from_the_parent_is_the_weights():
    """The claim the whole lane rests on, asserted rather than trusted."""
    child, parent = _engine_cls(), eng_ltx25.Ltx25FoleyPlusEngine
    assert child.__bases__ == (parent,), (
        "%s no longer subclasses its parent: %r" % (ENGINE_ID, child.__bases__))
    own = set(vars(child)) - {"__module__", "__qualname__", "__doc__"}
    assert own <= {"name", "engine_version", "default_roles", "_dit_name"}, (
        "the Q5 sibling overrides more than its identity and its weights: %r"
        % sorted(own))
    assert child._dit_name(child.__new__(child)) == eng_ltx25.LTX25_DIT_GGUF_24GB
    assert (child._dit_name(child.__new__(child))
            != parent._dit_name(parent.__new__(parent))), (
        "the Q5 sibling resolves the same DiT as its parent, so it is not a "
        "different lane at all")


# --------------------------------------------------------------------------- #
# the fast sibling, which G2 pins on the same terms
# --------------------------------------------------------------------------- #
FAST_ID = "ltx25_foley_plus_32gb"


def test_the_fast_sibling_declares_the_same_canvas():
    """It inherits the canvas and must keep it: only encoder PLACEMENT moves."""
    fast = eng_ltx25.Ltx25FoleyPlusFast32gbEngine
    assert FAST_ID in set(vreg.all_engine_names()), FAST_ID
    assert fast.name == FAST_ID
    declared = (getattr(fast, "render_canvas", None)
                or getattr(eng_ltx25.Ltx25FoleyPlusEngine, "render_canvas", None))
    assert tuple(declared) == DECLARED, (
        "%s declares %r, this pin says %r" % (FAST_ID, tuple(declared), DECLARED))


def test_the_fast_sibling_changes_only_where_the_encoder_runs():
    """The claim it rests on, asserted rather than trusted.

    It must load the SAME weights as its parent -- the difference is the
    encoder, not the model -- and it must decline the pin while its parent
    keeps it.
    """
    fast = eng_ltx25.Ltx25FoleyPlusFast32gbEngine
    parent = eng_ltx25.Ltx25FoleyPlus24gbEngine
    assert fast.__bases__ == (parent,), fast.__bases__

    own = set(vars(fast)) - {"__module__", "__qualname__", "__doc__"}
    assert own <= {"name", "engine_version", "default_roles",
                   "_wrap_text_encoder"}, (
        "the fast sibling overrides more than its identity and the encoder "
        "placement: %r" % sorted(own))

    assert (fast._dit_name(fast.__new__(fast))
            == parent._dit_name(parent.__new__(parent))), (
        "it must load the SAME DiT as its parent; the lever is the encoder")

    sentinel = type("Sentinel", (), {})
    assert fast._wrap_text_encoder(fast.__new__(fast), sentinel) is sentinel, (
        "the fast lane must NOT pin the encoder -- that is its whole point")
    assert parent._wrap_text_encoder(parent.__new__(parent), sentinel) is not sentinel, (
        "and its parent must still pin, or a working lane was muddied")
