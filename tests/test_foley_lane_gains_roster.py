# -*- coding: utf-8 -*-
"""Every engine that HARVESTS a foley bed must have a row that MIXES it.

PBUG-20260923-05. Five shipping foley lanes were missing from
``FOLEY_LANE_GAINS``, and the consequence was total and silent: the sound was
generated, muxed into every beat clip and written to a durable stem, and then
the episode master was compiled without it -- because membership of that table
is how ``is_foley_route`` decides an episode is a foley episode at all.

A published episode proved it. Nothing failed on the way: the beat mp4 had
audio, the stem was on disk, the render reported success, the receipt was
clean, and the delivered file had healthy levels because the programme audio
was all there. Every check that existed passed. The only question nobody asked
was whether the bed reached the master.

So this file asks it, structurally, at import time. It does not re-measure the
mix; it asserts that the two halves of the contract cannot drift apart -- if a
class inherits the foley harvest, it inherits the obligation to declare how its
bed is mixed.
"""
import pytest


def _engine_classes():
    """Every registered engine, by id -> instance.

    NO SKIP PATH. An earlier draft guessed at ``registry.ENGINES`` and skipped
    when it was absent, which made this guard report green while checking
    nothing -- the same shape of silent pass the bug it guards against had.
    The registry is populated by importing the PACKAGE, then walked with
    ``all_engine_names`` / ``get_engine``, exactly as the other roster tests
    do (see tests/test_frame_contract.py).
    """
    import nodes._otr_video_engines  # noqa: F401 -- populates the registry
    from nodes._otr_video_engines import registry as vreg
    out = {}
    for name in sorted(vreg.all_engine_names()):
        try:
            out[name] = vreg.get_engine(name)
        except Exception:                    # an engine that will not build
            continue                          # is another test's problem
    assert out, "the registry walked to zero engines; this guard would be vacuous"
    return out


def _foley_parent():
    from nodes._otr_video_engines.eng_ltx25 import Ltx25FoleyPlusEngine
    return Ltx25FoleyPlusEngine


def _gains():
    from nodes._otr_video_engines.foley_stems import (
        FOLEY_LANE_GAINS, GLOBAL_MASTER_GAIN_LANES)
    return FOLEY_LANE_GAINS, GLOBAL_MASTER_GAIN_LANES


def test_every_foley_harvesting_engine_declares_its_mix():
    """A lane that harvests a bed and cannot say how to mix it ships silence."""
    engines = _engine_classes()
    parent = _foley_parent()
    gains, _ = _gains()

    missing = []
    for engine_id, cls in sorted(engines.items()):
        klass = cls if isinstance(cls, type) else type(cls)
        if not (isinstance(klass, type) and issubclass(klass, parent)):
            continue
        if engine_id not in gains:
            missing.append(engine_id)

    assert not missing, (
        "these engines subclass the foley lane -- so they harvest an audio "
        "latent, decode it and write a durable stem -- but have no "
        "FOLEY_LANE_GAINS row, which means is_foley_route() returns False for "
        "an episode that selects them and the bed is NEVER mixed into the "
        "master: %s. Add a row (foley_gain, master_gain). See "
        "PBUG-20260923-05." % (", ".join(missing),))


def test_every_global_master_gain_lane_has_gains_to_apply():
    """A lane cannot set a global floor it never declared."""
    gains, global_lanes = _gains()
    orphans = sorted(set(global_lanes) - set(gains))
    assert not orphans, (
        "these lanes are in GLOBAL_MASTER_GAIN_LANES but have no "
        "FOLEY_LANE_GAINS row, so the global floor reads a gain that does not "
        "exist: %s" % (", ".join(orphans),))


def test_mime_lanes_are_per_window_not_global():
    """Mime REPLACES the programme; a global zero would silence the episode.

    The distinction is a ruling rather than an implementation detail, and it is
    the one thing about mime that cannot be inherited from its foley parent --
    the parent is global precisely because it beds UNDER the programme.
    """
    gains, global_lanes = _gains()
    wrong = sorted(lane for lane in gains
                   if "mime" in lane and lane in global_lanes)
    assert not wrong, (
        "mime lanes must stay OUT of GLOBAL_MASTER_GAIN_LANES -- engines are "
        "ROLE-WIDE, so a global zero silences every role sharing the one "
        "master WAV, not just the mimed beats: %s" % (", ".join(wrong),))


def test_every_ltx25_lane_binds_its_own_prompt_formatter():
    """A lane that only INHERITS compose_prompt is treated as having none.

    `render_driver` dispatches with ``type(engine).__dict__.get`` -- not
    ``hasattr`` -- so inheritance does not count. Three assignments covered
    three ids while nine tier lanes silently used the legacy composer: no named
    sounds, no "No speech, no voices." terminator, on every beat they rendered.

    Same mechanism as PBUG-20260923-05 one level up, and just as invisible:
    an unfinished prompt still renders a clip.
    """
    engines = _engine_classes()
    parent = _foley_parent()
    missing = []
    for engine_id, eng in sorted(engines.items()):
        klass = eng if isinstance(eng, type) else type(eng)
        if not (isinstance(klass, type) and issubclass(klass, parent)):
            continue
        if "compose_prompt" not in klass.__dict__:
            missing.append(engine_id)
    assert not missing, (
        "these LTX 2.5 lanes inherit compose_prompt but do not BIND it, so "
        "render_driver's `type(engine).__dict__.get('compose_prompt')` finds "
        "nothing and they fall through to the legacy composer: %s"
        % (", ".join(missing),))


def test_every_prompt_binding_lane_is_in_the_joint_av_set():
    """Binding a formatter and being finished are two halves of one contract.

    `finish_joint_av_positive` returns the positive UNCHANGED for any id
    outside `_JOINT_AV_ENGINES`, so a lane can compose a correct joint-AV
    prompt and still ship it without its terminator. The driver imports this
    same tuple rather than keeping a second literal copy -- two copies is how
    the sets drifted to three-versus-twelve in the first place.
    """
    from nodes._otr_video_engines.eng_ltx25 import _JOINT_AV_ENGINES
    engines = _engine_classes()
    parent = _foley_parent()
    missing = [eid for eid, eng in sorted(engines.items())
               if issubclass(eng if isinstance(eng, type) else type(eng), parent)
               and eid not in _JOINT_AV_ENGINES]
    assert not missing, (
        "these lanes compose a joint-AV prompt but are absent from "
        "_JOINT_AV_ENGINES, so finish_joint_av_positive leaves their positive "
        "unfinished -- no named sounds, no terminator: %s" % (", ".join(missing),))
