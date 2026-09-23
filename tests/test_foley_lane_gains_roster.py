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


def _is_audio_in(engine):
    """An audio-IN lane, which is EXCLUDED from the generate-audio contract.

    `finish_joint_av_positive` states the split in its own docstring: the lanes
    it finishes "GENERATE audio but are not audio-IN lanes: nothing spoken may
    reach them, and the clause forbids voices outright". So an audio-in lane
    must NOT bind the foley formatter and must NOT be in `_JOINT_AV_ENGINES` --
    `ltx_audio_in` has declined both since 2026-08-27 on purpose, to keep the
    driver's proven talking register.

    An earlier version of the two guards below walked every foley subclass and
    so DEMANDED the wrong thing of these lanes: a test that pins a defect is
    worse than no test, because it makes the correct fix look like a
    regression.
    """
    return str(getattr(engine, "family", "") or "") == "audio_conditioned_video"


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
        if _is_audio_in(eng):
            continue            # audio-IN keeps the driver's talking register
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
               and not _is_audio_in(eng)
               and eid not in _JOINT_AV_ENGINES]
    assert not missing, (
        "these lanes compose a joint-AV prompt but are absent from "
        "_JOINT_AV_ENGINES, so finish_joint_av_positive leaves their positive "
        "unfinished -- no named sounds, no terminator: %s" % (", ".join(missing),))


def test_a_declared_role_set_is_not_a_restriction_it_cannot_enforce():
    """A lane may not declare FEWER roles than it is capability-eligible for.

    `role_compat.engine_fits_role` is "PURELY capability -- every token in the
    engine's required_inputs must be available in the role" and IGNORES this
    list. So narrowing `roles` restricts nothing: the director still selects
    the lane for any role its inputs fit, and if a separate literal table does
    not know it there, the combination refuses at plan time.

    That is precisely what the native audio-in lanes did -- declared cabinet
    roles only, believing it kept them off character faces, while
    `render_driver._AUDIO_IN_CHARACTER_ENGINES` did not list them. Found by a
    codex review.

    DECLARING NO ROLES IS DIFFERENT AND STAYS LEGAL. An empty tuple states no
    preference rather than a false restriction, and the cloud
    audio_conditioned lanes rely on it: aimed at a character beat they refuse
    DELIBERATELY, which `test_wire_w7_mouth_ownership` pins on purpose.
    """
    from nodes._otr_shared.role_compat import ROLE_AVAILABLE_INPUTS

    engines = _engine_classes()
    parent = _foley_parent()
    lying = []
    for engine_id, eng in sorted(engines.items()):
        klass = eng if isinstance(eng, type) else type(eng)
        if not (isinstance(klass, type) and issubclass(klass, parent)):
            continue
        declared = set(tuple(getattr(eng, "roles", ()) or ()))
        if not declared:
            continue                          # states no preference: legal
        required = set(tuple(getattr(eng, "required_inputs", ()) or ()))
        eligible = {r for r, avail in ROLE_AVAILABLE_INPUTS.items()
                    if required <= set(avail)}
        unclaimed = eligible - declared
        if unclaimed:
            lying.append("%s declares %s but is eligible for %s"
                         % (engine_id, sorted(declared), sorted(unclaimed)))
    assert not lying, (
        "these lanes declare a narrower role set than their required_inputs "
        "make them eligible for, which restricts NOTHING -- role_compat is "
        "capability-only. Either widen the declaration and make the lane work "
        "there, or change required_inputs so the capability gate actually "
        "excludes the role: %s" % ("; ".join(lying),))
