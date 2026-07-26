"""THE ROSTER AUDIT: every registered engine declares a real frame contract.

Multi-clip coverage chunk 7a (2026-07-26). The operator's ruling that ended the
opt-in design: "this architecture should work with all video and still models.
There's no gate with opt in or opt out... I don't like any hidden opt-ins. It
either works or it fails."

A declaration sweep is only worth what its audit is worth. Without this file a
future adapter can be registered with no contract, silently resolve to
``SINGLE_ONLY``, and be planned against a ladder that permits every length --
which is exactly the state chunk 7a exists to leave behind, restored one engine
at a time and invisibly.

So: this asks the LIVE registry, not a hand-kept list. A new engine that
declares nothing fails here, by name, the moment it is registered.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg


def _engines():
    out = []
    for name in sorted(vreg.all_engine_names()):
        try:
            out.append((name, vreg.get_engine(name)))
        except Exception as exc:  # noqa: BLE001
            pytest.fail("engine %r cannot be built: %r" % (name, exc))
    return out


def test_the_registry_is_not_empty():
    """Guard the guard: an empty roster would make every test below vacuous."""
    assert len(_engines()) >= 25


def test_every_registered_engine_declares_a_real_contract():
    """No engine resolves to SINGLE_ONLY, the undeclared default."""
    undeclared = [name for name, eng in _engines()
                  if fc.frame_contract_for(eng) is fc.SINGLE_ONLY]
    assert not undeclared, (
        "these engines declare no frame_contract and would be planned against "
        "an open ladder that accepts any length: %s" % ", ".join(undeclared))


def test_no_declaration_is_merely_a_default_shaped_contract():
    """Equality, not identity -- a contract can BE the default without BEING it.

    ``frame_contract_for`` returns the SINGLE_ONLY object for an undeclared
    engine, so the identity check above is what catches an omission. This
    catches the other half: an adapter that declares ``FrameContract()``
    literally, which passes the identity check while asserting nothing.
    """
    empty = [name for name, eng in _engines()
             if fc.frame_contract_for(eng) == fc.FrameContract()]
    assert not empty, (
        "these engines declare a contract identical to the undeclared default, "
        "which asserts nothing at all: %s" % ", ".join(empty))


@pytest.mark.parametrize("name,eng", _engines(), ids=lambda v: v if isinstance(v, str) else "")
def test_each_contracts_ceiling_is_itself_on_the_ladder(name, eng):
    """``max_frames`` must be a length the engine will actually accept.

    A ceiling off the quantum grid is a number no render can ever request --
    ``largest_legal_at_most(max_frames)`` would land below it forever, so the
    top rung of the declared ladder would be unreachable and the declaration
    would overstate the engine.
    """
    c = fc.frame_contract_for(eng)
    if not c.max_frames or c.discrete_frames:
        return
    assert c.is_legal_length(c.max_frames), (
        "%s declares max_frames=%d, which is not on its own ladder "
        "(min=%d quantum=%d)" % (name, c.max_frames, c.min_frames, c.quantum))


@pytest.mark.parametrize("name,eng", _engines(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_discrete_menu_is_declared_in_FRAMES_not_seconds(name, eng):
    """The unit trap, closed by arithmetic rather than by hope.

    ``discrete_frames`` is compared against a beat's ``target_visible_frames``,
    so the numbers are FRAMES -- but every provider publishes its menu in
    SECONDS, and ``(4, 6, 8)`` is a perfectly well-formed frame menu. Nothing
    in the dataclass can tell the two apart. What CAN: a real clip is never a
    handful of frames long. Any menu entry below one second at the declared
    rate is a seconds-for-frames substitution.
    """
    c = fc.frame_contract_for(eng)
    if not c.discrete_frames:
        return
    assert c.native_fps, (
        "%s declares a discrete menu with no native_fps, so its unit is "
        "unknowable" % name)
    too_short = [d for d in c.discrete_frames if int(d) < int(c.native_fps)]
    assert not too_short, (
        "%s declares menu entries %r shorter than one second at %d fps -- "
        "that is a seconds menu written where frames were required"
        % (name, too_short, c.native_fps))


@pytest.mark.parametrize("name,eng", _engines(), ids=lambda v: v if isinstance(v, str) else "")
def test_strict_first_frame_is_only_claimed_where_a_first_frame_is_taken(name, eng):
    """CHAIN is earned, and only an image-consuming lane can earn it.

    ``strict_first_frame`` is the one continuity mode that lets segment N+1
    begin exactly on segment N's terminal frame. An engine that never receives
    an image has no mechanism to honour that, so the claim would produce a
    visible jump at a join the plan promised was seamless.
    """
    c = fc.frame_contract_for(eng)
    if c.continuity != fc.CONTINUITY_STRICT_FIRST_FRAME:
        return
    required = tuple(getattr(eng, "required_inputs", ()) or ())
    assert "init_image" in required or bool(getattr(eng, "accepts_still", False)), (
        "%s claims strict_first_frame but neither requires an init_image nor "
        "accepts a still, so it has nothing to begin a successor segment from"
        % name)


@pytest.mark.parametrize("name,eng", _engines(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_declared_native_fps_matches_the_engines_own_target_fps(name, eng):
    """One rate, not two. An adapter that publishes ``target_fps`` and a
    contradicting ``native_fps`` is two policies over one state -- and the one
    the planner reads is not the one the render honours."""
    c = fc.frame_contract_for(eng)
    target = getattr(eng, "target_fps", None)
    if not c.native_fps or not target:
        return
    assert int(c.native_fps) == int(target), (
        "%s declares native_fps=%d but target_fps=%d"
        % (name, c.native_fps, target))


# ---------------------------------------------------------------------------
# The declarations are LITERALS on purpose. These pin them to the constants
# they were read from, so a hardware re-measurement moves both or neither.
# ---------------------------------------------------------------------------

def test_the_local_ladders_match_their_adapters_named_constants():
    from nodes._otr_video_engines import eng_humo as _humo
    from nodes._otr_video_engines import eng_ltx_8gb as _l8
    from nodes._otr_video_engines import eng_ltx_av as _lav
    from nodes._otr_video_engines import eng_ltx_video as _lv
    from nodes._otr_video_engines import eng_wan_i2v as _wi
    from nodes._otr_video_engines import eng_wan_ti2v as _wt

    def bounds(engine_name):
        c = fc.frame_contract_for(vreg.get_engine(engine_name))
        return (c.min_frames, c.max_frames)

    assert bounds("wan_i2v") == (_wi._WAN_MIN_FRAMES, _wi._WAN_MAX_FRAMES)
    assert bounds("wan_ti2v") == (_wt._TI2V_MIN_FRAMES, _wt._TI2V_MAX_FRAMES)
    assert bounds("humo") == (_humo._HUMO_MIN_FRAMES, _humo._HUMO_MAX_FRAMES)
    assert bounds("ltx_8gb") == (_l8._LTX8_MIN_FRAMES, 161)
    assert bounds("ltx_video") == (_lv._LTX_MIN_FRAMES, 169)
    assert bounds("ltx_audio_in") == (_lav._LTX_AV_MIN_FRAMES, 497)


def test_an_env_override_may_not_silently_move_a_declared_ceiling():
    """The rule the declarations were written to enforce.

    Three ceilings in this tree are env-overridable -- ``OTR_LTX_MAX_FRAMES``,
    ``OTR_LTX_8GB_MAX_FRAMES``, ``OTR_LTX_AV_MAX_FRAMES``. A FrameContract is
    STATIC by design, because stills are minted against it before the render
    phase begins: a partition computed from mutable state is a partition the
    image phase could not have planned for. So the contract carries the
    LITERAL, and the environment disagreeing with it must become a refusal
    rather than a quiet re-plan.

    This test pins the ONE unavoidable case -- ``_LTX_AV_MAX_FRAMES`` is
    resolved at import, so with the env unset it must equal the declared
    literal. If a box needs a different ceiling, the declaration moves and the
    stills follow; the environment does not get to move it alone.
    """
    from nodes._otr_video_engines import eng_ltx_av as _lav
    import os

    if os.environ.get("OTR_LTX_AV_MAX_FRAMES"):
        pytest.skip("box sets OTR_LTX_AV_MAX_FRAMES; the refusal is 7b's")
    declared = fc.frame_contract_for(vreg.get_engine("ltx_audio_in")).max_frames
    assert _lav._LTX_AV_MAX_FRAMES == declared, (
        "ltx_audio_in declares max_frames=%d but its module constant resolved "
        "to %d -- the environment moved a ceiling the image phase already "
        "planned against" % (declared, _lav._LTX_AV_MAX_FRAMES))


def test_the_google_menus_are_counted_at_the_CANVAS_rate_not_the_providers():
    """The conversion this chunk got wrong twice, pinned in both directions.

    Veo publishes 4/6/8 SECONDS and GENERATES at 24 fps. Writing the seconds
    straight in as frames would assert sixth-of-a-second clips -- the first
    error. Multiplying by 24 asserts frame counts that are UNREACHABLE -- the
    second, caught by the chunk-7a QA panel. ``canonicalize()`` resamples every
    delivered clip to ``canvas.fps`` and then counts ``duration_s * asset.fps``
    at that same rate, so by the time the coverage path sees a Veo clip its
    length is 4/6/8 s * 25 = 100/150/200 frames and 192 never occurs.

    Both wrong answers are pinned out: not the seconds, and not the provider's
    own rate either.
    """
    from nodes._otr_video_engines import eng_google_omni_video as _omni
    from nodes._otr_video_engines import eng_google_veo_video as _veo

    veo = fc.frame_contract_for(vreg.get_engine("google_veo_video"))
    assert _veo.OUTPUT_FPS == 24, "Veo still generates at 24"
    assert veo.native_fps == _veo.CANVAS_FPS == 25, "but it is COUNTED at 25"
    assert veo.discrete_frames == tuple(
        s * _veo.CANVAS_FPS for s in _veo.SUPPORTED_DURATIONS_S)
    assert veo.discrete_frames == (100, 150, 200)
    assert veo.discrete_frames != _veo.SUPPORTED_DURATIONS_S      # not seconds
    assert veo.discrete_frames != (96, 144, 192)                  # not 24 fps

    omni = fc.frame_contract_for(vreg.get_engine("google_omni_video"))
    lo, hi = _omni.OUTPUT_DURATION_RANGE_S
    assert _omni.OUTPUT_FPS == 24
    assert omni.native_fps == _omni.CANVAS_FPS == 25
    assert (omni.min_frames, omni.max_frames) == (lo * 25, hi * 25) == (75, 250)


def test_a_capped_subclass_declares_its_OWN_ceiling():
    """humo_14B_169 renders at a VRAM-safe cap its three siblings do not have.

    Found by the chunk-7a QA panel. The 14B landscape tier sets
    ``safe_render_frames = 49``, and ``HuMoEngine.render_clip`` does
    ``render_max = cap if cap is not None else _HUMO_MAX_FRAMES`` -- so its real
    ceiling is 49 while the contract it INHERITED advertised 177. A beat over 49
    frames raises today. The roster audit could not catch this: it checks each
    contract against itself, never against the subclass's runtime behaviour.
    So the relationship is pinned here, explicitly.
    """
    from nodes._otr_video_engines import eng_humo as _humo

    for name in ("humo", "humo_1.7B", "humo_1.7B_169"):
        engine = vreg.get_engine(name)
        assert getattr(engine, "safe_render_frames", None) is None, name
        assert fc.frame_contract_for(engine).max_frames == _humo._HUMO_MAX_FRAMES

    capped = vreg.get_engine("humo_14B_169")
    cap = capped.safe_render_frames
    assert cap == _humo._HUMO_14B_SAFE_RENDER_FRAMES == 49
    assert fc.frame_contract_for(capped).max_frames == cap, (
        "humo_14B_169 renders at most %d frames but declares a different "
        "ceiling" % cap)


def test_every_engine_with_a_safe_render_cap_declares_it_as_its_ceiling():
    """The general rule, so the next capped tier cannot repeat the trap.

    ``safe_render_frames`` is the adapter's own name for "this is the most I
    will render". An engine that has one and declares a different ceiling is
    telling the planner two different numbers.
    """
    for name, engine in _engines():
        cap = getattr(engine, "safe_render_frames", None)
        if cap is None:
            continue
        declared = fc.frame_contract_for(engine).max_frames
        assert declared == int(cap), (
            "%s renders at most safe_render_frames=%s but declares "
            "max_frames=%s" % (name, cap, declared))


def test_the_sfx_lanes_share_their_base_adapters_ladder_object():
    """The SFX engines call into the Veo/Omni adapters, so restating their
    numbers would be two declarations over one fact. Read, do not copy."""
    veo = fc.frame_contract_for(vreg.get_engine("google_veo_video"))
    for name in ("google_vid_sfx_veo_lite", "google_vid_sfx_veo_fast",
                 "google_vid_sfx_veo_pro"):
        c = fc.frame_contract_for(vreg.get_engine(name))
        assert c.discrete_frames == veo.discrete_frames, name
        assert c.native_fps == veo.native_fps, name

    omni = fc.frame_contract_for(vreg.get_engine("google_omni_video"))
    sfx = fc.frame_contract_for(vreg.get_engine("google_vid_sfx_omni"))
    assert (sfx.min_frames, sfx.max_frames) == (omni.min_frames, omni.max_frames)
