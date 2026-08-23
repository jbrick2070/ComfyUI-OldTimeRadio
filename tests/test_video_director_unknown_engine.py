"""Order 3 (lean-mean): the director is now the membership boundary.

Before this chunk, OTR_VideoDirector deliberately skipped engine ids the
registry did not know (the CW-1 carve-out validates only ``engine_id in
known``), so an unknown id sailed through and was caught one node later by
OTR_ImageDirector's 3D granularity lock -- with a message that misdiagnoses a
stale workflow as an adapter missing ``requires_mesh_portrait``. Downstream of
that lock, ShotLock / MetaBrief / the dispatcher tolerate unknown ids by
design, which is why order 4 (retiring the dormant 3D family the lock rides
on) is FORBIDDEN until this boundary exists: without it, an unknown id would
plan an episode for minutes and die mid-render at ``assert_usable``.

Six pins, each naming the arrival path it closes or preserves:

  1. unknown CUSTOM id        -> named ValueError at the director (this is the
                                 proof the boundary MOVED -- it passes today)
  2. force map -> soak_oom_heavy -> named error at the director (closes the
                                 ENGINE_FAMILY escape: in the family table for
                                 the soak harness, never registered, and the
                                 harness injects at ledger level so no soak leg
                                 ever passes through this node)
  3. canonical-shape picks    -> unchanged: effective == picked, no raise
  4. '+ Add Custom Model'
     with no JSON entry       -> still a WARNING (declare-later), never a raise
  5. humo with hosts unset    -> the redirect's OUTPUT is what must be
                                 registered; the check never fires on it
  6. a RETIRED id             -> still RetiredEngineError, not the new generic
                                 message (precedence pinned)
"""
from __future__ import annotations

import json

import pytest

from nodes import otr_video_director as vd
from nodes._otr_shared.public_engines import (
    RETIRED_ENGINE_IDS,
    RetiredEngineError,
)


def _direct(monkeypatch, announcer="still_flat", music="still_flat",
            character="still_flat", custom_models_json="{}", **env):
    """Drive the real OTR_VideoDirector and return its parsed policy.

    Same harness shape as tests/test_route_freeze_wiring.py: routing env
    cleared first so a developer's shell cannot leak a force map into a pin.
    """
    for key in ("OTR_FORCE_ENGINE_MAP", "OTR_ENABLE_HUMO_HOSTS"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    (raw,) = vd.OTRVideoDirector().direct(
        announcer_video_model=announcer,
        music_video_model=music,
        character_video_model=character,
        announcer_image_model="flux_gen1",
        music_image_model="flux_gen1",
        character_image_model="flux_gen1",
        fps=25, canvas_w=832, canvas_h=480,
        seed_mode="request_hash", request_seed=0,
        custom_models_json=custom_models_json,
    )
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# 1. The proof the boundary moved: this input PASSES the director today.
# --------------------------------------------------------------------------- #
def test_unknown_custom_engine_fails_at_the_director_with_both_names(monkeypatch):
    custom = json.dumps({"character_video_model": "totally_unknown_engine"})
    with pytest.raises(ValueError) as exc:
        _direct(monkeypatch,
                character=vd.ADD_CUSTOM, custom_models_json=custom)
    msg = str(exc.value)
    assert "totally_unknown_engine" in msg          # the offending id
    assert "character_video_model" in msg           # where it came from
    assert "not a registered video engine" in msg   # what is wrong
    assert "OTR_VideoDirector" in msg               # who is speaking
    # The message must help, not just refuse: the registered menu is included
    # so a stale-workflow user can see what to re-pick.
    assert "still_flat" in msg


# --------------------------------------------------------------------------- #
# 2. The ENGINE_FAMILY escape is closed at this boundary.
# --------------------------------------------------------------------------- #
def test_force_map_to_the_soak_stub_fails_at_the_director(monkeypatch):
    """`soak_oom_heavy` is in ENGINE_FAMILY (so parse_force_map admits it) but is
    NOT registered. The soak harness injects it at the ledger-fixture level and
    never routes through this node -- verified before this test was written --
    so refusing it here cannot break a soak leg, and DOES stop a leaked
    OTR_FORCE_ENGINE_MAP from planning an episode on a synthetic stub."""
    with pytest.raises(ValueError) as exc:
        _direct(monkeypatch,
                OTR_FORCE_ENGINE_MAP="character_video=soak_oom_heavy")
    msg = str(exc.value)
    assert "soak_oom_heavy" in msg
    # The dual-knob hint: a force map was active, and the message says so,
    # because the stale half may be the map rather than the pick.
    assert "OTR_FORCE_ENGINE_MAP is active" in msg


# --------------------------------------------------------------------------- #
# 3. Known ids behave byte-for-byte as before.
# --------------------------------------------------------------------------- #
def test_registered_picks_pass_and_freeze_to_themselves(monkeypatch):
    policy = _direct(monkeypatch)
    eff = policy["effective_video_models"]
    assert eff["announcer_visual"] == "still_flat"
    assert eff["music_visual"] == "still_flat"
    assert eff["character_video"] == "still_flat"


# --------------------------------------------------------------------------- #
# 4. Declare-later stays a warning, exactly as the carve-out above promises.
# --------------------------------------------------------------------------- #
def test_unresolved_custom_slot_is_still_a_warning_not_a_raise(monkeypatch):
    """An '+ Add Custom Model' pick with no custom_models_json entry resolves
    to {"engine_id": "", "custom": True} -- an EMPTY id the boundary must skip.
    Raising here would turn the documented declare-later flow into a wall."""
    policy = _direct(monkeypatch, character=vd.ADD_CUSTOM,
                     custom_models_json="{}")
    assert any("left unresolved" in w for w in policy.get("warnings", [])), (
        "the declare-later warning disappeared -- the membership boundary "
        "must not have eaten the unresolved-custom carve-out")


# --------------------------------------------------------------------------- #
# 5. A redirect's OUTPUT is validated; the check never fires on a valid one.
# --------------------------------------------------------------------------- #
def test_humo_redirect_output_is_registered_and_passes(monkeypatch):
    """With OTR_ENABLE_HUMO_HOSTS unset, a humo pick freezes to the
    ltx_audio_in redirect. Both the pick (humo, registered) and the effective
    engine (ltx_audio_in, registered) satisfy the boundary -- pinning that
    validation runs on the union without double-charging the redirect.

    EXACT on purpose (QA finding, 2026-08-23): the harness deletes
    OTR_ENABLE_HUMO_HOSTS before every call, so the redirect is deterministic
    here and `eff` can never legitimately be "humo". The first draft asserted
    membership in {"ltx_audio_in", "humo"}, under which a regression that
    silently disabled the radio-is-host redirect would have slipped through."""
    policy = _direct(monkeypatch, announcer="humo")
    eff = policy["effective_video_models"]["announcer_visual"]
    assert eff == "ltx_audio_in"
    # and the director accepted both halves of the redirect pair -- reaching
    # this line IS the assertion that no ValueError fired.


# --------------------------------------------------------------------------- #
# 6. Retired precedence: the OLD named error, never the new generic one.
# --------------------------------------------------------------------------- #
def test_retired_id_still_raises_the_named_retired_error(monkeypatch):
    retired = sorted(RETIRED_ENGINE_IDS)[0]
    custom = json.dumps({"character_video_model": retired})
    with pytest.raises(RetiredEngineError):
        _direct(monkeypatch,
                character=vd.ADD_CUSTOM, custom_models_json=custom)


# --------------------------------------------------------------------------- #
# 7. The force-map retired path fails with ITS named error, before the boundary.
# --------------------------------------------------------------------------- #
def test_retired_id_in_force_map_fails_before_the_boundary(monkeypatch):
    """QA finding, 2026-08-23: the two retired paths raise DIFFERENT named
    errors, and the force-map one was untested. A retired force-map value
    raises RouteFreezeError from parse_force_map -- which catches everything
    except its own type and re-wraps, embedding the retired message in its
    text -- NOT RetiredEngineError, and never the boundary's generic message.
    Both fire inside freeze_role_engines, strictly before the new check runs."""
    retired = sorted(RETIRED_ENGINE_IDS)[0]
    with pytest.raises(ValueError) as exc:
        _direct(monkeypatch,
                OTR_FORCE_ENGINE_MAP=f"character_video={retired}")
    msg = str(exc.value)
    assert "retired" in msg                          # the real diagnosis travels
    assert "not a registered video engine" not in msg  # never the generic one
