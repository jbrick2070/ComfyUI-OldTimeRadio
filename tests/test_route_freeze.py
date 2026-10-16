"""THE ONE ROUTE-FREEZE AUTHORITY -- contract tests.

``nodes/_otr_shared/route_freeze.py`` replaced FOUR independent copies of the
same routing rule. Two of those copies swallowed a malformed force map that
the render path treats as terminal. These tests pin the unified contract so
the copies cannot grow back.

The force map is the ONLY rewrite (operator ruling 2026-10-15, "the picked
lane is the lane -- no switcheroo"): a HuMo pick on a radio-host bookend stays
HuMo. The pure core needs no registry, no ComfyUI and no environment at all.
"""

from __future__ import annotations

import pytest

from nodes._otr_shared import route_freeze as rf


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------

def test_snapshot_normalizes_absent_env():
    snap = rf.routing_env_snapshot({})
    assert snap == {"force_engine_map": ""}


def test_snapshot_strips_and_normalizes():
    snap = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "  *=viz_camera  "})
    assert snap == {"force_engine_map": "*=viz_camera"}


def test_fingerprint_changes_with_the_captured_variable():
    """``IS_CHANGED`` must see the force map or ComfyUI serves a stale lock."""
    base = rf.routing_env_snapshot({})
    forced = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "*=viz_camera"})
    prints = {rf.snapshot_fingerprint(s) for s in (base, forced)}
    assert len(prints) == 2


def test_snapshots_agree_is_exact():
    a = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "*=viz_camera"})
    assert rf.snapshots_agree(a, dict(a))
    assert not rf.snapshots_agree(a, rf.routing_env_snapshot({}))


# ---------------------------------------------------------------------------
# The pure core -- no env, no registry
# ---------------------------------------------------------------------------

_NONE = {"force_engine_map": ""}


def _resolve(role, engine_id, mapping=None):
    return rf.resolve_effective_engine(
        role, engine_id, force_mapping=mapping or {})


def test_no_env_is_identity():
    assert _resolve("character_video", "ltx_8gb") == "ltx_8gb"


def test_empty_pick_stays_empty():
    """A blank slot must never acquire a route."""
    assert _resolve("character_video", "") == ""


def test_force_map_star_is_the_catch_all():
    assert _resolve("character_video", "humo",
                    mapping={"*": "viz_camera"}) == "viz_camera"


def test_explicit_role_beats_star():
    assert _resolve("character_video", "humo",
                    mapping={"*": "viz_camera",
                             "character_video": "ltx_8gb"}) == "ltx_8gb"


def test_humo_pick_stays_humo_on_every_role():
    """THE PICKED LANE IS THE LANE: no role is rewritten off a HuMo pick."""
    for role in ("announcer_visual", "music_visual", "character_video"):
        for humo in ("humo", "humo_1.7B", "humo_14B"):
            assert _resolve(role, humo) == humo


def test_cloud_audio_driven_face_stays_cloud():
    """A cloud id is never rewritten onto a local engine (that would create
    the hybrid cloud/local behaviour the cloud profiles exist to avoid)."""
    assert _resolve("announcer_visual", "cloud_example_avatar") == "cloud_example_avatar"


def test_force_map_overrides_a_humo_bookend_pick():
    """The force map is the one operator-chosen rewrite, in either direction."""
    assert _resolve("announcer_visual", "humo",
                    mapping={"*": "viz_camera"}) == "viz_camera"
    assert _resolve("announcer_visual", "ltx_8gb",
                    mapping={"*": "humo"}) == "humo"


def test_resolution_is_idempotent():
    once = _resolve("announcer_visual", "humo", mapping={"*": "viz_camera"})
    assert _resolve("announcer_visual", once) == once


# ---------------------------------------------------------------------------
# Fail-closed parsing -- the contract that was inverted in chunk 1a
# ---------------------------------------------------------------------------

def test_empty_spec_parses_to_empty_map_without_touching_the_driver():
    assert rf.parse_force_map("", driver=object()) == {}
    assert rf.parse_force_map("   ", driver=object()) == {}


def test_malformed_spec_is_terminal():
    class _Driver:
        @staticmethod
        def parse_engine_override(spec):
            raise ValueError("entry %r is not role=engine" % spec)

    with pytest.raises(rf.RouteFreezeError) as exc:
        rf.parse_force_map("garbage-no-equals", driver=_Driver())
    # The message must tell the operator what to do and must refuse to
    # promise a fallback -- a silently unforced render is indistinguishable
    # from a working one.
    assert "NO FALLBACK" in str(exc.value)


def test_parse_delegates_the_grammar_to_the_render_driver():
    """route_freeze owns the POLICY; render_driver still owns the GRAMMAR.

    Keeping one grammar owner is why a public menu id or a legacy id still
    resolves to its internal engine id identically at every reader.
    """
    seen = {}

    class _Driver:
        @staticmethod
        def parse_engine_override(spec):
            seen["spec"] = spec
            return {"*": "viz_camera"}

    assert rf.parse_force_map("*=ltx098_low_video",
                              driver=_Driver()) == {"*": "viz_camera"}
    assert seen["spec"] == "*=ltx098_low_video"


# ---------------------------------------------------------------------------
# The frozen map, against the REAL registry
# ---------------------------------------------------------------------------

def test_freeze_role_engines_omits_empty_slots():
    """A role with no pick is OMITTED, never defaulted."""
    import nodes._otr_video_engines  # noqa: F401
    frozen = rf.freeze_role_engines({}, snapshot=_NONE)
    assert frozen == {}


def test_freeze_role_engines_keeps_a_humo_pick_on_every_role():
    """The operator's bug: a HuMo run must NEVER plan ltx25_audio_in_16gb.

    announcer, music and character all picking humo_1.7B freeze to humo_1.7B
    for every role -- no bookend is rewritten onto another lane."""
    import nodes._otr_video_engines  # noqa: F401

    frozen = rf.freeze_role_engines(
        {"announcer_video_model": {"engine_id": "humo_1.7B"},
         "music_video_model": {"engine_id": "humo_1.7B"},
         "character_video_model": {"engine_id": "humo_1.7B"}},
        snapshot=_NONE)
    assert frozen["announcer_visual"] == "humo_1.7B"
    assert frozen["music_visual"] == "humo_1.7B"
    assert frozen["character_video"] == "humo_1.7B"
    assert "ltx25_audio_in_16gb" not in frozen.values()


def test_every_former_mirror_now_agrees_with_the_authority(monkeypatch):
    """The call sites that used to disagree must now return one answer.

    This is the regression that would have caught the original defect: the
    dispatcher, MetaBrief and the authority all resolving the same
    (role, engine) pair to the same effective engine under the same env.
    """
    import nodes._otr_video_engines  # noqa: F401
    from nodes import otr_image_gen_dispatcher as disp
    from nodes import otr_meta_brief_image_prompt as mbp

    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)

    video_models = {"announcer_video_model": {"engine_id": "humo"}}
    expected = "humo"

    assert rf.effective_engine_for_role("announcer_visual", "humo") == expected
    assert disp._effective_video_engine_for_role("announcer_visual", "humo") == expected
    assert mbp._effective_prompt_engine_for_role(
        video_models, "announcer_visual") == expected
    assert rf.freeze_role_engines(video_models)["announcer_visual"] == expected
