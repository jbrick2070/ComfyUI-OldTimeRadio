"""THE ONE ROUTE-FREEZE AUTHORITY -- contract tests.

``nodes/_otr_shared/route_freeze.py`` replaced FOUR independent copies of the
same two routing rules (force map + radio-is-host redirect). Two of those
copies hard-coded the redirect target as a bare literal and two swallowed a
malformed force map that the render path treats as terminal. These tests pin
the unified contract so the copies cannot grow back.

The pure core takes every predicate INJECTED, so most of this file needs no
registry, no ComfyUI and no environment at all.
"""

from __future__ import annotations

import pytest

from nodes._otr_shared import route_freeze as rf


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------

def test_snapshot_normalizes_absent_env():
    snap = rf.routing_env_snapshot({})
    assert snap == {"force_engine_map": "", "enable_humo_hosts": False}


def test_snapshot_strips_and_normalizes():
    snap = rf.routing_env_snapshot({
        "OTR_FORCE_ENGINE_MAP": "  *=viz_camera  ",
        "OTR_ENABLE_HUMO_HOSTS": "1",
    })
    assert snap["force_engine_map"] == "*=viz_camera"
    assert snap["enable_humo_hosts"] is True


@pytest.mark.parametrize("raw", ["0", "", "true", "yes", "2"])
def test_humo_hosts_is_strictly_the_string_one(raw):
    """Only the exact string "1" opts the toggle ON.

    This mirrors every existing reader (``render_driver.py:1439``,
    ``otr_meta_brief_image_prompt.py:300``). A looser truthiness rule here
    would silently enable HuMo hosts for an operator who exported
    ``OTR_ENABLE_HUMO_HOSTS=true``.
    """
    assert rf.routing_env_snapshot(
        {"OTR_ENABLE_HUMO_HOSTS": raw})["enable_humo_hosts"] is False


def test_fingerprint_changes_with_each_captured_variable():
    """``IS_CHANGED`` must see BOTH variables or ComfyUI serves a stale lock."""
    base = rf.routing_env_snapshot({})
    forced = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "*=viz_camera"})
    hosts = rf.routing_env_snapshot({"OTR_ENABLE_HUMO_HOSTS": "1"})
    prints = {rf.snapshot_fingerprint(s) for s in (base, forced, hosts)}
    assert len(prints) == 3


def test_snapshots_agree_is_exact():
    a = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "*=viz_camera"})
    assert rf.snapshots_agree(a, dict(a))
    assert not rf.snapshots_agree(a, rf.routing_env_snapshot({}))


# ---------------------------------------------------------------------------
# The pure core -- injected predicates, no env, no registry
# ---------------------------------------------------------------------------

_OFF = {"force_engine_map": "", "enable_humo_hosts": False}
_ON = {"force_engine_map": "", "enable_humo_hosts": True}

#: Stand-in for ``render_driver._radio_is_host_redirect_applies``: LOCAL
#: audio-driven-face only. ``cloud_kling_avatar`` is deliberately excluded --
#: it IS an audio-driven-face engine but it is provider-side.
def _redirects(engine_id):
    return engine_id in ("humo", "humo_1.7B", "humo_14B")


def _resolve(role, engine_id, snapshot=_OFF, mapping=None):
    return rf.resolve_effective_engine(
        role, engine_id,
        snapshot=snapshot,
        force_mapping=mapping or {},
        redirect_applies=_redirects,
        redirect_target="ltx_audio_in",
    )


def test_no_env_is_identity():
    assert _resolve("character_video", "wan_i2v") == "wan_i2v"


def test_empty_pick_stays_empty():
    """A blank slot must never acquire a route."""
    assert _resolve("character_video", "") == ""


def test_force_map_star_is_the_catch_all():
    assert _resolve("character_video", "humo",
                    mapping={"*": "viz_camera"}) == "viz_camera"


def test_explicit_role_beats_star():
    assert _resolve("character_video", "humo",
                    mapping={"*": "viz_camera",
                             "character_video": "wan_i2v"}) == "wan_i2v"


def test_redirect_fires_for_bookend_roles_only():
    assert _resolve("announcer_visual", "humo") == "ltx_audio_in"
    assert _resolve("music_visual", "humo") == "ltx_audio_in"
    # character_video is never subject to the guard
    assert _resolve("character_video", "humo") == "humo"


def test_hosts_on_disables_the_redirect():
    assert _resolve("announcer_visual", "humo", snapshot=_ON) == "humo"


def test_cloud_audio_driven_face_stays_cloud():
    """``cloud_kling_avatar`` has a ``cloud_`` id and NO ``provider_side``
    attribute, so it is caught by the id prefix alone in the real predicate.
    Redirecting it to local LTX would create the hybrid cloud/local behaviour
    the cloud profiles exist to avoid."""
    assert _resolve("announcer_visual", "cloud_kling_avatar") == "cloud_kling_avatar"


def test_order_is_force_map_then_redirect():
    """ORDER IS THE CONTRACT, and it matches the render path exactly."""
    # Forcing a HuMo onto a bookend still redirects.
    assert _resolve("announcer_visual", "wan_i2v",
                    mapping={"*": "humo"}) == "ltx_audio_in"
    # Forcing a non-HuMo onto a bookend that WOULD have redirected does not.
    assert _resolve("announcer_visual", "humo",
                    mapping={"*": "viz_camera"}) == "viz_camera"


def test_resolution_is_idempotent():
    once = _resolve("announcer_visual", "humo")
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

    assert rf.parse_force_map("*=wan_8gb", driver=_Driver()) == {"*": "viz_camera"}
    assert seen["spec"] == "*=wan_8gb"


# ---------------------------------------------------------------------------
# The frozen map, against the REAL registry
# ---------------------------------------------------------------------------

def test_freeze_role_engines_omits_empty_slots():
    """A role with no pick is OMITTED, never defaulted."""
    import nodes._otr_video_engines  # noqa: F401
    frozen = rf.freeze_role_engines({}, snapshot=_OFF)
    assert frozen == {}


def test_freeze_role_engines_applies_the_redirect_live():
    import nodes._otr_video_engines  # noqa: F401
    from nodes._otr_video_engines import render_driver as rd

    frozen = rf.freeze_role_engines(
        {"announcer_video_model": {"engine_id": "humo"},
         "character_video_model": {"engine_id": "humo"}},
        snapshot=_OFF)
    assert frozen["announcer_visual"] == rd._NEVER_HUMO_REDIRECT_ENGINE
    assert frozen["character_video"] == "humo"


def test_frozen_target_is_the_constant_not_a_literal():
    """Two former mirrors hard-coded ``"ltx_audio_in"``. If that constant is
    ever renamed, this test fails instead of the product silently splitting
    into an image phase and a render phase that disagree."""
    import nodes._otr_video_engines  # noqa: F401
    from nodes._otr_video_engines import render_driver as rd

    frozen = rf.freeze_role_engines(
        {"announcer_video_model": {"engine_id": "humo"}}, snapshot=_OFF)
    assert frozen["announcer_visual"] == rd._NEVER_HUMO_REDIRECT_ENGINE


def test_every_former_mirror_now_agrees_with_the_authority(monkeypatch):
    """The four call sites that used to disagree must now return one answer.

    This is the regression that would have caught the original defect: the
    dispatcher, MetaBrief and the authority all resolving the same
    (role, engine) pair to the same effective engine under the same env.
    """
    import nodes._otr_video_engines  # noqa: F401
    from nodes import otr_image_gen_dispatcher as disp
    from nodes import otr_meta_brief_image_prompt as mbp
    from nodes._otr_video_engines import render_driver as rd

    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)

    video_models = {"announcer_video_model": {"engine_id": "humo"}}
    expected = rd._NEVER_HUMO_REDIRECT_ENGINE

    assert rf.effective_engine_for_role("announcer_visual", "humo") == expected
    assert disp._effective_video_engine_for_role("announcer_visual", "humo") == expected
    assert mbp._effective_prompt_engine_for_role(
        video_models, "announcer_visual") == expected
    assert rf.freeze_role_engines(video_models)["announcer_visual"] == expected
