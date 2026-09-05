"""`provider_side` is a CLAUSE of the cloud rule, never the rule itself.

`_is_cloud_video_engine` (`nodes/_otr_video_engines/render_driver.py`) answers
one question the render path cannot get wrong: does this engine load local
video weights? A false NO sends a provider-side avatar down the local VRAM
path; a false YES skips the local residue cleanup a real local engine needs.

It answers it with THREE clauses -- a `cloud_` id prefix, a `provider_side`
attribute, or a `cloud_` `node_key` -- and the reason each one is load-bearing
is not a style preference, it is two shipped engines that disagree:

* `cloud_kling_avatar` declares NO `provider_side` attribute at all, so a
  builder written as `getattr(eng, "provider_side", False)` classifies a Kling
  avatar row as LOCAL and routes an audio-driven face to the local lane.
* `google_veo_video` declares `provider_side = True` but carries NO `cloud_`
  prefix and NO `node_key`, so a builder written on the prefix alone
  classifies a Veo row as LOCAL.

Neither engine is hypothetical and neither is deprecated. These tests pin the
disagreement itself, so that the next person who reaches for a one-line
`getattr` sees the two engines it would break before they write it.
"""
import pytest

from nodes._otr_shared import route_freeze as rf
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd


#: The engine that fails the ATTRIBUTE clause and is carried by the prefix.
KLING_AVATAR = "cloud_kling_avatar"
#: The engine that fails the PREFIX clause and is carried by the attribute.
VEO = "google_veo_video"


def _engine(engine_id):
    assert vreg.is_registered(engine_id), (
        "%s is not registered; this test is about a SHIPPED engine and a "
        "missing one makes it vacuous" % engine_id)
    return vreg.get_engine(engine_id)


# --------------------------------------------------------------------------- #
# 1. The verdict itself, on the picked path.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("engine_id", [KLING_AVATAR, VEO])
def test_a_provider_side_engine_is_cloud_however_it_declares_itself(engine_id):
    assert rd._is_cloud_video_engine(engine_id) is True


def test_a_local_engine_is_not_cloud():
    assert rd._is_cloud_video_engine("ltx25_high_video") is False
    assert rd._is_cloud_video_engine("still_flat") is False
    assert rd._is_cloud_video_engine("") is False
    assert rd._is_cloud_video_engine(None) is False


# --------------------------------------------------------------------------- #
# 2. Each clause is load-bearing -- MEASURED on the two engines, not asserted.
# --------------------------------------------------------------------------- #

def test_the_bare_attribute_alone_would_misroute_the_kling_avatar():
    """This is the defect the three-part rule exists to prevent."""
    eng = _engine(KLING_AVATAR)
    assert bool(getattr(eng, "provider_side", False)) is False, (
        "cloud_kling_avatar has grown a provider_side attribute -- good, but "
        "this test's premise is now stale: re-pin it on whichever engine "
        "still declares none, or retire it if none do")
    # ... and the rule still gets it right, because two other clauses see it.
    assert KLING_AVATAR.startswith("cloud_")
    assert str(getattr(eng, "node_key", "")).startswith("cloud_")
    assert rd._is_cloud_video_engine(KLING_AVATAR) is True


def test_the_prefix_alone_would_misroute_veo():
    eng = _engine(VEO)
    assert not VEO.startswith("cloud_")
    assert not str(getattr(eng, "node_key", "") or "").startswith("cloud_")
    assert bool(getattr(eng, "provider_side", False)) is True
    assert rd._is_cloud_video_engine(VEO) is True


def test_no_single_clause_covers_both_shipped_engines():
    """The summary claim, stated once so a reader cannot miss it."""
    def by_prefix(eid):
        return str(eid).startswith("cloud_")

    def by_attribute(eid):
        eng = vreg.get_engine(eid)
        return bool(getattr(eng, "provider_side", False))

    for clause in (by_prefix, by_attribute):
        verdicts = {eid: clause(eid) for eid in (KLING_AVATAR, VEO)}
        assert not all(verdicts.values()), (
            "a single clause now covers both engines (%r) -- if that is "
            "deliberate, simplify the rule and delete this test; if it is "
            "accidental, the other clauses are still what protect the third "
            "engine nobody has added yet" % verdicts)
    assert all(rd._is_cloud_video_engine(eid) for eid in (KLING_AVATAR, VEO))


# --------------------------------------------------------------------------- #
# 3. The FORCED path reaches the same verdict as the picked one.
# --------------------------------------------------------------------------- #

def test_a_forced_kling_avatar_classifies_exactly_as_a_picked_one(monkeypatch):
    """`OTR_FORCE_ENGINE_MAP` must not be able to smuggle a cloud row local."""
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    snap = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "*=%s" % KLING_AVATAR})
    forced = rf.effective_engine_for_role(
        "announcer", "ltx25_high_video", snapshot=snap)
    assert forced == KLING_AVATAR
    assert rd._is_cloud_video_engine(forced) is True


def test_a_forced_veo_classifies_exactly_as_a_picked_one(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    snap = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "*=%s" % VEO})
    forced = rf.effective_engine_for_role(
        "announcer", "ltx25_high_video", snapshot=snap)
    assert forced == VEO
    assert rd._is_cloud_video_engine(forced) is True


def test_forcing_a_local_engine_over_a_cloud_pick_flips_the_verdict(monkeypatch):
    """The verdict follows the EFFECTIVE engine, not the picked one."""
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    snap = rf.routing_env_snapshot({"OTR_FORCE_ENGINE_MAP": "*=still_flat"})
    forced = rf.effective_engine_for_role("announcer", KLING_AVATAR,
                                          snapshot=snap)
    assert forced == "still_flat"
    assert rd._is_cloud_video_engine(KLING_AVATAR) is True
    assert rd._is_cloud_video_engine(forced) is False
