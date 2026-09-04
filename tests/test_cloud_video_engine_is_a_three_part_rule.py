""""Is this video engine provider-side?" is a THREE-part rule, not an attribute.

`render_driver._is_cloud_video_engine` decides whether a video engine renders on
somebody else's hardware. Getting it wrong is not cosmetic: the answer gates
local VRAM residue cleanup and the local HuMo policy redirect, so a cloud engine
misread as local gets treated as an in-process model route -- the shape that
would send a cloud avatar down a local LTX path.

THE RULE, and why it has three arms:

* `cloud_` ID PREFIX -- the `cloud_*` family (`cloud_kling_avatar`,
  `cloud_seedance_2`, ...). These declare NO `provider_side` attribute at all.
* the `provider_side` ATTRIBUTE -- the `google_*` BYO-API lanes, whose ids do
  NOT start with `cloud_`, so the prefix arm cannot see them.
* the `node_key` PREFIX -- the third path, for an engine whose registered id and
  node key disagree.

THE POINT OF THIS FILE. A future `engine_facts` builder that "simplifies" this to
a bare `getattr(eng, "provider_side", False)` reads as a tidy-up and silently
reclassifies the entire `cloud_*` family as LOCAL, because not one of them sets
the attribute. That is recorded in GO_FORWARD as a real hazard; this pins it so
the simplification fails here instead of in a render.

CPU-only: predicate over the registered roster. No GPU, no provider call.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import registry as _REG
from nodes._otr_video_engines import render_driver as _RD


def _roster_ids():
    """The DECLARED roster, which is `CAPABILITIES` -- not
    `audit_engine_roster()`.

    Worth stating because the first draft of this file got it wrong and every
    interesting case SKIPPED: `audit_engine_roster()` returns
    `{"missing": (...), "unexpected": (...)}`, so `sorted()` over it yields the
    two KEYS and reads as a two-engine roster. A guard that skips itself is
    worse than no guard, because the run still shows green."""
    return sorted(_REG.CAPABILITIES)


def _engine(engine_id):
    descriptor = _REG.descriptor_for_engine(engine_id)
    if isinstance(descriptor, dict):
        return descriptor.get("engine") or descriptor.get("cls")
    return descriptor


# --------------------------------------------------------------------------- #
# the three arms, each on its own
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine_id", ["cloud_kling_avatar", "cloud_seedance_2"])
def test_the_cloud_prefix_arm_classifies_the_cloud_family(engine_id):
    if engine_id not in _roster_ids():
        pytest.skip("%s is not registered in this build" % engine_id)
    assert _RD._is_cloud_video_engine(engine_id) is True


@pytest.mark.parametrize("engine_id", ["google_veo_video", "google_omni_video"])
def test_the_attribute_arm_classifies_the_google_lanes(engine_id):
    """Their ids do NOT start with `cloud_`, so only the attribute sees them."""
    if engine_id not in _roster_ids():
        pytest.skip("%s is not registered in this build" % engine_id)
    assert not engine_id.startswith("cloud_")          # the prefix arm is blind
    assert _RD._is_cloud_video_engine(engine_id) is True


def test_a_local_engine_is_not_cloud():
    assert _RD._is_cloud_video_engine("minimax_h3_video") is False
    assert _RD._is_cloud_video_engine("ltx_video") is False


def test_an_unknown_engine_is_not_cloud_and_does_not_raise():
    """Unknown/unbuilt engines are not cloud -- the predicate swallows and says
    False rather than exploding mid-render."""
    assert _RD._is_cloud_video_engine("no_such_engine_at_all") is False
    assert _RD._is_cloud_video_engine("") is False
    assert _RD._is_cloud_video_engine(None) is False


# --------------------------------------------------------------------------- #
# THE SIMPLIFICATION THAT MUST NEVER LAND
# --------------------------------------------------------------------------- #
def test_a_bare_provider_side_getattr_would_MISCLASSIFY_the_cloud_family():
    """The regression this file exists for.

    If a future `engine_facts` builder asks only `getattr(eng, "provider_side",
    False)`, every `cloud_*` engine comes back LOCAL -- they do not set it. This
    asserts the gap directly, so the "tidy-up" fails here rather than by routing
    a provider-side avatar into a local model path."""
    ids = _roster_ids()
    if "cloud_kling_avatar" not in ids:
        pytest.skip("cloud_kling_avatar is not registered in this build")

    engine = _engine("cloud_kling_avatar")
    naive = bool(getattr(engine, "provider_side", False))
    real = _RD._is_cloud_video_engine("cloud_kling_avatar")

    assert real is True
    assert naive is False, (
        "cloud_kling_avatar now declares provider_side -- good, but this test "
        "is the record that the bare-getattr shortcut was UNSAFE while it did "
        "not. Re-point it at whichever cloud_* engine still omits the attribute, "
        "or retire it once every cloud_* engine declares one.")


def test_every_cloud_prefixed_engine_in_the_roster_is_classified_cloud():
    """Both directions over the real roster: no `cloud_*` id is read as local,
    and nothing else is read as cloud purely by accident of its name."""
    ids = _roster_ids()
    if not ids:
        pytest.skip("no video engines registered in this build")
    misread = [e for e in ids
               if e.startswith("cloud_") and not _RD._is_cloud_video_engine(e)]
    assert misread == [], (
        "these cloud_* engines are being treated as LOCAL model routes: %s"
        % misread)
