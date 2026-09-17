"""EVERY provider-side engine a dropdown can reach must be FLOORABLE.

Operator, 2026-09-16: *"any fixes need to be in canonical and all 17+ [workflow
JSONs]. They may be dormant in dropdowns but the user can flip, so they need to
be resilient."*

That instinct is right, and this file is the durable form of it. The cloud floor
itself is Python, so it applies to every workflow without a JSON edit -- but the
floor is GATED on ``_is_cloud_video_engine`` / ``_is_cloud_image_engine``, and an
engine that is provider-side while failing its gate is a dormant landmine: the
user flips the dropdown to it, one job fails, and the whole paid run dies with no
floor. That is the beat-40 defect again, on a lane nobody happened to test.

Only 13 of the ~49 registered engine ids appear in any workflow JSON at the time
of writing. The other three quarters are reachable ONLY by flipping a dropdown,
so pinning coverage to the shipped JSONs would miss most of the surface. These
tests walk the REGISTRIES instead, which is the real dropdown.

THE EVIDENCE IS INDEPENDENT OF THE GATE ON PURPOSE. ``_provider_side_evidence``
reads the engine's OWN declarations -- name, provider_side, native, node_key,
defining module. If it simply called the gate, the test would agree with itself
and prove nothing.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import cloud_media_backend as cmb
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes import otr_image_gen_dispatcher as disp

try:
    from nodes._otr_image_engines import registry as ireg
except ImportError:  # pragma: no cover -- image registry is optional
    ireg = None


def _provider_side_evidence(engine_id, eng):
    """Reasons to believe this engine talks to a PROVIDER, gate excluded."""
    reasons = []
    if str(engine_id or "").startswith("cloud_"):
        reasons.append("id starts with cloud_")
    if getattr(eng, "provider_side", False):
        reasons.append("provider_side=True")
    if getattr(eng, "native", True) is False:
        reasons.append("native=False")
    node_key = str(getattr(eng, "node_key", "") or "")
    if node_key.startswith("cloud_"):
        reasons.append("node_key=%s" % node_key)
    return reasons


def _registered(registry_obj):
    return sorted(getattr(registry_obj, "_registry", {}) or {})


VIDEO_IDS = _registered(getattr(vreg, "_VIDEO_REGISTRY", None))
IMAGE_IDS = _registered(getattr(ireg, "_IMAGE_REGISTRY", None)) if ireg else []


@pytest.mark.parametrize("engine_id", VIDEO_IDS)
def test_every_provider_side_video_engine_is_recognized_by_the_floor(engine_id):
    eng = vreg.get_engine(engine_id)
    evidence = _provider_side_evidence(engine_id, eng)
    if not evidence:
        # A local engine MUST NOT be gated as cloud: its failures are our own
        # bugs and have to keep failing LOUD under NO FALLBACKS.
        assert not rd._is_cloud_video_engine(engine_id), (
            "%s shows no provider-side evidence but the cloud gate claims it; "
            "a local fault would be floored instead of failing loud" % engine_id)
        return
    assert rd._is_cloud_video_engine(engine_id), (
        "%s is provider-side (%s) but the cloud floor does NOT recognize it. "
        "Flip a dropdown to this engine and one failed job kills the whole "
        "paid run with no floor." % (engine_id, "; ".join(evidence)))


@pytest.mark.parametrize("engine_id", IMAGE_IDS)
def test_every_provider_side_image_engine_is_recognized_by_the_floor(engine_id):
    eng = ireg.get_engine(engine_id)
    evidence = _provider_side_evidence(engine_id, eng)
    if not evidence:
        assert not disp._is_cloud_image_engine(engine_id), (
            "%s shows no provider-side evidence but the cloud gate claims it"
            % engine_id)
        return
    assert disp._is_cloud_image_engine(engine_id), (
        "%s is provider-side (%s) but the still floor does NOT recognize it"
        % (engine_id, "; ".join(evidence)))


@pytest.mark.parametrize("engine_id", VIDEO_IDS)
def test_every_cloud_video_engine_can_actually_be_floored(engine_id):
    """Recognition is not enough -- prove the floor RETURNS a reason.

    The gate and the floor are two different functions, and a beat is only
    saved when ``_cloud_floor_reason`` yields something truthy.
    """
    if not rd._is_cloud_video_engine(engine_id):
        return
    shot = {"shot_id": "s1", "engine_id": engine_id}
    for code in cmb.JOB_SCOPED_CODES:
        err = cmb.CloudMediaError(code, "%s: no clip" % engine_id)
        wrap = rd.RenderError("fallbacks are disabled")
        wrap.__cause__ = err
        assert rd._cloud_floor_reason("s1", {"s1": wrap}, shot) == code.value, (
            "%s cannot be floored on %s" % (engine_id, code.value))
    # ...and a run-scoped verdict on the same engine must still NOT floor.
    for code in cmb.RUN_SCOPED_CODES:
        err = cmb.CloudMediaError(code, "%s: stop" % engine_id)
        assert rd._cloud_floor_reason("s1", {"s1": err}, shot) == "", (
            "%s floors on run-scoped %s, which would publish an empty show"
            % (engine_id, code.value))


@pytest.mark.parametrize("engine_id", IMAGE_IDS)
def test_every_cloud_image_engine_can_actually_be_floored(engine_id):
    if not disp._is_cloud_image_engine(engine_id):
        return
    for code in cmb.JOB_SCOPED_CODES:
        err = cmb.CloudMediaError(code, "%s: no image" % engine_id)
        assert disp._cloud_still_job_failure(err, engine_id) == code.value, (
            "%s cannot be floored on %s" % (engine_id, code.value))
    for code in cmb.RUN_SCOPED_CODES:
        err = cmb.CloudMediaError(code, "%s: stop" % engine_id)
        assert disp._cloud_still_job_failure(err, engine_id) == "", (
            "%s floors on run-scoped %s" % (engine_id, code.value))


def test_the_registries_are_not_empty():
    """A registry that failed to import would make every test above vacuous."""
    assert len(VIDEO_IDS) >= 20, VIDEO_IDS
    assert len(IMAGE_IDS) >= 5, IMAGE_IDS


def test_a_local_engine_is_never_floored_on_either_funnel():
    """The negative control for the whole file."""
    local_video = [e for e in VIDEO_IDS if not rd._is_cloud_video_engine(e)]
    assert local_video, "no local video engine to test the gate against"
    err = cmb.CloudMediaError(cmb.CloudErrorCode.TIMEOUT, "x")
    for engine_id in local_video:
        assert rd._cloud_floor_reason(
            "s1", {"s1": err}, {"engine_id": engine_id}) == ""
    local_image = [e for e in IMAGE_IDS if not disp._is_cloud_image_engine(e)]
    for engine_id in local_image:
        assert disp._cloud_still_job_failure(err, engine_id) == ""


# --------------------------------------------------------------------------- #
# The CONVENTION the gate rests on, defended at its root
# --------------------------------------------------------------------------- #
def test_every_partner_row_key_starts_with_cloud():
    """The gate reads a PREFIX; this is what keeps that honest.

    ``_is_cloud_video_engine`` / ``_is_cloud_image_engine`` decide an engine is
    provider-side partly by ``node_key.startswith("cloud_")``. That is a naming
    convention, and an undefended convention is a landmine with a timer: add a
    partner row called ``acme_v3_i2v``, wire an engine to it, and the engine
    reaches a real provider while both floors look straight past it. One failed
    job then kills a whole paid run -- the beat-40 defect, on a lane nobody
    tested because the engine was dormant in a dropdown until someone flipped it.

    If this test fails, do NOT just rename the row to make it pass unless the
    rename is genuinely right: the alternative fix is to make the gates consult
    ``partner_rows()`` directly instead of trusting the prefix.
    """
    from nodes._otr_shared.cloud_media_invoke import partner_rows
    rows = partner_rows()
    assert rows, "partner_nodes.yaml produced no rows"
    offenders = [k for k in rows if not str(k).startswith("cloud_")]
    assert not offenders, (
        "partner row key(s) %s do not start with 'cloud_', so an engine wired "
        "to them is invisible to the cloud floor gate" % offenders)


def test_every_engine_wired_to_a_partner_row_is_gated():
    """An engine that can reach a PROVIDER must be floorable, whatever it is called.

    This is the authoritative check: ``partner_rows()`` is the real list of
    things that spend money, so any engine whose ``node_key`` is in it must be
    recognized by its funnel's gate regardless of what the engine id looks like.
    """
    from nodes._otr_shared.cloud_media_invoke import partner_rows
    rows = set(partner_rows())
    checked = 0
    for engine_id in VIDEO_IDS:
        node_key = str(getattr(vreg.get_engine(engine_id), "node_key", "") or "")
        if node_key in rows:
            checked += 1
            assert rd._is_cloud_video_engine(engine_id), (
                "%s is wired to partner row %r -- it spends real money -- but "
                "the video floor does not recognize it" % (engine_id, node_key))
    for engine_id in IMAGE_IDS:
        node_key = str(getattr(ireg.get_engine(engine_id), "node_key", "") or "")
        if node_key in rows:
            checked += 1
            assert disp._is_cloud_image_engine(engine_id), (
                "%s is wired to partner row %r but the still floor does not "
                "recognize it" % (engine_id, node_key))
    assert checked, "no engine is wired to a partner row -- test is vacuous"


# --------------------------------------------------------------------------- #
# The DIRECT BYO lanes, which never touch the partner boundary
# --------------------------------------------------------------------------- #
def test_google_direct_api_failures_are_floorable_by_their_real_error_type():
    """THE FALSE CONFIDENCE THIS FILE NEARLY SHIPPED WITH.

    Every test above injects a ``CloudMediaError`` and proves the floor reads
    it. That is fine for PARTNER engines, whose failures really do arrive
    stamped from ``invoke_partner_node``. It proves nothing about Google:
    Google is a DIRECT BYO API lane that never passes through that boundary
    and raises ``GoogleAPIError`` subclasses instead. So the parametrized
    tests said "cloud_google_* is floorable" while the engine's ACTUAL
    exception carried no code and could not be floored at all -- a whole
    provider that merely looked covered.

    This test uses the real classifier, on the real exception type.
    """
    from nodes._otr_google_api import client as gclient

    def _classified(status, body=None):
        exc = gclient.GoogleAPIError("google says no")
        gclient._attach_evidence(
            exc, status=status,
            response_json=body if body is not None else {
                "error": {"message": "boom"}})
        return exc

    # JOB-SCOPED: about THIS call, so the beat floors and the run continues.
    for status in (500, 502, 503, None):
        exc = _classified(status)
        assert cmb.cloud_job_failure_code(exc) is not None, (
            "Google HTTP %s is not floorable; one bad call would take a whole "
            "paid episode down" % status)

    # RUN-SCOPED: nothing will ever render, so these must NOT floor.
    for status in (401, 403, 429, 400, 404, 422):
        exc = _classified(status)
        assert cmb.cloud_job_failure_code(exc) is None, (
            "Google HTTP %s floors, which would publish an empty show" % status)

    # A safety block arrives in the BODY, often on an ordinary status, so the
    # status alone would misname it.
    blocked = _classified(
        200, {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})
    assert blocked.code is cmb.CloudErrorCode.CONTENT_REFUSED
    assert cmb.is_content_refusal(blocked)
    assert cmb.cloud_job_failure_code(blocked) is not None


def test_stamping_a_google_failure_never_masks_it():
    """Classification must not become its own failure mode."""
    from nodes._otr_google_api import client as gclient
    exc = gclient.GoogleAPIError("original message")

    class _Unserializable:
        def __repr__(self):
            raise ValueError("even repr explodes")

    gclient._attach_evidence(exc, status=500,
                             response_json=_Unserializable(), raw_body=None)
    assert "original message" in str(exc)
    assert exc.failure_kind == "server"

