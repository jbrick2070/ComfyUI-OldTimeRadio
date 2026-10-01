"""A floored cloud beat shows its own scene still -- never a black hole.

THE DEFECT (measured 2026-10-01). The Spanish "the_keel" episode on the
cloud_vidu_q2_pro_fast_720p lane published with about 136 seconds of black:
five Vidu jobs timed out at the 900 s watchdog and one was rejected, each beat
was floored, and a floored beat reached OTR_SilentComposite with no clip -- so
the composite filled it with its black gap segment, because a cloud lane has no
procgen floor video.

These tests drive the REAL render loop (``run_episode``) with stub cloud
engines and a real scene still on disk, and decode the floored beat's frames.
No network, no GPU, no ComfyUI server. ffmpeg is required for the decode and
for the still_pan render itself; without it the frame tests skip.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

from nodes._otr_shared import cloud_media_backend as cmb
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd

_HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")


# --------------------------------------------------------------------------- #
# stub cloud engines
# --------------------------------------------------------------------------- #
class _CloudStubBase:
    family = "abstract"
    roles = ("retired_role_b",)
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    provider_side = True

    def __init__(self):
        self.calls = 0

    def load(self):
        pass

    def unload(self):
        pass

    def assert_usable(self, host_caps=None, profile=None, request_template=None):
        return self.name

    def prepare(self, host_caps=None, profile=None, session_ctx=None):
        return {"engine_id": self.name}

    def canonicalize(self, raw, request, profile=None):
        return {"clip_id": request["shot_id"], "engine_id": self.name,
                "family": self.family, "frame_count": 25, "path": ""}

    def teardown(self, prepared):
        pass


class _CloudFloorOK(_CloudStubBase):
    name = "cloud_floorstub_ok"

    def render_clip(self, request, prepared):
        self.calls += 1
        return {"raw": True}


class _CloudFloorTimeout(_CloudStubBase):
    name = "cloud_floorstub_timeout"

    def render_clip(self, request, prepared):
        self.calls += 1
        raise cmb.CloudMediaError(cmb.CloudErrorCode.TIMEOUT,
                                  "cloud_floorstub exceeded timeout_s=900")


class _CloudFloorRejected(_CloudStubBase):
    name = "cloud_floorstub_rejected"

    def render_clip(self, request, prepared):
        self.calls += 1
        raise cmb.CloudMediaError(cmb.CloudErrorCode.PROVIDER_REJECTED,
                                  "cloud_floorstub: HTTP 400")


@pytest.fixture
def floor_engines():
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    engines = {"ok": _CloudFloorOK(), "timeout": _CloudFloorTimeout(),
               "rejected": _CloudFloorRejected()}
    for eng in engines.values():
        vreg.register(eng)
    try:
        yield engines
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def _bright_still(path):
    """A deterministic, clearly non-black 16:9 scene still."""
    from PIL import Image
    img = Image.new("RGB", (640, 360), (210, 150, 60))
    for x in range(0, 640, 40):           # some structure for the pan to move
        for y in range(360):
            img.putpixel((x, y), (40, 90, 200))
    img.save(str(path))
    return str(path)


def _ledger(tmp_path, engines_by_index, *, with_stills=True):
    shots, images = [], []
    for i, eng in enumerate(engines_by_index):
        bid = "b%03d" % i
        shots.append({
            "shot_id": "shot_%s" % bid, "beat_id": bid,
            "source_line_ids": [bid], "role": "retired_role_b",
            "engine_id": eng, "family": "abstract", "group_id": "g%d" % i,
            "target_frame_count": 25, "degradation_trail": [],
        })
        if with_stills:
            images.append({
                "kind": "scene_beat", "beat_id": bid,
                "path": _bright_still(tmp_path / ("scene_%s.png" % bid)),
            })
    led = rd.build_full_ledger({"video_revision": 1, "fps": 25, "shots": shots})
    led["images"] = {"images": images}
    return led


def _mean_luma(clip_path):
    """Average luma of the clip's middle frame, decoded by ffmpeg (0..255)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip_path, "-vf",
         "select=eq(n\\,12),scale=64:36,format=gray", "-frames:v", "1",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    assert raw, "ffmpeg decoded no frame from %s" % clip_path
    return sum(raw) / float(len(raw))


# --------------------------------------------------------------------------- #
# 1. the floor content
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_a_timed_out_cloud_beat_shows_its_scene_still_on_the_fanout_walk(
        floor_engines, tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "2")
    led = _ledger(tmp_path, [
        "cloud_floorstub_ok", "cloud_floorstub_timeout", "cloud_floorstub_ok"])
    assert rd._should_fanout_cloud_episode(led["video"], set()) is True
    out = rd.run_episode(led)
    shot = out["ledger"]["video"]["shots"][1]
    assert shot["cloud_floor"] == "timeout"
    assert shot["floor_render"] == rd.CLOUD_FLOOR_RENDER_ENGINE == "still_pan"
    clip = out["clips"]["shot_b001"]
    assert clip["engine_id"] == "still_pan"
    assert os.path.isfile(clip["path"])
    assert int(clip["frame_count"]) == 25          # the beat's own frame budget
    # NOT BLACK: the composite's black gap segment is luma 0-16.
    assert _mean_luma(clip["path"]) > 60.0


@needs_ffmpeg
def test_a_rejected_cloud_beat_shows_its_scene_still_on_the_serial_walk(
        floor_engines, tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "1")
    led = _ledger(tmp_path, [
        "cloud_floorstub_rejected", "cloud_floorstub_ok"])
    assert rd._should_fanout_cloud_episode(led["video"], set()) is False
    out = rd.run_episode(led)
    shot = out["ledger"]["video"]["shots"][0]
    assert shot["cloud_floor"] == "provider_rejected"
    assert shot["floor_render"] == "still_pan"
    assert _mean_luma(out["clips"]["shot_b000"]["path"]) > 60.0
    # The floor clip says what made it, so no receipt claims the cloud engine.
    assert out["clips"]["shot_b000"]["receipt"]["status"] == "cloud_floor"


@needs_ffmpeg
def test_a_budget_floor_shows_its_scene_still_too(floor_engines, tmp_path,
                                                 monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "1")

    class _Broke(_CloudStubBase):
        name = "cloud_floorstub_broke"

        def render_clip(self, request, prepared):
            raise cmb.CloudMediaError(cmb.CloudErrorCode.BUDGET, "HTTP 402")

    vreg.register(_Broke())
    led = _ledger(tmp_path, ["cloud_floorstub_broke", "cloud_floorstub_ok"])
    out = rd.run_episode(led)
    shots = out["ledger"]["video"]["shots"]
    # The halt floors the beat behind it as well; both show their stills.
    assert rd._shot_is_budget_floor(shots[0]) and rd._shot_is_budget_floor(shots[1])
    for sid in ("shot_b000", "shot_b001"):
        assert _mean_luma(out["clips"][sid]["path"]) > 60.0


def test_a_floor_with_no_scene_still_keeps_its_black_gap_and_the_episode(
        floor_engines, tmp_path, monkeypatch, caplog):
    """Operator 2026-10-01: a failed clip whose still cannot be drawn still
    gets its black gap -- one bad beat must never abort the episode and throw
    away the clips already paid for (QA on 0385d3b1)."""
    import logging
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "2")
    led = _ledger(tmp_path, ["cloud_floorstub_timeout", "cloud_floorstub_ok"],
                  with_stills=False)
    with caplog.at_level(logging.ERROR):
        out = rd.run_episode(led)
    shots = out["ledger"]["video"]["shots"]
    assert shots[0]["cloud_floor"] == "timeout"
    assert shots[0]["floor_render"] == "none"
    assert "shot_b000" not in out["clips"]        # the composite's black gap
    assert "shot_b001" in out["clips"]            # the paid beat survives
    assert any("FLOOR STILL FAILED shot shot_b000" in r.getMessage()
               for r in caplog.records)


class _LocalFloorStub(_CloudStubBase):
    """A LOCAL engine (not provider-side, no cloud_ prefix)."""
    name = "localfloorstub_render"
    provider_side = False

    def render_clip(self, request, prepared):
        self.calls += 1
        return {"raw": True}


def test_a_spend_cap_halt_floors_cloud_beats_only(floor_engines, tmp_path,
                                                 monkeypatch):
    """Operator 2026-10-01: "cloud budget should only be counted against cloud
    beats". After a 402 halt the next cloud beat is floored, but a LOCAL beat
    costs nothing and renders normally."""
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "1")

    class _Broke(_CloudStubBase):
        name = "cloud_floorstub_broke2"

        def render_clip(self, request, prepared):
            raise cmb.CloudMediaError(cmb.CloudErrorCode.BUDGET, "HTTP 402")

    local = _LocalFloorStub()
    vreg.register(_Broke())
    vreg.register(local)
    led = _ledger(tmp_path, ["cloud_floorstub_broke2", "localfloorstub_render",
                             "cloud_floorstub_ok"])
    out = rd.run_episode(led)
    shots = out["ledger"]["video"]["shots"]
    assert rd._shot_is_budget_floor(shots[0])
    assert not rd._shot_is_budget_floor(shots[1])   # local beat not floored
    assert local.calls == 1                          # and it actually rendered
    assert rd._shot_is_budget_floor(shots[2])       # later cloud beat floored


# --------------------------------------------------------------------------- #
# 2. what is and is not retried
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_the_driver_never_resubmits_a_timeout_or_a_rejection(
        floor_engines, tmp_path, monkeypatch):
    """No recovery key exists, so a resubmitted timeout can bill twice.

    ``invoke_partner_node`` calls ``session.submit(rid, None)`` -- the provider
    job id is never surfaced -- and the watchdog cancels only the local poll.
    A rejection is a verdict, and resubmitting it verbatim buys the same one.
    """
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "1")
    led = _ledger(tmp_path, ["cloud_floorstub_timeout",
                             "cloud_floorstub_rejected"])
    rd.run_episode(led)
    assert floor_engines["timeout"].calls == 1
    assert floor_engines["rejected"].calls == 1


def test_a_failed_download_is_retried_once_and_lands(tmp_path, monkeypatch):
    """The ONE retry that cannot pay twice: the job is done, the URL is in hand."""
    from nodes._otr_shared import cloud_media_invoke as cmi
    calls = {"n": 0}

    class _Body:
        def __init__(self):
            self._chunks = [b"video-bytes", b""]

        def read(self, _n):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("reset by peer")
        return _Body()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    dest = tmp_path / "clip.mp4"
    cmi._stream_to_temp("https://cdn.example.invalid/clip.mp4", dest)
    assert calls["n"] == 2
    assert dest.read_bytes() == b"video-bytes"


def test_a_download_that_fails_twice_is_billed_not_released(tmp_path,
                                                           monkeypatch):
    """The provider finished and charged before the download began."""
    from nodes._otr_shared import cloud_media_invoke as cmi
    import urllib.request

    def _urlopen(req, timeout=None):
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    with pytest.raises(cmb.CloudMediaError) as ei:
        cmi._stream_to_temp("https://cdn.example.invalid/clip.mp4",
                            tmp_path / "clip.mp4")
    err = ei.value
    assert err.code is cmb.CloudErrorCode.RETRYABLE_TRANSPORT
    assert getattr(err, "provider_charged", False) is True
    assert not (tmp_path / "clip.mp4").exists()

    class _Session:
        def __init__(self):
            self.billed, self.released, self.rows = [], [], []

        def bill(self, rid):
            self.billed.append(rid)

        def release(self, rid):
            self.released.append(rid)

        def ledger_append(self, row):
            self.rows.append(row)

    sess = _Session()
    cmi._settle_failure(sess, "rid1", err, "cloud_vidu_q2_i2v")
    assert sess.billed == ["rid1"] and sess.released == []
    assert sess.rows[0]["settlement"] == "billed_estimate"
    # An ordinary transport failure (never reached the provider) still releases.
    sess2 = _Session()
    cmi._settle_failure(sess2, "rid2", cmb.CloudMediaError(
        cmb.CloudErrorCode.RETRYABLE_TRANSPORT, "connect refused"),
        "cloud_vidu_q2_i2v")
    assert sess2.released == ["rid2"] and sess2.billed == []


# --------------------------------------------------------------------------- #
# 3. reporting and accounting
# --------------------------------------------------------------------------- #
def _floor_manifest(tmp_path):
    clip = tmp_path / "floor.mp4"
    clip.write_bytes(b"x")
    ok = tmp_path / "ok.mp4"
    ok.write_bytes(b"x")
    led = rd.build_full_ledger({"video_revision": 1, "fps": 25, "shots": [
        {"shot_id": "shot_b000", "role": "character_video",
         "engine_id": "cloud_vidu_q2_pro_fast_720p", "target_frame_count": 25},
        dict(rd._stamp_cloud_floor_shot(
            {"shot_id": "shot_b001", "role": "character_video",
             "engine_id": "cloud_vidu_q2_pro_fast_720p",
             "target_frame_count": 25}, "timeout"),
            floor_render="still_pan"),
    ]})
    result = {"ledger": led, "clips": {
        "shot_b000": {"engine_id": "cloud_vidu_q2_pro_fast_720p",
                      "frame_count": 25, "path": str(ok)},
        "shot_b001": {"engine_id": "still_pan", "frame_count": 25,
                      "path": str(clip), "extension_mode": "none"},
    }}
    return rd.build_clip_manifest(result, episode_id="ep_floor")


def test_the_manifest_carries_the_floor_and_the_planned_engine(tmp_path):
    m = _floor_manifest(tmp_path)
    row = m["clips"][1]
    assert row["exists"] is True
    assert row["status"] == "sanctioned_gap"
    assert row["floor_render"] == "still_pan"
    assert row["planned_engine_id"] == "cloud_vidu_q2_pro_fast_720p"
    assert m["floor_clip_count"] == 1
    assert "floor_render" not in m["clips"][0]


def test_a_still_floor_is_degraded_and_never_billed_to_an_engine(tmp_path):
    from nodes import otr_video_render_batch as vrb
    m = _floor_manifest(tmp_path)
    payload = vrb._build_render_engines_payload(m, None)
    assert payload["sanctioned_gap_shot_ids"] == ["shot_b001"]
    reason = payload["sanctioned_gap_reasons"][0]
    assert reason["reason"] == "timeout"
    assert reason["planned_engine"] == "cloud_vidu_q2_pro_fast_720p"
    assert reason["floor_render"] == "still_pan"
    # The credits receipt does not claim the cloud engine (or still_pan)
    # rendered the floored beat as motion.
    delivered = [r["shot_id"] for r in payload["per_clip"]]
    assert delivered == ["shot_b000"]
    counts = vrb._beat_accounting(m["clips"])
    assert counts == {"delivered": 1, "sanctioned": 1, "unaccounted": 0}


def test_acceptance_does_not_grade_a_floor_as_a_multiclip_render():
    from nodes._otr_video_engines import acceptance as acc
    led = {"video": {"shots": [{
        "shot_id": "shot_b001", "target_frame_count": 50,
        "coverage_plan": {"segments": [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 0, "trim_tail": 0}],
            "target_visible_frames": 50, "join_mode": "jump"}}]}}
    manifest = {"clips": [{"shot_id": "shot_b001", "exists": True,
                           "status": "sanctioned_gap", "engine_id": "still_pan",
                           "extension_mode": "none", "frame_count": 50}]}
    assert acc.grade_multiclip_honesty(led, manifest) == []


# --------------------------------------------------------------------------- #
# 4. the wiring, at the real call sites
# --------------------------------------------------------------------------- #
def test_every_cloud_floor_site_in_run_episode_renders_the_still():
    """A helper nothing calls is this repo's most repeated defect."""
    src = pathlib.Path(rd.__file__).read_text(encoding="utf-8")
    start = src.index("def run_episode(")
    end = src.index("\ndef _log_if_legacy_render_plan_present", start)
    body = src[start:end]
    stamps = body.count("_stamp_cloud_floor_shot(") + body.count(
        "_stamp_budget_floor_shot(")
    committed = body.count("_commit_still_floor(")
    # Every stamped floor (job, predecessor, budget; both walks) is committed
    # through the still floor -- one definition plus one call per stamp.
    assert stamps >= 7
    assert committed == stamps + 1


def test_the_download_retry_is_on_the_materialize_path():
    from nodes._otr_shared import cloud_media_invoke as cmi
    src = pathlib.Path(cmi.__file__).read_text(encoding="utf-8")
    mat = src[src.index("def _materialize("):src.index("def _normalize_result(")]
    assert "_stream_to_temp(value, dest)" in mat
    assert "_DOWNLOAD_ATTEMPTS" in src
