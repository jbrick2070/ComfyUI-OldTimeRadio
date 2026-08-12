"""THE RECEIPT HAS TO SURVIVE THE PATH THAT ACTUALLY RENDERS (lane 9).

`7afe40e5` landed the L4 receipt fields on this adapter "because lane 9 cannot
measure without them", and it landed them on ONE of the two render paths. The
one it landed on is the one this box does not run: the installed unet is
`ltx-2.3-22b-dev-Q3_K_M.gguf`, `_detect_recipe` routes a dev-family unet to
`hq_two_stage`, and `_render_clip_hq` had no `VramPeakProbe` and never called
`_clip_telemetry`.

That is not a missing number. `render_driver` reads::

    clip_peak = clip.get("vram_peak_mb")
    return ..., (clip_peak or _mc.vram_used_mb())

so the absence was silently filled in by an INSTANTANEOUS post-render sample,
shaped exactly like a peak and readable as one. Lesson L10 PROVENANCE: a sample
is a LOWER BOUND, it may never seed a cost row, and a cost row built on a lower
bound under-predicts -- which admits renders that then OOM. Lane 9's entire
deliverable is a measured low/high marker, so the lane could not have produced
an honest one.

And the SECOND seam, which is `eng_ltx_8gb`'s bug repeated inside the file that
quotes it: `_clip_from_raw` copied `native_frame_count` and `extension_mode` and
dropped the other five, under a comment stating that a field this method does
not copy is a field `render_beat_coverage` never sees. So even the single-pass
path's correctly-probed peak died one method short of the ledger.

These tests are BEHAVIOURAL on both counts. The peak assertion pins the value to
a probe SPY's sentinel rather than to "is not None", because `vram_used_mb()`
would satisfy "is not None" on the GPU box -- which is precisely the substitution
under test. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import eng_ltx_video as mod  # noqa: E402
from nodes._otr_video_engines import motion_common as mc  # noqa: E402
from nodes._otr_video_engines import wrapper_bridge as wb  # noqa: E402
from nodes._otr_video_engines.eng_ltx_video import (  # noqa: E402
    RECIPE_HQ_TWO_STAGE,
    LtxVideoEngine,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
DEV_UNET = "ltx-2.3-22b-dev-Q3_K_M.gguf"

#: The value the spy probe reports. Deliberately not a plausible VRAM reading
#: from this box, so a test that passes because a real sampler happened to
#: return something is impossible.
PROBE_SENTINEL = 424242

#: Every key the ledger reads off a clip. `_clip_summary` (render_driver) names
#: exactly these, so the list is the consumer's, not this test's opinion.
RECEIPT_KEYS = ("vram_peak_mb", "recipe", "quant", "render_canvas",
                "native_frame_count", "extension_mode")


class _ProbeSpy:
    """Stands in for VramPeakProbe and records that the window was opened.

    `started`/`stopped` prove the probe SPANS the render rather than being
    constructed and dropped -- a probe stopped before `run_graph` would report a
    pre-render reading and still pass a value-only assertion.
    """

    instances = []

    def __init__(self, interval_s=1.0):
        self.interval_s = interval_s
        self.started = False
        self.stopped = False
        self.order = []
        _ProbeSpy.instances.append(self)

    def start(self):
        self.started = True
        self.order.append("start")
        return self

    def stop(self):
        self.stopped = True
        self.order.append("stop")
        return PROBE_SENTINEL


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """An engine whose graph executes without ComfyUI, a GPU or real weights.

    Only the wrapper seam is faked. `images_to_uint8`, the ffmpeg encode and the
    ffprobe silence proof all run for real, so the raw dict under test is the
    dict a live render would build.
    """
    np = pytest.importorskip("numpy")
    _ProbeSpy.instances.clear()
    monkeypatch.setenv("OTR_LTX_VIDEO_UNET", DEV_UNET)
    monkeypatch.delenv("OTR_LTX_VIDEO_RECIPE", raising=False)
    monkeypatch.delenv("OTR_LTX_MIN_DECODE_FRAMES", raising=False)
    monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
    monkeypatch.delenv(mod.PREQUALIFICATION_ENV, raising=False)

    still = tmp_path / "scene_still.png"
    still.write_bytes(b"not-really-a-png-the-stager-is-faked")

    frames = {"n": 0}

    def _run_graph(graph, classes, **kwargs):
        # The decoded IMAGE batch is sized from the length the engine actually
        # wired into the graph, so a frame count is an OBSERVATION rather than a
        # fixture constant (the `_ltx8_fakes` rule).
        length = int(graph["emptylatent"]["inputs"]["length"]) \
            if "emptylatent" in graph else int(graph["latent"]["inputs"]["length"])
        frames["n"] = length
        images = np.zeros((length, 16, 16, 3), dtype="float32")
        return {"decode": (images,), LtxVideoEngine._TERMINAL: (images,),
                "lora": (None,)}

    monkeypatch.setattr(wb, "resolve_graph_classes", lambda cands: {})
    monkeypatch.setattr(wb, "run_graph", _run_graph)
    monkeypatch.setattr(wb, "stage_into_comfy_input",
                        lambda p: os.path.basename(p))
    monkeypatch.setattr(mc, "VramPeakProbe", _ProbeSpy)

    eng = LtxVideoEngine()
    request = {"shot_id": "s1", "text_prompt": "a rain-slicked street at night",
               "canvas": {"w": 1024, "h": 576},
               "asset_refs": {"still": str(still)},
               "timing": {"target_frame_count": 169},
               "seed_bundle": {"request_seed": 7}}
    return eng, request, frames


def _render_hq(rig):
    eng, request, _frames = rig
    plan = eng._build_render_request(request)
    raw = eng._render_clip_hq(request, {}, plan)
    return eng, request, raw


# -- the peak, on the path that renders -------------------------------------


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_the_HQ_path_reports_the_PROBE_maximum_not_a_post_render_sample(rig):
    """The defect, stated as the assertion that would have caught it.

    Pinned to the SENTINEL: `assert raw["vram_peak_mb"] is not None` would pass
    on the GPU box against the very fallback sample this exists to forbid.
    """
    _eng, _request, raw = _render_hq(rig)
    try:
        assert raw["vram_peak_mb"] == PROBE_SENTINEL, (
            "the hq_two_stage path must report the VramPeakProbe MAXIMUM. A "
            "missing peak is filled in by render_driver from an instantaneous "
            "vram_used_mb() sample, which is a LOWER BOUND and may never seed a "
            "cost row (L10 PROVENANCE)")
        assert len(_ProbeSpy.instances) == 1
        probe = _ProbeSpy.instances[0]
        assert probe.started and probe.stopped
        assert probe.order == ["start", "stop"]
    finally:
        if raw.get("out_path") and os.path.exists(raw["out_path"]):
            os.unlink(raw["out_path"])


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_the_HQ_path_stamps_the_L4_TELEMETRY(rig):
    """recipe / quant / use_lora / render_canvas -- the four fields the lane's
    receipt work added and routed only to the sibling."""
    _eng, _request, raw = _render_hq(rig)
    try:
        assert raw["recipe"] == RECIPE_HQ_TWO_STAGE
        assert raw["quant"] == "Q3_K_M"
        assert raw["use_lora"] is True
        # The DECLARED canvas, and derived from the declaration rather than
        # written as a literal, so the number can move without this test lying
        # (lesson L10).
        w, h = LtxVideoEngine.render_canvas
        assert raw["render_canvas"] == "%dx%d" % (w, h)
    finally:
        if raw.get("out_path") and os.path.exists(raw["out_path"]):
            os.unlink(raw["out_path"])


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_a_MEASUREMENT_clip_off_the_HQ_path_NAMES_its_departures(rig,
                                                                 monkeypatch):
    """The consent act's safety property, on the path that can use it.

    `OTR_LTX_VIDEO_PREQUALIFICATION` exists so lane 9 can render rungs below the
    169 decode floor. Its guarantee is that the resulting clips are
    SELF-IDENTIFYING -- a sweep artifact must never be mistakable for a
    production one. That guarantee lived entirely in `_clip_telemetry`, which
    the measuring path never called, so every clip the consent act produced
    would have been stamped exactly like a production clip.
    """
    monkeypatch.setenv(mod.PREQUALIFICATION_ENV, "1")
    # A DEPARTURE FROM THE SHIPPED DEFAULT, derived rather than written.
    # A first draft pinned "9" as the swept value; lane 9 then MOVED the default
    # floor to 9, so the test kept passing while departing from nothing. Derive
    # the value from the constant it must differ from, and the test cannot go
    # hollow when the default moves again.
    departed_floor = mod._LTX_DECODE_FLOOR_DEFAULT + 88   # 97 today
    monkeypatch.setenv("OTR_LTX_MIN_DECODE_FRAMES", str(departed_floor))
    eng, request, _frames = rig
    # THE SHAPE LEG 1 RUNS: a SHORT ask against a MOVED floor, so the floor is
    # the thing deciding the length. Asking for a length above the floor departs
    # from nothing -- the floor only ever raises a SHORT ask -- and an earlier
    # draft did exactly that, proving the departure receipt by asserting a length
    # the departure had not changed.
    request = dict(request, timing={"target_frame_count": 17})
    plan = eng._build_render_request(request)
    raw = eng._render_clip_hq(request, {}, plan)
    try:
        assert raw["recipe"].startswith(RECIPE_HQ_TWO_STAGE)
        assert mod._RD.PREQUALIFICATION_SUFFIX in raw["recipe"], raw["recipe"]
        assert "min_decode_frames" in raw["recipe"], raw["recipe"]
        # And the departure was REAL: the moved floor, not the ask, decided the
        # length. A receipt naming a departure that did not happen would be its
        # own defect.
        assert raw["frame_count"] == departed_floor
        assert raw["native_frame_count"] == departed_floor
    finally:
        if raw.get("out_path") and os.path.exists(raw["out_path"]):
            os.unlink(raw["out_path"])


# -- the canonicalize seam ---------------------------------------------------


@pytest.mark.parametrize("key", RECEIPT_KEYS)
def test_the_canonicalize_seam_carries_EVERY_receipt_field(key):
    """`_clip_from_raw` is the second producer seam and it dropped five of six.

    Parameterised so a future key that is added to the raw returns and forgotten
    here fails BY NAME rather than as one line of a compound assertion.
    """
    eng = LtxVideoEngine()
    raw = {"out_path": "x.mp4", "frame_count": 169,
           "vram_peak_mb": PROBE_SENTINEL, "recipe": RECIPE_HQ_TWO_STAGE,
           "quant": "Q3_K_M", "use_lora": True, "render_canvas": "1024x576",
           "native_frame_count": 169, "extension_mode": "none"}
    clip = eng._clip_from_raw(raw, {"shot_id": "s1"})
    assert clip[key] == raw[key], (
        "%s is produced by the engine and dropped by canonicalize, so "
        "render_beat_coverage never sees it" % key)


def test_the_seam_carries_use_lora_too():
    """`use_lora` gets its OWN test rather than a trailing line on the
    parameterised one.

    A first draft asserted it at the end of every parameterised case, which made
    all six cases fail together pre-fix -- including `native_frame_count` and
    `extension_mode`, which were never dropped. A case that fails for a reason
    other than its own parameter reports the wrong defect.

    It is not in RECEIPT_KEYS because `_clip_summary` does not carry it; the
    identity rollup in `otr_video_render_batch` groups on it, so the seam still
    must not drop it.
    """
    eng = LtxVideoEngine()
    clip = eng._clip_from_raw({"out_path": "x.mp4", "use_lora": True},
                              {"shot_id": "s1"})
    assert clip["use_lora"] is True


def test_the_seam_does_not_INVENT_a_peak_for_a_raw_that_has_none():
    """The other direction, and the one that matters more.

    A raw with no peak must canonicalize to None -- never to a default, a zero,
    or a sample. `_clip_summary` documents the rule ("a missing peak must stay
    NULL here rather than be filled in from a watcher's sample") and this is the
    seam upstream of it.
    """
    eng = LtxVideoEngine()
    clip = eng._clip_from_raw({"out_path": "x.mp4", "frame_count": 9},
                              {"shot_id": "s1"})
    for key in RECEIPT_KEYS:
        assert clip[key] is None, key
