"""WIRE-W3b -- wan_ti2v: the beat session, and the PING-PONG NARROWING.

The session half is W3a's, mirrored: one 5B UNET load per beat instead of one
per segment, counted on the LOADER rather than on ``prepare()`` because
declaring a ``session_identity()`` alone silences ``BeatSession``'s refusal
while the segment graph keeps reloading.

THE HALF THAT IS THIS ADAPTER'S OWN is the ping-pong. ``wan_ti2v`` renders a
VRAM-bounded SHORT clip and mirror-extends it up to the beat target -- on
purpose, and it is the shipped 8 GB tier contract (``PBUG-20260723-02``). That
is fine for a single-clip beat and forbidden for a coverage-planned segment:
the pad wears the right frame count, so a render that did not happen passes
``render_driver``'s ``got != segment.render_frames`` gate wearing the number of
one that did, and on a ``strict_first_frame`` lane the next chained segment
would begin on a mirrored frame.

So this file pins BOTH sides: the tier still pads, the plan never does.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import wan_shared as _WS  # noqa: F401
from nodes._otr_video_engines.eng_wan_ti2v import (RECIPE_WAN_TI2V,
                                                   WanTi2vEngine)


def _mk_node(fn):
    return type("FakeNode", (), {"FUNCTION": "f", "f": fn})


class _FakeModel:
    def detach(self, unpatch_all=False):
        return None


class _FakeProbe:
    PEAK_MB = 4242

    def start(self):
        return self

    def stop(self):
        return self.PEAK_MB


def _counting_fakes(np, counts, decoded=None):
    """The CORE Wan 2.2 TI2V node classes, faked, with every LOADER counting.

    The decoder HONOURS the length the latent node was asked for, because
    ``render_clip`` now refuses a decode that returns a different count -- a
    fake that always emitted a fixed number would have been exercising the pad
    on every render rather than the render. ``decoded`` overrides that with a
    deliberate under-delivery, which is the one thing the refusal is for.
    """
    asked = {"length": 17}

    def _count(key, out):
        def _f(self, **_kw):
            counts[key] = counts.get(key, 0) + 1
            return out() if callable(out) else out
        return _mk_node(_f)

    def _latent(self, **k):
        asked["length"] = max(1, int(k.get("length") or asked["length"]))
        return (("latent",),)

    def _decode(self, **_k):
        n = asked["length"] if decoded is None else int(decoded)
        return (np.zeros((n, 8, 8, 3), dtype="float32"),)

    return {
        "unet": _count("unet", lambda: (_FakeModel(),)),
        "clip": _count("clip", lambda: (object(),)),
        "vae": _count("vae", lambda: (object(),)),
        "modelsampling": _mk_node(lambda self, **k: (_FakeModel(),)),
        "pos": _mk_node(lambda self, **k: (("c",),)),
        "neg": _mk_node(lambda self, **k: (("c",),)),
        "loadimage": _mk_node(lambda self, **k: (object(), object())),
        "latent": _mk_node(_latent),
        "ksampler": _count("ksampler", lambda: (("latent",),)),
        "vaedecode": _mk_node(_decode),
    }


def _no_ffmpeg(monkeypatch, mod, tmp_path):
    from nodes._otr_video_engines import wrapper_bridge as _wb
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"not-a-real-mp4")
    monkeypatch.setattr(_wb, "encode_frames_to_silent_mp4",
                        lambda frames, path, fps: (str(out), len(frames)))
    monkeypatch.setattr(mod, "ffprobe_clip_fields", lambda p: {})
    monkeypatch.setattr(mod, "validate_silent_clip_contract",
                        lambda fields, fps: None)
    monkeypatch.setattr(mod._MC, "VramPeakProbe", lambda **kw: _FakeProbe())
    monkeypatch.setattr(_wb, "comfy_input_dir", lambda: str(tmp_path))


def _init_png(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "p.png"
    Image.new("RGB", (832, 480), (10, 20, 30)).save(src)
    return str(src)


def _request(tmp_path, seed=3, frames=33):
    return {"shot_id": "s1", "asset_refs": {"init_image": _init_png(tmp_path)},
            "text_prompt": "subtle motion", "init_w": 832, "init_h": 480,
            "canvas": {"w": 832, "h": 480, "aspect_policy": "pad"},
            "timing": {"target_frame_count": frames},
            "seed_bundle": {"request_seed": seed}}


def _ctx(segments):
    """The session_ctx BeatSession builds -- multi_clip is derived from the
    STAMPED plan's segment count, which is why it is the honest discriminator:
    a segment's request looks exactly like a single-clip beat's."""
    return {"beat_id": "b1", "segment_count": segments,
            "multi_clip": segments > 1}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_wan_ti2v as mod
    counts = {}
    eng = WanTi2vEngine()
    eng._classes = _counting_fakes(np, counts)
    monkeypatch.setattr(eng, "load", lambda: None)      # classes already set
    _no_ffmpeg(monkeypatch, mod, tmp_path)
    return eng, counts, tmp_path


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------

def test_wan_ti2v_can_name_its_own_handles():
    """Until this existed, BeatSession refused EVERY multi-segment wan_ti2v
    beat -- one of the six NO_RENDER engines on the 2026-07-28 campaign."""
    from nodes._otr_video_engines import beat_session as bs
    identity = bs.session_identity(WanTi2vEngine())
    assert identity is not None
    assert isinstance(identity, tuple) and identity[0] == "wan_ti2v"
    assert RECIPE_WAN_TI2V in identity[1]


def test_the_identity_carries_BOTH_loader_modes(monkeypatch):
    """Unlike wan_i2v this adapter really has two, and their node schemas
    DIFFER (CLIPLoaderGGUF takes no ``device``). A beat that switched either
    class mid-way would be reusing a handle its own graph no longer knows how
    to have produced -- so BOTH have to move the identity, not just the UNET's.
    """
    eng = WanTi2vEngine()
    monkeypatch.setenv("OTR_WAN_TI2V_LOADER", "safetensors")
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_LOADER", "safetensors")
    base = eng.session_identity()
    monkeypatch.setenv("OTR_WAN_TI2V_LOADER", "gguf")
    assert eng.session_identity() != base, "the UNET mode must move it"
    monkeypatch.setenv("OTR_WAN_TI2V_LOADER", "safetensors")
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_LOADER", "gguf")
    assert eng.session_identity() != base, "the CLIP mode must move it too"


def test_the_identity_moves_when_a_WEIGHT_moves(monkeypatch):
    """Size+mtime, not a content hash: it is re-read before every segment, and
    hashing the weights per segment would cost more than the render it
    protects. A rebuilt weight still moves it."""
    eng = WanTi2vEngine()
    monkeypatch.setattr(eng, "_wan_file_receipt",
                        lambda p: ("unet.gguf", 111, 222))
    before = eng.session_identity()
    monkeypatch.setattr(eng, "_wan_file_receipt",
                        lambda p: ("unet.gguf", 111, 999))       # rebuilt
    assert eng.session_identity() != before


# ---------------------------------------------------------------------------
# THE ACCEPTANCE: loader invocations, not prepare() calls
# ---------------------------------------------------------------------------

def test_a_THREE_SEGMENT_beat_loads_the_UNET_EXACTLY_ONCE(wired):
    """THE test the session half exists to pass. Three segments, one load."""
    eng, counts, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(3))
    try:
        assert counts["unet"] == 1, "prepare() must load the UNET exactly once"
        for seg in range(3):
            eng.render_clip(_request(tmp_path, seed=seg), prepared)
        assert counts["unet"] == 1, (
            "the UNET loaded %d time(s) across a 3-segment beat -- the hoist "
            "is not reaching the segment graph" % counts["unet"])
    finally:
        eng.teardown(prepared)


def test_the_AUXILIARIES_may_reload_and_that_is_the_narrowed_contract(wired):
    """``render_clip`` runs with ``free_after_use=True`` precisely so the umt5
    text encode frees before the 5B UNET and the 2.2 video VAE decode. Hoisting
    the CLIP would pin it resident for the whole beat and undo the one
    mitigation this tier has, so the reload is BY DESIGN, not a shortfall."""
    eng, counts, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(3))
    try:
        for seg in range(3):
            eng.render_clip(_request(tmp_path, seed=seg), prepared)
        assert counts["unet"] == 1
        assert counts["clip"] == 3, "the CLIP is per-segment BY DESIGN"
        assert counts["vae"] == 3, "the VAE is per-segment BY DESIGN"
    finally:
        eng.teardown(prepared)


def test_a_caller_that_PREPARED_NOTHING_renders_exactly_as_before(wired):
    """The single-clip path is what production runs today, and every sibling
    test hand-builds ``prepared={"patchers": []}``. It must still get its own
    loader nodes."""
    eng, counts, tmp_path = wired
    eng.render_clip(_request(tmp_path), {"patchers": []})
    assert counts["unet"] == 1, "the un-prepared path must load its own UNET"
    assert counts["clip"] == 1 and counts["vae"] == 1


def test_the_graph_OMITS_a_hoisted_node_and_KEEPS_every_wire_that_reads_it():
    """``run_graph`` REFUSES a graph that also defines an id the caller
    supplied -- two parties claiming to produce one output -- so omission is a
    contract, not an optimisation. The WIRES must survive."""
    eng = WanTi2vEngine()
    req = {"text_prompt": "x", "canvas": {"w": 832, "h": 480}}
    plain = eng._build_graph(req, "i.png", {"seed": 1}, 33, 832, 480)
    hoisted = eng._build_graph(req, "i.png", {"seed": 1}, 33, 832, 480,
                               external_results={"unet": (object(),)})
    assert "unet" in plain
    assert "unet" not in hoisted
    assert set(plain) - set(hoisted) == {"unet"}
    assert hoisted["modelsampling"]["inputs"]["model"].src == "unet"


def test_teardown_DROPS_the_hoisted_handles_before_the_base_runs(wired):
    """The base teardown detaches patchers, unloads, releases the lease and
    then WAITS for machine-wide VRAM to settle -- with this dict still pinning
    the tensors it is reclaiming, the stability wait watches its own
    referent."""
    eng, _counts, _tmp = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    assert "external_results" in prepared
    eng.teardown(prepared)
    assert "external_results" not in prepared


def test_a_failure_DURING_the_hoist_does_not_strand_the_lease(monkeypatch,
                                                              tmp_path):
    """A raise after ``super().prepare()`` has taken the cross-process GPU
    lease strands it for the life of the server -- the C-3 lesson."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_wan_ti2v as mod
    eng = WanTi2vEngine()
    counts = {}
    fakes = _counting_fakes(np, counts)

    def _boom(self, **_kw):
        raise RuntimeError("the UNET did not load")

    fakes["unet"] = _mk_node(_boom)
    eng._classes = fakes
    monkeypatch.setattr(eng, "load", lambda: None)
    _no_ffmpeg(monkeypatch, mod, tmp_path)

    torn = {"n": 0}
    real_teardown = eng.teardown

    def _spy(prepared):
        torn["n"] += 1
        return real_teardown(prepared)

    monkeypatch.setattr(eng, "teardown", _spy)
    with pytest.raises(Exception):
        eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    assert torn["n"] == 1, "a failed hoist must tear down what it took"


# ---------------------------------------------------------------------------
# THE PING-PONG: still there for the tier, forbidden inside a plan
# ---------------------------------------------------------------------------

def test_a_SINGLE_CLIP_beat_STILL_PING_PONGS(wired, monkeypatch):
    """THE HALF THAT MUST NOT MOVE. The 8 GB tier pins a 17-frame render
    ceiling and fills the rest of the beat with a mirror extension -- that is
    ``PBUG-20260723-02``, shipped and load-bearing. If this goes red the chunk
    has ripped the low-VRAM tier instead of narrowing it."""
    eng, _counts, tmp_path = wired
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "17")
    raw = eng.render_clip(_request(tmp_path, frames=33), {"patchers": []})
    assert raw["native_frame_count"] == 17, "the tier renders SHORT on purpose"
    assert raw["extension_mode"] == "ping_pong"
    assert raw["frame_count"] == 33, "and the beat is still FILLED"


def test_a_COVERAGE_PLANNED_segment_RENDERS_ITS_WHOLE_LENGTH(wired):
    """The narrowing. A planned segment's length came off ``partition_beat``
    and the rest of the beat was planned around it, so it is not a preference
    the VRAM predictor may trade down -- every frame is rendered."""
    eng, _counts, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(3))
    try:
        raw = eng.render_clip(_request(tmp_path, frames=33), prepared)
    finally:
        eng.teardown(prepared)
    assert raw["native_frame_count"] == 33
    assert raw["extension_mode"] == "none", "a planned segment never pads"
    assert raw["frame_count"] == 33


def test_the_SAME_REQUEST_pads_single_clip_and_does_not_inside_a_plan(
        wired, monkeypatch):
    """The two paths above, driven by ONE request, so the discriminator itself
    is what is under test rather than two differently-shaped inputs. This is
    the whole point of reading ``session_ctx`` instead of the request: a
    coverage-planned segment's request is shaped exactly like a single-clip
    beat's, so nothing else here COULD tell them apart."""
    eng, _counts, tmp_path = wired
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "33")
    req = _request(tmp_path, frames=33)

    solo = eng.render_clip(dict(req), {"patchers": []})

    monkeypatch.delenv("OTR_WAN_TI2V_MAX_FRAMES", raising=False)
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    try:
        planned = eng.render_clip(dict(req), prepared)
    finally:
        eng.teardown(prepared)

    assert solo["frame_count"] == planned["frame_count"] == 33
    assert planned["extension_mode"] == "none"
    assert planned["native_frame_count"] == 33


def test_a_PLANNED_segment_REFUSES_a_tier_ceiling_below_its_length(
        wired, monkeypatch):
    """WAN is deliberately OUT of ``frame_contract.PLANNING_CAP_ENGINES``: its
    ceiling is a RENDER cap the single-clip path fills around, and narrowing
    the planner by it would turn every WAN beat into a pile of 17-frame renders
    and rewrite the tier PBUG-20260723-02 fixed. The price is that a ceiling
    and a plan CAN contradict each other -- and when they do, the honest answer
    is to say so by name, not to render short and mirror the difference.

    IT REFUSES BEFORE THE SEGMENT GRAPH RUNS (the eng_ltx_8gb B4 ordering): the
    condition is knowable from pure numbers, so no init image is staged and no
    GPU work is paid for.
    """
    eng, counts, tmp_path = wired
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "17")
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    try:
        with pytest.raises(Exception) as caught:
            eng.render_clip(_request(tmp_path, frames=33), prepared)
    finally:
        eng.teardown(prepared)
    msg = str(caught.value)
    assert "17" in msg and "33" in msg
    assert "NO FALLBACK" in msg
    assert counts.get("ksampler", 0) == 0, (
        "the refusal must arrive before the segment graph runs")


def test_a_PLANNED_length_OFF_THE_LADDER_is_refused(wired):
    """A planned length that is not on this adapter's own declared ladder means
    the stamped plan and the frame_contract disagree. Snapping to a nearby rung
    would make the assembled beat a different length than the plan says, which
    is the class of silent fallback this build removes."""
    eng, _counts, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    try:
        with pytest.raises(Exception) as caught:
            eng.render_clip(_request(tmp_path, frames=30), prepared)
    finally:
        eng.teardown(prepared)
    assert "ladder" in str(caught.value)


def test_the_PING_PONG_GUARD_holds_even_if_the_length_decision_is_wrong(
        wired, monkeypatch):
    """Defence in depth, and "unreachable" is a claim about today's callers.

    Force ``_planned_length`` to hand back a length SHORTER than the segment's
    target -- the exact state a future edit to the length decision would
    produce -- and the pad must still refuse rather than mirror its way to the
    right frame count inside a beat claiming real multi-clip coverage."""
    eng, _counts, tmp_path = wired
    monkeypatch.setattr(eng, "_planned_length", lambda target: 17)
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    try:
        with pytest.raises(Exception) as caught:
            eng.render_clip(_request(tmp_path, frames=33), prepared)
    finally:
        eng.teardown(prepared)
    assert "COVERAGE-PLANNED" in str(caught.value)


def test_a_DECODE_THAT_COMES_UP_SHORT_is_refused_rather_than_padded(
        monkeypatch, tmp_path):
    """The pipeline invariant, on the SINGLE-CLIP path where the pad still
    lives. The extension used to absorb an under-delivery for ANY reason, not
    just the deliberate VRAM cap, so a decode that silently returned short was
    indistinguishable from one that did what it was told."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_wan_ti2v as mod
    eng = WanTi2vEngine()
    counts = {}
    eng._classes = _counting_fakes(np, counts, decoded=9)
    monkeypatch.setattr(eng, "load", lambda: None)
    _no_ffmpeg(monkeypatch, mod, tmp_path)
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "17")
    with pytest.raises(Exception) as caught:
        eng.render_clip(_request(tmp_path, frames=33), {"patchers": []})
    msg = str(caught.value)
    assert "17" in msg and "9" in msg and "NO FALLBACK" in msg


# ---------------------------------------------------------------------------
# The receipt the W5 grader will read
# ---------------------------------------------------------------------------

def test_the_receipt_SURVIVES_canonicalisation_into_the_manifest_dict(
        wired, monkeypatch):
    """A field the engine returns and ``_clip_from_raw`` drops is a field the
    grader never sees. Both WAN adapters share that method, which is why the
    passthrough lives there and not in each adapter."""
    eng, _counts, tmp_path = wired
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "17")
    req = _request(tmp_path, frames=33)
    clip = eng.canonicalize(eng.render_clip(req, {"patchers": []}), req, {})
    assert clip["native_frame_count"] == 17
    assert clip["extension_mode"] == "ping_pong"
    assert clip["frame_count"] == 33, (
        "frame_count ALONE cannot answer the question -- a padded clip and a "
        "real one carry the same number, which is what the new fields are for")


def test_BOTH_wan_adapters_ANSWER_the_extension_question():
    """An ABSENT field is indistinguishable from an unanswered one. wan_i2v has
    never extended, but a family where one adapter declares "none" and its
    sibling declares nothing gives the grader no lane it can trust."""
    from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine
    raw = {"out_path": "x.mp4", "frame_count": 33,
           "native_frame_count": 33, "extension_mode": "none"}
    for eng in (WanTi2vEngine(), WanI2VEngine()):
        clip = eng.canonicalize(raw, {"shot_id": "s"}, {})
        assert clip["native_frame_count"] == 33
        assert clip["extension_mode"] == "none"


# ---------------------------------------------------------------------------
# The budget, taken with the memory state every segment sees
# ---------------------------------------------------------------------------

def test_the_HOIST_COST_is_added_back_or_the_predictor_refuses_real_renders(
        monkeypatch):
    """r3's third bullet, and it is not cosmetic -- without it the session half
    BREAKS the lane it was meant to fix.

    The cost model's ``overhead`` term is documented as "the resident model +
    fixed buffers". Hoisting the UNET moves those GB out of *free* before
    ``_floor_length`` reads it, so the same weights are charged twice and the
    predictor raises ``MotionBudgetError`` on renders that fit. Adding the
    MEASURED hoist cost back restores exactly the memory state the pre-hoist
    code measured."""
    from nodes._otr_video_engines import eng_wan_ti2v as mod
    monkeypatch.setattr(mod._MC, "free_vram_mb", lambda: 4000.0)
    eng = WanTi2vEngine()
    with pytest.raises(mod._MC.MotionBudgetError):
        eng._floor_length(33, 832, 480)                  # post-hoist, uncorrected
    assert eng._floor_length(33, 832, 480, hoisted_vram_mb=8000.0) == 33


def test_the_hoist_cost_is_MEASURED_and_never_INVENTS_headroom():
    """``None`` whenever either read is unavailable (the CPU box), so the
    budget keeps its geometry-only path. A NEGATIVE delta -- another process
    freeing memory mid-load -- reads as 0.0: handing the budget MORE room than
    the box has is the one direction this correction must never move."""
    assert WanTi2vEngine._hoist_cost_mb(None, 1000.0) is None
    assert WanTi2vEngine._hoist_cost_mb(1000.0, None) is None
    assert WanTi2vEngine._hoist_cost_mb(9000.0, 3000.0) == 6000.0
    assert WanTi2vEngine._hoist_cost_mb(3000.0, 9000.0) == 0.0


def test_prepare_MEASURES_the_hoist_and_hands_it_to_the_segments(monkeypatch,
                                                                 tmp_path):
    """The number has to reach ``render_clip``, and it has to be the SAME one
    for every segment -- a budget re-derived per segment from a drifting free
    read is how two segments of one beat pick different lengths."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_wan_ti2v as mod
    reads = iter([12000.0, 5000.0])
    monkeypatch.setattr(mod._MC, "free_vram_mb", lambda: next(reads, 5000.0))
    eng = WanTi2vEngine()
    eng._classes = _counting_fakes(np, {})
    monkeypatch.setattr(eng, "load", lambda: None)
    _no_ffmpeg(monkeypatch, mod, tmp_path)
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    try:
        assert prepared["hoisted_vram_mb"] == 7000.0
        assert prepared["session_ctx"]["multi_clip"] is True
    finally:
        eng.teardown(prepared)
