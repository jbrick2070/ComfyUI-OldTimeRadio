"""WIRE-W3a -- the wan_i2v BEAT SESSION: one UNET load per beat, not per segment.

WHY THE LOAD COUNT IS THE ONLY ACCEPTANCE THAT MATTERS. ``BeatSession`` refuses
a multi-segment beat from an engine with no ``session_identity()``, and that
refusal is why `wan_i2v` came back NO_RENDER on the 2026-07-28 campaign. It is
also a trap: declaring an identity SILENCES the refusal, and the segment graph
then executes ``UNETLoader`` every single segment. The beat would look fixed,
render, and reload a 14B UNET three times.

So these tests count LOADER INVOCATIONS on the adapter, never ``prepare()``
calls. ``BeatSession`` deliberately carries no call counters for exactly this
reason -- a counter there measures the bracket, which is obviously correct,
instead of the loading, which is the thing that is not.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import wan_shared as _WS  # noqa: F401
from nodes._otr_video_engines.eng_wan_i2v import RECIPE_WAN_I2V, WanI2VEngine


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


def _counting_fakes(np, counts, n=4):
    """The CORE Wan node classes, faked, with every LOADER counting its calls."""
    img = np.zeros((n, 8, 8, 3), dtype="float32")

    def _count(key, out):
        def _f(self, **_kw):
            counts[key] = counts.get(key, 0) + 1
            return out() if callable(out) else out
        return _mk_node(_f)

    return {
        "unet": _count("unet", lambda: (_FakeModel(),)),
        "clip": _count("clip", lambda: (object(),)),
        "vae": _count("vae", lambda: (object(),)),
        "modelsampling": _mk_node(lambda self, **k: (_FakeModel(),)),
        "pos": _mk_node(lambda self, **k: (("c",),)),
        "neg": _mk_node(lambda self, **k: (("c",),)),
        "loadimage": _mk_node(lambda self, **k: (object(), object())),
        "wan": _mk_node(lambda self, **k: (("p",), ("n",), ("latent",))),
        "ksampler": _mk_node(lambda self, **k: (("latent",),)),
        "vaedecode": _mk_node(lambda self, **k: (img,)),
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


def _request(tmp_path, seed=3):
    return {"shot_id": "s1", "asset_refs": {"init_image": _init_png(tmp_path)},
            "text_prompt": "subtle motion", "init_w": 832, "init_h": 480,
            "canvas": {"w": 832, "h": 480, "aspect_policy": "pad"},
            "timing": {"target_frame_count": 33},
            "seed_bundle": {"request_seed": seed}}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """An engine whose classes are fakes and whose residency is stubbed."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_wan_i2v as mod
    counts = {}
    eng = WanI2VEngine()
    eng._classes = _counting_fakes(np, counts)
    monkeypatch.setattr(eng, "load", lambda: None)      # classes already set
    _no_ffmpeg(monkeypatch, mod, tmp_path)
    return eng, counts, tmp_path


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------

def test_wan_i2v_can_name_its_own_handles():
    """Until this existed, BeatSession refused EVERY multi-segment wan_i2v beat
    -- which is 2 of the 6 NO_RENDER engines on the 2026-07-28 campaign."""
    from nodes._otr_video_engines import beat_session as bs
    identity = bs.session_identity(WanI2VEngine())
    assert identity is not None
    assert isinstance(identity, tuple) and identity[0] == "wan_i2v"
    assert RECIPE_WAN_I2V in identity[1]


def test_the_identity_moves_when_a_WEIGHT_moves(monkeypatch):
    """Size+mtime, not a content hash: the identity is re-read before every
    segment and hashing a 14B UNET per segment would cost more than the render
    it protects. A rebuilt weight still moves it."""
    eng = WanI2VEngine()
    monkeypatch.setattr(eng, "_wan_file_receipt",
                        lambda p: ("unet.safetensors", 111, 222))
    before = eng.session_identity()
    monkeypatch.setattr(eng, "_wan_file_receipt",
                        lambda p: ("unet.safetensors", 111, 999))   # rebuilt
    assert eng.session_identity() != before


def test_the_identity_carries_the_VAE_even_though_the_VAE_IS_NOT_HOISTED():
    """r4/A5, and it is not pedantry. Both WAN adapters load a VAE
    independently and TI2V explicitly distinguishes INCOMPATIBLE VAE
    GENERATIONS, so a VAE swapped between segments is exactly the drift the
    session exists to catch -- even though the handle itself stays
    per-segment. An identity that named only what it HOISTED would miss it."""
    import os
    eng = WanI2VEngine()
    os.environ["OTR_WAN_I2V_VAE_NAME"] = "wan_2.1_vae.safetensors"
    try:
        before = eng.session_identity()
        os.environ["OTR_WAN_I2V_VAE_NAME"] = "wan_2.2_vae.safetensors"
        assert eng.session_identity() != before
    finally:
        os.environ.pop("OTR_WAN_I2V_VAE_NAME", None)


def test_the_identity_carries_the_UNET_LOADER_MODE(monkeypatch):
    """The mode picks the loader CLASS (UNETLoader vs UnetLoaderGGUF). A beat
    that switched class mid-way would be reusing a handle its own graph no
    longer knows how to have produced."""
    eng = WanI2VEngine()
    monkeypatch.setenv("OTR_WAN_I2V_LOADER", "safetensors")
    before = eng.session_identity()
    monkeypatch.setenv("OTR_WAN_I2V_LOADER", "gguf")
    assert eng.session_identity() != before


# ---------------------------------------------------------------------------
# THE ACCEPTANCE: loader invocations, not prepare() calls
# ---------------------------------------------------------------------------

def test_a_THREE_SEGMENT_beat_loads_the_UNET_EXACTLY_ONCE(wired):
    """THE test this chunk exists to pass.

    Three segments, one UNET load. Before the hoist this was three -- 14B fp8
    weights read from disk once per segment, plus three stability waits, to
    render one beat. A ten-line ``session_identity()`` on its own would have
    made this test read 3 while every other test in the suite went green,
    which is precisely why the count is asserted on the LOADER and not on the
    bracket around it.
    """
    eng, counts, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={},
                           session_ctx={"beat_id": "b1", "segment_count": 3,
                                        "multi_clip": True})
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
    """BeatSession's docstring promised model + CLIP + VAE acquired once. W3
    narrows that to "primary diffusion patcher once per beat; auxiliaries may
    reload", and this pins the narrowing so the next reader does not file the
    reload as a defect.

    It is not laziness. ``render_clip`` runs with ``free_after_use=True``
    precisely so umt5-fp8 and the 14B UNET are never co-resident through the
    sampler -- the bare smoke without it peaked at 14,499 MB on a 16 GiB card.
    Hoisting the CLIP would pin ~9 GB for the whole beat and delete the one
    mitigation standing between this lane and an OOM.
    """
    eng, counts, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={},
                           session_ctx={"beat_id": "b1", "segment_count": 3,
                                        "multi_clip": True})
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
    loader nodes -- if this goes red the chunk has broken the majority path to
    serve the new one."""
    eng, counts, tmp_path = wired
    eng.render_clip(_request(tmp_path), {"patchers": []})
    assert counts["unet"] == 1, "the un-prepared path must load its own UNET"
    assert counts["clip"] == 1 and counts["vae"] == 1


def test_the_graph_OMITS_a_hoisted_node_and_KEEPS_every_wire_that_reads_it():
    """``run_graph`` REFUSES a graph that also defines an id the caller
    supplied -- two parties claiming to produce one output -- so omission is a
    contract, not an optimisation. The WIRES must survive: the sampler still
    reads the model through ModelSamplingSD3."""
    eng = WanI2VEngine()
    req = {"text_prompt": "x", "canvas": {"w": 832, "h": 480}}
    plain = eng._build_graph(req, "i.png", {"seed": 1}, 33, 832, 480)
    hoisted = eng._build_graph(req, "i.png", {"seed": 1}, 33, 832, 480,
                               external_results={"unet": (object(),)})
    assert "unet" in plain
    assert "unet" not in hoisted
    # every other node, and every wire, is untouched
    assert set(plain) - set(hoisted) == {"unet"}
    assert hoisted["modelsampling"]["inputs"]["model"].src == "unet"


def test_teardown_DROPS_the_hoisted_handles_before_the_base_runs(wired):
    """The base teardown detaches patchers, unloads, releases the lease and
    then WAITS for machine-wide VRAM to settle. Every one of those steps would
    run with this dict still pinning the very tensors it is reclaiming, so the
    stability wait would be watching its own referent."""
    eng, _counts, _tmp = wired
    prepared = eng.prepare(host_caps={}, profile={},
                           session_ctx={"beat_id": "b1", "segment_count": 2,
                                        "multi_clip": True})
    assert "external_results" in prepared
    eng.teardown(prepared)
    assert "external_results" not in prepared


def test_a_failure_DURING_the_hoist_does_not_strand_the_lease(monkeypatch,
                                                              tmp_path):
    """A raise after ``super().prepare()`` has taken the cross-process GPU
    lease strands it for the life of the server -- the C-3 lesson. The hoist
    is wrapped so anything that fails mid-load tears down what it took."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_wan_i2v as mod
    eng = WanI2VEngine()
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
        eng.prepare(host_caps={}, profile={},
                    session_ctx={"beat_id": "b1", "segment_count": 2,
                                 "multi_clip": True})
    assert torn["n"] == 1, "a failed hoist must tear down what it took"
