"""WIRE-W4a -- the HuMo BEAT SESSION: one load of everything per beat.

WHY ALL THREE HuMo LANES CAME BACK NO_RENDER. ``BeatSession`` refuses a
multi-segment beat from an engine with no ``session_identity()``, and no HuMo
adapter declared one. HuMo beats are dialogue, so they are long, so they are
exactly the beats the partitioner splits.

WHY THE HOIST IS WIDER HERE THAN ON THE WAN LANES, and why that is a property
of the family rather than a preference: WAN renders with ``free_after_use=True``
so umt5 and the diffusion UNET are never co-resident, and hoisting its CLIP
would delete that mitigation. HuMo renders FULLY RESIDENT by contract (BUG-265:
forcing inter-node eviction fragmented the allocator into an OOM), so every
loader is held for the whole render anyway -- hoisting changes how many times
they are READ, not how much is held.

THE RECLAIM IS THE OTHER HALF, and neither half works alone. ``render_clip``
ends with a LOUD ``reclaim_idle_models`` whose stated job is "so the resident
stack drops back down before the NEXT SOAK BEAT starts". Run between two
segments of ONE beat it would ``detach(unpatch_all=True)`` the very handles
``prepare`` hoisted -- the load count would stay 1 while the weights bounced to
CPU and back every segment. Drop it without the hoist and segment 2 builds NEW
umt5 and whisper handles beside segment 1's. So: hoist, skip between segments,
reclaim once at teardown, and the cross-beat promise is unchanged.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import beat_session as bs
from nodes._otr_video_engines.eng_humo import (HuMo14BLandscapeEngine,
                                               HuMo17BEngine, HuMoEngine)


def _mk_node(fn):
    return type("FakeNode", (), {"FUNCTION": "f", "f": fn})


class _FakeModel:
    def detach(self, unpatch_all=False):
        return None


def _counting_fakes(np, counts, n_frames=33):
    """The proven HuMo node classes, faked, with every LOADER counting."""
    img = np.zeros((n_frames, 32, 24, 3), dtype="float32")

    def _count(key, out):
        def _f(self, **_kw):
            counts[key] = counts.get(key, 0) + 1
            return out() if callable(out) else out
        return _mk_node(_f)

    return {
        "unet": _count("unet", lambda: (_FakeModel(),)),
        "lora": _count("lora", lambda: (_FakeModel(),)),
        "clip": _count("clip", lambda: (object(),)),
        "vae": _count("vae", lambda: (object(),)),
        "audioenc_loader": _count("audioenc_loader", lambda: (object(),)),
        "modelsampling": _count("modelsampling", lambda: (_FakeModel(),)),
        "pos": _mk_node(lambda self, **k: (("cond_pos",),)),
        "neg": _mk_node(lambda self, **k: (("cond_neg",),)),
        "loadaudio": _count("loadaudio", lambda: (object(),)),
        "audioenc": _count("audioenc", lambda: (object(),)),
        "loadimage": _mk_node(lambda self, **k: (object(), object())),
        "humo": _mk_node(lambda self, **k: (("p",), ("n",), ("latent",))),
        "ksampler": _mk_node(lambda self, **k: (("latent",),)),
        "vaedecode": _mk_node(lambda self, **k: (img,)),
    }


def _no_ffmpeg(monkeypatch, mod, tmp_path, reclaims):
    from nodes._otr_video_engines import wrapper_bridge as _wb
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"not-a-real-mp4")
    monkeypatch.setattr(_wb, "encode_frames_to_silent_mp4",
                        lambda frames, path, fps: (str(out), len(frames)))
    monkeypatch.setattr(mod, "ffprobe_clip_fields", lambda p: {})
    monkeypatch.setattr(mod, "validate_silent_clip_contract",
                        lambda fields, fps: None)
    monkeypatch.setattr(_wb, "comfy_input_dir", lambda: str(tmp_path))
    monkeypatch.setattr(_wb, "stage_into_comfy_input",
                        lambda p: "staged_" + str(p).rsplit("\\", 1)[-1])
    # THE RECLAIM IS COUNTED, NOT SUPPRESSED. Its call count IS the contract
    # this chunk changes, so a stub that swallowed it would erase the evidence.
    monkeypatch.setattr(_wb, "reclaim_idle_models",
                        lambda reason="": reclaims.append(reason) or 0)


def _request(tmp_path, seed=3, frames=33):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFFfake")
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNGfake")
    return {"shot_id": "s1", "asset_refs": {"init_image": str(png)},
            "audio_ref": {"path": str(audio)},
            "text_prompt": "a calm anchor",
            "init_w": 480, "init_h": 832,
            "canvas": {"w": 480, "h": 832, "aspect_policy": "pad"},
            "timing": {"target_frame_count": frames},
            "seed_bundle": {"request_seed": seed}}


def _ctx(segments):
    return {"beat_id": "b1", "segment_count": segments,
            "multi_clip": segments > 1}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_humo as mod
    counts, reclaims = {}, []
    eng = HuMoEngine()
    eng._classes = _counting_fakes(np, counts)
    monkeypatch.setattr(eng, "load", lambda: None)      # classes already set
    _no_ffmpeg(monkeypatch, mod, tmp_path, reclaims)
    return eng, counts, reclaims, tmp_path


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------

def test_every_HuMo_TIER_can_name_its_own_handles():
    """Not just the base class. The three subclasses are separate registry
    entries -- separate NO_RENDER rows on the campaign -- and each overrides
    ``_loader_names``/``_ckpt_path``, so each has to produce its own answer
    from the inherited mechanism."""
    for engine in (HuMoEngine(), HuMo17BEngine(), HuMo14BLandscapeEngine()):
        identity = bs.session_identity(engine)
        assert identity is not None, "%s declares no identity" % engine.name
        assert identity[0] == engine.name


def test_two_TIERS_never_share_an_identity():
    """Two beats on different tiers pass ``BeatSession``'s engine check only
    trivially -- the identity is what keeps a 1.7B beat from being served the
    14B's handles."""
    seen = [bs.session_identity(e)
            for e in (HuMoEngine(), HuMo17BEngine(), HuMo14BLandscapeEngine())]
    assert len(set(seen)) == len(seen)


def test_the_identity_moves_when_a_WEIGHT_moves(monkeypatch):
    """Size+mtime, not a content hash: it is re-read before every segment and
    hashing a 14B UNET per segment would cost more than the render it
    protects. A rebuilt weight still moves it."""
    eng = HuMoEngine()
    monkeypatch.setattr(eng, "_humo_file_receipt",
                        lambda p: ("humo.safetensors", 111, 222))
    before = eng.session_identity()
    monkeypatch.setattr(eng, "_humo_file_receipt",
                        lambda p: ("humo.safetensors", 111, 999))
    assert eng.session_identity() != before


def test_the_identity_says_whether_the_LORA_IS_MERGED(monkeypatch):
    """The lightx2v distill LoRA is merged INTO the hoisted patcher, so a beat
    that turned it off midway would be reusing a model its own later graph no
    longer describes. The 1.7B tiers really do run LoRA-free, so this is a
    live distinction, not a hypothetical."""
    eng = HuMoEngine()
    monkeypatch.setenv("OTR_HUMO_LORA_NAME", "lightx2v_rank64_bf16.safetensors")
    merged = eng.session_identity()
    monkeypatch.setenv("OTR_HUMO_LORA_NAME", "none")
    assert eng.session_identity() != merged
    assert any("lora_merged=False" in part for part in eng.session_identity())


def test_the_identity_carries_the_ENCODERS_that_are_hoisted_with_the_model():
    """umt5 and whisper are hoisted here (unlike the WAN lanes), so a swapped
    encoder is drift in the most literal sense -- later segments would render
    from a handle earlier segments never used."""
    eng = HuMoEngine()
    labels = [row[0] for row in eng._humo_session_receipts()]
    assert labels == ["unet", "lora", "clip", "vae", "whisper"]


# ---------------------------------------------------------------------------
# THE ACCEPTANCE: loader invocations, not prepare() calls
# ---------------------------------------------------------------------------

def test_a_THREE_SEGMENT_beat_loads_EVERY_HEAVY_HANDLE_EXACTLY_ONCE(wired):
    """THE test this chunk exists to pass.

    Before the hoist this was three of each: a 14B UNET, a LoRA merge, umt5 and
    whisper, read from disk once per segment to render one beat. Counted on the
    LOADER and never on ``prepare()`` -- an adapter that loads lazily inside
    ``render_clip`` shows one perfect prepare and three loads, which is the
    trap ``BeatSession`` deliberately carries no counters for.
    """
    eng, counts, _reclaims, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(3))
    try:
        for key in ("unet", "lora", "clip", "vae", "audioenc_loader"):
            assert counts[key] == 1, "prepare() must load %s exactly once" % key
        for seg in range(3):
            eng.render_clip(_request(tmp_path, seed=seg), prepared)
        for key in ("unet", "lora", "clip", "vae", "audioenc_loader"):
            assert counts[key] == 1, (
                "%s loaded %d time(s) across a 3-segment beat -- the hoist is "
                "not reaching the segment graph" % (key, counts[key]))
    finally:
        eng.teardown(prepared)


def test_the_AUDIO_NODES_stay_per_segment_and_that_is_the_whole_point(wired):
    """``loadaudio``/``audioenc`` are the one thing that genuinely DIFFERS per
    segment on a lip-synced lane, so hoisting them would be the bug rather than
    the optimisation. (They are handed the same file today -- the per-segment
    SLICE is W4b -- but the nodes must already be per-segment for that to have
    anywhere to land.)"""
    eng, counts, _reclaims, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(3))
    try:
        for seg in range(3):
            eng.render_clip(_request(tmp_path, seed=seg), prepared)
        assert counts["loadaudio"] == 3
        assert counts["audioenc"] == 3
        assert counts["modelsampling"] == 3, (
            "ModelSamplingSD3 is a cheap clone+patch of an already-resident "
            "MODEL and is correctly per-render")
    finally:
        eng.teardown(prepared)


def test_a_caller_that_PREPARED_NOTHING_renders_exactly_as_before(wired):
    """The single-clip path is what production runs today, and every sibling
    test hand-builds ``prepared={"patchers": []}``."""
    eng, counts, reclaims, tmp_path = wired
    eng.render_clip(_request(tmp_path), {"patchers": []})
    for key in ("unet", "lora", "clip", "vae", "audioenc_loader"):
        assert counts[key] == 1, "the un-prepared path must load its own %s" % key
    assert reclaims == ["humo post-decode"], (
        "the single-clip path keeps its post-decode reclaim, unchanged")


def test_the_graph_OMITS_the_hoisted_loaders_and_KEEPS_every_wire(wired):
    """``run_graph`` REFUSES a graph that also defines an id the caller
    supplied -- two parties claiming to produce one output -- so omission is a
    contract. Every WIRE must survive: the sampler still reaches the model
    through ModelSamplingSD3, and the encode still reaches the VAE."""
    eng, _counts, _reclaims, _tmp = wired
    plan = {"seed": 1, "text_prompt": "x"}
    plain = eng._build_graph("p.png", "a.wav", plan, 33, 480, 832)
    hoisted = eng._build_graph("p.png", "a.wav", plan, 33, 480, 832,
                               external_results={
                                   nid: (object(),)
                                   for nid in eng._session_node_ids()})
    assert set(plain) - set(hoisted) == set(eng._session_node_ids())
    assert hoisted["modelsampling"]["inputs"]["model"].src == "lora"
    assert hoisted["vaedecode"]["inputs"]["vae"].src == "vae"
    assert hoisted["audioenc"]["inputs"]["audio_encoder"].src == "audioenc_loader"


def test_the_LORA_FREE_tier_hoists_a_DIFFERENT_SET(monkeypatch):
    """A session that hoisted a ``lora`` node the graph never defines would
    wire a handle nothing reads; one that skipped a node the graph DOES define
    would let it reload every segment. Both readings now come from
    ``_lora_is_skipped``, so they cannot disagree."""
    eng = HuMoEngine()
    monkeypatch.setenv("OTR_HUMO_LORA_NAME", "none")
    ids = eng._session_node_ids()
    assert "lora" not in ids
    graph = eng._build_graph("p.png", "a.wav", {"seed": 1}, 33, 480, 832)
    assert "lora" not in graph
    assert set(ids) <= set(graph), "every hoisted id must exist in the graph"
    monkeypatch.setenv("OTR_HUMO_LORA_NAME", "lightx2v_rank64_bf16.safetensors")
    assert "lora" in eng._session_node_ids()


def test_prepare_REFUSES_to_hoist_a_node_its_own_graph_does_not_define(
        wired, monkeypatch):
    """The session and the graph are two readings of one lane. If they ever
    disagree the honest answer is to say so before any weights land -- a
    hoisted id the graph never defines is a handle nothing would read."""
    eng, _counts, _reclaims, _tmp = wired
    monkeypatch.setattr(eng, "_session_node_ids",
                        lambda: ("unet", "not_a_real_node"))
    with pytest.raises(Exception) as caught:
        eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    assert "not_a_real_node" in str(caught.value)
    assert "NO FALLBACK" in str(caught.value)


def test_prepare_REFUSES_a_loader_that_produced_NO_HANDLE(monkeypatch,
                                                          tmp_path):
    """A loader node that returns nothing leaves the session with nothing to
    reuse, and every segment would then quietly build its own -- a beat that
    reports a hoist while reloading is worse than one that never hoisted, and
    the "1 load per beat" acceptance would still read 1 because the loader
    node had run once. Named, so the log says WHICH handle is missing."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_humo as mod
    eng = HuMoEngine()
    counts, reclaims = {}, []
    fakes = _counting_fakes(np, counts)
    fakes["vae"] = _mk_node(lambda self, **_k: ())      # produced no handle
    eng._classes = fakes
    monkeypatch.setattr(eng, "load", lambda: None)
    _no_ffmpeg(monkeypatch, mod, tmp_path, reclaims)
    with pytest.raises(Exception) as caught:
        eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    assert "vae" in str(caught.value)
    assert "silently reload" in str(caught.value)


# ---------------------------------------------------------------------------
# THE RECLAIM: once per beat, not once per segment
# ---------------------------------------------------------------------------

def test_the_RECLAIM_does_not_run_BETWEEN_the_segments_of_one_beat(wired):
    """Its stated job is "so the resident stack drops back down before the NEXT
    SOAK BEAT starts". Between two segments of the SAME beat there is nothing
    to drop back for -- and running it would detach(unpatch_all=True) the very
    handles ``prepare`` hoisted, so the load count would stay 1 while the
    weights bounced to CPU and back every segment."""
    eng, _counts, reclaims, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(3))
    try:
        for seg in range(3):
            eng.render_clip(_request(tmp_path, seed=seg), prepared)
        assert reclaims == [], "no reclaim may run between segments"
    finally:
        eng.teardown(prepared)
    assert reclaims == ["humo beat teardown"], (
        "the cross-beat promise is kept by running it exactly once, at the end "
        "of the beat")


def test_a_SINGLE_CLIP_beat_still_reclaims_ONCE_and_only_once(wired):
    """A single-clip beat opens a session too (render_driver always does), so
    the teardown branch must not double-reclaim what render_clip already did.
    """
    eng, _counts, reclaims, tmp_path = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(1))
    try:
        eng.render_clip(_request(tmp_path), prepared)
    finally:
        eng.teardown(prepared)
    assert reclaims == ["humo post-decode"]


def test_teardown_DROPS_the_hoisted_handles_before_the_base_runs(wired):
    """The base teardown detaches patchers, unloads, releases the lease and
    then waits for machine-wide VRAM to settle -- with this dict still pinning
    the tensors it is reclaiming, the wait watches its own referent."""
    eng, _counts, _reclaims, _tmp = wired
    prepared = eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    assert set(prepared["external_results"]) == set(eng._session_node_ids())
    eng.teardown(prepared)
    assert "external_results" not in prepared


def test_a_failure_DURING_the_hoist_does_not_strand_the_lease(monkeypatch,
                                                              tmp_path):
    """A raise after ``super().prepare()`` has taken the cross-process GPU
    lease strands it for the life of the server -- the C-3 lesson. Every later
    heavy render then blocks its full timeout for a reason that has nothing to
    do with it."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import eng_humo as mod
    eng = HuMoEngine()
    counts, reclaims = {}, []
    fakes = _counting_fakes(np, counts)

    def _boom(self, **_kw):
        raise RuntimeError("the whisper encoder did not load")

    fakes["audioenc_loader"] = _mk_node(_boom)
    eng._classes = fakes
    monkeypatch.setattr(eng, "load", lambda: None)
    _no_ffmpeg(monkeypatch, mod, tmp_path, reclaims)

    torn = {"n": 0}
    real_teardown = eng.teardown

    def _spy(prepared):
        torn["n"] += 1
        return real_teardown(prepared)

    monkeypatch.setattr(eng, "teardown", _spy)
    with pytest.raises(Exception):
        eng.prepare(host_caps={}, profile={}, session_ctx=_ctx(2))
    assert torn["n"] == 1, "a failed hoist must tear down what it took"
