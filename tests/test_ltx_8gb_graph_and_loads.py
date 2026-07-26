"""The regression net `ltx_8gb` never had -- written BEFORE the loader hoist.

All six of its sibling adapters (`ltx_video`, `ltx_av`, `wan_i2v`, `wan_ti2v`,
`humo`, `mesh_stage`) have graph-shape tests. `ltx_8gb` had NONE: no test in the
suite built its graph, and no test drove its `render_clip`. A wrong graph shape
here turned nothing red -- it surfaced on a live GPU render.

Chunk B1b will hoist `CheckpointLoaderSimple` out of the per-segment graph so a
multi-segment beat loads the checkpoint ONCE. Landing that on top of no coverage
would put the engine's first graph tests and the codebase's first `prepare()`
override in one diff, where a red says nothing about which half is wrong. So
this file lands FIRST and pins what is true TODAY.

WHAT THE HOIST MAY CHANGE HERE, stated exactly -- an earlier draft of this
docstring got it WRONG and two independent reviewers caught it. The hoist keeps
`_build_graph` CONDITIONAL: it omits `ckpt` ONLY when the caller supplied one
through `external_results`, and still emits it when the caller prepared nothing.
Every test below either builds the graph directly or hands `render_clip` a
prepared dict it built by hand, so every one of them stays on the UNSUPPLIED
branch. That makes them CONTROLS on the hoist, not observers of it -- the load
count in `test_THE_LOAD_COUNT_...` is expected to STAY 3.

EXACTLY ONE assertion is expected to flip:
`test_the_executor_is_called_with_the_current_keep_contract`'s
`"external_results" not in seen["kwargs"]`, because `render_clip` will forward
the caller's externals on every call. If the hoist turns anything ELSE in this
file red it has over-reached -- it has broken a caller that prepared nothing.

WHAT THE HOIST CHUNK OWES, because nothing here can prove it: a test that calls
`Ltx8gbEngine.prepare()` once, reuses the dict it returns across several
`render_clip` calls, and pins ONE checkpoint load. The mechanism that test will
stand on is proven here against this engine's REAL graph, ahead of the wiring,
by `test_the_forward_runs_with_the_checkpoint_supplied_EXTERNALLY`.

Fakes + a real ffmpeg encode, the idiom already proven in
`tests/test_video_motion_forward.py`. ffmpeg-running tests skip cleanly.
UTF-8, no BOM, ASCII-only source.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from nodes._otr_video_engines import eng_ltx_8gb as m
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_ltx_8gb import Ltx8gbEngine

_HAS_FFMPEG = shutil.which("ffmpeg") is not None

_ENVS = (
    "OTR_LTX_8GB_CKPT", "OTR_LTX_8GB_CKPT_DIR", "OTR_LTX_8GB_CKPT_NAME",
    "OTR_LTX_8GB_T5_DIR", "OTR_LTX_8GB_T5_NAME", "OTR_LTX_8GB_T5_DEVICE",
    "OTR_LTX_8GB_TILED_VAE", "OTR_LTX_8GB_STEPS", "OTR_LTX_8GB_CFG",
    "OTR_LTX_8GB_SAMPLER", "OTR_LTX_8GB_MAX_SHIFT", "OTR_LTX_8GB_BASE_SHIFT",
    "OTR_LTX_8GB_TERMINAL", "OTR_LTX_8GB_MAX_FRAMES", "OTR_LTX_8GB_NEGATIVE",
    "OTR_LTX_8GB_VAE_TILE", "OTR_LTX_8GB_VAE_OVERLAP",
    "OTR_LTX_8GB_VAE_TEMPORAL", "OTR_LTX_8GB_VAE_TEMPORAL_OVERLAP",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


def _mk(fn):
    return type("FakeNode", (), {"FUNCTION": "f", "f": fn})


class _FakeModel:
    """Stands in for a ComfyUI MODEL patcher: the harvest loop keeps anything
    with a callable ``detach``, and teardown is what calls it."""

    def __init__(self, tag="model"):
        self.tag = tag
        self.detached = 0

    def detach(self, unpatch_all=False):
        self.detached += 1
        return None


class _Counter:
    """Counts how many times each fake node class actually EXECUTED.

    This is the observable the hoist moves: today `ckpt` executes once per
    render; after B1b it executes once per BEAT.
    """

    def __init__(self):
        self.calls = {}

    def bump(self, nid):
        self.calls[nid] = self.calls.get(nid, 0) + 1


def _ltx8_fakes(np, counter, n=9):
    """The LTX 0.9.8 node classes, faked.

    The decode fake sizes its IMAGE batch from the `length` the graph actually
    asked `LTXVImgToVideo` for, so a decoded frame count is an OBSERVATION of
    what the engine wired rather than a fixture constant. With a fixed-size
    array every `frame_count` assertion in this file would be true by
    construction -- `_ltx8_frame_length` could return anything and the forward
    tests would still pass. `n` is only the fallback for a graph that never
    reaches `img2vid`.
    """
    asked = {"length": n}

    def _node(nid, out):
        def f(self, **kwargs):
            counter.bump(nid)
            if nid == "img2vid":
                asked["length"] = int(kwargs.get("length", n))
            return out() if callable(out) else out
        return _mk(f)

    return {
        # CheckpointLoaderSimple -> (MODEL, CLIP, VAE); the 0.9.8 all-in-one
        # carries the VAE at slot 2 and NO text encoder.
        "ckpt": _node("ckpt", lambda: (_FakeModel("ckpt"), object(), object())),
        "clip": _node("clip", lambda: (object(),)),          # CLIPLoader (T5)
        "pos": _node("pos", (("c",),)),
        "neg": _node("neg", (("c",),)),
        "loadimage": _node("loadimage", lambda: (object(), object())),
        "modelsampling": _node("modelsampling",
                               lambda: (_FakeModel("modelsampling"),)),
        "img2vid": _node("img2vid", (("p",), ("n",), ("latent",))),
        "cond": _node("cond", (("p",), ("n",))),
        "sched": _node("sched", (("sigmas",),)),
        "sampler": _node("sampler", (("sampler",),)),
        # SamplerCustom returns (output, denoised_output) -- two LATENTs. The
        # graph reads slot 0 only, but a fake that under-models the real class
        # would IndexError the day someone wires the second one.
        "sample": _node("sample", (("latent",), ("denoised",))),
        "decode": _node("decode", lambda: (
            np.zeros((asked["length"], 24, 32, 3), dtype="float32"),)),
    }


def _request(init_png, frames=9):
    return {"shot_id": "s1", "text_prompt": "a rain-slicked street at night",
            "asset_refs": {"init_image": str(init_png)},
            "init_w": 512, "init_h": 288,
            "canvas": {"w": 512, "h": 288, "fps": 25, "aspect_policy": "pad"},
            "timing": {"target_frame_count": frames},
            "seed_bundle": {"request_seed": 7}}


# --- the node classes the graph resolves ----------------------------------- #
def test_node_candidates_are_the_discovery_verified_core_LTX_nodes():
    """Pinned against the LIVE /object_info capture of 2026-07-20. These are
    CORE ComfyUI classes -- no custom pack -- which is why the 8 GB tier can be
    a normal selectable row with no vendor gate."""
    cand = Ltx8gbEngine()._node_candidates()
    assert cand["ckpt"] == ("CheckpointLoaderSimple",)
    assert cand["clip"] == ("CLIPLoader",)
    assert cand["pos"] == ("CLIPTextEncode",)
    assert cand["neg"] == ("CLIPTextEncode",)
    assert cand["loadimage"] == ("LoadImage",)
    assert cand["modelsampling"] == ("ModelSamplingLTXV",)
    assert cand["img2vid"] == ("LTXVImgToVideo",)
    assert cand["cond"] == ("LTXVConditioning",)
    assert cand["sched"] == ("LTXVScheduler",)
    assert cand["sampler"] == ("KSamplerSelect",)
    assert cand["sample"] == ("SamplerCustom",)
    assert cand["decode"] == ("VAEDecode",)          # tiled has its own test


def test_the_decode_CLASS_and_its_INPUTS_are_chosen_by_the_same_switch(
        monkeypatch):
    """`_node_candidates` and `_decode_inputs` read the tiled-VAE flag
    INDEPENDENTLY. Nothing stops the two reads from disagreeing, and a
    disagreement is a render that asks `VAEDecode` for tile arguments it does
    not accept. Pin the agreement so a later refactor cannot split them
    silently -- this is the same class of defect as a contract declared in one
    channel and consumed from another."""
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})

    assert eng._node_candidates()["decode"] == ("VAEDecode",)
    plain = eng._build_graph({}, "p.png", plan, 9, 512, 288)["decode"]["inputs"]
    assert set(plain) == {"samples", "vae"}

    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "1")
    assert eng._node_candidates()["decode"] == ("VAEDecodeTiled",)
    tiled = eng._build_graph({}, "p.png", plan, 9, 512, 288)["decode"]["inputs"]
    assert tiled["tile_size"] == 512 and tiled["overlap"] == 64
    assert tiled["temporal_size"] == 16 and tiled["temporal_overlap"] == 8


# --- the graph shape ------------------------------------------------------- #
def test_the_graph_carries_ITS_OWN_loader_nodes_today(monkeypatch):
    """CONTROL on B1b -- this must NOT flip, and an earlier draft said it would.

    Today every segment's graph defines its own `CheckpointLoaderSimple` and
    `CLIPLoader`, which is exactly why a multi-segment beat re-reads a 6.34 GiB
    checkpoint per segment. The hoist removes `ckpt` ONLY when the caller
    supplied one through `external_results`. This call supplies nothing, so the
    loader must still be here afterwards: the day this goes red, the hoist has
    broken every caller that prepared nothing.

    The ABSENT case -- the hoist's actual payoff -- cannot be observed from
    here, because `_build_graph` has no external-results channel until the
    hoist adds one. `test_the_forward_runs_with_the_checkpoint_supplied_
    EXTERNALLY` proves the executor half of that mechanism today; the graph
    half is owed by the hoist chunk itself.
    """
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288)

    assert g["ckpt"]["inputs"]["ckpt_name"] == m._LTX8_DEFAULT_CKPT
    assert g["clip"]["inputs"]["clip_name"] == m._LTX8_DEFAULT_T5
    assert g["clip"]["inputs"]["type"] == "ltxv"
    assert g["clip"]["inputs"]["device"] == "cpu"     # the 8 GB tier default


def test_every_wire_that_reads_the_loaders_is_pinned():
    """These wires MUST SURVIVE the hoist unchanged -- that is the whole trick:
    the node definitions go away, the wires stay and resolve against
    `external_results` instead. If a hoist edit drops a wire, this fails."""
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 3}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288)

    assert g["modelsampling"]["inputs"]["model"] == wb.Wire("ckpt", 0)
    assert g["img2vid"]["inputs"]["vae"] == wb.Wire("ckpt", 2)   # embedded VAE
    assert g["decode"]["inputs"]["vae"] == wb.Wire("ckpt", 2)
    assert g["pos"]["inputs"]["clip"] == wb.Wire("clip", 0)
    assert g["neg"]["inputs"]["clip"] == wb.Wire("clip", 0)
    assert g["sample"]["inputs"]["model"] == wb.Wire("modelsampling", 0)
    assert g["img2vid"]["inputs"]["image"] == wb.Wire("loadimage", 0)
    assert g["cond"]["inputs"]["positive"] == wb.Wire("img2vid", 0)
    assert g["sched"]["inputs"]["latent"] == wb.Wire("img2vid", 2)
    assert g["sample"]["inputs"]["latent_image"] == wb.Wire("img2vid", 2)
    assert g["sample"]["inputs"]["sigmas"] == wb.Wire("sched", 0)
    assert g["sample"]["inputs"]["sampler"] == wb.Wire("sampler", 0)


def test_the_PROMPT_POLARITY_is_pinned_on_every_hop():
    """The wrong-graph-shape defect this file exists to catch and did not.

    A positive/negative swap renders the negative prompt. It does not crash,
    does not shorten the clip, and no FORWARD test in this file can see it --
    the fakes return canned tuples and never look at what they were handed. So
    polarity has to be pinned statically, on every hop the conditioning takes:
    text -> pos/neg -> img2vid -> cond -> sample -> decode.
    """
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 5}})
    g = eng._build_graph({"text_prompt": "a lit doorway",
                          "negative_prompt": "watermark"},
                         "p.png", plan, 9, 512, 288)

    assert g["pos"]["inputs"]["text"] == "a lit doorway"
    assert g["neg"]["inputs"]["text"] == "watermark"
    assert g["img2vid"]["inputs"]["positive"] == wb.Wire("pos", 0)
    assert g["img2vid"]["inputs"]["negative"] == wb.Wire("neg", 0)
    assert g["cond"]["inputs"]["negative"] == wb.Wire("img2vid", 1)
    assert g["sample"]["inputs"]["positive"] == wb.Wire("cond", 0)
    assert g["sample"]["inputs"]["negative"] == wb.Wire("cond", 1)
    assert g["decode"]["inputs"]["samples"] == wb.Wire("sample", 0)


def test_the_RESOLVED_RECIPE_VALUES_reach_the_nodes_that_consume_them():
    """A configured value with a severed channel is this build's recurring
    defect class -- three instances in four days. `_resolve_render_config` is
    the ONE authority for steps / cfg / shift / sampler / terminal, so this
    pins that each one ARRIVES at the node that consumes it, and that the
    geometry the caller asked for is what `LTXVImgToVideo` is told.

    It compares against the resolver, not against literals, on purpose: the
    VALUES are the recipe's to change, the DELIVERY is not. What this catches
    is a crossed key (`"steps": cfg["max_shift"]`) or a hard-coded literal --
    the shapes that survive every other assertion in this file.
    """
    eng = Ltx8gbEngine()
    cfg = eng._resolve_render_config()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9},
         "seed_bundle": {"request_seed": 11}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 17, 512, 288)

    assert g["modelsampling"]["inputs"]["max_shift"] == cfg["max_shift"]
    assert g["modelsampling"]["inputs"]["base_shift"] == cfg["base_shift"]
    assert g["sched"]["inputs"]["steps"] == cfg["steps"]
    assert g["sched"]["inputs"]["max_shift"] == cfg["max_shift"]
    assert g["sched"]["inputs"]["base_shift"] == cfg["base_shift"]
    assert g["sched"]["inputs"]["terminal"] == cfg["terminal"]
    assert g["sampler"]["inputs"]["sampler_name"] == cfg["sampler"]
    assert g["sample"]["inputs"]["cfg"] == cfg["cfg"]
    assert g["sample"]["inputs"]["noise_seed"] == 11      # the request's seed
    assert g["img2vid"]["inputs"]["length"] == 17
    assert g["img2vid"]["inputs"]["width"] == 512
    assert g["img2vid"]["inputs"]["height"] == 288
    assert g["cond"]["inputs"]["frame_rate"] == float(eng.target_fps)


def test_the_graph_topologically_sorts_with_no_unknown_sources():
    """The executor refuses a wire whose source is neither a graph node nor a
    declared external. Proving the graph sorts TODAY is what makes the hoist's
    failure mode legible: drop the `ckpt` definition without supplying the
    external and this same call raises `wires from unknown source`."""
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288)

    order = wb._topo_order(g)
    assert set(order) == set(g)
    assert order.index("ckpt") < order.index("modelsampling")
    assert order.index("clip") < order.index("pos")
    assert order.index("img2vid") < order.index("sample")
    assert order[-1] == eng._TERMINAL


def test_a_graph_missing_its_loader_is_REFUSED_by_name():
    """The exact failure the hoist must not ship. Delete the producer, keep the
    wires, supply no external -> a NAMED refusal, never a silent partial
    render."""
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288)
    del g["ckpt"]

    with pytest.raises(wb.GraphExecutionError) as e:
        wb._topo_order(g)
    assert "unknown source" in str(e.value) and "ckpt" in str(e.value)


def test_the_same_graph_SORTS_once_the_loader_is_declared_external():
    """The mechanism B1b will use, proven against THIS engine's real graph
    rather than a synthetic fixture: the definition goes, the wires stay, and
    the sort succeeds because the caller declared it owns that id."""
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288)
    del g["ckpt"]

    order = wb._topo_order(g, external_keys={"ckpt"})
    assert "ckpt" not in order                   # externals are never executed
    assert order.index("clip") < order.index("pos")
    assert order[-1] == eng._TERMINAL


def test_the_forward_runs_with_the_checkpoint_supplied_EXTERNALLY():
    """The MECHANISM B1b will stand on, proven against THIS engine's real graph
    before the hoist wires it -- and the one test here that would catch a hoist
    which silently did nothing.

    Delete the `ckpt` definition, keep every wire, hand the executor an
    already-produced output tuple for that id. The loader class must NEVER be
    constructed; the downstream nodes must receive the SUPPLIED handles, not
    ones of their own; and the handle must survive a `free_after_use` pass --
    the executor adds every external to `keep`, which is what stops segment 0's
    cleanup from evicting the checkpoint segment 1 still needs. No ffmpeg: this
    stops at the executor, which is where the mechanism lives.
    """
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    classes = _ltx8_fakes(np, counter, n=9)
    supplied = (_FakeModel("hoisted"), object(), object())
    seen = {}

    def _spy(nid, out):
        def f(self, **kwargs):
            counter.bump(nid)
            seen[nid] = kwargs
            return out
        return _mk(f)

    classes["modelsampling"] = _spy("modelsampling", (_FakeModel("ms"),))
    classes["img2vid"] = _spy("img2vid", (("p",), ("n",), ("latent",)))
    classes["decode"] = _spy(
        "decode", (np.zeros((9, 24, 32, 3), dtype="float32"),))

    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288)
    del g["ckpt"]

    results = wb.run_graph(g, classes, free_after_use=True,
                           external_results={"ckpt": supplied},
                           keep={"modelsampling", eng._TERMINAL})

    assert "ckpt" not in counter.calls               # the loader NEVER ran
    assert counter.calls["clip"] == 1                # the T5 still does
    assert seen["modelsampling"]["model"] is supplied[0]
    assert seen["img2vid"]["vae"] is supplied[2]
    assert seen["decode"]["vae"] is supplied[2]
    assert results["ckpt"] is supplied               # survived free_after_use
    assert len(results[eng._TERMINAL][0]) == 9


# --- the frame-length ladder ----------------------------------------------- #
def test_the_frame_length_ladder_floors_caps_and_snaps_to_8n_plus_1():
    """`_ltx8_frame_length` had ZERO coverage anywhere in the suite, and B3/B4
    are built on it: once ping-pong is deleted a non-`8n+1` segment becomes a
    hard RenderError, so the snap IS the contract rather than a nicety.

    Floor at the 9-frame legal minimum, clamp to the static per-hardware cap,
    then snap DOWN to the ladder. Down, never up: rounding up would hand LTX a
    length past the ceiling the tier's VRAM budget was chosen for.
    """
    f = m._ltx8_frame_length
    assert m._LTX8_MIN_FRAMES == 9
    assert f(0, 161) == 9 and f(None, 161) == 9      # floor, not a crash
    assert f(5, 161) == 9                            # below the minimum
    assert f(9, 161) == 9                            # already legal
    assert f(16, 161) == 9                           # snaps DOWN
    assert f(17, 161) == 17                          # 8*2+1
    assert f(65, 161) == 65                          # 8*8+1
    assert f(500, 161) == 161                        # the cap, itself 8n+1
    assert f(500, 60) == 57                          # a cap off the ladder
    assert f(500, 2) == 9                            # a cap under the floor
    for target in range(1, 200):
        n = f(target, 161)
        assert n >= 9 and n <= 161 and (n - 1) % 8 == 0


# --- the forward, through a real ffmpeg encode ----------------------------- #
@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A real PNG plus a redirected ComfyUI input dir. Pillow is required by
    the init-image staging (S10 fails LOUD on an unreadable source rather than
    staging raw), so this is not mockable without weakening the test."""
    Image = pytest.importorskip("PIL.Image")
    in_dir = tmp_path / "comfy_input"
    in_dir.mkdir()
    src = tmp_path / "still.png"
    Image.new("RGB", (512, 288), (40, 40, 55)).save(src)
    monkeypatch.setattr(wb, "comfy_input_dir", lambda: str(in_dir))
    return src


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_render_clip_produces_a_silent_bt709_clip_with_its_receipt(staged):
    """The whole forward: stage the still, execute the graph, encode a SILENT
    mp4, ffprobe-prove the contract, and carry the recipe receipt. None of this
    may change under the hoist -- only WHERE the checkpoint comes from.

    `frame_count` is an OBSERVATION here, not a restatement of the ask: the
    decode fake sizes its batch from the `length` the graph requested, so a
    broken `_ltx8_frame_length` or a mis-wired `length` moves this number.
    """
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)
    prepared = {"patchers": []}
    path = None

    try:
        clip = eng.canonicalize(
            eng.render_clip(_request(staged), prepared), _request(staged), {})
        path = pathlib.Path(clip["path"])
        assert path.exists()
        assert clip["frame_count"] == 9
        assert clip["has_audio"] is False
        assert clip["engine_id"] == "ltx_8gb"
        assert clip["recipe"] == m.RECIPE_LTX8_I2V
        assert clip["render_canvas"] == "512x288"
        assert clip["fps"] == 25
        assert clip["color_primaries"] == "bt709"
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_the_patcher_harvest_keeps_what_teardown_must_detach(staged):
    """`teardown` detaches exactly what `render_clip` put in
    `prepared["patchers"]`. Anything that stops being harvested stops being
    detached -- and a MODEL patcher that is never detached holds VRAM for the
    life of the ComfyUI process. Two distinct handles today: the checkpoint's
    MODEL and the per-render ModelSamplingLTXV clone."""
    np = pytest.importorskip("numpy")
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, _Counter(), n=9)
    prepared = {"patchers": []}
    path = None

    try:
        path = pathlib.Path(
            eng.render_clip(_request(staged), prepared)["out_path"])
        tags = sorted(p.tag for p in prepared["patchers"])
        assert tags == ["ckpt", "modelsampling"]
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_the_harvest_registers_a_REUSED_handle_exactly_once(staged):
    """The `seen` id-dedupe, driven under the condition the hoist creates.

    Today every render builds its own checkpoint object, so a broken dedupe
    would change nothing and this test would be decorative. So SIMULATE the
    post-hoist condition through today's path: a fake loader returning the SAME
    MODEL object every time is what a hoisted `external_results` handle will
    look like to the harvest loop. It does not exercise `prepare()` -- nothing
    in this file can, until the hoist exists -- it exercises the loop the hoist
    keeps relying on. Three renders must leave ONE checkpoint entry and three
    per-render ModelSampling clones, because `teardown` detaches every entry
    and detaching one patcher three times is not a no-op.
    """
    np = pytest.importorskip("numpy")
    shared = _FakeModel("ckpt")
    fakes = _ltx8_fakes(np, _Counter(), n=9)
    fakes["ckpt"] = _mk(lambda self, **k: (shared, object(), object()))
    eng = Ltx8gbEngine()
    eng._classes = fakes
    prepared = {"patchers": []}
    made = []
    try:
        for _ in range(3):
            made.append(pathlib.Path(
                eng.render_clip(_request(staged), prepared)["out_path"]))
        tags = sorted(p.tag for p in prepared["patchers"])
        assert tags == ["ckpt", "modelsampling", "modelsampling",
                        "modelsampling"]
        assert sum(1 for p in prepared["patchers"] if p is shared) == 1
    finally:
        for p in made:
            p.unlink(missing_ok=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_THE_LOAD_COUNT_every_render_reloads_the_checkpoint_today(staged):
    """The defect, stated as a number -- and a CONTROL on B1b, not an observer
    of it. An earlier draft of this docstring said the 3 would become a 1. It
    will not, and two independent reviewers caught that before the hoist.

    Three renders sharing ONE `prepared` dict, exactly as three segments of one
    beat do. Today the checkpoint loads three times and the T5 loads three
    times, because both loaders live in the per-segment graph. This `prepared`
    dict is HAND-BUILT and carries no `external_results`, so after the hoist
    `_build_graph` still emits the loader here and this stays 3. If it ever
    reads 1, `render_clip` has started hoisting behind the caller's back --
    which is a different, worse bug than the one B1b fixes.

    The 1-load proof belongs to the hoist chunk, which must call `prepare()`
    and reuse the dict it returns. `test_the_forward_runs_with_the_checkpoint_
    supplied_EXTERNALLY` already proves the executor half.

    A count of 3 for `decode` is what makes any future 1 meaningful. Without
    it, a hoist that accidentally rendered once would also "pass".
    """
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)
    prepared = {"patchers": []}
    made = []
    try:
        for _ in range(3):
            made.append(pathlib.Path(
                eng.render_clip(_request(staged), prepared)["out_path"]))
        assert counter.calls["ckpt"] == 3            # stays 3: no externals
        assert counter.calls["clip"] == 3            # the T5 is NOT hoisted
        assert counter.calls["modelsampling"] == 3   # stays 3: a cheap clone
        assert counter.calls["decode"] == 3          # the render itself
        assert counter.calls["sample"] == 3
    finally:
        for p in made:
            p.unlink(missing_ok=True)


def test_the_executor_is_called_with_the_current_keep_contract(staged,
                                                               monkeypatch):
    """`free_after_use=True` plus a `keep` set is what frees the ~9 GB T5 mid
    render, once `pos`/`neg` have consumed it. An external is added to `keep`
    unconditionally by the executor, so hoisting a node CHANGES when it can be
    freed. Pin today's call so that change has to be deliberate.

    THE ONE ASSERTION IN THIS FILE B1b IS EXPECTED TO FLIP is the last one:
    once `render_clip` forwards the caller's externals, `external_results`
    appears in every call's kwargs -- with or without a prepared handle in it.
    The `keep` set itself need not change (the executor unions the externals
    in), so if THAT line moves, the hoist has widened something it should not.

    No ffmpeg needed: the spy stops the forward at the executor.
    """
    np = pytest.importorskip("numpy")
    seen = {}

    class _Stop(Exception):
        pass

    def _spy(graph, classes=None, **kwargs):
        seen["graph"] = graph
        seen["kwargs"] = kwargs
        raise _Stop

    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, _Counter(), n=9)
    monkeypatch.setattr(wb, "run_graph", _spy)

    with pytest.raises(_Stop):
        eng.render_clip(_request(staged), {"patchers": []})

    assert seen["kwargs"]["free_after_use"] is True
    assert seen["kwargs"]["keep"] == {"ckpt", "modelsampling", eng._TERMINAL}
    assert "external_results" not in seen["kwargs"]   # -> supplied after B1b
    assert "ckpt" in seen["graph"] and "clip" in seen["graph"]
