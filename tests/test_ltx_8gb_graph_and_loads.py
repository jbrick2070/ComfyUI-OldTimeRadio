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
from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_ltx_8gb import Ltx8gbEngine
from nodes._otr_video_engines.registry import EngineUnusable

_HAS_FFMPEG = shutil.which("ffmpeg") is not None

_ENVS = (
    "OTR_LTX_8GB_CKPT", "OTR_LTX_8GB_CKPT_DIR", "OTR_LTX_8GB_CKPT_NAME",
    "OTR_LTX_8GB_T5_DIR", "OTR_LTX_8GB_T5_NAME", "OTR_LTX_8GB_T5_DEVICE",
    "OTR_LTX_8GB_TILED_VAE", "OTR_LTX_8GB_STEPS", "OTR_LTX_8GB_CFG",
    "OTR_LTX_8GB_SAMPLER", "OTR_LTX_8GB_MAX_SHIFT", "OTR_LTX_8GB_BASE_SHIFT",
    "OTR_LTX_8GB_TERMINAL", "OTR_LTX_8GB_MAX_FRAMES", "OTR_LTX_8GB_NEGATIVE",
    "OTR_LTX_8GB_VAE_TILE", "OTR_LTX_8GB_VAE_OVERLAP",
    "OTR_LTX_8GB_VAE_TEMPORAL", "OTR_LTX_8GB_VAE_TEMPORAL_OVERLAP",
    m.PREQUALIFICATION_ENV,
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
    # v2 froze tiled decode ON (measured 2026-07-27), so the DEFAULT decode
    # class is the tiled one. Still a core class -- no vendor gate.
    assert cand["decode"] == ("VAEDecodeTiled",)


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

    # Prequalification: the tiled-VAE lever is only LIVE there since B6, and the
    # property under test is that the two readers AGREE, which needs a knob that
    # can actually move. A production leg's freeze has its own test below.
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    # Set the flag EXPLICITLY in both directions. Leaving it unset would take
    # the recipe default, which v2 flipped to True -- so the "plain" half would
    # silently become a second tiled assertion.
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "0")
    assert eng._node_candidates()["decode"] == ("VAEDecode",)
    plain = eng._build_graph({}, "p.png", plan, 9, 512, 288)["decode"]["inputs"]
    assert set(plain) == {"samples", "vae"}

    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "1")
    assert eng._node_candidates()["decode"] == ("VAEDecodeTiled",)
    tiled = eng._build_graph({}, "p.png", plan, 9, 512, 288)["decode"]["inputs"]
    assert tiled["tile_size"] == 512 and tiled["overlap"] == 64
    assert tiled["temporal_size"] == 16 and tiled["temporal_overlap"] == 8


def test_a_production_leg_decodes_through_the_FROZEN_class_whatever_the_env_says(
        monkeypatch):
    """B6 -- the decode class is part of the frozen recipe on a production leg.

    An episode is submitted to an ALREADY-BOOTED server (PBUG-20260723-02), so
    `OTR_LTX_8GB_TILED_VAE` in that server's environment is whatever the box was
    started with -- possibly months ago, possibly by a different sweep. It must
    not decide which decoder a published render used. Both readers follow the
    RECIPE, so they still agree; the freeze does not split them.

    v2 (measured 2026-07-27) froze tiled decode ON, so the env value here is
    "0" -- it has to oppose the recipe or this proves nothing."""
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "0")     # no prequalification

    assert m.LTX8_RECIPE["tiled_vae"] is True            # what it opposes
    assert eng._tiled_vae() is True
    assert eng._node_candidates()["decode"] == ("VAEDecodeTiled",)
    inputs = eng._build_graph({}, "p.png", plan, 9, 512, 288)["decode"]["inputs"]
    assert set(inputs) == {"samples", "vae", "tile_size", "overlap",
                           "temporal_size", "temporal_overlap"}


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


def test_the_RESOLVED_RECIPE_VALUES_reach_the_nodes_that_consume_them(
        monkeypatch):
    """A configured value with a severed channel is this build's recurring
    defect class -- three instances in four days. `_resolve_render_config` is
    the ONE authority for steps / cfg / shift / sampler / terminal, so this
    pins that each one ARRIVES at the node that consumes it, and that the
    geometry the caller asked for is what `LTXVImgToVideo` is told.

    It compares against the resolver, not against literals, on purpose: the
    VALUES are the recipe's to change, the DELIVERY is not. What this catches
    is a crossed key (`"steps": cfg["max_shift"]`) or a hard-coded literal --
    the shapes that survive every other assertion in this file.

    SINCE B6 THAT NEEDS THE CONSENT ACT. With the recipe frozen, a clean
    environment makes the resolver return the frozen constants, so a
    `_build_graph` that hard-coded `8` / `1.0` / `"euler"` would compare EQUAL
    to it and the literal-catching half of this test would be decorative.
    Opening the knobs and setting values that differ from every frozen one
    restores the discrimination.
    """
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    for env, value in (("OTR_LTX_8GB_STEPS", "13"),
                       ("OTR_LTX_8GB_CFG", "2.5"),
                       ("OTR_LTX_8GB_MAX_SHIFT", "3.25"),
                       ("OTR_LTX_8GB_BASE_SHIFT", "1.75"),
                       ("OTR_LTX_8GB_TERMINAL", "0.35"),
                       ("OTR_LTX_8GB_SAMPLER", "dpmpp_2m")):
        monkeypatch.setenv(env, value)

    eng = Ltx8gbEngine()
    cfg = eng._resolve_render_config()
    # Every value must differ from the frozen one, or a literal would still
    # match and the test would be back where it started.
    for key in ("steps", "cfg", "max_shift", "base_shift", "terminal",
                "sampler"):
        assert cfg[key] != m.LTX8_RECIPE[key], key

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
def test_the_render_LENGTH_is_owned_by_the_declared_contract_now():
    """`_ltx8_frame_length` is DELETED (B4) and its ladder job moved HERE.

    The old helper snapped an ask DOWN and clamped it to the env cap, which is
    precisely why the ping-pong existed -- something had to put the missing
    frames back. The adapter now asks its OWN declared contract for the
    smallest legal length at or above the ask, renders that many real frames,
    and trims the surplus. Same rungs, opposite direction, and the rungs are
    now the same object the coverage planner partitions against, so the two
    cannot disagree about what a legal length is.

    This test replaces the ladder test rather than deleting it: the arithmetic
    still needs pinning, it just has a different owner.
    """
    contract = m.Ltx8gbEngine.frame_contract
    assert m._LTX8_MIN_FRAMES == 9
    # The module constant and the contract declare the same floor in two
    # places. The constant now has exactly one job -- the lower bound
    # `_resolve_render_config` range-checks OTR_LTX_8GB_MAX_FRAMES against --
    # so if the contract's floor ever moves alone, that config check would
    # start accepting a cap below the real minimum and the refusal would fire
    # for a reason nobody could trace back. Post-code panel, both seats.
    assert m._LTX8_MIN_FRAMES == int(contract.min_frames)
    assert not hasattr(m, "_ltx8_frame_length"), (
        "the retired helper is back -- if a length rule is needed again it "
        "belongs on the contract, not beside it")
    f = contract.smallest_legal_at_least
    assert f(0) == 9 and f(5) == 9                   # floor at the minimum
    assert f(9) == 9                                 # already legal
    assert f(10) == 17                               # snaps UP now, not down
    assert f(16) == 17
    assert f(65) == 65                               # 8*8+1
    assert f(100) == 105                             # the off-grid case
    assert f(161) == 161                             # the declared ceiling
    assert f(162) is None                            # past it: no legal rung
    for target in range(1, 162):
        n = f(target)
        assert n >= target and n >= 9 and n <= 161 and (n - 1) % 8 == 0


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
    # THE FLIP: pre-B1b this kwarg was absent. It is now always forwarded, and
    # it is None exactly when the caller prepared nothing -- which is why the
    # loader nodes are still in the graph on the very next line.
    assert seen["kwargs"]["external_results"] is None
    assert "ckpt" in seen["graph"] and "clip" in seen["graph"]


# --- B1b: the beat-scoped loader hoist ------------------------------------- #
class _LeaseSpy:
    """Records the shared single-heavy-engine lease instead of taking it.

    A unit test that took the real cross-process lease would block a live
    render, and a stranded one blocks every later beat for its full 120s
    timeout. Recording it is also what lets these tests assert WHEN it was
    taken, which is the whole point of the ordering.

    Patched onto the three lease FUNCTIONS of the gpu-resources module, never
    over the module itself: the same module owns `nvml_available`, which the
    VRAM peak probe calls on every render.
    """

    def __init__(self):
        self.acquired = 0
        self.released = []
        self.waits = 0

    def acquire(self, timeout_s=None):
        self.acquired += 1
        return "LEASE-%d" % self.acquired

    def release(self, lease):
        self.released.append(lease)

    def wait_until_stable(self, attempts=3, sleep_s=2.0):
        self.waits += 1


class _Rig:
    def __init__(self, eng, counter, fakes, lease, ckpt):
        self.eng = eng
        self.counter = counter
        self.fakes = fakes
        self.lease = lease
        self.ckpt = ckpt


@pytest.fixture
def hoistable(tmp_path, monkeypatch):
    """An engine that can actually run `prepare()` on a box with no ComfyUI.

    Three substitutions, each deliberate and each leaving the code under test
    intact: model files resolve inside `tmp_path` through the same
    `_resolve_model_file` seam the session-config suite uses; node classes come
    back as fakes so `load()` does not need a live ComfyUI registry; and the
    GPU lease is recorded rather than taken.

    The 4 GiB checkpoint floor is lowered to 1 KiB so a fixture does not have
    to write 4 GiB. Its VALUE is not what these tests are about -- its POSITION
    is, and `test_the_checkpoint_FLOOR_is_checked_BEFORE_the_lease_and_the_load`
    proves that against this same lowered floor.
    """
    np = pytest.importorskip("numpy")
    models = tmp_path / "models"
    models.mkdir()
    ckpt = models / m._LTX8_DEFAULT_CKPT
    ckpt.write_bytes(b"c" * 4096)
    (models / m._LTX8_DEFAULT_T5).write_bytes(b"t" * 1024)
    monkeypatch.setattr(m, "_LTX8_CKPT_MIN_BYTES", 1024)

    counter = _Counter()
    fakes = _ltx8_fakes(np, counter, n=9)
    monkeypatch.setattr(wb, "resolve_graph_classes", lambda candidates: fakes)
    lease = _LeaseSpy()
    monkeypatch.setattr(mc._GR, "acquire", lease.acquire)
    monkeypatch.setattr(mc._GR, "release", lease.release)
    monkeypatch.setattr(mc._GR, "wait_until_stable", lease.wait_until_stable)

    eng = Ltx8gbEngine()
    monkeypatch.setattr(
        eng, "_resolve_model_file",
        lambda categories, name, env_dir: (
            str(models / name) if (models / name).exists() else None))
    return _Rig(eng, counter, fakes, lease, ckpt)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_prepare_loads_the_checkpoint_ONCE_for_the_whole_beat(hoistable, staged):
    """THE PAYOFF, and the debt the pre-hoist net named and could not pay.

    Three renders through ONE prepared session, exactly as three segments of a
    beat do -- but this `prepared` came from `prepare()`, so it carries the
    hoisted handle and the graph omits the loader. One checkpoint load for the
    beat instead of three, which at 6.34 GiB is the difference the whole chunk
    exists for.

    The other counts are what make the 1 mean something. `clip` stays at 3
    because the T5 is deliberately NOT hoisted; `decode` and `sample` stay at 3
    because three segments really rendered. A hoist that accidentally rendered
    once would show 1 everywhere and would be caught here, not on the GPU.
    """
    rig = hoistable
    prepared = rig.eng.prepare({}, {}, {})
    made = []
    try:
        for _ in range(3):
            made.append(pathlib.Path(
                rig.eng.render_clip(_request(staged), prepared)["out_path"]))
        assert rig.counter.calls["ckpt"] == 1            # the whole point
        assert rig.counter.calls["clip"] == 3            # T5 NOT hoisted
        assert rig.counter.calls["modelsampling"] == 3   # a cheap clone
        assert rig.counter.calls["decode"] == 3          # three real renders
        assert rig.counter.calls["sample"] == 3
    finally:
        for p in made:
            p.unlink(missing_ok=True)
        rig.eng.teardown(prepared)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_the_hoisted_checkpoint_is_harvested_in_prepare_and_detached_once(
        hoistable, staged):
    """A hoisted MODEL that nothing harvests is VRAM held for the life of the
    ComfyUI process. `prepare()` registers it the moment it lands -- before it
    has returned, and before any render -- so a failure between `prepare()` and
    the first `render_clip` (the baseline identity read is one) still has an
    owner for it.

    Then three renders must add three ModelSampling clones and NOT a second
    checkpoint entry: teardown detaches every entry, and detaching one patcher
    four times is not a no-op.
    """
    rig = hoistable
    prepared = rig.eng.prepare({}, {}, {})
    assert [p.tag for p in prepared["patchers"]] == ["ckpt"]
    ckpt_model = prepared["patchers"][0]
    # The harvested patcher must be the SAME OBJECT the segments will render
    # through. A copy would detach something no render ever used.
    assert ckpt_model is prepared["external_results"]["ckpt"][0]
    made = []
    try:
        for _ in range(3):
            made.append(pathlib.Path(
                rig.eng.render_clip(_request(staged), prepared)["out_path"]))
        assert sorted(p.tag for p in prepared["patchers"]) == [
            "ckpt", "modelsampling", "modelsampling", "modelsampling"]
        assert sum(1 for p in prepared["patchers"] if p is ckpt_model) == 1
    finally:
        for p in made:
            p.unlink(missing_ok=True)
        rig.eng.teardown(prepared)
    assert ckpt_model.detached == 1
    assert prepared["patchers"] == []
    assert "external_results" not in prepared
    assert rig.lease.released == ["LEASE-1"]


def test_the_segment_graph_OMITS_the_loader_the_caller_supplied():
    """The other half of the hoist, and the one the pre-hoist net could not
    reach: with a handle supplied, the `ckpt` DEFINITION is gone while every
    wire that reads it stays. Omitting is not an optimisation -- `run_graph`
    REFUSES a graph that also defines an id the caller supplied.

    `clip` must still be here. The T5 is not hoisted, and a hoist that took it
    too would pin ~9 GB for the whole beat.
    """
    eng = Ltx8gbEngine()
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    g = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288,
                         external_results={"ckpt": (object(), object(), object())})

    assert "ckpt" not in g
    assert "clip" in g
    assert g["modelsampling"]["inputs"]["model"] == wb.Wire("ckpt", 0)
    assert g["img2vid"]["inputs"]["vae"] == wb.Wire("ckpt", 2)
    assert g["decode"]["inputs"]["vae"] == wb.Wire("ckpt", 2)
    order = wb._topo_order(g, external_keys={"ckpt"})
    assert "ckpt" not in order and order[-1] == eng._TERMINAL


def test_the_checkpoint_FLOOR_is_checked_BEFORE_the_lease_and_the_load(
        hoistable):
    """The blocker this chunk had to clear, proved by ORDER rather than by the
    refusal alone.

    The 4 GiB floor is the only check that tells a truncated download from the
    real all-in-one, and it lived in `assert_usable` -- which runs PER SEGMENT,
    after the session is already open. Hoisting the load into `prepare()` moved
    the real load ahead of it. So: the loader must never have run, and the
    cross-process lease must never have been taken (C-3: a raise after the
    lease strands it for the life of the server).
    """
    rig = hoistable
    rig.ckpt.write_bytes(b"c" * 8)

    with pytest.raises(EngineUnusable) as excinfo:
        rig.eng.prepare({}, {}, {})

    assert "is only 8 bytes" in str(excinfo.value)
    assert "re-fetch" in str(excinfo.value)
    assert rig.lease.acquired == 0, "refused AFTER taking the lease -- C-3"
    assert "ckpt" not in rig.counter.calls, "the loader ran before the floor"


def test_assert_usable_still_owns_the_same_floor_with_the_same_message(
        hoistable):
    """CONTROL on the extraction. The floor moved into a shared helper; the
    single-clip preflight must still raise it, with the same reason and the
    same wording, and still AFTER the missing-file verdict."""
    rig = hoistable
    rig.ckpt.write_bytes(b"c" * 8)

    with pytest.raises(EngineUnusable) as excinfo:
        rig.eng.assert_usable({}, {})

    assert "is only 8 bytes" in str(excinfo.value)
    assert "0.9.8 distilled 2B all-in-one" in str(excinfo.value)


def test_an_honest_checkpoint_still_PREPARES(hoistable):
    """CONTROL. A tightening that also refuses honest input is not a fix: the
    floor, the arity gate and the tripwire must all wave a good config
    through."""
    rig = hoistable
    prepared = rig.eng.prepare({}, {}, {})
    try:
        assert rig.counter.calls["ckpt"] == 1
        assert set(prepared["external_results"]) == {"ckpt"}
        assert len(prepared["external_results"]["ckpt"]) == 3
        assert rig.lease.acquired == 1
    finally:
        rig.eng.teardown(prepared)


def test_a_FAILING_hoist_releases_the_lease_and_reraises_the_cause(hoistable):
    """`prepare()` takes the lease before it loads, so anything that raises
    after that point must give it back or every later beat blocks its full
    timeout. And the error the caller sees must be the CAUSE -- the teardown
    call is wrapped precisely so a raising `unload()` cannot replace it."""
    rig = hoistable
    rig.fakes["ckpt"] = _mk(
        lambda self, **k: (_ for _ in ()).throw(
            RuntimeError("checkpoint vanished mid-load")))

    with pytest.raises(wb.GraphExecutionError, match="vanished mid-load"):
        rig.eng.prepare({}, {}, {})

    assert rig.lease.released == ["LEASE-1"]


def test_a_RAISING_unload_cannot_mask_the_hoist_failure(hoistable,
                                                        monkeypatch):
    """The reason the teardown call inside `prepare()` has its own try/except.
    `unload()` is engine-overridable; one that raises during the unwind would
    otherwise become the exception the operator sees, and the real cause -- the
    thing that actually broke the beat -- would never be reported."""
    rig = hoistable
    rig.fakes["ckpt"] = _mk(
        lambda self, **k: (_ for _ in ()).throw(RuntimeError("the real cause")))
    monkeypatch.setattr(
        type(rig.eng), "unload",
        lambda self: (_ for _ in ()).throw(RuntimeError("noisy unload")))

    with pytest.raises(wb.GraphExecutionError, match="the real cause"):
        rig.eng.prepare({}, {}, {})


def test_prepare_REFUSES_a_checkpoint_that_yields_too_few_outputs(hoistable):
    """The 0.9.8 all-in-one must give MODEL(0), CLIP(1) and the embedded
    VAE(2); the segment graph wires slot 2. A two-output checkpoint would sail
    through `prepare()` and fail deep inside the first segment's `img2vid`
    instead, with nothing naming the cause.

    The handle it DID produce must still be detached -- which is what proves
    `on_result` takes ownership as each output lands rather than after the
    graph returns."""
    rig = hoistable
    short = _FakeModel("short")
    rig.fakes["ckpt"] = _mk(lambda self, **k: (short, object()))

    with pytest.raises(wb.GraphExecutionError, match="returned 2 output"):
        rig.eng.prepare({}, {}, {})

    assert short.detached == 1, (
        "the loaded MODEL was never detached -- on_result did not take "
        "ownership before the refusal, so this is a live VRAM leak")
    assert rig.lease.released == ["LEASE-1"]


def test_teardown_clears_the_external_handles_BEFORE_the_base_runs(hoistable,
                                                                   monkeypatch):
    """Ordering, not just end state. The base teardown detaches, unloads,
    releases the lease and then waits for machine-wide VRAM to settle. If
    `external_results` were still holding the MODEL and the embedded VAE
    through that sequence, the stability wait would be watching its own
    referent."""
    rig = hoistable
    seen = {}
    base = mc.MotionEngineBase.teardown

    def _spy(self, prepared):
        seen["had_external"] = "external_results" in (prepared or {})
        return base(self, prepared)

    monkeypatch.setattr(mc.MotionEngineBase, "teardown", _spy)
    prepared = rig.eng.prepare({}, {}, {})
    assert "external_results" in prepared

    rig.eng.teardown(prepared)
    assert seen["had_external"] is False
    assert rig.lease.released == ["LEASE-1"]


# --- B4: the ping-pong is gone, and what stands in its place ---------------- #
#
# The CLIP-FILL block mirror-extended a short render up to the ask. That let a
# render the engine could not deliver PASS the plan-vs-output count gate
# wearing the right number -- part real motion, part mirrored frames -- and on
# this lane, which declares strict_first_frame, the next chained segment would
# begin on a MIRRORED tail frame. WAN keeps its extension; it renders short on
# purpose (PBUG-20260723-02).

def test_an_ask_ABOVE_the_cap_is_refused_before_anything_is_staged(
        staged, monkeypatch):
    """The env-vs-plan disagreement, and it must cost nothing to discover.

    A server booted with a low OTR_LTX_8GB_MAX_FRAMES against a plan that asked
    for more is the exact case the ping-pong used to launder: render short, pad
    back to the ask, pass the count gate. Now it refuses -- and refuses BEFORE
    an init image is written or a graph is built, because a request this
    adapter cannot render is knowable from pure numbers.
    """
    np = pytest.importorskip("numpy")
    counter = _Counter()
    monkeypatch.setenv("OTR_LTX_8GB_MAX_FRAMES", "30")
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)

    def _must_not_stage(*_a, **_k):
        raise AssertionError("the init image was staged before the refusal")

    monkeypatch.setattr(eng, "_materialize_init_image", _must_not_stage)
    with pytest.raises(wb.GraphExecutionError) as excinfo:
        eng.render_clip(_request(staged, frames=49), {"patchers": []})
    assert "NO FALLBACK" in str(excinfo.value)
    assert "49" in str(excinfo.value)
    # ORDERING, pinned two ways: no graph node ran (the counter is empty) and
    # the init image was never written (the staging detonates). Move the
    # refusal below either one and this goes red rather than merely slower.
    assert dict(counter.calls) == {}


def test_an_ask_ABOVE_the_declared_maximum_is_refused_too(staged):
    """No env involved -- 200 frames is past the contract's own 161 ceiling,
    so there is no legal rung at all and `smallest_legal_at_least` says None.
    A caller that asks for it (`render_single`, an HTTP body) gets an error
    instead of a clip that is 161 real frames wearing a 200-frame count."""
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)
    with pytest.raises(wb.GraphExecutionError):
        eng.render_clip(_request(staged, frames=200), {"patchers": []})
    assert dict(counter.calls) == {}


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_an_OFF_GRID_ask_renders_the_next_rung_up_and_trims_REAL_frames(
        staged):
    """100 is not 8n+1. The engine renders 105 real frames and keeps 100.

    Every frame delivered is a rendered frame in order: no mirror, no loop, no
    held frame. The decode fake sizes its batch from the `length` the graph
    actually asked for, so 105 here is an OBSERVATION of what was wired -- if
    the engine went back to snapping DOWN, the graph would be asked for 97 and
    this goes red on both numbers.
    """
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)
    path = None
    try:
        raw = eng.render_clip(_request(staged, frames=100), {"patchers": []})
        path = pathlib.Path(raw["out_path"])
        assert raw["frame_count"] == 100
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_an_ON_GRID_ask_is_rendered_exactly_and_never_trimmed(staged):
    """The planned path: every coverage-plan segment length is already legal,
    so nothing is rendered spare and nothing is cut. This is the CONTROL for
    the trim test above -- if trimming ever fired here it would be silently
    shortening a planned segment."""
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)
    path = None
    try:
        raw = eng.render_clip(_request(staged, frames=65), {"patchers": []})
        path = pathlib.Path(raw["out_path"])
        assert raw["frame_count"] == 65
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_an_ask_BELOW_the_declared_minimum_is_left_at_the_minimum(staged):
    """The trim stops at the contract's own floor.

    9 is the shortest thing this adapter renders, so a 5-frame ask gets 9 real
    frames rather than a 5-frame clip -- cutting below the declared minimum
    would be inventing a length the declaration forbids, and it is the one
    place the trim must NOT fire.
    """
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)
    path = None
    try:
        raw = eng.render_clip(_request(staged, frames=5), {"patchers": []})
        path = pathlib.Path(raw["out_path"])
        assert raw["frame_count"] == 9
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def test_a_SHORT_decode_is_refused_rather_than_padded(staged, monkeypatch):
    """The invariant the pad was standing in front of.

    The old block padded whenever the decode came up short FOR ANY REASON, not
    just a cap disagreement. Deleting it without this check would send the
    short clip on to the composite, which loop-fills it with the underrun
    warning suppressed -- trading a logged mirror for a silent jump-cut
    repeat. The pre-code panel found that, and this is the test for it.
    """
    np = pytest.importorskip("numpy")
    counter = _Counter()
    eng = Ltx8gbEngine()
    classes = _ltx8_fakes(np, counter, n=9)
    short = np.zeros((7, 24, 32, 3), dtype="float32")     # asked 9, decoded 7
    classes["decode"] = _mk(lambda self, **kw: (short,))
    eng._classes = classes
    with pytest.raises(wb.GraphExecutionError) as excinfo:
        eng.render_clip(_request(staged, frames=9), {"patchers": []})
    assert "decoded 7" in str(excinfo.value)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_the_ping_pong_helper_is_never_called_on_this_lane(staged,
                                                           monkeypatch):
    """Behavioural, not a source grep: make the helper explode and render.

    A grep for the call site passes a reversed-argument mutation and passes a
    call that is merely unreachable. Detonating the helper proves the lane does
    not reach it -- while `wrapper_bridge.extend_frames_to_target` itself stays
    exactly where it is for WAN, which is the point of a lane-specific rip.
    """
    np = pytest.importorskip("numpy")

    def _boom(*_a, **_k):
        raise AssertionError("ltx_8gb reached the ping-pong extender")

    monkeypatch.setattr(wb, "extend_frames_to_target", _boom)
    counter = _Counter()
    eng = Ltx8gbEngine()
    eng._classes = _ltx8_fakes(np, counter, n=9)
    path = None
    try:
        raw = eng.render_clip(_request(staged, frames=100), {"patchers": []})
        path = pathlib.Path(raw["out_path"])
        assert raw["frame_count"] == 100
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def test_the_shared_extender_still_EXISTS_for_the_lanes_that_need_it():
    """The rip is lane-specific. `wrapper_bridge.extend_frames_to_target` is
    WAN's mechanism for filling a beat from a deliberately short native render,
    and deleting it would take the shipped 8GB WAN tier with it."""
    assert callable(getattr(wb, "extend_frames_to_target", None))
