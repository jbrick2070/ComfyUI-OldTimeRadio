"""``render_clip``'s cache branch, driven end to end with a stubbed graph.

WHY THIS FILE EXISTS SEPARATELY. The sibling
``tests/test_ltx25_encoder_cache.py`` tests the cache HELPERS -- the key, the
liveness check, the defensive copy, the ownership toggles -- and
``tests/test_episode_encoder_scope_wiring.py`` tests the DRIVER boundary. A
review lane pointed out that between them nothing actually called
``render_clip``, so the branch that decides HIT vs MISS, pops graph nodes, wires
``external_results`` and publishes the cache had no coverage at all. Helpers all
green and the feature still broken is exactly the shape that ships.

Everything heavy is stubbed: no graph executes, no weights load, no ffmpeg or
ffprobe runs. What is exercised is the decision logic and the transaction
boundary.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_video_engines import eng_ltx25
from nodes._otr_video_engines import wrapper_bridge as wb


class _FakeTensor:
    shape = (1, 4, 8)
    dtype = "torch.float16"

    def numel(self):
        return 32

    def sum(self):
        return self

    def item(self):
        return 1.5


class _ImageTensor:
    shape = (97, 960, 1664, 3)


def _cond():
    return ([[_FakeTensor(), {"pooled": 1}]],)


class _Patcher:
    load_device = "cpu"
    offload_device = "cpu"
    model = object()


class _Clip:
    patcher = _Patcher()


def _clip_out():
    return (_Clip(),)


@pytest.fixture()
def eng(tmp_path, monkeypatch):
    """A real engine with every heavy collaborator replaced."""
    engine = eng_ltx25.Ltx25VideoEngine()

    weight = tmp_path / "gemma-encoder.gguf"
    weight.write_bytes(b"x" * 4096)
    monkeypatch.setattr(eng_ltx25, "_resolve",
                        lambda folder, name: str(weight))

    still = tmp_path / "still.png"
    still.write_bytes(b"p" * 16)

    monkeypatch.setattr(wb, "stage_into_comfy_input", lambda p: "staged.png")
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": 0)
    monkeypatch.setattr(wb, "images_to_uint8",
                        lambda images: [object()] * eng_ltx25.R.LTX25_FRAMES)
    monkeypatch.setattr(wb, "encode_frames_to_silent_mp4",
                        lambda frames, out, fps, **kw: (str(out), len(frames)))
    monkeypatch.setattr(eng_ltx25, "ffprobe_clip_fields", lambda p: {})
    monkeypatch.setattr(eng_ltx25, "validate_silent_clip_contract",
                        lambda fields, fps: None)

    class _Probe:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            return self

        def stop(self):
            return 0

    monkeypatch.setattr(eng_ltx25._MC, "VramPeakProbe", _Probe)
    # The classes never execute; run_graph is stubbed. Resolution must not
    # reach ComfyUI's real NODE_CLASS_MAPPINGS on a CPU box.
    engine._classes = {k: type("N", (), {"FUNCTION": "f", "f": lambda s: None})
                       for k in engine._node_candidates()}
    monkeypatch.setattr(eng_ltx25, "_cpu_pinned_clip_loader", lambda base: base)
    engine._still = str(still)
    return engine


def _request(still):
    return {"shot_id": "s0", "beat_id": "b0",
            "asset_refs": {"init_image": still},
            "target_frame_count": eng_ltx25.R.LTX25_FRAMES,
            "seed_bundle": {"video_seed": 7},
            "text_prompt": "a scene"}


@pytest.fixture()
def run_graph_spy(monkeypatch):
    """Capture every ``run_graph`` call and answer it with a fake terminal."""
    calls = []

    def _fake(graph, classes=None, **kw):
        calls.append({"graph": dict(graph), "kwargs": dict(kw)})
        assert kw.get("audit_node_ids") == {
            "latent_upscale", "refine_sampler", "decode"}
        on_result = kw.get("on_result")
        external = kw.get("external_results") or {}
        results = dict(external)
        if "te" in graph:
            results["te"] = _clip_out()
            if on_result:
                on_result("te", results["te"])
        if "neg" in graph:
            results["neg"] = _cond()
            if on_result:
                on_result("neg", results["neg"])
        records = kw.get("execution_records")
        if records is not None:
            assert records == [], "only the real executor may mint evidence"
            records.extend([
                {"class_name": "LTXVLatentUpsampler",
                 "node_id": "latent_upscale", "ordinal": 1,
                 "output_shapes": [[1, 128, 13, 30, 52]]},
                {"class_name": "SamplerCustomAdvanced",
                 "node_id": "refine_sampler", "ordinal": 2,
                 "output_shapes": [[1, 128, 13, 30, 52]]},
                {"class_name": "VAEDecodeTiled",
                 "node_id": "decode", "ordinal": 3,
                 "output_shapes": [[97, 960, 1664, 3]]},
            ])
        results[eng_ltx25.Ltx25VideoEngine._TERMINAL] = (_ImageTensor(),)
        return results

    monkeypatch.setattr(wb, "run_graph", _fake)
    return calls


def _render(eng, prepared=None):
    return eng.render_clip(_request(eng._still), prepared or {"patchers": []})


# --- MISS then HIT --------------------------------------------------------- #
def test_the_first_shot_MISSES_and_publishes_both_halves(eng, run_graph_spy):
    eng.begin_encoder_scope()
    _render(eng)
    graph = run_graph_spy[0]["graph"]
    assert "te" in graph and "neg" in graph, "a MISS must execute both nodes"
    assert not run_graph_spy[0]["kwargs"].get("external_results")
    scope = eng._encoder_scope
    assert scope.get("clip") is not None and scope.get("neg") is not None, (
        "a completed render must publish BOTH halves or neither")


def test_the_second_shot_HITS_and_drops_both_nodes_from_the_graph(
        eng, run_graph_spy):
    eng.begin_encoder_scope()
    _render(eng)
    _render(eng)
    graph = run_graph_spy[1]["graph"]
    external = run_graph_spy[1]["kwargs"]["external_results"]
    assert "te" not in graph and "neg" not in graph, (
        "a HIT must remove the nodes, not merely supply them")
    assert "te" in external and "neg" in external
    assert external["te"] is eng._encoder_scope["clip"], (
        "the cached CLIP itself must be handed over, not a copy")


def test_a_HIT_hands_over_a_PRIVATE_conditioning_container(eng, run_graph_spy):
    eng.begin_encoder_scope()
    _render(eng)
    _render(eng)
    external = run_graph_spy[1]["kwargs"]["external_results"]
    cached = eng._encoder_scope["neg"]
    assert external["neg"][0] is not cached[0], "the outer list must be private"
    assert external["neg"][0][0][0] is cached[0][0][0], (
        "the tensor is shared on purpose -- copying it defeats the cache")


def test_an_id_is_never_in_BOTH_the_graph_and_the_externals(eng,
                                                            run_graph_spy):
    """``run_graph`` raises a NAMED error on that collision, so a desync
    between the pop and the external is a hard failure, not a slow render."""
    eng.begin_encoder_scope()
    _render(eng)
    _render(eng)
    for call in run_graph_spy:
        overlap = set(call["graph"]) & set(call["kwargs"].get(
            "external_results") or {})
        assert not overlap, "ids in both graph and externals: %r" % overlap


# --- the transaction boundary ---------------------------------------------- #
def test_a_FAILED_graph_publishes_NOTHING(eng, monkeypatch):
    """``on_result`` fires as each node lands, so a naive publish would commit
    a partial transaction from a graph that then died."""
    eng.begin_encoder_scope()

    def _boom(graph, classes=None, **kw):
        on_result = kw.get("on_result")
        if on_result and "te" in graph:
            on_result("te", _clip_out())      # the encoder DID land...
        raise wb.GraphExecutionError("sampler died")   # ...then the graph died

    monkeypatch.setattr(wb, "run_graph", _boom)
    with pytest.raises(wb.GraphExecutionError):
        _render(eng)
    assert eng._encoder_scope == {}, (
        "state from an unsuccessful graph was committed to the cache")


def test_no_open_scope_never_publishes_and_never_caches(eng, run_graph_spy):
    """Caching is a property of running an EPISODE. Outside one, the lane must
    behave exactly as it did before this change."""
    assert eng._encoder_scope is None
    _render(eng)
    _render(eng)
    assert eng._encoder_scope is None
    for call in run_graph_spy:
        assert "te" in call["graph"] and "neg" in call["graph"]
        assert not call["kwargs"].get("external_results")


def test_the_kill_switch_closes_an_already_open_scope(eng, run_graph_spy,
                                                      monkeypatch):
    eng.begin_encoder_scope()
    _render(eng)
    assert eng._encoder_scope.get("clip") is not None
    monkeypatch.setenv(eng_ltx25._ENCODER_CACHE_ENV, "0")
    _render(eng)
    assert eng._encoder_scope is None, (
        "flipping the switch off must RELEASE the 8.86 GiB, not just stop "
        "reading it")


# --- invalidation releases, rather than merely missing --------------------- #
def test_a_CHANGED_weight_drops_the_stale_entry_before_reloading(
        eng, run_graph_spy, tmp_path, monkeypatch):
    """Two 8.86 GiB encoders must never be resident at once. A stale entry is
    released the moment it is known to be stale, not at episode end."""
    eng.begin_encoder_scope()
    _render(eng)
    first_clip = eng._encoder_scope["clip"]

    replacement = tmp_path / "gemma-encoder-v2.gguf"
    replacement.write_bytes(b"y" * 8192)
    monkeypatch.setattr(eng_ltx25, "_resolve",
                        lambda folder, name: str(replacement))

    _render(eng)
    assert eng._encoder_scope["clip"] is not first_clip, (
        "the stale CLIP survived a weight change")
    assert "te" in run_graph_spy[1]["graph"], "the new weight must be loaded"


def test_an_unresolvable_weight_drops_the_cache_entirely(eng, run_graph_spy,
                                                         monkeypatch):
    eng.begin_encoder_scope()
    _render(eng)
    assert eng._encoder_scope.get("clip") is not None
    monkeypatch.setattr(eng_ltx25, "_resolve", lambda folder, name: "")
    _render(eng)
    assert not eng._encoder_scope.get("clip"), (
        "an unresolvable weight left a strong reference behind")


def test_a_DEAD_cached_clip_is_dropped_and_reloaded(eng, run_graph_spy):
    eng.begin_encoder_scope()
    _render(eng)

    class _OffCpu:
        load_device = "cuda:0"
        offload_device = "cuda:0"
        model = object()

    eng._encoder_scope["clip"][0].patcher = _OffCpu()
    _render(eng)
    assert "te" in run_graph_spy[1]["graph"], (
        "a CLIP that drifted off the CPU must be reloaded, not reused")
