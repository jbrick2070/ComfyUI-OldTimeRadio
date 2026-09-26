# -*- coding: utf-8 -*-
"""What every LTX 2.5 lane owes its operator.

Every registered LTX 2.5 lane loads native safetensors through STOCK ComfyUI
loaders. Each rule below was a real defect on this family once, invisible to
the checks that ran at the time:

  * operator-facing text named a third-party pack at a site where the failure
    had nothing to do with it, sending the reader to the wrong fix;
  * a lane that stopped pinning its text encoder without saying so wrote a
    cache entry it then rejected on every read (found live 2026-09-21);
  * the lanes had no publication shortcode, so an episode they dominate
    published its video identity as ``unk``.
"""
import pytest

from nodes._otr_video_engines import eng_ltx25
from nodes._otr_video_engines import registry as _registry
from nodes._otr_video_engines.registry import EngineUnusable
from nodes._otr_shared import shortcodes

LANES = (
    eng_ltx25.Ltx25VideoEngine,
    eng_ltx25.Ltx25NativeFoley16gbEngine,
    eng_ltx25.Ltx25NativeFoleyWideEngine,
    eng_ltx25.Ltx25NativeFoleyBlackwellEngine,
    eng_ltx25.Ltx25NativeMime16gbEngine,
    eng_ltx25.Ltx25NativeMime24gbEngine,
    eng_ltx25.Ltx25NativeAudioIn16gbEngine,
    eng_ltx25.Ltx25NativeAudioIn24gbEngine,
)


#: The audio-in lanes stage a real reference waveform inside ``_build_graph``,
#: so their graphs are built by tests/test_ltx25_audio_in_*.py with a real WAV.
#: The loader nodes they emit come from the same base method as these.
GRAPH_LANES = tuple(c for c in LANES
                    if not issubclass(c, eng_ltx25.Ltx25NativeAudioInMixin))


def _plan(**kw):
    return kw


def test_every_lane_under_test_is_registered():
    for cls in LANES:
        assert _registry.is_registered(cls.name), cls.name


@pytest.mark.parametrize("cls", LANES)
def test_the_loaders_are_stock_comfyui_nodes(cls):
    cand = cls()._node_candidates()
    assert cand["unet"] == ("UNETLoader",)
    assert cand["te"] == ("CLIPLoader",)


@pytest.mark.parametrize("cls", GRAPH_LANES)
def test_the_encoder_placement_is_a_widget_value_that_tracks_the_cache(cls):
    """The pin is requested in the graph, and the cache expects what was asked."""
    eng = cls()
    graph = eng._build_graph(_plan(text_prompt="a hand turns a dial", seed=7),
                             "still.png", 97, 832, 480)
    assert graph["te"]["inputs"]["device"] == eng._native_te_device
    assert graph["te"]["inputs"]["type"] == "ltxv"
    assert graph["unet"]["inputs"]["weight_dtype"] == "default"
    assert eng._encoder_cache_expects_cpu == (eng._native_te_device == "cpu")


@pytest.mark.parametrize("cls", LANES)
def test_the_cache_expectation_tracks_the_requested_placement(cls):
    """Every lane, audio-in included: the two flags are one fact."""
    eng = cls()
    assert eng._native_te_device in ("cpu", "default")
    assert eng._encoder_cache_expects_cpu == (eng._native_te_device == "cpu")


@pytest.mark.parametrize("cls", GRAPH_LANES)
def test_the_graph_loads_the_lanes_own_weights(cls):
    eng = cls()
    graph = eng._build_graph(_plan(text_prompt="x", seed=1), "s.png",
                             97, 832, 480)
    assert graph["unet"]["inputs"]["unet_name"] == eng._dit_name()
    assert graph["te"]["inputs"]["clip_name"] == eng._text_encoder_name()
    assert eng._dit_name().endswith(".safetensors")
    assert eng._text_encoder_name().endswith(".safetensors")


def test_the_24gb_and_blackwell_tiers_declare_the_same_pinned_canvas():
    """G2.2's per-lane pin, for the four tiers that inherit ``render_canvas``
    from ``Ltx25VideoEngine`` without overriding it -- 16gb Foley and Mime
    are pinned by name in tests/test_ltx25_video_lane.py; these four never
    got the same explicit name+dims pairing, which is what the lane
    preflight matrix (G2) reads for before trusting the declaration.
    832x480, 32-legal on both axes: see
    test_the_declared_canvas_is_832x480_and_32_legal for why (768x432
    corrupts the tensor, 1024x576 OOMs)."""
    for name, cls in (
            ("ltx25_foley_24gb", eng_ltx25.Ltx25NativeFoleyWideEngine),
            ("ltx25_foley_blackwell",
             eng_ltx25.Ltx25NativeFoleyBlackwellEngine),
            ("ltx25_mime_24gb", eng_ltx25.Ltx25NativeMime24gbEngine),
            ("ltx25_audio_in_24gb",
             eng_ltx25.Ltx25NativeAudioIn24gbEngine),
    ):
        eng = cls()
        assert eng.name == name
        assert tuple(eng.render_canvas) == (832, 480)
        assert eng.render_canvas[0] % 32 == 0
        assert eng.render_canvas[1] % 32 == 0


def test_the_silent_lane_loads_the_16gb_weight():
    eng = eng_ltx25.Ltx25VideoEngine()
    assert eng._dit_name() == eng_ltx25.LTX25_NATIVE_DIT_16GB
    assert eng._native_te_device == "cpu"
    assert eng._quant_label() == "mix4x8"


@pytest.mark.parametrize("cls", LANES)
def test_a_missing_node_names_comfyui_itself_as_the_fix(cls, monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge as _wb
    from nodes._otr_video_engines import motion_common as _MC

    monkeypatch.setattr(_wb, "node_class_mappings", lambda: {})
    monkeypatch.setattr(_MC, "assert_sage_not_patched", lambda *a, **k: None)
    with pytest.raises(EngineUnusable) as info:
        cls().assert_usable({}, {})
    msg = str(info.value)
    assert "update ComfyUI itself" in msg
    assert "UNETLoader" in msg and "CLIPLoader" in msg


@pytest.mark.parametrize("cls", LANES)
def test_every_lane_has_a_publication_shortcode(cls):
    code = shortcodes.VIDEO_LANE.get(cls.name)
    assert code, "%s would publish as 'unk'" % cls.name
    assert len(code) == 4 and code.isalnum()


def test_the_shortcodes_collide_with_nothing():
    codes = list(shortcodes.VIDEO_LANE.values())
    assert len(codes) == len(set(codes))
