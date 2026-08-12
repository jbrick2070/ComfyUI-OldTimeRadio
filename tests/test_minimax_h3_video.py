"""LANE 19 -- `minimax_h3_video` (`h3_low_video`), the MiniMax H3 FL2VA lane.

The two things this lane owns that nothing before it had, and one correction:

* **A frame grid that is the MODEL's, not the canvas's.** The tests below
  RE-DERIVE the 17k+5 rule from an independent copy of the installed node's own
  ``align_frame_count`` rather than comparing the adapter against a transcribed
  list. A test that asserts a tuple equals the tuple in the module next to it
  proves only that nobody typo'd; this one fails if the rule itself moves.
* **A 24 -> 25 delivery conversion.** The local encoder can only LABEL a rate,
  never resample, so this is the one lane where a frame-batch index map stands
  between the render and an honest duration. 192 model frames must deliver 200
  canvas frames at exactly 8.000 s.
* **The boot contract's reserve clamp.** Lane 19 moved ``h3``'s
  ``reserve_vram_gb`` from None to 12.0 on the lab receipts; it is pinned here
  so a later tidy-up cannot quietly drop the knob that decides whether this lane
  fits at all.

CPU-safe: no CUDA, no ComfyUI server, no weights, no render.
"""
from __future__ import annotations

import ast
import inspect
import json
import os

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_shared import boot_contracts as bc
from nodes._otr_shared import public_engines as pub
from nodes._otr_shared import still_plan_helpers as sph
from nodes._otr_video_engines import eng_minimax_h3 as h3
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.registry import EngineUnusable


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = os.path.join(
    REPO_ROOT, "config", "profiles", "otr_h3_low_video.json")

#: The boot contract this lane requires, as a profile fragment. Every test that
#: needs preflight to get PAST the boot gate uses this, so a test can never pass
#: by accidentally selecting `default`.
H3_PROFILE = {"launch": {"boot_contract": "h3"}}

#: The server state the `h3` contract is satisfied by, injected rather than
#: probed (there is no ComfyUI in the CPU suite).
H3_SERVER_STATE = {"available": True, "reserve_vram_gb": 12.0,
                   "disable_pinned_memory": True, "sage_attention": False}


@pytest.fixture()
def engine():
    return vreg.get_engine("minimax_h3_video")


def _profile():
    with open(PROFILE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# THE GRID -- derived independently, never transcribed
# ---------------------------------------------------------------------------
def _independent_align(n):
    """The installed node's rule, written from its source rather than imported.

    ``comfy_extras/nodes_minimax_h3.py`` cannot be imported here (it pulls in
    torch, torchaudio and ComfyUI's node registry), so the RULE is restated --
    but restated from the node, not from the adapter. If the adapter's copy ever
    drifts from the node's arithmetic, these two disagree.
    """
    while n % 17 != 5:
        n += 1
    return n


def test_the_grid_rule_matches_the_installed_nodes_own_arithmetic():
    for n in range(0, 400):
        assert h3.align_frame_count(n) == _independent_align(n), n


def test_every_published_rung_is_a_fixed_point_of_the_grid_rule():
    for model_frames in h3.H3_MODEL_RUNGS:
        assert h3.align_frame_count(model_frames) == model_frames
        assert model_frames % 17 == 5


def test_the_model_ladder_spans_the_trained_band_and_stops_there():
    assert h3.H3_MODEL_RUNGS[0] == 124
    assert h3.H3_MODEL_RUNGS[-1] == 362
    # 107 is the rung BELOW the trained floor, and it is the one the evidence
    # manifest records as MEASURED_BELOW_RANGE_FAILURE. It is on the grid, so
    # only the trained band keeps it out of the menu.
    assert _independent_align(107) == 107
    assert 107 not in h3.H3_MODEL_RUNGS


# ---------------------------------------------------------------------------
# THE 24 -> 25 CONVERSION
# ---------------------------------------------------------------------------
def test_the_canvas_menu_is_the_24_to_25_conversion_of_the_model_grid():
    assert h3.H3_CANVAS_RUNGS == tuple(
        h3.canvas_frames_for_model(m) for m in h3.H3_MODEL_RUNGS)
    # The value the spec published, independently reached.
    assert h3.H3_CANVAS_RUNGS == (
        129, 146, 164, 182, 200, 217, 235, 253, 270, 288, 306, 323, 341, 359,
        377)


def test_192_model_frames_deliver_200_canvas_frames_at_exactly_8_seconds():
    assert h3.canvas_frames_for_model(192) == 200
    assert 192 / float(h3.H3_MODEL_FPS) == 8.0
    assert 200 / float(h3.H3_CANVAS_FPS) == 8.0


def test_the_conversion_never_publishes_a_frame_past_the_end_of_the_render():
    """FLOOR, not round: a canvas frame must exist inside the rendered duration.

    Rounding up would invent a frame the model never produced -- exactly the
    manufactured-frame condition ``native_frame_count`` exists to expose.
    """
    for model_frames in h3.H3_MODEL_RUNGS:
        canvas = h3.canvas_frames_for_model(model_frames)
        assert canvas / float(h3.H3_CANVAS_FPS) <= (
            model_frames / float(h3.H3_MODEL_FPS)) + 1e-9
        assert (canvas + 1) / float(h3.H3_CANVAS_FPS) > (
            model_frames / float(h3.H3_MODEL_FPS))


def test_the_model_lookup_is_the_exact_inverse_over_the_published_menu():
    for model_frames, canvas in zip(h3.H3_MODEL_RUNGS, h3.H3_CANVAS_RUNGS):
        assert h3.model_frames_for_canvas(canvas) == model_frames
    # A length that is not on the menu has no model rung, and answering None is
    # what makes render_clip refuse instead of inventing a length the node would
    # then silently snap somewhere else.
    assert h3.model_frames_for_canvas(130) is None
    assert h3.model_frames_for_canvas(0) is None


def test_the_index_map_selects_real_rendered_frames_only():
    for model_frames, canvas in zip(h3.H3_MODEL_RUNGS, h3.H3_CANVAS_RUNGS):
        idx = h3.canvas_index_map(model_frames, canvas)
        assert len(idx) == canvas
        assert idx[0] == 0                      # the first frame is the first
        assert min(idx) >= 0
        assert max(idx) <= model_frames - 1     # never past the render
        assert list(idx) == sorted(idx)         # never runs backwards
        # It SELECTS; it never synthesises. Every emitted frame is a rendered
        # frame, which is what lets the receipt say native_frame_count == n.
        assert set(idx) <= set(range(model_frames))


def test_the_conversion_REPEATS_about_one_frame_in_24_and_that_is_stated():
    """25/24 is not 1, and the receipt must not read as 129 distinct frames.

    Drawing 129 canvas frames from 124 source frames MUST show some of them
    twice -- 5 repeats at the base rung, 15 at f377. Nothing is invented (every
    delivered frame is a rendered frame) and the duration is exact, which is why
    the lane stamps ``native_frame_count = emitted`` and ``extension_mode:
    none``. This test pins the ARITHMETIC so the claim in the receipt cannot
    quietly become wrong, and so nobody "fixes" the stamp to 124 -- which
    ``acceptance.py`` would read as "frames 124..128 are the manufactured ones",
    since its producer contract is that manufactured frames are appended at the
    TAIL. A rate conversion's repeats are spread evenly through the clip.
    """
    from collections import Counter

    for model_frames, canvas in zip(h3.H3_MODEL_RUNGS, h3.H3_CANVAS_RUNGS):
        idx = h3.canvas_index_map(model_frames, canvas)
        counts = Counter(idx)
        repeats = sum(v - 1 for v in counts.values() if v > 1)
        # Every delivered frame comes from a real rendered frame ...
        assert set(counts) <= set(range(model_frames))
        # ... and the repeats are exactly the frames the rate ratio demands.
        assert repeats == canvas - len(counts)
        # About one in 24, never a mirrored block: no source frame is shown
        # more than twice at this ratio.
        assert max(counts.values()) <= 2, (model_frames, canvas)
    assert sum(v - 1 for v in Counter(h3.canvas_index_map(124, 129)).values()
               if v > 1) == 5


def test_the_index_map_ends_within_one_frame_of_the_end_of_the_render():
    """It does NOT always land exactly on the final model frame, and that is
    correct rather than a rounding slip.

    The canvas count is a FLOOR, so the published clip is very slightly shorter
    than the render: at model f141 the render is 5.875 s and the 146-frame
    delivery is 5.840 s, which leaves the last model frame unpublished. Dropping
    under one frame at the tail is the price of never publishing a frame that
    does not exist; the alternative -- rounding the count up -- manufactures one.
    """
    for model_frames, canvas in zip(h3.H3_MODEL_RUNGS, h3.H3_CANVAS_RUNGS):
        last = h3.canvas_index_map(model_frames, canvas)[-1]
        assert 0 <= (model_frames - 1) - last <= 1, (model_frames, canvas, last)


def test_the_integer_map_is_exactly_round_and_no_tie_can_ever_arise():
    """The map is integer arithmetic so half-to-EVEN can never reach it.

    Python's ``round`` is banker's rounding, so a tie would resolve differently
    at even and odd indices. There is no tie here -- ``j*24/25`` is never an
    exact half -- and this test asserts BOTH halves of that: the integer form
    agrees with ``round`` everywhere, and no index is ever a tie in the first
    place, so the agreement does not depend on which way ties go.
    """
    for j in range(0, 400):
        assert (48 * j + 25) // 50 == round(j * 24 / 25.0)
        assert (j * 24 * 2) % 50 != 25          # never an exact .5


# ---------------------------------------------------------------------------
# THE CONTRACT AND THE CANVAS
# ---------------------------------------------------------------------------
def test_the_lane_DECLARES_its_canvas_864x480(engine):
    assert engine.render_canvas == (864, 480)
    w, h = engine.render_canvas
    # Every H3 node takes width/height at step=32 and the latent is H/16 x W/16.
    assert w % 32 == 0 and h % 32 == 0


def test_the_profile_carries_the_same_canvas_as_the_declaration(engine):
    """The declaration is applied LAST and overrules the profile, so a profile
    saying something else is a config an operator reads and is misled by."""
    render = _profile()["render"]
    assert (render["canvas_w"], render["canvas_h"]) == engine.render_canvas
    assert render["fps"] == engine.target_fps


def test_the_contract_publishes_canvas_frames_at_the_canvas_rate(engine):
    contract = fc.frame_contract_for(engine)
    assert contract.discrete_frames == h3.H3_CANVAS_RUNGS
    assert contract.native_fps == 25
    assert engine.target_fps == 25
    assert contract.allow_tail_trim is True


def test_continuity_is_DECLARED_not_defaulted(engine):
    assert fc.declares_continuity_kwarg(engine)
    assert fc.frame_contract_for(engine).continuity == (
        fc.CONTINUITY_STRICT_FIRST_FRAME)
    assert fc.can_chain(engine)


def test_a_sub_floor_ask_is_lifted_to_the_trained_floor_never_cut_below(engine):
    contract = fc.frame_contract_for(engine)
    assert contract.smallest_legal_at_least(1) == 129
    assert contract.smallest_legal_at_least(129) == 129
    assert contract.smallest_legal_at_least(130) == 146
    # Above the ceiling there is no rung at all, which is what makes the engine
    # refuse rather than render something shorter than the ask.
    assert contract.smallest_legal_at_least(378) is None
    assert not contract.is_legal_length(107)


# ---------------------------------------------------------------------------
# THE GRAPH -- and the audio half that is deliberately not in it
# ---------------------------------------------------------------------------
def test_the_graph_carries_NO_audio_decoder_at_all(engine):
    """H3 emits audio natively; this lane's silence starts in the topology.

    The lab's official H3 graph decodes both halves of the NestedTensor latent
    (``VAEDecodeAudio`` off a second ``VAELoader``) and muxes them with
    ``CreateVideo``. None of that may appear here: OTR's V-1 invariant is that
    only ``OTR_MasterAudioMux`` ever adds audio.

    STRUCTURAL, NOT LEXICAL, and the first draft of this test proves why. It
    grepped the adapter's SOURCE for "VAEDecodeAudio" and failed -- on the
    ``_build_graph`` docstring that EXPLAINS why the audio decoder is absent.
    The same instrument that lets a comment satisfy a gate lets a comment break
    one. Ask the graph instead.
    """
    candidates = engine._node_candidates()
    flat = {c for pair in candidates.values() for c in pair}
    for forbidden in ("VAEDecodeAudio", "CreateVideo", "SaveVideo", "LoadAudio"):
        assert forbidden not in flat
    assert sorted(flat).count("VAELoader") == 1
    # And the BUILT graph may use no class the candidate set does not name, so
    # an audio decoder cannot arrive by a literal class in the spec either.
    graph = engine._build_graph({}, "still.png", {"seed": 0}, 124, 864, 480)
    assert set(graph) == set(candidates)
    for node_id, spec in graph.items():
        assert spec["class"] == node_id      # every class is a candidate key
    assert engine._TERMINAL == "decode"
    assert candidates["decode"] == ("VAEDecode",)


def test_the_graph_wires_the_still_as_the_FIRST_frame_and_never_the_last(engine):
    graph = engine._build_graph({"text_prompt": "x"}, "still.png",
                                {"seed": 43}, 124, 864, 480)
    h3_node = graph["h3"]["inputs"]
    assert h3_node["first_frame"] == wb.Wire("loadimage", 0)
    # last_frame is H3's first/LAST interpolation endpoint, a different
    # capability from the first-frame chaining this lane declares.
    assert "last_frame" not in h3_node
    assert h3_node["length"] == 124
    assert (h3_node["width"], h3_node["height"]) == (864, 480)
    assert graph["noise"]["inputs"]["noise_seed"] == 43


def test_the_graph_matches_the_frozen_recipe(engine):
    graph = engine._build_graph({}, "still.png", {"seed": 0}, 124, 864, 480)
    assert graph["sampler"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["sched"]["inputs"]["scheduler"] == "simple"
    assert graph["sched"]["inputs"]["steps"] == 20
    assert graph["sched"]["inputs"]["denoise"] == 1.0
    assert graph["unet"]["inputs"]["weight_dtype"] == "default"
    assert graph["clip"]["inputs"]["type"] == "minimax"


def test_no_environment_variable_can_move_the_frozen_recipe(monkeypatch, engine):
    """A production episode is submitted to an already-booted server, so a knob
    exported at launch cannot bind the work (PBUG-20260723-02). The recipe has
    no env channel at all, and this asserts the absence rather than trusting it.
    """
    for name in ("OTR_MINIMAX_H3_STEPS", "OTR_MINIMAX_H3_SAMPLER",
                 "OTR_MINIMAX_H3_CFG", "OTR_MINIMAX_H3_SCHEDULER"):
        monkeypatch.setenv(name, "999")
    graph = engine._build_graph({}, "still.png", {"seed": 0}, 124, 864, 480)
    assert graph["sched"]["inputs"]["steps"] == 20
    assert graph["sampler"]["inputs"]["sampler_name"] == "res_multistep"


# ---------------------------------------------------------------------------
# PREFLIGHT -- ordering, and the refusals
# ---------------------------------------------------------------------------
def test_the_sage_gate_fires_FIRST_before_any_weight_is_resolved(
        engine, monkeypatch):
    """An ORDERING claim needs a test that can fail (lane 8's lesson).

    Weight resolution is made to raise a distinctive error. If the Sage gate
    ever moves below it, this test fails with that error instead of the refusal
    -- so the order is proven, not assumed.
    """
    class Tripwire(RuntimeError):
        pass

    def _boom(*a, **k):
        raise Tripwire("weight resolution ran before the Sage gate")

    monkeypatch.setattr(type(engine), "_weight_paths", _boom, raising=True)
    monkeypatch.setattr(mc, "assert_sage_not_patched",
                        lambda *a, **k: (_ for _ in ()).throw(
                            EngineUnusable("minimax_h3_video", "image_to_video",
                                           "malformed_config", "SAGE",
                                           kind="video")))
    with pytest.raises(EngineUnusable):
        engine.assert_usable({}, H3_PROFILE)


def test_a_missing_weight_is_a_NAMED_refusal_that_lists_what_is_missing(
        engine, monkeypatch):
    monkeypatch.setattr(mc, "assert_sage_not_patched", lambda *a, **k: None)
    monkeypatch.setattr(bc, "running_server_boot_state",
                        lambda: dict(H3_SERVER_STATE))
    monkeypatch.setattr(type(engine), "_weight_paths",
                        lambda self: {"UNET": ("u.safetensors", None),
                                      "CLIP": ("c.safetensors", "/tmp/c"),
                                      "VAE": ("v.safetensors", "/tmp/v")})
    with pytest.raises(EngineUnusable) as excinfo:
        engine.assert_usable({}, H3_PROFILE)
    assert "u.safetensors" in str(excinfo.value)


def test_a_truncated_weight_is_refused_by_SIZE_not_traced_by_the_loader(
        engine, monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "assert_sage_not_patched", lambda *a, **k: None)
    monkeypatch.setattr(bc, "running_server_boot_state",
                        lambda: dict(H3_SERVER_STATE))
    stub = tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    stub.write_bytes(b"not the 21 GB artifact")
    monkeypatch.setattr(
        type(engine), "_weight_paths",
        lambda self: {"UNET": (stub.name, str(stub)),
                      "CLIP": ("c.safetensors", str(stub)),
                      "VAE": ("v.safetensors", str(stub))})
    with pytest.raises(EngineUnusable) as excinfo:
        engine.assert_usable({}, H3_PROFILE)
    assert "truncated" in str(excinfo.value)


def test_the_lane_REQUIRES_its_own_boot_contract_and_refuses_the_stock_one(
        engine):
    assert engine.compatible_boot_contracts == ("h3",)
    assert bc.check_engine_against_profile(engine, H3_PROFILE) == []
    problems = bc.check_engine_against_profile(engine, {"launch": {}})
    assert problems and "h3" in problems[0]


def test_a_wrong_boot_is_refused_at_PREFLIGHT_by_name(engine, monkeypatch):
    monkeypatch.setattr(mc, "assert_sage_not_patched", lambda *a, **k: None)
    monkeypatch.setattr(
        bc, "running_server_boot_state",
        lambda: {"available": True, "reserve_vram_gb": None,
                 "disable_pinned_memory": False, "sage_attention": False})
    with pytest.raises(bc.BootContractError) as excinfo:
        engine.assert_usable({}, H3_PROFILE)
    assert "reserve-vram" in str(excinfo.value)


# ---------------------------------------------------------------------------
# THE BOOT CONTRACT ITSELF -- lane 19 moved a number in it
# ---------------------------------------------------------------------------
def test_the_h3_contract_carries_the_MEASURED_reserve_clamp():
    """12.0 is a measurement, not a preference.

    Every MiniMax H3 leg on this box that cleared the 14.5 GiB gate held
    ``--reserve-vram 12``, including the trained 1344x768 canvas at 9.15 GiB.
    The one I2V leg without it peaked 15.39 GiB on a SMALLER canvas and a
    SHORTER length, and failed. Dropping this knob would let the lane load its
    way over the ceiling while the contract still passed its own check.
    """
    spec = bc.contract_spec("h3")
    assert spec["reserve_vram_gb"] == 12.0
    assert spec["disable_pinned_memory"] is True
    assert spec["sage_attention"] is False


def test_the_reserve_clamp_REACHES_the_launcher_and_the_profile_carries_it():
    """A boot pin that no launcher turns into argv clamps nothing (lesson L6)."""
    env = bc.launch_env_for("h3")
    assert env["OTR_HEADLESS_RESERVE_VRAM_GB"] == "12"
    assert env["OTR_HEADLESS_DISABLE_PINNED"] == "1"
    launch = _profile()["launch"]
    assert launch["boot_contract"] == "h3"
    assert launch["sage_attention"] is False
    # The profile's env must BE the contract's env -- not a hand-typed echo of
    # it that can drift the next time the contract moves.
    assert launch["env"] == env


def test_the_solo_smoke_path_SELECTS_this_lanes_contract_instead_of_default():
    """``render_single`` is how every lane smoke runs, and it invents its own
    request -- so anything it does not ask for is absent from every smoke.

    It passed ``profile=None``, which selects ``default``, so the first lane to
    REQUIRE its own boot could not be smoked on the boot it declares. Same seam
    and same shape as lane 7's canvas fix one commit earlier.
    """
    from nodes._otr_video_engines import render_driver as rd

    captured = {}

    def _fake_render_one(engine_name, request, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here -- the profile is what is under test")

    real = rd._render_one
    rd._render_one = _fake_render_one
    try:
        out = rd.render_single("minimax_h3_video", frame_count=129)
    finally:
        rd._render_one = real
    assert out["ok"] is False               # the deliberate stop, not a render
    assert captured["profile"] == {"launch": {"boot_contract": "h3"}}


def test_a_lane_with_more_than_one_contract_is_left_alone_by_that_selection():
    """Selecting for a lane with a real CHOICE would be guessing, so it does
    not: HuMo declares both `default` and `humo_diet` and keeps stock behaviour.
    """
    from nodes._otr_video_engines import render_driver as rd

    captured = {}

    def _fake_render_one(engine_name, request, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")

    real = rd._render_one
    rd._render_one = _fake_render_one
    try:
        rd.render_single("humo", frame_count=33)
    finally:
        rd._render_one = real
    assert captured["profile"] is None


def test_selecting_a_contract_does_not_weaken_the_proof_against_the_server():
    """The selection is a CLAIM; the running server is still the judge.

    If this ever stopped being true the smoke path would be a way to render on a
    boot the lane refuses in production, which is the opposite of what a lane
    smoke is for.
    """
    problems = bc.check_running_server(
        "h3", state={"available": True, "reserve_vram_gb": None,
                     "disable_pinned_memory": True, "sage_attention": False})
    assert problems and "reserve-vram" in problems[0]


def test_no_shared_module_reaches_a_sibling_by_an_absolute_nodes_import():
    """THE LIVE BUG LANE 19'S SMOKE FOUND, swept rather than fixed once (L13).

    ``nodes`` resolves against ``sys.path``. In the CPU suite that IS this
    package, so an absolute ``from nodes._otr_video_engines import ...`` works
    here; inside a running ComfyUI server ``nodes`` is ComfyUI's OWN
    node-registry module and the same line raises. Three such lines existed and
    each failed differently:

    * ``boot_contracts.running_server_boot_state`` -- the Sage probe. Raised,
      was caught, left ``sage_attention`` UNKNOWN, and UNKNOWN is correctly not
      a pass -- so the H3 lane refused on every server for a reason that was
      about an import. Dormant until now because ``h3`` is the first contract
      that constrains Sage at all.
    * ``content_oracle.family_for_engine`` -- the WORST, because it failed
      SOFTLY into ``_FAMILY_FALLBACK``. Every engine added since that table was
      written resolved to family "" in production, which is not in
      MOTION_FAMILIES, so those lanes were silently motion-EXEMPT.
    * ``slot_matrix.eligible_engines_for_role`` -- raised outright.

    AST-BASED, and that is the point. The first draft grepped the source text
    and failed on the fixed code -- tripped by the COMMENT that explains the
    fix, which necessarily quotes the broken line. Comments are not AST nodes.
    (Same lesson the repo already learned in ``declares_continuity_kwarg``, and
    the second time today a lexical check read a comment as code.)
    """
    import pathlib

    offenders = []
    for pkg in ("_otr_shared", "_otr_video_engines"):
        root = pathlib.Path(REPO_ROOT) / "nodes" / pkg
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 0 and (node.module or "").startswith("nodes."):
                        offenders.append("%s/%s:%d: from %s import ..."
                                         % (pkg, path.name, node.lineno,
                                            node.module))
                elif isinstance(node, ast.Import):
                    # `imported`, not `alias`: the S30 forbidden-pattern sweep
                    # treats a bare `alias` identifier as a RUNTIME marker for a
                    # deleted subsystem, so the obvious name for an `ast.alias`
                    # node turns this file into two sweep hits (the CW-6 gotcha
                    # two other test files already carry a note about).
                    for imported in node.names:
                        if imported.name.startswith("nodes."):
                            offenders.append("%s/%s:%d: import %s"
                                             % (pkg, path.name, node.lineno,
                                                imported.name))
    assert not offenders, (
        "absolute `nodes.` imports of a SIBLING package resolve against "
        "sys.path -- this package in the CPU suite, ComfyUI's own node registry "
        "on a live server. Use a relative import:\n  " + "\n  ".join(offenders))


def test_the_family_oracle_answers_from_the_REGISTRY_not_the_stale_table():
    """The soft-failure half of the same bug, asserted on its consequence.

    A lane the fallback table has never heard of must still resolve its family,
    because ``motion_required_for_engine`` reads that family and an unknown one
    reads as motion-EXEMPT -- a frozen clip would pass the motion check by never
    being asked.
    """
    from nodes._otr_shared import content_oracle as co

    assert co.family_for_engine("minimax_h3_video") == "image_to_video"
    assert co.motion_required_for_engine("minimax_h3_video") is True
    # And every registered local motion lane agrees with the registry, not with
    # whatever the table happens to still say.
    for name in vreg.all_engine_names():
        engine = vreg.get_engine(name)
        assert co.family_for_engine(name) == getattr(engine, "family", ""), name


def test_an_unverifiable_boot_is_not_a_pass():
    problems = bc.check_running_server(
        "h3", state={"available": False, "error": "no comfy"})
    assert problems and "UNKNOWN is not satisfied" in problems[0]


# ---------------------------------------------------------------------------
# V-1 -- silence is PROVED on this lane's own emitted file
# ---------------------------------------------------------------------------
def test_canonicalize_PROBES_the_emitted_file_before_writing_has_audio_false(
        engine, monkeypatch):
    """H3 is the first local engine that natively produces audio, so on this
    lane ``has_audio: False`` is the one field in the roster that could lie."""
    probed = []

    def _fields(path, **kwargs):
        probed.append(path)
        return {"codec_types": ["video"], "video_codec": "h264",
                "pix_fmt": "yuv420p", "color_space": "bt709", "fps": 25}

    monkeypatch.setattr(h3, "ffprobe_clip_fields", _fields)
    clip = engine.canonicalize({"out_path": "/tmp/h3.mp4", "frame_count": 129},
                               {"shot_id": "s1"}, {})
    assert probed == ["/tmp/h3.mp4"]
    assert clip["has_audio"] is False
    assert clip["frame_count"] == 129
    assert clip["fps"] == 25
    assert clip["engine_id"] == "minimax_h3_video"


def test_canonicalize_REFUSES_a_clip_that_actually_carries_audio(
        engine, monkeypatch):
    monkeypatch.setattr(
        h3, "ffprobe_clip_fields",
        lambda path, **kw: {"codec_types": ["video", "audio"],
                            "video_codec": "h264", "pix_fmt": "yuv420p",
                            "color_space": "bt709", "fps": 25})
    with pytest.raises(wb.GraphExecutionError) as excinfo:
        engine.canonicalize({"out_path": "/tmp/h3.mp4", "frame_count": 129},
                            {"shot_id": "s1"}, {})
    assert "AUDIO stream" in str(excinfo.value)


# ---------------------------------------------------------------------------
# THE PUBLIC SURFACE
# ---------------------------------------------------------------------------
def test_exactly_one_public_id_resolves_to_this_lane():
    assert pub.resolve_engine_id("h3_low_video") == "minimax_h3_video"
    assert pub.resolve_engine_id("minimax_h3_video") == "minimax_h3_video"
    hits = [k for k, v in pub._PUBLIC_ENGINES.items()
            if v == "minimax_h3_video"]
    assert hits == ["h3_low_video"]


def test_lane_20s_adapter_is_NOT_registered_by_this_lane():
    """Scope discipline: one internal engine cannot carry both H3 modes, and two
    public ids on one internal id trips the bijection assert at import time."""
    assert "minimax_h3_audio_in" not in vreg.all_engine_names()
    assert "minimax_h3_audio_in" not in vreg.CAPABILITIES
    assert "h3_low_audio_in" not in pub._PUBLIC_ENGINES


def test_the_still_plan_is_declared_and_validates(engine):
    assert sph.engine_has_still_plan(engine)
    assert sph.engine_has_valid_still_plan(engine)


def test_the_capabilities_row_describes_the_stack_this_lane_actually_loads():
    row = vreg.CAPABILITIES["minimax_h3_video"]
    assert row["device_backends"] == ["cuda"]
    assert row["practical_without_gpu"] is False
    assert len(row["model_requirements"]) == 3
    # The audio VAE is NOT a requirement of this lane: its graph never loads one.
    assert not any("audio" in r for r in row["model_requirements"])


# ---------------------------------------------------------------------------
# THE DOCSTRING FIX THIS PACKET ALSO CARRIED
# ---------------------------------------------------------------------------
def test_the_discrete_frames_docstring_teaches_the_value_veo_really_declares():
    """``frame_contract``'s worked example is the first thing an adapter author
    reads, and it taught the exact number a test asserts is wrong."""
    doc = fc.FrameContract.__doc__ or ""
    assert "(100, 150, 200)" in doc
    assert "(96, 144, 192)" in doc          # kept, but named as the OLD error
    assert doc.index("(100, 150, 200)") < doc.index("(96, 144, 192)")


# ---------------------------------------------------------------------------
# COLD IMPORT -- the adapter must answer the grid question with no ComfyUI
# ---------------------------------------------------------------------------
def test_the_module_imports_nothing_heavy_at_module_scope():
    tree = ast.parse(inspect.getsource(h3))
    imported = []
    for node in tree.body:                   # module scope only, not inside defs
        if isinstance(node, ast.Import):
            imported += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.append((node.module or "").split(".")[0])
    for heavy in ("torch", "numpy", "PIL", "folder_paths", "comfy", "nodes"):
        assert heavy not in imported, heavy
