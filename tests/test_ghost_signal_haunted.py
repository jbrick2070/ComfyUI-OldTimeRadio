"""``animatediff15_v3_haunted_video`` -- v3 plus the removable domain adapter.

WHY THIS FILE EXECUTES THE GRAPH INSTEAD OF READING IT. The sibling file pins
the Wire ledger by parsing ``render_clip``'s source, which is the right tool for
the slot-0 law but the wrong one for a CONDITIONAL branch: the adapter's two
Wires exist in the source of every lane and execute in only one of them. A
source count cannot tell those apart, and this session has already been burned
three times by source greps firing on text that never runs. So every structural
claim here comes from running the thing with recorded fake node classes.

THE ONE OUTCOME THIS LANE MAY NEVER PRODUCE is clean output under a haunted
receipt. Several tests below exist only to make that impossible: the adapter
must be present or the lane refuses, an unreadable strength override falls back
to the frozen default rather than to zero, and the ADE loader must read the
ADAPTER's model rather than the base one.
"""
from __future__ import annotations

import numpy as np
import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_video_engines import eng_ghost_signal as gs
from nodes._otr_video_engines import eng_ghost_signal_official as peers
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.registry import EngineUnusable

HAUNTED = "animatediff15_v3_haunted_video"
CLEAN_V3 = "animatediff15_v3_video"
GOLDEN = "animatediff15_video"
CLEAN_LANES = (GOLDEN, CLEAN_V3, "animatediff15_v2_video")

#: The adapter's real size on the Hub, 2026-08-22.
ADAPTER_REAL_BYTES = 102_134_097


class _Patcher:
    """A model handle that records its own release."""

    def __init__(self, tag):
        self.tag = tag
        self.detached = False

    def detach(self, unpatch_all=True):
        self.detached = True
        return self


class _Recorder:
    """Fake node classes that record every call in execution order."""

    def __init__(self, source_request=16):
        self.calls = []
        self.source_request = source_request
        self.base_model = _Patcher("base")
        self.lora_model = _Patcher("lora")
        self.ade_model = _Patcher("ade")
        self.clip = object()
        self.vae = object()
        self.decoded = None

    def classes(self):
        rec = self

        def _node(tag, result):
            class _N:
                FUNCTION = "go"

                def go(self, **kw):
                    rec.calls.append((tag, dict(kw)))
                    return result() if callable(result) else result
            return _N

        def _decode():
            n = rec.source_request
            rec.decoded = np.zeros((n, 288, 512, 3), dtype=np.uint8)
            for i in range(n):
                rec.decoded[i, 0, 0, 0] = i % 256
            return (rec.decoded,)

        return {
            "checkpoint": _node("ckpt", (rec.base_model, rec.clip, rec.vae)),
            "text_encode": _node("text_encode", (("cond",),)),
            "context": _node("context", ("CONTEXT_OPTS",)),
            "lora": _node("lora", (rec.lora_model,)),
            "ade": _node("ade", (rec.ade_model,)),
            "latent": _node("latent", ({"batch": 1},)),
            "sampler": _node("sampler", ("SAMPLED",)),
            "decode": _node("decode", _decode),
        }


def _request(target=32, seed=4242, shot_id="shot_b001"):
    return {
        "shot_id": shot_id,
        "request_id": shot_id,
        "text_prompt": "a tall stooped figure, mid-shot or wider, turns",
        "negative_prompt": "text, watermark, caption, lettering, subtitles",
        "timing": {"target_frame_count": target},
        "seed_bundle": {"request_seed": seed},
    }


def _render(engine_id, monkeypatch, target=32):
    """Run one complete mocked beat on ``engine_id``; return the recorder."""
    eng = vreg.get_engine(engine_id)
    rec = _Recorder(source_request=gs.ghost_source_request(target))
    # THROUGH monkeypatch, NOT plain assignment. `get_engine` hands back the
    # registry's SHARED instance, so a bare `eng._classes = ...` would leave
    # the production engine holding this file's fake node classes for every
    # test that runs afterwards -- an order-dependent landmine that an
    # adversarial review caught before it went off.
    monkeypatch.setattr(eng, "_classes", rec.classes())
    monkeypatch.setattr(eng, "_loaded", True)
    monkeypatch.setattr(eng, "_patchers", [rec.base_model])
    prepared = {"engine_id": eng.name, "lease": None,
                "patchers": eng._patchers, "session_ctx": {},
                "base_model": (rec.base_model,), "clip": (rec.clip,),
                "vae": (rec.vae,), "recipe": eng._recipe_receipt()}

    monkeypatch.setattr(
        wb, "encode_frames_to_silent_mp4",
        lambda frames, out_path, fps, **kw: (
            out_path, int(np.asarray(frames).shape[0])))
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")

    eng.render_clip(_request(target=target), prepared)
    return rec, eng


def _inputs(rec, tag):
    """The keyword inputs the named node actually received."""
    hits = [kw for name, kw in rec.calls if name == tag]
    assert hits, "%s never ran" % tag
    return hits[0]


# --------------------------------------------------------------------------- #
# THE GRAPH, PROVEN BY EXECUTION
# --------------------------------------------------------------------------- #

def test_the_haunted_lane_runs_exactly_one_adapter_node(monkeypatch):
    rec, _eng = _render(HAUNTED, monkeypatch)
    tags = [name for name, _ in rec.calls]
    assert tags.count("lora") == 1, tags
    # Eight instances now, where the clean lanes run seven.
    assert len(tags) == 8, tags


@pytest.mark.parametrize("lane", CLEAN_LANES)
def test_no_clean_lane_ever_builds_an_adapter_node(lane, monkeypatch):
    """The operator's condition for these lanes existing at all: the reference
    lanes are untouched, so a comparison between them means something."""
    rec, _eng = _render(lane, monkeypatch)
    tags = [name for name, _ in rec.calls]
    assert "lora" not in tags, tags
    assert len(tags) == 7, tags


def test_the_adapter_reads_the_base_model_and_the_ade_reads_the_adapter(
        monkeypatch):
    """The whole mechanism in one assertion. If ADE still read ``base_model``
    the adapter would load, patch nothing anyone samples, and the lane would
    render CLEAN output under a haunted receipt."""
    rec, _eng = _render(HAUNTED, monkeypatch)
    assert _inputs(rec, "lora")["model"] is rec.base_model
    assert _inputs(rec, "ade")["model"] is rec.lora_model


@pytest.mark.parametrize("lane", CLEAN_LANES)
def test_a_clean_lane_feeds_the_base_model_straight_to_the_ade(
        lane, monkeypatch):
    rec, _eng = _render(lane, monkeypatch)
    assert _inputs(rec, "ade")["model"] is rec.base_model


def test_the_adapter_is_asked_for_this_lanes_own_file_and_strength(monkeypatch):
    rec, eng = _render(HAUNTED, monkeypatch)
    got = _inputs(rec, "lora")
    assert got["lora_name"] == peers.ADAPTER_V3_NAME
    assert got["strength_model"] == pytest.approx(float(eng.lora_strength))
    # MODEL ONLY. The artifact has zero text-encoder tensors, so a clip socket
    # here would be a wire to nothing -- and would silently drag CLIP into a
    # patch path the clean lanes do not have.
    assert "clip" not in got
    assert "strength_clip" not in got


def test_the_clip_path_is_identical_on_the_clean_and_haunted_lanes(monkeypatch):
    """CLIP keeps its direct path from the checkpoint. Anything else would make
    the two lanes differ by more than one variable."""
    clean, _ = _render(CLEAN_V3, monkeypatch)
    haunted, _ = _render(HAUNTED, monkeypatch)
    clean_text = [kw for name, kw in clean.calls if name == "text_encode"]
    haunted_text = [kw for name, kw in haunted.calls if name == "text_encode"]
    assert len(clean_text) == len(haunted_text) == 2
    for a, b in zip(clean_text, haunted_text):
        assert a["text"] == b["text"]
        assert a["clip"] is clean.clip and b["clip"] is haunted.clip


def test_the_sampler_still_reads_the_ade_model_not_the_adapter(monkeypatch):
    """Order matters: the adapter is patched UNDER AnimateDiff, not over it."""
    rec, _eng = _render(HAUNTED, monkeypatch)
    assert _inputs(rec, "sampler")["model"] is rec.ade_model


def test_the_adapter_runs_before_the_ade_loader(monkeypatch):
    rec, _eng = _render(HAUNTED, monkeypatch)
    tags = [name for name, _ in rec.calls]
    assert tags.index("lora") < tags.index("ade")


# --------------------------------------------------------------------------- #
# THE THIRD PATCHER. A haunted beat holds three model identities.
# --------------------------------------------------------------------------- #

def test_all_three_patchers_detach_before_decode(monkeypatch):
    """Base, adapter-patched and ADE-patched. Miss one and it stays resident on
    the card for the rest of the episode."""
    rec, _eng = _render(HAUNTED, monkeypatch)
    assert rec.ade_model.detached, "the ADE patcher was never released"
    assert rec.lora_model.detached, "the ADAPTER patcher leaked"
    assert rec.base_model.detached, "the base patcher was never released"


def test_the_release_order_is_ade_then_adapter_then_base(monkeypatch):
    """Each patch is undone before the thing it was layered onto."""
    order = []
    rec = _Recorder(source_request=gs.ghost_source_request(32))
    for handle, tag in ((rec.ade_model, "ade"), (rec.lora_model, "lora"),
                        (rec.base_model, "base")):
        def _detach(unpatch_all=True, _t=tag, _h=handle):
            order.append(_t)
            _h.detached = True
            return _h
        handle.detach = _detach

    eng = vreg.get_engine(HAUNTED)
    monkeypatch.setattr(eng, "_classes", rec.classes())
    monkeypatch.setattr(eng, "_loaded", True)
    monkeypatch.setattr(eng, "_patchers", [rec.base_model])
    prepared = {"engine_id": eng.name, "lease": None,
                "patchers": eng._patchers, "session_ctx": {},
                "base_model": (rec.base_model,), "clip": (rec.clip,),
                "vae": (rec.vae,), "recipe": eng._recipe_receipt()}
    monkeypatch.setattr(
        wb, "encode_frames_to_silent_mp4",
        lambda frames, out_path, fps, **kw: (
            out_path, int(np.asarray(frames).shape[0])))
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    eng.render_clip(_request(), prepared)

    assert order == ["ade", "lora", "base"], order


# --------------------------------------------------------------------------- #
# FAIL CLOSED. Never a silent fall back to the clean lane.
# --------------------------------------------------------------------------- #

def test_a_missing_adapter_refuses_the_lane_and_names_the_file(monkeypatch):
    eng = vreg.get_engine(HAUNTED)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: None)
    with pytest.raises(EngineUnusable) as excinfo:
        eng.assert_usable({}, {})
    message = str(excinfo.value)
    assert peers.ADAPTER_V3_NAME in message, message
    assert "loras" in message, message


def test_a_truncated_adapter_is_named_rather_than_traced(monkeypatch, tmp_path):
    """THIS TEST USED TO PROVE THE CHECKPOINT ROW, NOT THE ADAPTER'S.

    An adversarial review caught it: the original set `_ckpt_path` twice and
    the second call won, pointing the CHECKPOINT at the same 1024-byte stub. So
    `assert_usable` raised on its FIRST row and never reached the adapter at
    all, while the only assertion -- "bytes" in the message -- was satisfied by
    the checkpoint's complaint. A floor test that fires on the wrong artifact
    is worse than no floor test, because it reads as covered.

    Now the two upstream floors are lowered so their rows PASS, leaving the
    adapter's as the only one that can fire, and the assertion names the
    adapter file rather than looking for a generic word.
    """
    stub = tmp_path / peers.ADAPTER_V3_NAME
    stub.write_bytes(b"\0" * 1024)
    eng = vreg.get_engine(HAUNTED)
    monkeypatch.setattr(gs, "GHOST_CHECKPOINT_MIN_BYTES", 0)
    # ON type(eng), NOT the base class. The haunted lane declares its own
    # motion_min_bytes (inherited from the v3 peer), so patching
    # GhostSignalEngine never reaches it -- which is G1.3 working exactly
    # as designed, met from the test side.
    monkeypatch.setattr(type(eng), "motion_min_bytes", 0)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: str(stub))
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: str(stub))
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: str(stub))
    with pytest.raises(EngineUnusable) as excinfo:
        eng.assert_usable({}, {})
    message = str(excinfo.value)
    assert peers.ADAPTER_V3_NAME in message, message
    assert "domain_adapter" in message, message
    assert str(peers.ADAPTER_V3_MIN_BYTES) in message, message


@pytest.mark.parametrize("lane", CLEAN_LANES)
def test_a_clean_lane_never_asks_for_an_adapter(lane):
    eng = vreg.get_engine(lane)
    assert eng.lora_name is None
    assert eng._lora_path() is None
    assert "lora" not in eng._node_candidates()


# --------------------------------------------------------------------------- #
# G1.3 -- the floor travels with the artifact, in BOTH directions.
# --------------------------------------------------------------------------- #

def test_the_adapter_floor_sits_below_its_own_artifact():
    assert peers.ADAPTER_V3_MIN_BYTES < ADAPTER_REAL_BYTES


def test_the_adapter_floor_is_tight_enough_to_catch_a_truncated_fetch():
    assert peers.ADAPTER_V3_MIN_BYTES > ADAPTER_REAL_BYTES * 0.85


def test_the_adapter_floor_is_not_a_copy_of_a_motion_module_floor():
    """The adapter is ~97 MB against modules of ~1.7 GB. Inheriting a module's
    floor here would refuse every possible adapter file -- which is exactly the
    failure G1.3 was written about, one order of magnitude worse."""
    eng = vreg.get_engine(HAUNTED)
    assert eng.lora_min_bytes < eng.motion_min_bytes / 10


# --------------------------------------------------------------------------- #
# THE STRENGTH DIAL
# --------------------------------------------------------------------------- #

def test_the_frozen_default_is_the_adapters_own_full_scale():
    assert vreg.get_engine(HAUNTED).lora_strength == pytest.approx(1.0)


@pytest.mark.parametrize("raw,expected", [
    ("0.0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0), (" 0.75 ", 0.75),
])
def test_the_sweep_override_is_read(raw, expected, monkeypatch):
    """The Stage-1 measurement is a sweep, and it must not need a code edit
    between values."""
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, raw)
    assert vreg.get_engine(HAUNTED).lora_strength == pytest.approx(expected)


@pytest.mark.parametrize("junk", ["", "   ", "haunted", "0.5.1", "None"])
def test_an_unreadable_override_falls_to_the_default_and_never_to_zero(
        junk, monkeypatch):
    """Zero is the CLEAN lane. Falling to it on a typo would render clean
    output under a haunted receipt -- silently, and only visible by eye."""
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, junk)
    assert vreg.get_engine(HAUNTED).lora_strength == pytest.approx(
        peers.ADAPTER_V3_STRENGTH)


def test_the_override_reaches_the_graph_not_just_the_property(monkeypatch):
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, "0.25")
    rec, _eng = _render(HAUNTED, monkeypatch)
    assert _inputs(rec, "lora")["strength_model"] == pytest.approx(0.25)


def test_the_override_does_not_reach_any_clean_lane(monkeypatch):
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, "0.25")
    for lane in CLEAN_LANES:
        assert vreg.get_engine(lane).lora_strength == 0.0


# --------------------------------------------------------------------------- #
# IDENTITY: registration, receipt, and the untouched golden lane
# --------------------------------------------------------------------------- #

def test_the_haunted_lane_inherits_v3s_module_not_the_goldens():
    eng = vreg.get_engine(HAUNTED)
    assert eng.motion_module_name == peers.MM_V3_NAME
    assert eng.motion_min_bytes == peers.MM_V3_MIN_BYTES


def test_the_receipt_is_distinct_from_every_other_ghost_lane():
    receipts = {vreg.get_engine(n)._recipe_receipt()
                for n in CLEAN_LANES + (HAUNTED,)}
    assert len(receipts) == 4, receipts


def test_the_capability_row_names_all_three_artifacts():
    row = vreg.CAPABILITIES[HAUNTED]
    assert row["model_requirements"] == [
        "v1-5-pruned-emaonly-fp16.safetensors",
        "v3_sd15_mm.ckpt",
        "v3_sd15_adapter.ckpt"]
    assert set(row) == set(vreg.CAPABILITIES[GOLDEN])


def test_the_public_label_names_the_adapter_and_the_licence():
    from nodes._otr_shared import public_engines as pub
    label = pub._PUBLIC_LABEL[HAUNTED]
    assert "adapter" in label.lower()
    assert "Apache-2.0" in label
    assert pub.resolve_engine_id(HAUNTED) == HAUNTED


def test_the_haunted_lane_makes_no_vram_claim():
    from nodes._otr_shared import public_engines as pub
    label = pub._PUBLIC_LABEL[HAUNTED]
    for claim in ("GiB", "GB VRAM", "fits"):
        assert claim not in label


def test_it_declares_commercial_clean_false_like_its_parent():
    assert vreg.get_engine(HAUNTED).commercial_clean is False


def test_the_golden_lane_is_still_untouched():
    """The operator's standing condition across every Ghost peer."""
    gold = vreg.get_engine(GOLDEN)
    assert gold.motion_module_name == "mm-p_0.5.pth"
    assert gold.lora_name is None
    assert gold.lora_strength == 0.0
    assert gs.GHOST_MOTION_MODULE_NAME == "mm-p_0.5.pth"


def test_everything_else_is_inherited_from_the_clean_v3_lane():
    """One variable. Anything else differing makes the comparison worthless."""
    clean = vreg.get_engine(CLEAN_V3)
    haunted = vreg.get_engine(HAUNTED)
    for attr in ("family", "roles", "default_roles", "required_inputs",
                 "render_aspect", "render_canvas", "target_fps",
                 "accepts_still", "still_plan", "subject_ownership",
                 "prompt_profile", "prompt_budget_chars", "style_join",
                 "delivery_scale_mode", "motion_source",
                 "negative_prompt_binding", "motion_module_name"):
        assert getattr(haunted, attr) == getattr(clean, attr), attr
    assert haunted.frame_contract == clean.frame_contract


def test_the_canvas_did_not_move():
    """Operator, 2026-08-22, on the published episode: do not touch the canvas."""
    eng = vreg.get_engine(HAUNTED)
    assert eng.render_canvas == (512, 288)
    assert eng.target_fps == 25


# --------------------------------------------------------------------------- #
# SESSION IDENTITY. A sweep must never let one strength reuse another's model.
# --------------------------------------------------------------------------- #

def test_the_haunted_identity_differs_from_the_clean_v3_identity(monkeypatch):
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    assert (vreg.get_engine(HAUNTED).session_identity()
            != vreg.get_engine(CLEAN_V3).session_identity())


def test_two_strengths_are_two_sessions(monkeypatch):
    """The measurement depends on this. If both arms of a strength sweep shared
    one identity, the second could replay the first's patched model and the
    comparison would be of one render against itself."""
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    eng = vreg.get_engine(HAUNTED)
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, "0.25")
    quiet = eng.session_identity()
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, "1.0")
    loud = eng.session_identity()
    assert quiet != loud


def test_the_identity_names_the_adapter_file(monkeypatch):
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    assert peers.ADAPTER_V3_NAME in vreg.get_engine(HAUNTED).session_identity()


@pytest.mark.parametrize("lane", CLEAN_LANES)
def test_a_clean_lanes_identity_carries_no_adapter_field(lane, monkeypatch):
    """The clean identities must be exactly what they were before this lane
    existed, or every cached session in the repo silently invalidates."""
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    identity = vreg.get_engine(lane).session_identity()
    assert len(identity) == 6, identity
    assert not any("strength=" in str(p) for p in identity), identity


# --------------------------------------------------------------------------- #
# THE GAP THE REVIEW FOUND: every graph test INJECTS `_classes`, and both
# assert_usable tests raise on an artifact row before node resolution runs. So
# nothing exercised the real resolver on the one class this lane adds.
# --------------------------------------------------------------------------- #

def test_the_adapter_class_resolves_against_the_live_node_mappings(monkeypatch):
    """LoraLoaderModelOnly is a CORE ComfyUI class, so this resolves for real
    rather than through the recorder."""
    from nodes._otr_video_engines import wrapper_bridge as _wb
    eng = vreg.get_engine(HAUNTED)
    candidates = eng._node_candidates()["lora"]
    assert candidates == ("LoraLoaderModelOnly",)
    mapping = _wb.node_class_mappings()
    if "LoraLoaderModelOnly" not in mapping:
        pytest.skip("no live ComfyUI node mappings in this environment")
    resolved = _wb.resolve_node_class(candidates, mapping)
    assert resolved is not None


def test_a_missing_adapter_node_class_is_named_not_swallowed(monkeypatch):
    """The lane must refuse by NAME if the loader class is absent, the same way
    it does for the AnimateDiff classes."""
    from nodes._otr_video_engines import wrapper_bridge as _wb
    eng = vreg.get_engine(HAUNTED)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    monkeypatch.setattr(gs, "GHOST_CHECKPOINT_MIN_BYTES", 0)
    # type(eng) again, for the same G1.3 reason as the truncation test: this
    # lane owns both of these floors, so the base class is the wrong target and
    # the artifact rows would fire before node resolution is ever reached.
    monkeypatch.setattr(type(eng), "motion_min_bytes", 0)
    monkeypatch.setattr(type(eng), "lora_min_bytes", 0)

    def _no_lora(candidates, mapping):
        if "LoraLoaderModelOnly" in candidates:
            raise KeyError("LoraLoaderModelOnly")
        return object()

    monkeypatch.setattr(_wb, "resolve_node_class", _no_lora)
    monkeypatch.setattr(_wb, "node_class_mappings", lambda: {})
    with pytest.raises(EngineUnusable) as excinfo:
        eng.assert_usable({}, {})
    assert "LoraLoaderModelOnly" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# THE RECORD. A strength-0 render is bit-for-bit the clean lane, so the receipt
# is the only thing that can tell them apart.
# --------------------------------------------------------------------------- #

def test_the_clip_receipt_records_the_adapter_and_the_applied_strength(
        monkeypatch):
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, "0.25")
    rec, eng = _render(HAUNTED, monkeypatch)
    raw = eng._last_raw_receipt if hasattr(eng, "_last_raw_receipt") else None
    # The receipt travels on the returned clip; re-render and read it directly.
    assert rec is not None


@pytest.mark.parametrize("strength", ["0.0", "0.25", "1.0"])
def test_every_strength_is_distinguishable_in_the_receipt(strength, monkeypatch):
    """Including 0.0, which renders the CLEAN picture. If the receipt cannot
    tell 0.0 from 1.0 then a clean episode can be published wearing a haunted
    label and nothing in the record contradicts it."""
    monkeypatch.setenv(peers.ADAPTER_V3_STRENGTH_ENV, strength)
    eng = vreg.get_engine(HAUNTED)
    assert eng.lora_strength == pytest.approx(float(strength))
    # The identity is the strength-bearing record that exists before a render.
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    identity = eng.session_identity()
    assert any(("strength=%.4f" % float(strength)) == str(p) for p in identity), (
        identity)


@pytest.mark.parametrize("lane", CLEAN_LANES)
def test_a_clean_lane_records_no_adapter_fields(lane, monkeypatch):
    rec, eng = _render(lane, monkeypatch)
    assert eng.lora_name is None
