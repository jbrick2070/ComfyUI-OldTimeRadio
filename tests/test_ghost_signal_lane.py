"""Ghost Signal (``animatediff15_video``) -- the lane's contract, pinned.

DETERMINISTIC CONTRACT CHECKS, NOT A MEASUREMENT CAMPAIGN. Nothing here loads a
weight, times anything, or judges a picture: the operator declined the campaign,
so every claim this file makes is structural.

THE GRAPH TESTS RUN THE REAL EXECUTOR. Fake node classes are substituted for the
seven live ComfyUI/ADE classes, but ``wrapper_bridge.run_graph`` itself is the
real one -- so the wiring, the topological order, the ``Wire`` slot indexing and
the ``on_result`` ownership handoff are all genuinely exercised. Mocking the
executor would leave exactly the wiring this lane is most likely to get wrong
untested.
"""
from __future__ import annotations

import ast
import inspect
import os

import numpy as np
import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_shared import public_engines as pub
from nodes._otr_video_engines import eng_ghost_signal as gs
from nodes._otr_video_engines import ghost_signal_prompt as gsp
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas as vschemas
from nodes._otr_video_engines import wrapper_bridge as wb

NAME = "animatediff15_video"


# --------------------------------------------------------------------------- #
# Registration and declarations
# --------------------------------------------------------------------------- #

def test_the_lane_is_registered_with_a_capabilities_row():
    assert vreg.is_registered(NAME)
    row = vreg.CAPABILITIES[NAME]
    assert row["required_toolchain"] is None
    assert row["requires_sidecar"] is False
    assert row["device_backends"] == ["cuda"]
    assert row["requires_vendor"] is None
    assert row["needs_fp8_te"] is False
    assert row["needs_fp4_te"] is False
    assert row["practical_without_gpu"] is False
    assert row["sidecar_conditional"] is False
    assert row["model_requirements"] == [
        "v1-5-pruned-emaonly-fp16.safetensors", "mm-p_0.5.pth"]
    # No novel capability keys: the row's shape must match its siblings exactly.
    assert set(row) == set(vreg.CAPABILITIES["minimax_h3_video"])


def test_the_bare_id_passes_through_and_carries_a_friendly_label():
    """No `_PUBLIC_ENGINES` self-alias -- the resolver already passes a bare
    internal id through, and a self-alias would put a duplicate in the
    bijection for nothing."""
    assert pub.resolve_engine_id(NAME) == NAME
    assert NAME not in pub._PUBLIC_ENGINES
    assert pub._PUBLIC_LABEL[NAME] == "AnimateDiff -- Ghost Signal"
    # The public id carries no unmeasured cost token.
    assert "low" not in NAME and "high" not in NAME


def test_all_three_roles_are_eligible():
    eng = vreg.get_engine(NAME)
    assert set(eng.roles) == {
        "announcer_visual", "music_visual", "character_video"}
    # Opt-in: it never seizes a role by declaring itself the default for one.
    assert eng.default_roles == ()
    assert eng.requires_flag is None


def test_the_declared_contract_is_exact():
    eng = vreg.get_engine(NAME)
    assert eng.name == NAME
    assert eng.family == "text_to_video"
    assert eng.required_inputs == ("text_prompt",)
    assert eng.render_aspect == "wide"
    # G2: THIS is the test that pins the declared canvas to the graph. The
    # EmptyLatentImage widget vector below reads the same two constants, so a
    # drift between the declaration and what the graph actually samples fails
    # here rather than silently rendering at a size nobody declared.
    assert eng.render_canvas == (512, 288)
    assert (gs.GHOST_CANVAS_W, gs.GHOST_CANVAS_H) == (512, 288)
    # /32-legal on BOTH axes (the retired 384x216 was not: 216 % 32 == 24).
    assert gs.GHOST_CANVAS_W % 32 == 0 and gs.GHOST_CANVAS_H % 32 == 0
    assert eng.target_fps == 25
    assert eng.prompt_profile == "ghost_signal_v1"
    assert eng.prompt_budget_chars == 320
    assert eng.style_join == "compose"
    assert eng.delivery_scale_mode == "lanczos_clean_full_frame"
    assert eng.subject_ownership == "prompt"
    assert eng.motion_source == "ledger_motion_clause"
    assert eng.commercial_clean is False


def test_the_frame_contract_is_delivered_frame_units_and_explicit():
    fc = vreg.get_engine(NAME).frame_contract
    assert fc.min_frames == 1
    assert fc.max_frames == 0          # unbounded: a beat is ONE timeline
    assert fc.quantum == 1
    assert fc.native_fps == 25
    assert fc.allow_tail_trim is True
    assert fc.continuity == "none"     # EXPLICIT, never inherited
    assert fc.discrete_frames == ()


def test_it_declares_no_still_and_an_empty_valid_plan():
    eng = vreg.get_engine(NAME)
    assert eng.accepts_still is False
    assert eng.still_plan == ()
    assert isinstance(eng.still_plan, tuple)


def _all_ghost_policy():
    """A policy routing all three video roles to Ghost.

    ``video_models`` is keyed by SLOT (`announcer_video_model`, ...), not by
    role -- `ROLE_TO_VIDEO_SLOT` is the map between them, and building the dict
    from role names instead yields a policy every lookup answers ``None`` to,
    which reads exactly like "no still is minted" while proving nothing.
    """
    from nodes._otr_shared import role_slots as _role_slots
    return {"video_models": {slot: NAME
                             for slot in _role_slots.ROLE_TO_VIDEO_SLOT.values()}}


def test_the_image_dispatcher_mints_no_still_for_any_ghost_role():
    """THE REAL CAPABILITY LOOKUP, not a re-reading of the declaration.

    `still_word` proved that a plan nothing consults is a plan that does not
    exist -- it declared portrait `never` for months while a portrait was minted
    for every cast member anyway. So this asks the seam that actually decides.
    """
    from nodes import otr_image_gen_dispatcher as dispatcher
    caps = dispatcher.still_consumer_capabilities(_all_ghost_policy())
    assert caps is not None, "the all-Ghost policy was rejected as malformed"
    for role, consumes in caps.items():
        assert consumes is False, (
            "role %r would have a still minted for it on an all-Ghost policy "
            "(capability said %r)" % (role, consumes))
    assert dispatcher.roles_requiring_stills(_all_ghost_policy()) == frozenset()


def test_which_g3_7_seam_actually_covers_ghost_and_which_does_not():
    """WHICH mechanism owns this lane, stated exactly, because the two look alike.

    G3.7 has two seams and Ghost is covered by the STRONGER one:

    * ``accepts_still = False`` at the image dispatcher -- NO still of any kind
      is minted for a Ghost role. Proven by the test above.
    * ``_portrait_free_roles_from_policy`` -- the fifth lane-derived role set,
      which exempts a role from PORTRAIT minting specifically. It looks for a
      plan ROW declaring ``kind=portrait required=never``, so it is INERT for a
      lane whose ``still_plan`` is empty, and it returns nothing for Ghost.

    That is correct, not a gap: the portrait-free set exists for a lane that
    consumes SOME stills but never a portrait (`still_word` is the case it was
    built for). A lane that consumes none has no portrait left to exempt.

    Pinned because the obvious "fix" -- adding a portrait/never row to Ghost's
    plan to make this set light up -- would be a false declaration. The plan is
    empty because the lane reads nothing, and a row claiming otherwise would be
    the exact class of unread-declaration defect G3.7 was written to end.
    """
    import json
    from nodes import otr_meta_brief_image_prompt as mbip
    free = mbip._portrait_free_roles_from_policy(
        json.dumps(_all_ghost_policy()))
    assert free == set(), (
        "the portrait-free set now claims Ghost roles (%s). Either a portrait "
        "row was added to an empty still_plan -- which would be a declaration "
        "the lane cannot honour -- or the enumerator changed meaning" % (free,))
    # And the reason it is empty is the empty plan, not a broken lookup.
    assert vreg.get_engine(NAME).still_plan == ()


# --------------------------------------------------------------------------- #
# Cold import
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("module", [gs, gsp])
def test_cold_import_pulls_no_heavy_dependency(module):
    """torch / numpy / transformers / diffusers / AnimateDiff must all be lazy.

    Checked against the AST rather than sys.modules: another test importing
    torch first would make a sys.modules check pass for the wrong reason.
    """
    forbidden = {"torch", "numpy", "transformers", "diffusers", "safetensors",
                 "animatediff", "comfy", "folder_paths", "cv2", "PIL"}
    tree = ast.parse(inspect.getsource(module))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.append((node.module or "").split(".")[0])
    offenders = sorted(set(top_level) & forbidden)
    assert not offenders, (
        "%s imports %s at MODULE SCOPE; the cold-import invariant (V-12) needs "
        "every heavy dependency inside a lifecycle method"
        % (module.__name__, offenders))


# --------------------------------------------------------------------------- #
# The node / link / widget contract (plan section 4.3)
# --------------------------------------------------------------------------- #

EXPECTED_CLASSES = {
    "checkpoint": ("CheckpointLoaderSimple",),
    "text_encode": ("CLIPTextEncode",),
    "context": ("ADE_StandardStaticContextOptions",),
    "ade": ("ADE_AnimateDiffLoaderGen1",),
    "latent": ("EmptyLatentImage",),
    "sampler": ("KSampler",),
    "decode": ("VAEDecode",),
}


def test_the_resolver_map_is_seven_names_one_per_alias():
    assert gs.GHOST_NODE_CANDIDATES == EXPECTED_CLASSES
    for logical, candidates in gs.GHOST_NODE_CANDIDATES.items():
        assert len(candidates) == 1, (
            "%s lists %d candidates; ONE NAME PER ALIAS -- an alternative "
            "spelling is runtime probing wearing a tuple" % (logical, candidates))


def test_the_frozen_recipe_values_are_exact():
    assert (gs.GHOST_STEPS, gs.GHOST_CFG) == (20, 8.0)
    assert gs.GHOST_SAMPLER_NAME == "euler"
    assert gs.GHOST_SCHEDULER == "normal"
    assert gs.GHOST_DENOISE == 1.0
    assert gs.GHOST_BETA_SCHEDULE == "autoselect"
    assert gs.GHOST_CONTEXT_LENGTH == 16
    assert gs.GHOST_CONTEXT_OVERLAP == 4
    assert gs.GHOST_CONTEXT_FUSE_METHOD == "pyramid"
    assert gs.GHOST_CONTEXT_USE_ON_EQUAL_LENGTH is False
    assert gs.GHOST_CONTEXT_START_PERCENT == 0.0
    assert gs.GHOST_CONTEXT_GUARANTEE_STEPS == 1
    assert gs.GHOST_SOURCE_FLOOR == 16
    assert gs.GHOST_CHECKPOINT_NAME == "v1-5-pruned-emaonly-fp16.safetensors"
    assert gs.GHOST_MOTION_MODULE_NAME == "mm-p_0.5.pth"


def test_the_cfg_keeps_the_negative_live():
    """cfg 1.0 would skip the unconditional branch and make the negative inert,
    which is exactly why an AnimateLCM checkpoint was refused: the lettering
    defense IS the negative."""
    assert gs.GHOST_CFG > 1.0


# --- fake node classes: real executor, substituted leaves ------------------ #

class _Handle:
    """A stand-in for a ComfyUI object with a detachable patcher."""

    def __init__(self, kind, detach_raises=False, detach_callable=True):
        self.kind = kind
        self.detached = 0
        self._raises = detach_raises
        if not detach_callable:
            self.detach = "not callable"

    def detach(self, unpatch_all=False):       # noqa: D401
        if self._raises:
            raise RuntimeError("detach refused for the test")
        self.detached += 1

    def __repr__(self):
        return "<%s>" % self.kind


class _Recorder:
    """Records every node call so the tests can assert order and arguments."""

    def __init__(self):
        self.calls = []
        self.base_model = _Handle("base_model")
        self.ade_model = _Handle("ade_model")
        self.clip = _Handle("clip")
        self.vae = _Handle("vae")
        self.decoded = None

    def classes(self, source_request=None):
        rec = self

        class Ckpt:
            FUNCTION = "go"

            def go(self, **kw):
                rec.calls.append(("ckpt", dict(kw)))
                return (rec.base_model, rec.clip, rec.vae)

        class TextEncode:
            FUNCTION = "go"

            def go(self, **kw):
                rec.calls.append(("text_encode", dict(kw)))
                return (("cond", kw.get("text")),)

        class Context:
            FUNCTION = "go"

            def go(self, **kw):
                rec.calls.append(("context", dict(kw)))
                return ("CONTEXT_OPTS",)

        class Ade:
            FUNCTION = "go"

            def go(self, **kw):
                rec.calls.append(("ade", dict(kw)))
                return (rec.ade_model,)

        class Latent:
            FUNCTION = "go"

            def go(self, **kw):
                rec.calls.append(("latent", dict(kw)))
                return ({"batch": kw.get("batch_size")},)

        class Sampler:
            FUNCTION = "go"

            def go(self, **kw):
                rec.calls.append(("sampler", dict(kw)))
                return ("SAMPLED",)

        class Decode:
            FUNCTION = "go"

            def go(self, **kw):
                rec.calls.append(("decode", dict(kw)))
                n = source_request if source_request is not None else 16
                rec.decoded = np.zeros((n, 288, 512, 3), dtype=np.uint8)
                # Distinguishable frames so the cadence selector is provable.
                for i in range(n):
                    rec.decoded[i, 0, 0, 0] = i % 256
                return (rec.decoded,)

        return {"checkpoint": Ckpt, "text_encode": TextEncode,
                "context": Context, "ade": Ade, "latent": Latent,
                "sampler": Sampler, "decode": Decode}


def _request(target=32, seed=4242, shot_id="shot_b001"):
    return {
        "shot_id": shot_id,
        "request_id": shot_id,
        "text_prompt": "a tall stooped figure, mid-shot or wider, turns",
        "negative_prompt": "text, watermark, caption, lettering, subtitles",
        "timing": {"target_frame_count": target},
        "seed_bundle": {"request_seed": seed},
    }


def _engine_with(rec, monkeypatch, source_request=None):
    eng = gs.GhostSignalEngine()
    eng._classes = rec.classes(source_request=source_request)
    eng._loaded = True
    return eng


def _prepared(eng, rec):
    """The `prepared` dict `prepare()` would have produced, without a lease."""
    eng._patchers.append(rec.base_model)
    return {"engine_id": eng.name, "lease": None,
            "patchers": eng._patchers, "session_ctx": {},
            "base_model": (rec.base_model,), "clip": (rec.clip,),
            "vae": (rec.vae,), "recipe": gs.GHOST_RECIPE_RECEIPT}


@pytest.fixture
def rendered(monkeypatch):
    """One complete mocked render; yields (recorder, raw, engine, prepared)."""
    rec = _Recorder()
    target = 32
    source_request = gs.ghost_source_request(target)
    eng = _engine_with(rec, monkeypatch, source_request=source_request)
    prepared = _prepared(eng, rec)

    captured = {}

    def fake_encode(frames, out_path, fps, **kw):
        captured["frames"] = np.array(frames)
        captured["fps"] = fps
        return (out_path, int(np.asarray(frames).shape[0]))

    monkeypatch.setattr(wb, "encode_frames_to_silent_mp4", fake_encode)
    monkeypatch.setattr(wb, "reclaim_idle_models",
                        lambda reason="": rec.calls.append(("reclaim", reason)))
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")

    raw = eng.render_clip(_request(target=target), prepared)
    return rec, raw, eng, prepared, captured


def test_exactly_eight_node_instances_run_and_only_one_ksampler(rendered):
    rec, _raw, _eng, _prepared, _cap = rendered
    node_calls = [c for c, _ in rec.calls if c != "reclaim"]
    assert node_calls.count("ckpt") == 0, (
        "the checkpoint belongs to prepare(), not to render_clip -- running it "
        "per beat would reload SD1.5 for every beat of the episode")
    assert node_calls.count("text_encode") == 2      # positive + negative
    assert node_calls.count("context") == 1
    assert node_calls.count("ade") == 1
    assert node_calls.count("latent") == 1
    assert node_calls.count("sampler") == 1
    assert node_calls.count("decode") == 1
    assert len(node_calls) == 7                      # + the ckpt in prepare = 8


def test_stage_order_is_encode_then_sample_then_decode(rendered):
    rec, _raw, _eng, _prepared, _cap = rendered
    order = [c for c, _ in rec.calls]
    assert order.index("text_encode") < order.index("sampler")
    assert order.index("context") < order.index("ade")
    assert order.index("ade") < order.index("sampler")
    assert order.index("latent") < order.index("sampler")
    assert order.index("sampler") < order.index("decode")
    # The reclaim seams sit between the stages, not after everything.
    reclaims = [i for i, (c, _) in enumerate(rec.calls) if c == "reclaim"]
    assert len(reclaims) == 2
    assert reclaims[0] > order.index("text_encode")
    assert reclaims[0] < order.index("sampler")
    assert reclaims[1] > order.index("sampler")
    assert reclaims[1] < order.index("decode")


def test_every_node_receives_its_exact_frozen_inputs(rendered):
    rec, _raw, _eng, _prepared, _cap = rendered
    by_node = {}
    for name, kw in rec.calls:
        by_node.setdefault(name, []).append(kw)

    ctx = by_node["context"][0]
    assert ctx == {
        "context_length": 16, "context_overlap": 4, "fuse_method": "pyramid",
        "use_on_equal_length": False, "start_percent": 0.0,
        "guarantee_steps": 1}

    ade = by_node["ade"][0]
    assert ade["model_name"] == "mm-p_0.5.pth"
    assert ade["beta_schedule"] == "autoselect"
    assert set(ade) == {"model", "model_name", "beta_schedule",
                        "context_options"}

    lat = by_node["latent"][0]
    assert lat["width"] == 512 and lat["height"] == 288
    assert lat["batch_size"] == gs.ghost_source_request(32)

    smp = by_node["sampler"][0]
    assert smp["seed"] == 4242
    assert smp["steps"] == 20 and smp["cfg"] == 8.0
    assert smp["sampler_name"] == "euler" and smp["scheduler"] == "normal"
    assert smp["denoise"] == 1.0
    assert set(smp) == {"model", "seed", "steps", "cfg", "sampler_name",
                        "scheduler", "positive", "negative", "latent_image",
                        "denoise"}

    assert set(by_node["decode"][0]) == {"samples", "vae"}


def test_the_ade_optional_sockets_are_OMITTED_not_passed_none(rendered):
    """Omission is the contract. An invented `None` widget is not omission --
    it hands the pinned implementation an explicit answer it never asked for."""
    rec, _raw, _eng, _prepared, _cap = rendered
    ade = next(kw for name, kw in rec.calls if name == "ade")
    for socket in gs.GHOST_ADE_OMITTED_SOCKETS:
        assert socket not in ade, (
            "ADE optional socket %r was passed (value %r); it must be absent "
            "from the input dict entirely" % (socket, ade.get(socket)))
    ctx = next(kw for name, kw in rec.calls if name == "context")
    for socket in gs.GHOST_CONTEXT_OMITTED_SOCKETS:
        assert socket not in ctx


FORBIDDEN_CLASSES = (
    "ADE_AnimateDiffSamplingSettings", "ADE_MultivalDynamic",
    "VHS_VideoCombine", "RIFE", "ImageUpscaleWithModel", "IPAdapter",
    "ControlNetApply", "LatentUpscale", "ADE_AnimateDiffKeyframe",
    "ADE_LoopedUniformContext", "VAELoader",
)


def test_no_forbidden_node_can_enter_the_graph():
    """Checked against the RESOLVER MAP, not against the module text.

    `GHOST_NODE_CANDIDATES` is the only door a class name can come through --
    every graph below resolves through it -- so this is both stricter and
    immune to the adapter's own docstring, which legitimately NAMES these
    classes to record that they are excluded. A source grep failed here on
    exactly that prose, and punishing a lane for documenting itself teaches
    authors to delete the documentation.
    """
    resolved = {n for names in gs.GHOST_NODE_CANDIDATES.values() for n in names}
    for forbidden in FORBIDDEN_CLASSES:
        assert not any(forbidden in name for name in resolved), (
            "%s is reachable through the resolver map %s" % (forbidden, resolved))
    assert len(resolved) == 7


def test_the_wire_ledger_uses_only_slot_zero_aliases():
    """`run_graph.external_results` normalises a value to a ONE-SLOT tuple, so
    a cross-stage alias can only ever be `Wire(name, 0)`. A graph addressing
    `ckpt[1]` after the tuple was split would read past the end of a 1-tuple."""
    import textwrap
    # dedent, NOT cleandoc: cleandoc strips the leading indent from the FIRST
    # line only, which leaves an unparseable `def` header over an indented body.
    src = textwrap.dedent(inspect.getsource(gs.GhostSignalEngine.render_clip))
    tree = ast.parse(src)
    wires = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "Wire"]
    assert len(wires) == 12, (
        "render_clip builds %d Wire(s); the SOURCE ledger is exactly twelve "
        "(2 encode + 6 sample + 2 decode, plus the 2 that exist only inside "
        "the domain-adapter branch). The number a lane EXECUTES is a different "
        "and stronger claim -- ten on a clean lane, twelve on a haunted one -- "
        "and tests/test_ghost_signal_haunted.py proves that behaviourally. "
        "This test owns the slot-0 law below, which must hold for every Wire "
        "in the file whether its branch runs or not." % len(wires))
    for call in wires:
        assert len(call.args) == 2, "every Wire must state its slot explicitly"
        slot = call.args[1]
        assert isinstance(slot, ast.Constant) and slot.value == 0, (
            "a cross-stage Wire addresses slot %r; externals are one-slot "
            "tuples, so nothing may address ckpt[1] or ckpt[2] after the "
            "checkpoint tuple has been split"
            % (getattr(slot, "value", slot),))


# --------------------------------------------------------------------------- #
# Ownership, release order, and failure law
# --------------------------------------------------------------------------- #

def test_both_sampling_patchers_detach_in_ade_then_base_order_before_decode(
        rendered):
    rec, _raw, eng, prepared, _cap = rendered
    assert rec.ade_model.detached == 1
    assert rec.base_model.detached == 1
    # Identity-removed IN PLACE from every tracked list.
    assert rec.ade_model not in eng._patchers
    assert rec.base_model not in eng._patchers
    assert rec.ade_model not in prepared["patchers"]
    assert rec.base_model not in prepared["patchers"]
    # The Ghost-owned references are cleared too.
    assert "ade_model" not in prepared
    assert "base_model" not in prepared
    # Successful sampling leaves NEITHER patcher in the teardown bucket.
    assert prepared["patchers"] == []


def test_a_duplicate_identity_detaches_exactly_once():
    """The candidate list is identity-deduplicated, so a lane whose ADE loader
    returned the base model unchanged must not detach it twice."""
    eng = gs.GhostSignalEngine()
    shared = _Handle("shared")
    eng._patchers.append(shared)
    prepared = {"patchers": eng._patchers, "ade_model": shared,
                "base_model": (shared,)}
    eng._release_sampling_patchers_before_decode(prepared)
    assert shared.detached == 1
    assert prepared["patchers"] == []


def test_a_raising_detach_keeps_the_patcher_tracked_and_blocks_decode(
        monkeypatch):
    rec = _Recorder()
    rec.ade_model = _Handle("ade_model", detach_raises=True)
    source_request = gs.ghost_source_request(32)
    eng = _engine_with(rec, monkeypatch, source_request=source_request)
    prepared = _prepared(eng, rec)
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)

    with pytest.raises(wb.GraphExecutionError) as excinfo:
        eng.render_clip(_request(), prepared)
    assert "detach" in str(excinfo.value).lower()
    # DECODE NEVER RAN.
    assert not any(name == "decode" for name, _ in rec.calls)
    # The failed candidate STAYS TRACKED so base teardown can try again --
    # dropping it would leave it resident with nobody holding a handle.
    assert rec.ade_model in prepared["patchers"]


def test_a_non_callable_detach_is_refused_by_name(monkeypatch):
    rec = _Recorder()
    rec.ade_model = _Handle("ade_model", detach_callable=False)
    eng = _engine_with(rec, monkeypatch,
                       source_request=gs.ghost_source_request(32))
    prepared = _prepared(eng, rec)
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    with pytest.raises(wb.GraphExecutionError):
        eng.render_clip(_request(), prepared)
    assert not any(name == "decode" for name, _ in rec.calls)


def test_teardown_releases_the_lease_even_when_unload_raises(monkeypatch):
    eng = gs.GhostSignalEngine()
    released = []

    class Boom(gs.GhostSignalEngine):
        def unload(self):
            raise RuntimeError("unload refused")

    eng = Boom()
    from nodes._otr_video_engines import motion_common as mc
    monkeypatch.setattr(mc._GR, "release", lambda lease: released.append(lease))
    monkeypatch.setattr(mc._GR, "wait_until_stable",
                        lambda **kw: None)
    with pytest.raises(RuntimeError):
        eng.teardown({"lease": "L1", "patchers": []})
    assert released == ["L1"], "the lease must be released in a finally"


def test_the_concrete_load_override_exists_because_base_prepare_calls_it():
    """`MotionEngineBase.prepare` calls `self.load()` unconditionally; the base
    implementation raises NotImplementedError, so inheriting it would kill
    every beat."""
    assert "load" in gs.GhostSignalEngine.__dict__
    from nodes._otr_video_engines import motion_common as mc
    assert gs.GhostSignalEngine.load is not mc.MotionEngineBase.load


def test_nothing_calls_unload_all_models():
    """V-4 / V-5. Asserted against CALL NODES in the AST, not a text grep: the
    adapter's docstrings say in words that it never calls this, and a grep
    cannot tell the promise from the breach."""
    tree = ast.parse(inspect.getsource(gs))
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            called.add(func.attr)
        elif isinstance(func, ast.Name):
            called.add(func.id)
    assert "unload_all_models" not in called


def test_compute_real_frame_budget_is_never_touched(monkeypatch):
    """A DETONATION TEST. This lane's cadence is quantum-1 and audio-derived;
    the inherited budget path only snaps and conditionally refuses, so touching
    it would refuse legal beats for a reason that does not apply here."""
    rec = _Recorder()
    eng = _engine_with(rec, monkeypatch,
                       source_request=gs.ghost_source_request(32))
    prepared = _prepared(eng, rec)

    def detonate(*a, **kw):
        raise AssertionError("compute_real_frame_budget must never be called")

    monkeypatch.setattr(eng, "compute_real_frame_budget", detonate)
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    monkeypatch.setattr(wb, "encode_frames_to_silent_mp4",
                        lambda frames, out, fps, **kw: (out, len(frames)))
    eng.render_clip(_request(), prepared)      # must not raise


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

def test_a_missing_artifact_is_a_named_engine_unusable(monkeypatch):
    from nodes._otr_video_engines.registry import EngineUnusable
    eng = gs.GhostSignalEngine()
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: None)
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    with pytest.raises(EngineUnusable) as excinfo:
        eng.assert_usable({}, {})
    text = str(excinfo.value)
    assert "v1-5-pruned-emaonly-fp16.safetensors" in text
    assert "checkpoints" in text


def test_a_truncated_artifact_is_named_rather_than_traced(monkeypatch, tmp_path):
    from nodes._otr_video_engines.registry import EngineUnusable
    stub = tmp_path / "v1-5-pruned-emaonly-fp16.safetensors"
    stub.write_bytes(b"nowhere near big enough")
    eng = gs.GhostSignalEngine()
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path",
                        lambda self: str(stub))
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path",
                        lambda self: str(stub))
    with pytest.raises(EngineUnusable) as excinfo:
        eng.assert_usable({}, {})
    assert "truncated" in str(excinfo.value)


def test_a_missing_node_class_names_the_pack(monkeypatch, tmp_path):
    from nodes._otr_video_engines.registry import EngineUnusable
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * 16)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: str(big))
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: str(big))
    monkeypatch.setattr(os.path, "getsize", lambda p: 10 ** 12)
    monkeypatch.setattr(wb, "node_class_mappings", lambda mapping=None: {})
    eng = gs.GhostSignalEngine()
    with pytest.raises(EngineUnusable) as excinfo:
        eng.assert_usable({}, {})
    text = str(excinfo.value)
    assert "ADE_AnimateDiffLoaderGen1" in text
    assert "AnimateDiff-Evolved" in text


def test_a_blank_negative_is_refused_and_no_constant_substitutes():
    eng = gs.GhostSignalEngine()
    req = _request()
    req["negative_prompt"] = ""
    with pytest.raises(RuntimeError) as excinfo:
        eng._assert_required_inputs(eng._build_render_request(req))
    assert "negative" in str(excinfo.value).lower()
    assert "NO ENGINE-SIDE CONSTANT" in str(excinfo.value)


def test_a_blank_positive_is_refused():
    eng = gs.GhostSignalEngine()
    req = _request()
    req["text_prompt"] = "   "
    with pytest.raises(RuntimeError):
        eng._assert_required_inputs(eng._build_render_request(req))


def test_a_decoded_count_mismatch_refuses_rather_than_pads(monkeypatch):
    rec = _Recorder()
    # Decode returns FEWER frames than were requested.
    eng = _engine_with(rec, monkeypatch, source_request=5)
    prepared = _prepared(eng, rec)
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    with pytest.raises(RuntimeError) as excinfo:
        eng.render_clip(_request(target=32), prepared)
    assert "decoded" in str(excinfo.value)
    assert "pads nothing" in str(excinfo.value)


def test_the_adapter_never_downloads_anything():
    src = inspect.getsource(gs)
    for token in ("urllib", "requests", "huggingface_hub", "snapshot_download",
                  "hf_hub_download", "urlretrieve", "wget", "curl"):
        assert token not in src, (
            "%s appears in the Ghost adapter; the offline invariant means a "
            "missing artifact fails closed, never fetches" % token)


# --------------------------------------------------------------------------- #
# Cadence (pure)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("target", [1, 2, 3, 12, 13, 49, 50])
def test_the_cadence_arithmetic_is_exact(target):
    unique = gs.ghost_unique_source_count(target)
    assert unique == -(-target // 2)                # ceil(T/2), integer-only
    requested = gs.ghost_source_request(target)
    assert requested == max(unique, 16)

    selector = gs.ghost_hold2_selector(target)
    assert len(selector) == target                  # exact output length
    assert all(0 <= i < unique for i in selector)   # in range
    assert selector == sorted(selector)             # monotonically nondecreasing

    counts = {i: selector.count(i) for i in range(unique)}
    for i in range(unique - 1):
        assert counts[i] == 2, "source %d appears %d times" % (i, counts[i])
    assert counts[unique - 1] == (2 if target % 2 == 0 else 1)

    receipts = gs.ghost_cadence_receipts(target, requested)
    assert receipts["cadence_tail_trim"] in (0, 1)
    assert receipts["cadence_tail_trim"] == (2 * unique) - target
    assert receipts["native_frame_count"] == target
    assert receipts["cadence_delivered_frame_count"] == target
    assert receipts["cadence_source_frame_count"] == unique
    assert receipts["model_frame_count"] == requested
    assert receipts["cadence_mode"] == "hold_2"
    assert receipts["extension_mode"] == "none"
    assert receipts["delivery_scale_mode"] == "lanczos_clean_full_frame"
    # Duration is EXACTLY T/25 -- a resample, never an extension.
    assert target / 25.0 == pytest.approx(
        receipts["cadence_delivered_frame_count"] / 25.0)


def test_the_structural_surplus_is_reported_separately_from_tail_trim():
    """`model_frame_count - cadence_source_frame_count` exposes frames generated
    only to satisfy the node's 16-frame minimum. `cadence_tail_trim` covers ONLY
    the hold-2 surplus, and conflating them would hide real model work."""
    receipts = gs.ghost_cadence_receipts(4)          # U=2, requested=16
    assert receipts["model_frame_count"] == 16
    assert receipts["cadence_source_frame_count"] == 2
    assert receipts["model_frame_count"] - receipts[
        "cadence_source_frame_count"] == 14
    assert receipts["cadence_tail_trim"] == 0


def test_a_zero_frame_beat_is_refused_not_papered_over():
    with pytest.raises(gs.GhostCadenceError):
        gs.ghost_unique_source_count(0)


def test_the_delivered_frames_are_the_declared_hold_two_pairs(rendered):
    rec, raw, _eng, _prepared, captured = rendered
    frames = captured["frames"]
    assert frames.shape[0] == 32
    assert captured["fps"] == 25
    marks = [int(frames[i, 0, 0, 0]) for i in range(frames.shape[0])]
    assert marks == gs.ghost_hold2_selector(32)
    assert raw["frame_count"] == 32
    assert raw["cadence_mode"] == "hold_2"
    assert raw["extension_mode"] == "none"


def test_two_beats_with_identical_text_get_distinct_identities(monkeypatch):
    """A prompt-only cache key would collapse two beats into one. The shot id
    and the resolved seed are in the identity precisely so it cannot."""
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    eng = gs.GhostSignalEngine()
    a = eng.shot_cache_identity(_request(shot_id="shot_b001", seed=11))
    b = eng.shot_cache_identity(_request(shot_id="shot_b002", seed=22))
    same = eng.shot_cache_identity(_request(shot_id="shot_b001", seed=11))
    assert a != b
    assert a == same
    assert "shot_b001" in a and "shot_b002" in b


# --------------------------------------------------------------------------- #
# canonicalize
# --------------------------------------------------------------------------- #

def test_canonicalize_refuses_a_canvas_that_is_not_exactly_512x288(monkeypatch):
    """The clean full-frame chain does NO pad or crop, so a source that is not
    exactly 16:9 would be silently distorted on the way to 1920x1080. Asserting
    it here is what earns the composite's right to skip its own probe."""
    eng = gs.GhostSignalEngine()
    monkeypatch.setattr(gs, "ffprobe_clip_fields",
                        lambda p: {"width": 512, "height": 320, "fps": 25})
    monkeypatch.setattr(gs, "validate_silent_clip_contract",
                        lambda fields, fps: None)
    with pytest.raises(RuntimeError) as excinfo:
        eng.canonicalize({"out_path": "x.mp4", "frame_count": 4},
                         _request(), {})
    assert "512x288" in str(excinfo.value)


def test_canonicalize_probes_once_and_carries_every_receipt(monkeypatch):
    probes = []
    eng = gs.GhostSignalEngine()
    monkeypatch.setattr(gs, "ffprobe_clip_fields",
                        lambda p: (probes.append(p),
                                   {"width": 512, "height": 288, "fps": 25})[1])
    monkeypatch.setattr(gs, "validate_silent_clip_contract",
                        lambda fields, fps: None)
    raw = {"out_path": "x.mp4", "frame_count": 32,
           "recipe": gs.GHOST_RECIPE_RECEIPT, "render_canvas": "512x288"}
    raw.update(gs.ghost_cadence_receipts(32))
    clip = eng.canonicalize(raw, _request(), {})
    assert len(probes) == 1, "ONE probe, fed to both checks"
    assert clip["has_audio"] is False
    assert clip["engine_id"] == NAME
    assert clip["fps"] == 25
    for key in ("model_frame_count", "cadence_mode",
                "cadence_source_frame_count", "cadence_delivered_frame_count",
                "cadence_tail_trim", "delivery_scale_mode"):
        assert key in clip, "canonicalize dropped %s" % key


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

def test_canonical_clip_accepts_the_new_receipts_and_still_forbids_extras():
    clip = vschemas.CanonicalClip(
        clip_id="c1", path="x.mp4",
        delivery_scale_mode="lanczos_clean_full_frame",
        cadence_mode="hold_2", cadence_source_frame_count=16,
        cadence_delivered_frame_count=32, cadence_tail_trim=0,
        model_frame_count=16)
    assert clip.cadence_mode == "hold_2"
    assert clip.model_frame_count == 16
    with pytest.raises(Exception):
        vschemas.CanonicalClip(clip_id="c1", path="x.mp4",
                               not_a_real_field="nope")


def test_canonical_clip_defaults_keep_every_receipt_absent():
    """Absence is load-bearing: a legacy row must not acquire six nulls."""
    clip = vschemas.CanonicalClip(clip_id="c1", path="x.mp4")
    for field in ("delivery_scale_mode", "cadence_mode",
                  "cadence_source_frame_count",
                  "cadence_delivered_frame_count", "cadence_tail_trim",
                  "model_frame_count"):
        assert getattr(clip, field) is None


def test_shot_row_subject_sigil_is_optional_string_data():
    row = vschemas.ShotRow(shot_id="shot_b001")
    assert row.subject_sigil is None
    row2 = vschemas.ShotRow(shot_id="shot_b001", subject_sigil="a tall figure")
    assert row2.subject_sigil == "a tall figure"
