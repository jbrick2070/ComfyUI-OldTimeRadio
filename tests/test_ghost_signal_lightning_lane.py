"""``animatediff15_lightning_video`` -- the distilled speed peer, pinned.

WHY THIS FILE EXISTS AT ALL. This lane adds no render code: the recipe seam
landed first (``8f882788``) and the parent reads all six cells through ``self``,
so the engine is a class declaration and nothing else. That is precisely the
condition under which a lane can be WRONG IN SILENCE -- there is no new code
path to fail, only declarations that can quietly disagree with the graph, the
weights on disk, or the receipt. Every assertion here is aimed at one of those
disagreements, and three of them pin defects this lane WOULD have shipped
without the seam:

  * the inherited 1.7 GB byte floor would have refused the byte-perfect 908 MB
    Lightning module as "truncated" -- the same false accusation that killed the
    v3 lane's first live leg;
  * inheriting even ONE recipe cell would sample on the golden 20-step / cfg-8.0
    recipe while stamping a Lightning receipt;
  * ``"sqrt_linear"`` is a REAL and DIFFERENT dropdown value from
    ``"sqrt_linear (AnimateDiff)"`` in the installed pack, so a plausible-looking
    string would silently select another beta schedule.

None of the three raises. All three produce a confident receipt over the wrong
picture, which is this repo's defining failure mode.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._otr_video_engines import eng_ghost_signal as gs  # noqa: E402
from nodes._otr_video_engines import eng_ghost_signal_lightning as L  # noqa: E402
from nodes._otr_video_engines import registry as vreg  # noqa: E402

ENGINE_ID = "animatediff15_lightning_video"


@pytest.fixture
def eng():
    return vreg.get_engine(ENGINE_ID)


# ---------------------------------------------------------------------------
# G2.2 -- the canvas pin. A `render_canvas` declaration needs a test naming the
# engine id AND the literal canvas, or the declaration can drift from the graph
# in silence (the preflight gate reads THIS).
# ---------------------------------------------------------------------------

def test_animatediff15_lightning_video_renders_512x288(eng):
    """512x288, the family's native canvas, unchanged.

    Distillation changes the number of STEPS, never the geometry, so this lane
    delivers by the same clean full-frame Lanczos as its siblings -- which only
    works from an EXACT source, hence the literal here and the exact-size
    refusal the parent's ``canonicalize`` performs.
    """
    assert eng.render_canvas == (512, 288)
    assert (gs.GHOST_CANVAS_W, gs.GHOST_CANVAS_H) == (512, 288)
    assert eng.delivery_scale_mode == gs.GHOST_DELIVERY_SCALE_MODE


# ---------------------------------------------------------------------------
# The byte floor -- the defect the seam was cut for
# ---------------------------------------------------------------------------

def test_the_floor_is_below_its_own_artifact_and_not_the_parents(eng):
    """THE PINNED DEFECT. The real file is 908,929,664 bytes: SMALLER than
    ``v3_sd15_mm.ckpt`` (1,673,262,583) and about half of ``mm-p_0.5``. The
    inherited 1.7 GB floor would refuse it as truncated while the file is
    byte-perfect -- the exact failure that killed the v3 lane's first live leg,
    and the reason ``docs/ADDING_IMAGE_AND_VIDEO_LANES.md`` says to ask what
    ELSE was sized for the parent."""
    real_bytes = 908_929_664
    assert eng.motion_min_bytes < real_bytes, (
        "the floor would refuse the real artifact")
    assert eng.motion_min_bytes != gs.GHOST_MOTION_MIN_BYTES, (
        "inheriting the golden floor is the whole defect this lane exists past")
    # G1.3: still tight enough that a truncated fetch is NAMED, not traced
    # through a loader stack. 15% is the sibling adapter's own margin.
    assert eng.motion_min_bytes >= real_bytes * 0.85, (
        "a floor far below the artifact stops catching truncation at all")


def test_the_module_name_and_the_step_count_agree(eng):
    """Upstream's own Note is a RULE, not advice: *"Make sure loading the
    correct Animatediff-Lightning checkpoint corresponding to the inference
    steps."* Two numbers that happen to agree today would drift; this makes the
    filename and the cell prove each other.

    AND NOTHING ELSE CAN. The 1/2/4/8-step ComfyUI checkpoints are ALL EXACTLY
    908,929,664 bytes upstream, so the byte floor is blind to the variant: load
    the 4-step file under an 8-step receipt and nothing raises, it simply
    samples a schedule those weights were not distilled for. This assertion is
    the only guard against that."""
    assert "%dstep" % eng.steps in eng.motion_module_name, (
        "%s samples %d steps -- swapping the artifact means changing BOTH"
        % (eng.motion_module_name, eng.steps))


# ---------------------------------------------------------------------------
# All six recipe cells -- inheriting any ONE of them is the silent defect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell,value,golden", [
    ("steps", 8, gs.GHOST_STEPS),
    ("cfg", 1.0, gs.GHOST_CFG),
    ("sampler_name", "euler", gs.GHOST_SAMPLER_NAME),
    ("scheduler", "sgm_uniform", gs.GHOST_SCHEDULER),
    ("beta_schedule", "sqrt_linear (AnimateDiff)", gs.GHOST_BETA_SCHEDULE),
])
def test_each_distilled_cell_is_the_upstream_value(eng, cell, value, golden):
    """Read off ByteDance's own ``comfyui/animatediff_lightning_workflow.json``,
    not its README: the ``ADE_AnimateDiffLoaderGen1`` widgets and the KSampler
    widgets. ``euler`` is the one cell that legitimately MATCHES the golden
    lane, and it is asserted by value rather than by difference for that
    reason."""
    assert getattr(eng, cell) == value
    if cell != "sampler_name":
        assert getattr(eng, cell) != golden, (
            "%s still holds the golden value -- this lane would sample on the "
            "20-step cfg-8.0 recipe while stamping a Lightning receipt" % cell)


def test_denoise_is_deliberately_unchanged(eng):
    """Not every cell moves. Lightning starts from an empty latent at full
    denoise exactly as the golden lane does; asserting this stops a future
    reader from "completing the set"."""
    assert eng.denoise == 1.0 == gs.GHOST_DENOISE


def test_the_beta_schedule_is_the_parenthesised_dropdown_value(eng):
    """``"sqrt_linear"`` alone is ``BetaSchedules.RAW_SQRT_LINEAR``, a REAL and
    DIFFERENT option in the installed pack. Dropping the parenthetical selects
    another schedule with no error anywhere."""
    assert eng.beta_schedule == "sqrt_linear (AnimateDiff)"
    assert eng.beta_schedule != "sqrt_linear"


# ---------------------------------------------------------------------------
# No adapter -- from the source graph, not from preference
# ---------------------------------------------------------------------------

def test_it_carries_no_domain_adapter(eng):
    """ByteDance's workflow is checkpoint -> loader -> KSampler with NO LoRA
    node. The v3 adapter is v3-PAIRED and composing it with a non-v3 distilled
    module has no upstream receipt, so this lane subclasses the GOLDEN base."""
    assert eng.lora_name is None
    assert isinstance(eng, gs.GhostSignalEngine)
    from nodes._otr_video_engines import eng_ghost_signal_official as off
    assert not isinstance(eng, off.GhostSignalV3HauntedEngine), (
        "subclassing the haunted lane would drag in the v3-paired adapter")


def test_the_registry_row_asks_for_the_three_it_loads_and_no_adapter():
    """THREE artifacts since 2026-09-09 -- checkpoint, motion module, external
    decoder. Still NOT the v3 domain adapter: that is v3-paired and this lane
    never loads it, so listing it would make the S5 wizard demand 97 MB for
    nothing. "Three artifacts" and "the haunted three" are different sets."""
    req = vreg.CAPABILITIES[ENGINE_ID]["model_requirements"]
    assert L.MM_LIGHTNING_NAME in req
    assert "v1-5-pruned-emaonly-fp16.safetensors" in req
    assert L.VAE_FT_MSE_NAME in req
    assert "v3_sd15_adapter.ckpt" not in req
    assert len(req) == 3


# ---------------------------------------------------------------------------
# THE COLLISION: cfg 1.0 deletes the unconditional pass
# ---------------------------------------------------------------------------

def test_negative_is_live_mirrors_comfyui_own_rule():
    """``comfy/samplers.py::sampling_function`` sets ``uncond_ = None`` when
    ``math.isclose(cond_scale, 1.0)``. At cfg 1.0 the negative prompt is not
    weakened -- it is never evaluated. This helper must agree with that rule
    exactly, because the receipt's honesty is computed from it."""
    assert L.negative_is_live(1.0) is False
    assert L.negative_is_live(1) is False
    assert L.negative_is_live(2.0) is True
    assert L.negative_is_live(gs.GHOST_CFG) is True
    # A malformed value is not "live": claiming a defence ran on an unparseable
    # cfg is the one direction that must never be optimistic.
    assert L.negative_is_live("nonsense") is False
    assert L.negative_is_live(None) is False


def test_the_default_recipe_knows_its_negative_is_inert(eng, monkeypatch):
    """The receipt must never claim the Ghost lettering defence ran when
    ComfyUI skipped the pass. This is the lane's single biggest honesty risk:
    the negative prompt is still COMPOSED and still stamped, it simply does
    nothing at cfg 1.0."""
    monkeypatch.delenv(L.LIGHTNING_CFG_ENV, raising=False)
    assert eng.cfg == 1.0
    assert L.negative_is_live(eng.cfg) is False


def test_the_cfg_dial_is_an_env_var_not_an_edit(eng, monkeypatch):
    """Getting the live negative back must cost an env var. It doubles the UNet
    passes, which is the operator's trade to make, not the code's."""
    monkeypatch.setenv(L.LIGHTNING_CFG_ENV, "2.0")
    assert eng.cfg == 2.0
    assert L.negative_is_live(eng.cfg) is True


@pytest.mark.parametrize("bad", ["", "  ", "abc", "-1", "0", "nan", "inf"])
def test_a_malformed_cfg_is_a_named_refusal_not_a_silent_default(
        eng, monkeypatch, bad):
    """Unlike ``lora_strength``, this dial REFUSES. cfg decides whether the
    unconditional pass runs at all, so a silently-defaulted value would print a
    receipt describing a render nobody asked for.

    The empty/whitespace cases are the exception and deliberately so: UNSET IS
    UNCHANGED is the repo's own sweep-knob law."""
    monkeypatch.setenv(L.LIGHTNING_CFG_ENV, bad)
    if not bad.strip():
        assert eng.cfg == 1.0
        return
    with pytest.raises(vreg.EngineUnusable) as exc:
        _ = eng.cfg
    assert L.LIGHTNING_CFG_ENV in str(exc.value)


def test_cfg_joins_the_shot_cache_identity(eng, monkeypatch):
    """A cfg sweep must not be served another arm's clip out of the shot cache.
    The parent's identity carries the recipe receipt, which does NOT move when
    the env dial does -- so without this the two arms collide."""
    import inspect
    src = inspect.getsource(type(eng).shot_cache_identity)
    assert "cfg" in src
    assert "super().shot_cache_identity" in src, (
        "the parent's shot id / seed / prompt digests must still be in there")


# ---------------------------------------------------------------------------
# Identity, declarations, and the class-level property footgun
# ---------------------------------------------------------------------------

def test_it_declares_accepts_still_explicitly(eng):
    """G3.6: declared, not inherited. An engine that stays silent mints no
    still, the operator's chosen image model is never invoked for the role, and
    the episode renders anyway with nothing reporting it."""
    assert "accepts_still" in type(eng).__dict__
    assert eng.accepts_still is False
    assert eng.still_plan == ()


def test_nothing_reads_cfg_off_the_class(eng):
    """``cfg`` is a PROPERTY here, which the parent explicitly warns about on
    ``hold_factor``: class-level access returns a ``<property object>``, not a
    number. That is safe ONLY while every reader uses an instance -- which
    ``register`` guarantees by instantiating at import
    (``engine_registry_base.py:148``) and keeping one shared instance. This
    test is the guarantee, so a future class-level read fails here rather than
    silently comparing a property object to 1.0."""
    assert isinstance(type(eng).__dict__["cfg"], property)
    assert isinstance(eng.cfg, float)
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in (repo / "nodes").rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "GhostSignalLightningEngine.cfg" in line:
                offenders.append("%s:%d" % (path.relative_to(repo), n))
    assert not offenders, (
        "class-level .cfg read(s) at %r would get a property object" % offenders)


def test_the_recipe_receipt_names_the_distillation(eng):
    """A receipt string that did not move would make Lightning renders
    indistinguishable from golden ones on disk."""
    assert eng.recipe_receipt_id != gs.GHOST_RECIPE_RECEIPT
    assert "lightning" in eng.recipe_receipt_id
    assert "512x288" in eng.recipe_receipt_id


def test_the_device_row_claims_only_what_has_been_proven():
    """MPS EARNED 2026-09-09, and the row moved only when the receipt existed.

    This test asserted `["cuda"]` from the lane's first commit until a complete
    episode rendered through the real OTR adapter path on an M4/16 GB -- 23
    beats, 2,736 delivered frames, 02:32:34, published to otr/obs/. The rule
    that kept it cuda-only for three commits is the same rule that lets it say
    mps now: a device row is a claim about PROVEN execution, and the proof is a
    file on disk, not a plausible reading of the source."""
    row = vreg.CAPABILITIES[ENGINE_ID]
    assert row["device_backends"] == ["mps", "cuda"]
    assert "cuda" in row["device_backends"], "removing cuda strands every NVIDIA profile"
    assert row["needs_fp8_te"] is False and row["needs_fp4_te"] is False, (
        "fp8/fp4 are the two things Metal cannot do; this stack needs neither, "
        "which is what makes the eventual mps claim plausible")


def test_it_is_not_a_default_for_any_role(eng):
    """Additive means additive: a new lane is opt-in through the director menu
    and never seizes a role."""
    assert eng.default_roles == ()
    assert eng.requires_flag is None


def test_the_golden_base_is_untouched():
    """The whole additive claim in one assertion -- read off the CLASS, because
    ``animatediff15_video`` IS NOT REGISTERED.

    That surprise is worth recording rather than working around. The operator
    retired every non-haunted AnimateDiff on 2026-08-23 (*"delete any
    animatediff that are not haunted"*), so the golden id is tombstoned in
    ``RETIRED_ENGINE_IDS`` and ``GhostSignalEngine`` survives only as the
    unregistered base three lanes inherit. The first draft of this test asked
    the registry for it and failed -- which is the tombstone working exactly as
    designed, and the reason this lane subclasses a BASE rather than a peer.
    """
    assert not vreg.is_registered("animatediff15_video")
    from nodes._otr_shared import public_engines as pe
    assert "animatediff15_video" in pe.RETIRED_ENGINE_IDS
    base = gs.GhostSignalEngine
    assert base.motion_module_name == gs.GHOST_MOTION_MODULE_NAME
    assert (base.steps, base.cfg) == (20, 8.0)
    assert base.scheduler == "normal"
    assert base.beta_schedule == "autoselect"
    assert base.motion_min_bytes == gs.GHOST_MOTION_MIN_BYTES


def test_the_two_shipping_siblings_are_untouched():
    """The lanes an operator can actually SELECT keep every recipe cell they
    had. Nothing about this lane reaches them."""
    for name in ("animatediff15_v3_haunted_video",
                 "animatediff15_v3_stillin_lab_video"):
        peer = vreg.get_engine(name)
        assert peer.motion_module_name == "v3_sd15_mm.ckpt"
        assert (peer.steps, peer.cfg) == (20, 8.0)
        assert peer.scheduler == "normal"
        assert peer.beta_schedule == "autoselect"
        assert peer.lora_name == "v3_sd15_adapter.ckpt"


# ---------------------------------------------------------------------------
# PROVISIONING -- two memberships, two questions, and getting one right broke
# the other
# ---------------------------------------------------------------------------

def _provision_module():
    import importlib.util
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_prov_for_test", str(repo / "scripts" / "otr_provision.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_it_is_not_routed_through_the_haunted_fetch_bundle():
    """THE DEFECT THIS LANE NEARLY SHIPPED. Every id in
    ``_ANIMATEDIFF_ENGINES`` routes to the ``"haunted"`` bundle, which installs
    ``v3_sd15_mm.ckpt`` + ``v3_sd15_adapter.ckpt`` -- 1.77 GB this lane never
    opens -- and NOT the one artifact it does. Adding the id to that set is the
    obvious move and it is wrong: provisioning would report success with the
    needed file absent, and the lane would then fail its byte floor against a
    file nobody fetched."""
    m = _provision_module()
    assert ENGINE_ID not in m._ANIMATEDIFF_ENGINES, (
        "membership there routes this lane to the v3 weights it never loads")
    # And prove the route it DOES take, so "not haunted" cannot quietly become
    # "not provisioned at all".
    import inspect
    src = inspect.getsource(m.profile_lanes)
    assert 'automatic.append("lightning")' in src
    assert src.index('== "%s"' % ENGINE_ID) < src.index("_ANIMATEDIFF_ENGINES"), (
        "the lightning branch must be evaluated BEFORE the _ANIMATEDIFF_ENGINES "
        "catch-all, or the catch-all wins and fetches the wrong weights")


def test_but_it_IS_in_the_no_still_set():
    """THE SECOND HALF, and it is the bug the first half caused.
    ``_NO_STILL_VIDEO_ENGINES`` is DERIVED from ``_ANIMATEDIFF_ENGINES``, so
    keeping this lane out of the fetch route silently kept it out of the
    no-still route too -- and a profile would then plan a 13-19 GB image
    download for stills an ``accepts_still = False`` engine never consumes.
    Two different questions; the lane needs its own answer to each."""
    m = _provision_module()
    assert ENGINE_ID in m._NO_STILL_VIDEO_ENGINES


def test_the_profile_plans_the_lightning_bundle_and_no_image_weights():
    """The end-to-end statement both memberships exist to produce."""
    import json
    import pathlib
    m = _provision_module()
    repo = pathlib.Path(__file__).resolve().parent.parent
    prof = json.loads((repo / "config" / "profiles" /
                       ("otr_w45_%s.json" % ENGINE_ID)).read_text(
                           encoding="utf-8"))
    plan = m.profile_lanes(prof)
    assert plan["automatic"] == ["lightning", "stable_audio_3"], plan
    assert not any("z_image" in lane for lane in plan["automatic"])
    assert not plan.get("manual"), plan


def test_the_haunted_lane_still_plans_exactly_what_it_did():
    """CLAUDE.md 0B in one assertion: the sibling is provably unchanged."""
    import json
    import pathlib
    m = _provision_module()
    repo = pathlib.Path(__file__).resolve().parent.parent
    prof = json.loads((repo / "config" / "profiles" /
                       "otr_w45_animatediff15_v3_haunted_video.json").read_text(
                           encoding="utf-8"))
    assert m.profile_lanes(prof)["automatic"] == ["haunted", "stable_audio_3"]


def test_the_fetch_bundle_is_fully_pinned():
    """``WeightSpec``'s own docstring: *"New promoted lanes must fill all
    three."* On THIS lane the SHA is not belt-and-braces -- it is the only
    check that can catch the wrong artifact, because upstream's 1/2/4/8-step
    ComfyUI checkpoints are all exactly 908,929,664 bytes."""
    import importlib.util
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_fetch_for_test", str(repo / "scripts" / "otr_fetch_lane_weights.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.LANES["lightning"]
    assert len(rows) == 3, "checkpoint + motion module + decoder, no adapter"
    for row in rows:
        assert row.revision and row.revision != "main", row
        assert row.expected_bytes, row
        assert row.expected_sha256 and len(row.expected_sha256) == 64, row
    module_row = [r for r in rows if r.destination == "animatediff_models"][0]
    assert module_row.path_in_repo == L.MM_LIGHTNING_NAME
    assert module_row.expected_bytes == 908_929_664
    assert module_row.expected_bytes >= L.MM_LIGHTNING_MIN_BYTES
    assert "lightning" in mod.LANE_INFO, "--list would print a blank row"


def test_the_cache_key_cannot_collide_across_a_negative_boundary():
    """THE ROUNDING DEFECT, PINNED. ``cfg=%.4f`` mapped 1.0 and 1.00004 to the
    same token while ``math.isclose`` puts them on OPPOSITE sides of the
    unconditional-pass optimization -- so one render could be served the
    other's clip despite differing in whether the negative prompt ran."""
    import math
    assert ("cfg=%.4f" % 1.0) == ("cfg=%.4f" % 1.00004), (
        "the rounding collision this guards against")
    assert math.isclose(1.0, 1.0) and not math.isclose(1.00004, 1.0)
    assert float(1.0).hex() != float(1.00004).hex(), "hex round-trips exactly"
    import inspect
    src = inspect.getsource(L.GhostSignalLightningEngine.shot_cache_identity)
    assert ".hex()" in src, "a rounded cfg token can collide"
    assert "negative_effective" in src


# ---------------------------------------------------------------------------
# THE FRAME MATH -- "every other lane does that" (operator, 2026-09-08)
# ---------------------------------------------------------------------------

def _upstream_windows(num_frames, length=16, overlap=4):
    """``create_windows_static_standard`` from the pinned AnimateDiff-Evolved,
    copied at build (2026-09-08) so this suite needs neither ComfyUI nor the
    node pack importable -- the same pattern as ``repeat_latent`` in the
    still-in lab lane. It is a REFERENCE, not a second rule: if upstream's
    scheduler changes, this copy is what makes the disagreement visible.
    """
    windows = []
    if num_frames <= length:
        return [list(range(num_frames))]
    delta = length - overlap
    for start_idx in range(0, num_frames, delta):
        ending = start_idx + length
        if ending >= num_frames:
            final_start = start_idx - (ending - num_frames)
            windows.append(list(range(final_start, final_start + length)))
            break
        windows.append(list(range(start_idx, start_idx + length)))
    return windows


def _overlaps(windows):
    return {windows[i - 1][-1] - windows[i][0] + 1
            for i in range(1, len(windows))}


def test_the_window_count_formula_matches_the_real_scheduler():
    """The claim "rounding up is free" rests entirely on this, so it is checked
    against the algorithm rather than asserted."""
    for n in range(1, 400):
        assert gs.ghost_context_window_count(n) == len(_upstream_windows(n)), n


def test_rounding_up_to_a_legal_count_never_adds_a_window():
    """THE WHOLE ARGUMENT. ``sampling.py`` invokes the model once per window on
    that window's slice, so sampler cost tracks the window COUNT, not the latent
    count -- and the aligned count has the identical count by construction,
    because rounding up IS taking that ceiling. Alignment is therefore free in
    sampler time; the only cost is decoding the surplus."""
    for n in range(1, 400):
        assert (len(_upstream_windows(gs.ghost_legal_source_count(n)))
                == len(_upstream_windows(n))), n


def test_a_legal_count_tiles_with_a_uniform_overlap():
    """What alignment actually buys."""
    for n in range(1, 400):
        laps = _overlaps(_upstream_windows(gs.ghost_legal_source_count(n)))
        assert laps in ({gs.GHOST_CONTEXT_OVERLAP}, set()), (n, laps)


def test_the_beat_this_lane_renders_is_currently_illegal():
    """PIN THE ACTUAL NUMBER. 125 source frames is what a 250-frame hold-2 beat
    asks for, and its final window overlaps the previous by 15 of 16 frames --
    it re-denoises almost entirely covered ground and the pyramid fuse weights
    the tail unevenly. NOT a crash: backing the final window up is upstream's
    deliberate clamp so every window stays full-length, and the operator reports
    never having seen it fail. It is simply not the arithmetic the rest of the
    pack does."""
    assert _overlaps(_upstream_windows(125)) == {4, 15}
    assert gs.ghost_legal_source_count(125) == 136
    assert _overlaps(_upstream_windows(136)) == {4}
    assert len(_upstream_windows(125)) == len(_upstream_windows(136)) == 11


def test_short_beats_pad_to_the_window_rather_than_shrinking_it():
    """The operator's other option -- "maybe we pad to the min" -- and the lane
    already did this half through ``GHOST_SOURCE_FLOOR``. Alignment is the same
    idea applied to the STRIDE instead of the FLOOR."""
    for n in (1, 5, 8, 15, 16):
        assert gs.ghost_legal_source_count(n) == gs.GHOST_CONTEXT_LENGTH


def test_the_stride_is_derived_not_a_fourth_literal():
    assert gs.GHOST_CONTEXT_STRIDE == (gs.GHOST_CONTEXT_LENGTH
                                       - gs.GHOST_CONTEXT_OVERLAP) == 12


def test_this_lane_opts_in_and_asks_for_the_legal_count(eng):
    assert eng.align_source_to_context_window is True
    assert eng._source_request_for(250) == 136, (
        "a 250-frame hold-2 beat needs 125 unique sources; the legal count is "
        "136 and costs the same eleven windows")


def test_the_siblings_did_not_opt_in_and_are_byte_identical():
    """CLAUDE.md 0B. Turning this on changes how many latents a beat asks for,
    which changes the picture -- and these lanes have PUBLISHED EPISODES. The
    default is False so they are untouched; flipping it for them is an operator
    decision backed by a 5080 comparison, not a driver one."""
    assert gs.GhostSignalEngine.align_source_to_context_window is False
    for name in ("animatediff15_v3_haunted_video",
                 "animatediff15_v3_stillin_lab_video"):
        peer = vreg.get_engine(name)
        assert peer.align_source_to_context_window is False
        # The exact number they asked for before the seam existed.
        assert peer._source_request_for(250) == 125
        assert peer._source_request_for(250) == gs.ghost_source_request(
            250, peer.hold_factor)


def test_the_surplus_is_reported_rather_than_hidden(eng):
    """``model_frame_count`` must name what the model really did, or the
    alignment becomes 11 frames of invisible work."""
    receipts = gs.ghost_cadence_receipts(250, eng._source_request_for(250),
                                         eng.hold_factor)
    assert receipts["model_frame_count"] == 136
    assert receipts["native_frame_count"] == 250
    assert receipts["cadence_source_frame_count"] == 125


def test_the_alignment_is_NOT_expressed_as_a_frame_contract_quantum(eng):
    """THE OBVIOUS "FIX" THAT WOULD BREAK THE LANE.

    ``FrameContract.quantum`` is the natural place to say ``16 + 12k``, and it
    is the WRONG place, because the two quantities are not the same one:

      * the contract governs DELIVERED frames -- the audio-derived
        ``target_frame_count`` the composite must match;
      * the context window constrains SOURCE frames -- what the sampler is
        handed, which at hold 2 is roughly half as many.

    Declaring ``quantum=12`` would therefore assert something false about
    delivered frames, and it would also make the lane splittable: a contract
    with a real quantum invites `PLANNING_CAP_ENGINES` membership, and with
    ``continuity=NONE`` the planner joins segments with ``join_mode="jump"`` --
    about fifteen jump cuts and ~240 latents where one continuous beat sampled
    125. `docs/MAC_PORTABILITY_GUIDE.md` 10.7 carries the arithmetic.

    So the alignment lives at the SOURCE REQUEST, and the contract stays
    unbounded. This test is the tripwire for anyone who tries to move it."""
    from nodes._otr_video_engines import frame_contract as fc
    contract = eng.frame_contract
    assert contract.quantum == 1, (
        "the window quantum belongs on the source request, not on the "
        "delivered-frame contract -- they are different quantities")
    assert contract.max_frames == 0, "a beat is ONE timeline"
    assert contract.continuity == fc.CONTINUITY_NONE
    assert ENGINE_ID not in fc.PLANNING_CAP_ENGINES, (
        "capping this lane produces jump cuts and MORE latents, not fewer")
    # And the source request is where it actually happens.
    assert eng._source_request_for(250) != gs.ghost_source_request(250, 2)


# ---------------------------------------------------------------------------
# THE EXTERNAL DECODER (2026-09-09) -- adopted on operator override
# ---------------------------------------------------------------------------

def test_it_decodes_with_ft_mse_and_not_the_checkpoint_vae(eng):
    """The A/B the operator asked for and then ruled on. Only the decoder
    changed between the two renders; latents were identical, which is exactly
    why this was the one recipe change safe to land after the fact."""
    assert eng.vae_name == "vae-ft-mse-840000-ema-pruned.safetensors"
    assert eng.vae_min_bytes < 334_641_190, "would refuse the real artifact"
    assert eng.vae_min_bytes >= 334_641_190 * 0.85, "too loose to catch a truncation"


def test_the_siblings_still_decode_with_the_checkpoints_own_vae():
    """CLAUDE.md 0B again. `vae_name` defaults to None on the parent, so the two
    lanes with PUBLISHED EPISODES load no decoder, require no third file and
    bind `ckpt_out[2]` exactly as they always did."""
    assert gs.GhostSignalEngine.vae_name is None
    assert gs.GhostSignalEngine.vae_min_bytes == 0
    for name in ("animatediff15_v3_haunted_video",
                 "animatediff15_v3_stillin_lab_video"):
        peer = vreg.get_engine(name)
        assert peer.vae_name is None
        assert peer._vae_path() is None, "a lane with no name is never asked"
        assert "vae_loader" not in peer._node_candidates(), (
            "no name means no loader node enters the graph at all")


def test_the_decoder_lane_gets_a_loader_node_and_the_others_do_not(eng):
    assert eng._node_candidates()["vae_loader"] == ("VAELoader",)


def test_prepare_rebinds_the_vae_only_when_a_name_is_declared():
    """The ONLY graph change this seam makes. `render_clip`'s decode node
    already consumes `owners["vae"]` as a one-slot tuple and does not care where
    it came from, so the rebinding is the whole wiring -- and the checkpoint's
    own VAE is simply never bound, rather than held resident alongside."""
    import inspect
    src = inspect.getsource(gs.GhostSignalEngine.prepare)
    assert 'prepared["vae"] = (ckpt_out[2],)' in src, "the default path stays"
    assert "if self.vae_name:" in src
    assert 'prepared["vae"] = (vae_out[0],)' in src


def test_the_decoder_is_in_the_identity_so_a_swap_opens_a_new_session(eng):
    """Two decoders are two sessions. Without this a cached handle from a
    baked-VAE run could be reused for an ft-mse render."""
    import inspect
    src = inspect.getsource(gs.GhostSignalEngine.session_identity)
    assert "self.vae_name" in src
    assert "_vae_path" in src


def test_the_receipt_id_was_REPOINTED_not_edited(eng):
    """Clips already exist on disk under the pre-decoder id. Editing that string
    in place would retroactively change what every older receipt means."""
    assert "ftmse" in eng.recipe_receipt_id
    assert eng.recipe_receipt_id != "animatediff_sd15_lightning8_static16_512x288_v1"


def test_the_lane_says_EXPERIMENTAL_out_loud(eng):
    """The operator named it one. It must not become a production path by
    accident: no default role, no proven device row, no qualified cost row."""
    import inspect
    doc = inspect.getdoc(type(eng)) or ""
    assert "EXPERIMENTAL" in inspect.getmodule(type(eng)).__doc__
    assert eng.default_roles == ()
    # NOT a device-row assertion any more. The row earned "mps" on 2026-09-09
    # with a published episode; experimental-ness is carried by the things that
    # actually keep it out of production -- no default role, no qualified cost
    # row, and the label itself.
    assert ENGINE_ID not in _QUALIFIED_COST_ROWS(), (
        "an experimental lane must make no VRAM-fit claim")


# ---------------------------------------------------------------------------
# ADAPTIVE HOLD -- PBUG-20260909-01, the beat that rebooted the machine
# ---------------------------------------------------------------------------

def test_the_beat_that_rebooted_the_machine_now_fits(eng):
    """THE PINNED INCIDENT. The first real episode rendered five beats at
    124-136 latents, then asked for 160 on a ~320-frame beat and took the whole
    Mac down -- unified memory, so an OOM is a reboot and not a process kill.

    At hold 3 the same beat needs 112: the SAME 320 delivered frames, the same
    audio sync, no jump cuts, and no reboot."""
    hold = eng._beat_hold(320)
    assert hold == 3, "hold 2 would ask for 160, which is the fatal count"
    assert eng._source_request_for(320, hold) == 112
    # and it is still a legal sliding-window count
    assert gs.ghost_legal_source_count(112) == 112


@pytest.mark.parametrize("target,expect_hold,expect_latents", [
    # The five beats that ACTUALLY RENDERED in that episode keep hold 2
    # untouched -- adaptation must not disturb what already worked.
    (250, 2, 136),   # shot_music_opening_001
    (243, 2, 124),   # shot_b001
    (267, 2, 136),   # shot_b002
    (239, 2, 124),   # shot_b003 / b004
])
def test_the_beats_that_already_worked_are_untouched(eng, target,
                                                     expect_hold, expect_latents):
    assert eng._beat_hold(target) == expect_hold
    assert eng._source_request_for(target, expect_hold) == expect_latents


def test_it_escalates_only_as_far_as_it_must(eng):
    """Hold rises one step at a time to the FIRST value that fits, never
    straight to the maximum -- every extra step costs unique frames per second
    (12.5 at hold 2, 8.3 at 3, 6.25 at 4, 5.0 at 5)."""
    assert eng._beat_hold(400) == 3
    assert eng._beat_hold(600) == 5


def test_an_impossible_beat_is_REFUSED_BY_NAME_not_attempted(eng):
    """The whole point. A named refusal is recoverable; a reboot is not."""
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng._beat_hold(4000)
    msg = str(exc.value)
    assert "PBUG-20260909-01" in msg
    assert "rebooted the machine" in msg


def test_an_unreadable_host_refuses_rather_than_guessing(eng, monkeypatch):
    """On a platform where the failure mode is a reboot, "I could not measure
    the memory" must never resolve to "go ahead"."""
    from nodes._otr_video_engines import motion_common as mc
    monkeypatch.setattr(mc, "latent_ceiling_for_host", lambda *a, **k: None)
    with pytest.raises(vreg.EngineUnusable):
        eng._beat_hold(250)


def test_the_ceiling_is_derived_from_the_host_not_hardcoded():
    """A 32 GB Mac or a discrete card must NOT inherit a 16 GB limit."""
    from nodes._otr_video_engines import motion_common as mc
    at16 = mc.latent_ceiling_for_host(512, 288, 16 * 1024)
    at32 = mc.latent_ceiling_for_host(512, 288, 32 * 1024)
    assert at16 == 136, "the largest count measured to SURVIVE, not to fail"
    # NOT `2 * at16`. This test asserted exactly that until 2026-09-09, which
    # encoded the linear scaling that was the defect: only the VARIABLE pool
    # doubles, because ~4.6 GB of weights and baseline is fixed. So twice the
    # RAM buys MORE than twice the latents, and half the RAM buys LESS than
    # half. A test that pins the arithmetic must pin the corrected arithmetic.
    assert at32 > 2 * at16, "the fixed cost amortises on a bigger machine"
    assert mc.latent_ceiling_for_host(512, 288, 8 * 1024) < at16 / 2
    # A bigger canvas costs batch, so the ceiling falls.
    assert mc.latent_ceiling_for_host(512, 512, 16 * 1024) < at16


def test_the_ceiling_anchors_on_survival_not_on_failure():
    """136 rendered; 160 rebooted the machine. The untested gap between them is
    treated as UNSAFE, which is the only defensible direction when being wrong
    costs a reboot."""
    from nodes._otr_video_engines import motion_common as mc
    assert mc.GHOST_ANCHOR_SAFE_LATENTS == 136
    assert mc.latent_ceiling_for_host(512, 288, 16 * 1024) < 160


def test_one_hold_governs_the_plan_the_selector_and_the_receipts(eng):
    """A receipt that named a cadence the render did not use would be worse than
    no receipt. The hold is resolved ONCE and carried on the plan."""
    import inspect
    plan_src = inspect.getsource(gs.GhostSignalEngine._build_render_request)
    assert 'hold = self._beat_hold(target)' in plan_src
    assert '"hold": int(hold)' in plan_src
    clip_src = inspect.getsource(gs.GhostSignalEngine.render_clip)
    assert 'plan.get("hold"' in clip_src, "the selector must use the beat's hold"
    rec_src = inspect.getsource(gs._ghost_cadence_receipts_for)
    assert "hold=None" in rec_src


def test_the_receipt_names_the_cadence_that_actually_ran(eng):
    """cadence_mode must say hold_3 on a beat that adapted."""
    hold = eng._beat_hold(320)
    receipts = gs.ghost_cadence_receipts(320, eng._source_request_for(320, hold),
                                         hold)
    assert receipts["cadence_mode"] == "hold_3"
    assert receipts["model_frame_count"] == 112
    assert receipts["native_frame_count"] == 320, "delivered count is UNCHANGED"


def test_hold_is_resolved_per_beat_and_not_stored_on_the_instance(eng):
    """The registry keeps ONE shared instance per engine for the whole process,
    so a hold cached on `self` would leak from one beat to the next. And driving
    this from the existing `OTR_GHOST_HOLD_FACTOR` env knob would re-cadence
    every sibling that shares this base -- which is why it does not."""
    before = eng.hold_factor
    assert eng._beat_hold(320) == 3
    assert eng.hold_factor == before == 2, "the instance must not be mutated"
    import inspect
    src = inspect.getsource(gs.GhostSignalEngine._beat_hold)
    # `GHOST_HOLD_FACTOR_MAX` is the legitimate loop bound and the docstring
    # names the env var on purpose, so match the ENV READ rather than the
    # substring: the knob must never be the mechanism here.
    assert "_resolve_hold_factor" not in src
    assert "GHOST_HOLD_FACTOR_ENV" not in src, (
        "the env knob is process-wide and would re-cadence the published lanes")
    assert "otr_env" not in src


def test_the_published_lanes_never_adapt():
    """CLAUDE.md 0B. Their cadence is frozen at whatever made their episodes."""
    assert gs.GhostSignalEngine.adaptive_hold_for_memory is False
    for name in ("animatediff15_v3_haunted_video",
                 "animatediff15_v3_stillin_lab_video"):
        peer = vreg.get_engine(name)
        assert peer.adaptive_hold_for_memory is False
        # the fatal beat still resolves to their frozen hold, unchanged
        assert peer._beat_hold(320) == peer.hold_factor == 2
        assert peer._source_request_for(320) == gs.ghost_source_request(320, 2)


def test_the_sampler_receipt_cannot_contradict_itself(eng):
    """CAUGHT IN REVIEW, and it was mine. `sampler_inputs_for` read
    `self.hold_factor` while `source_request` came from the beat's resolved
    hold -- so an adapted beat would have stamped `hold_factor: 2` beside a
    `source_request` of 112, which is impossible at hold 2 for a 320-frame
    beat. The receipt would have contradicted itself in the same dict."""
    class _Req(dict):
        pass
    req = {"shot_id": "s1", "text_prompt": "x", "negative_prompt": "y",
           "timing": {"target_frame_count": 320},
           "seed_bundle": {"request_seed": 42}}
    got = eng.sampler_inputs_for(req)
    assert got["hold_factor"] == 3, "the hold that actually ran"
    assert got["source_request"] == 112
    assert got["unique_source_count"] == 107
    # internally consistent: ceil(T / hold) == unique
    import math
    assert math.ceil(320 / got["hold_factor"]) == got["unique_source_count"]


def test_a_non_adapting_lane_reports_exactly_what_it_always_did():
    """The fix must be invisible to the published lanes."""
    peer = vreg.get_engine("animatediff15_v3_haunted_video")
    req = {"shot_id": "s1", "text_prompt": "x", "negative_prompt": "y",
           "timing": {"target_frame_count": 320},
           "seed_bundle": {"request_seed": 42}}
    got = peer.sampler_inputs_for(req)
    assert got["hold_factor"] == peer.hold_factor == 2
    assert got["source_fps"] == 12, "unchanged, including its truncation"


# ---------------------------------------------------------------------------
# EXECUTION tests for what was previously only string-matched. Review's point:
# an `inspect.getsource` assertion passes whether or not the code WORKS.
# ---------------------------------------------------------------------------

def test_the_ceiling_subtracts_the_fixed_resident_cost(eng):
    """CAUGHT IN REVIEW, and it was the dangerous direction. The first version
    scaled the 136-latent anchor LINEARLY by total RAM, which assumes the
    weights shrink with the machine. They do not: ~4.6 GB of checkpoint, motion
    module, decoder and OS baseline is fixed. On an 8 GB Mac that returned 68
    where the corrected figure is 40 -- a 1.7x overestimate on exactly the
    hardware the guard protects, in the direction that reboots it."""
    from nodes._otr_video_engines import motion_common as mc
    assert mc.latent_ceiling_for_host(512, 288, 16 * 1024) == 136, "anchor holds"
    at8 = mc.latent_ceiling_for_host(512, 288, 8 * 1024)
    assert at8 == 40, "linear scaling would have said 68"
    assert at8 < 68
    # Half the RAM must give LESS than half the latents, because the fixed cost
    # does not halve with it.
    assert at8 < 136 / 2


def test_a_machine_too_small_for_the_weights_gets_no_ceiling_at_all(eng):
    """Not a small number -- None. There is no safe batch size when the weights
    alone do not fit, and `_beat_hold` turns None into a named refusal."""
    from nodes._otr_video_engines import motion_common as mc
    assert mc.latent_ceiling_for_host(512, 288, 4 * 1024) is None
    assert mc.latent_ceiling_for_host(512, 288, 1) is None


def test_the_ceiling_does_not_collapse_while_a_render_is_resident():
    """WHY THE LIVE-MEMORY SIGNAL WAS REJECTED, pinned so it is not re-added.
    Review asked for live availability instead of total RAM. Implemented and
    measured, this host reported 1344 MB available mid-episode -- because the
    lane's own ~4.6 GB was already resident. Subtracting the fixed cost from a
    figure that has already paid it double-counts, and every beat after the
    first would be refused."""
    from nodes._otr_video_engines import motion_common as mc
    import inspect
    src = inspect.getsource(mc.latent_ceiling_for_host)
    assert "_available_ram_mb()" not in src.split("def _available_ram_mb")[0]
    # And the value must not move just because the machine is busy.
    assert (mc.latent_ceiling_for_host(512, 288)
            == mc.latent_ceiling_for_host(512, 288)), "must be reproducible"


def test_two_different_cfgs_really_do_produce_different_cache_keys(eng,
                                                                   monkeypatch):
    """EXECUTED, not string-matched. The previous test asserted `.hex()` appears
    in the source, which would pass even if the value were computed from the
    wrong object."""
    req = {"shot_id": "s1", "text_prompt": "x", "negative_prompt": "y",
           "timing": {"target_frame_count": 250},
           "seed_bundle": {"request_seed": 42}}
    monkeypatch.delenv(L.LIGHTNING_CFG_ENV, raising=False)
    a = eng.shot_cache_identity(req)
    monkeypatch.setenv(L.LIGHTNING_CFG_ENV, "2.0")
    b = eng.shot_cache_identity(req)
    assert a != b, "a cfg sweep must not be served the other arm's clip"
    assert any("negative_effective=False" in str(p) for p in a)
    assert any("negative_effective=True" in str(p) for p in b)


def test_two_different_holds_really_do_produce_different_cache_keys(eng):
    """Same point for the hold. A 320-frame beat resolves to hold 3 and a
    250-frame beat to hold 2, and their keys must differ by more than the shot
    id -- the hold token has to actually be in there."""
    base = {"shot_id": "same", "text_prompt": "x", "negative_prompt": "y",
            "seed_bundle": {"request_seed": 42}}
    short = eng.shot_cache_identity(dict(base, timing={"target_frame_count": 250}))
    long_ = eng.shot_cache_identity(dict(base, timing={"target_frame_count": 320}))
    assert any(p == "hold=2" for p in short)
    assert any(p == "hold=3" for p in long_)


def test_shot_cache_identity_raises_rather_than_returning_a_bogus_key(eng,
                                                                      monkeypatch):
    """REVIEW'S LANDMINE, pinned. `shot_cache_identity` now calls
    `_build_render_request`, which can raise `EngineUnusable` through
    `_beat_hold`. That is the CORRECT behaviour -- a key for a beat that cannot
    be rendered would be worse -- but it is a new exception on a method whose
    docstring calls it a description of handles, so it gets a test."""
    req = {"shot_id": "s1", "text_prompt": "x", "negative_prompt": "y",
           "timing": {"target_frame_count": 250},
           "seed_bundle": {"request_seed": 42}}
    assert eng.shot_cache_identity(req)          # fine normally
    from nodes._otr_video_engines import motion_common as mc
    monkeypatch.setattr(mc, "latent_ceiling_for_host", lambda *a, **k: None)
    with pytest.raises(vreg.EngineUnusable):
        eng.shot_cache_identity(req)


def test_the_published_lanes_cache_key_never_raises(monkeypatch):
    """The same unreadable host must NOT break the lanes that do not adapt."""
    from nodes._otr_video_engines import motion_common as mc
    monkeypatch.setattr(mc, "latent_ceiling_for_host", lambda *a, **k: None)
    req = {"shot_id": "s1", "text_prompt": "x", "negative_prompt": "y",
           "timing": {"target_frame_count": 320},
           "seed_bundle": {"request_seed": 42}}
    for name in ("animatediff15_v3_haunted_video",
                 "animatediff15_v3_stillin_lab_video"):
        assert vreg.get_engine(name).shot_cache_identity(req)



def _QUALIFIED_COST_ROWS():
    """The lanes that carry a qualified cost row, or an empty set.

    Read live rather than hardcoded: the point of the assertion above is that
    THIS lane makes no VRAM-fit claim, and that has to stay true as the table
    changes."""
    try:
        from nodes._otr_video_engines import frame_contract as fc
        return set(getattr(fc, "QUALIFIED_COST_ROWS", ()) or ())
    except Exception:  # noqa: BLE001
        return set()
