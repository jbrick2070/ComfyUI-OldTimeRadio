"""LANE 2 -- named boot contracts, and their first real consumer.

Spec S8. Some lanes only fit under the 14.5 GiB gate if the SERVER was started
with particular allocator flags: HuMo 14B measures 14.98 GiB unclamped and
13.06 GiB under reserve-VRAM + no pinned memory, same graph, same weights,
1.9 GiB apart. By the time a beat renders the server has been up for an hour,
so that is not fixable at render time -- it has to be declared, applied at
launch, and PROVED.

The mechanism ships WITH its consumer on purpose. Unused infrastructure is how
you get a "configured" knob that reaches nothing, which is the exact defect this
lane is cleaning up: `launch.extra_args` has been written into a markdown
documentation string for months while `--disable-pinned-memory` appeared in
ZERO non-doc files repo-wide.

CPU-safe: no CUDA, no model loads, no renders.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_video_director as vd
from nodes._otr_shared import boot_contracts as bc
from nodes._otr_shared.capability_profiles import ProfileError, load_profile
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd

LANE = "humo_14B_169"
PUBLIC = "humo14_high_audio_in_wide"
DECLARED_CANVAS = (832, 480)
REPO = pathlib.Path(__file__).resolve().parents[1]
PROFILE_ID = "otr_w45_humo_14b_169"


@pytest.fixture()
def engine():
    return vreg.get_engine(LANE)


@pytest.fixture()
def profile():
    return load_profile(PROFILE_ID)


# ---------------------------------------------------------------------------
# The contract table
# ---------------------------------------------------------------------------

def test_the_default_contract_constrains_nothing():
    """Every profile that shipped before this key existed means `default`, so
    `default` must be a real no-op or this mechanism would retire them all."""
    spec = bc.contract_spec(bc.DEFAULT)
    assert set(spec.values()) == {None}
    assert bc.launch_env_for(bc.DEFAULT) == {}


def test_dont_care_is_distinct_from_required_off():
    """`None` means the contract does not constrain that knob; `False` would
    mean it REQUIRES it off. Collapsing the two would let a contract silently
    forbid an unrelated flag."""
    assert bc.BOOT_CONTRACTS[bc.HUMO_DIET]["sage_attention"] is None
    assert bc.BOOT_CONTRACTS[bc.H3]["sage_attention"] is False


def test_the_humo_diet_is_the_measured_pair_not_a_round_number():
    """2.921 is a measurement, not a preference, and the diet is BOTH knobs --
    reserve-vram alone does not reproduce the 13.06 GiB envelope."""
    spec = bc.contract_spec(bc.HUMO_DIET)
    assert spec["reserve_vram_gb"] == 2.921
    assert spec["disable_pinned_memory"] is True


def test_an_unknown_contract_raises_rather_than_meaning_no_constraints():
    """A typo in a profile must not resolve to 'unconstrained' -- that would
    turn a misspelled clamp into a silently unclamped boot."""
    with pytest.raises(bc.BootContractError):
        bc.contract_spec("humo_deit")


def test_the_env_mapping_only_emits_knobs_a_launcher_actually_reads():
    """Lesson L6. `sage_attention` gets NO env row because no launcher passes
    an attention flag -- emitting one would be another configured knob that
    reaches nothing. Sage-sensitive lanes refuse at assert_usable instead,
    which is enforcement that runs."""
    env = bc.launch_env_for(bc.H3)
    assert env == {"OTR_HEADLESS_DISABLE_PINNED": "1"}
    assert not any("SAGE" in k.upper() for k in env)


# ---------------------------------------------------------------------------
# The channel: launch.env is live, launch.extra_args is documentation
# ---------------------------------------------------------------------------

def test_the_launcher_turns_both_diet_knobs_into_argv():
    """THE POINT OF THE WHOLE LANE. Until this commit the cmd had a hook for
    --reserve-vram and none for --disable-pinned-memory, so a profile that
    'configured' the diet clamped exactly one of its two knobs and the other
    was documentation."""
    cmd = (REPO / "scripts" / "_otr_soak_server_launch.cmd").read_text(
        encoding="utf-8", errors="replace")
    assert "OTR_HEADLESS_RESERVE_VRAM_GB set _OTR_RESERVE=--reserve-vram" in cmd
    assert ("if defined OTR_HEADLESS_DISABLE_PINNED set "
            "_OTR_PINNED=--disable-pinned-memory") in cmd
    # Declared is not applied: the variable must also reach the command line.
    assembly = cmd.split("main.py", 1)[1]
    assert "%_OTR_RESERVE%" in assembly and "%_OTR_PINNED%" in assembly


def test_the_hero_profile_carries_the_contract_in_the_live_channel(profile):
    assert bc.contract_for_profile(profile) == bc.HUMO_DIET
    assert profile["launch"]["env"] == bc.launch_env_for(bc.HUMO_DIET)
    assert not profile["launch"]["extra_args"], (
        "extra_args is documentation-only; anything load-bearing placed there "
        "is silently ignored at boot")


def test_every_shipped_profile_still_validates_with_the_new_optional_key():
    """`boot_contract` is OPTIONAL because the launch key set is
    closed-validated: a required key would have broken all ~20 profiles at
    once."""
    # widget_mapping.json shares the directory but is a WIDGET MAP, not a
    # profile -- load_profile refuses it by design.
    ids = sorted(p.stem for p in (REPO / "config" / "profiles").glob("*.json")
                 if p.stem != "widget_mapping")
    assert len(ids) >= 20
    for pid in ids:
        prof = load_profile(pid)
        name = bc.contract_for_profile(prof)
        assert bc.known_contract(name), (
            "profile %s selects unknown boot contract %r" % (pid, name))


def test_a_malformed_boot_contract_value_is_refused_by_the_schema():
    """An OPTIONAL key still gets VALIDATED when present. A typo'd value
    silently doing nothing is the drift class the closed validator kills."""
    from nodes._otr_shared import capability_profiles as cp
    prof = load_profile(PROFILE_ID)
    prof["launch"]["boot_contract"] = 17
    with pytest.raises(ProfileError) as exc:
        cp.validate_profile_shape(prof, source="synthetic")
    assert "boot_contract" in str(exc.value)


# ---------------------------------------------------------------------------
# Enforcement: against the RUNNING SERVER, never against the config text
# ---------------------------------------------------------------------------

def test_the_probe_reports_unavailable_off_a_comfy_server_rather_than_guessing():
    """A headroom-gated branch that is silently unreachable while the tests
    stay green is worse than no branch. Say so instead."""
    state = bc.running_server_boot_state()
    assert state["available"] is False
    assert state.get("error")


def test_unknowable_is_not_the_same_as_satisfied():
    """THIS TEST'S NAME WAS RIGHT AND ITS BODY WAS WRONG (retro bug hunt r1,
    2026-08-11 -- both reviewers reached the defect independently).

    It asserted `check_running_server(HUMO_DIET) == []` off a server, with a
    docstring explaining that the CALLER decides. No caller decided:
    `assert_running_server` treats an empty list as MET and raises nothing. So
    a contract constraining real VRAM clamps evaluated as COMPLIANT on any box
    where `comfy.cli_args` cannot be imported -- and the test that should have
    caught it pinned the bug under a name asserting the opposite.

    The rule the name always stated, now enforced: a contract that CONSTRAINS
    something is not satisfied by a server we cannot read. A contract that
    constrains nothing still is -- there is nothing to violate.
    """
    problems = bc.check_running_server(bc.HUMO_DIET, state={"available": False})
    assert problems, "a constrained contract may not pass on an unreadable server"
    assert "UNKNOWN is not satisfied" in problems[0]
    with pytest.raises(bc.BootContractError):
        bc.assert_running_server(bc.HUMO_DIET, state={"available": False})
    # ...and the stock contract constrains nothing, so it is genuinely met.
    assert bc.check_running_server(bc.DEFAULT, state={"available": False}) == []


def test_a_failed_sage_probe_is_not_a_pass():
    """`running_server_boot_state` recorded `sage_probe_error` and NOTHING read
    it, so a probe that raised left `sage_attention = None` and the comparison
    skipped -- silently passing a Sage-constrained contract on the exact lanes
    Sage silently corrupts. Recording an error nobody reads is swallowing it."""
    state = {"available": True, "disable_pinned_memory": True,
             "sage_attention": None, "sage_probe_error": "ImportError"}
    problems = bc.check_running_server(bc.H3, state=state)
    assert problems and "ImportError" in problems[0]
    assert "not a pass" in problems[0]


@pytest.mark.parametrize("state,needle", [
    ({"available": True, "reserve_vram_gb": None,
      "disable_pinned_memory": True}, "--reserve-vram"),
    ({"available": True, "reserve_vram_gb": 1.0,
      "disable_pinned_memory": True}, "--reserve-vram"),
    ({"available": True, "reserve_vram_gb": 2.921,
      "disable_pinned_memory": False}, "--disable-pinned-memory"),
])
def test_a_server_started_without_the_diet_is_named_knob_by_knob(state, needle):
    problems = bc.check_running_server(bc.HUMO_DIET, state=state)
    assert problems and any(needle in p for p in problems)
    with pytest.raises(bc.BootContractError) as exc:
        bc.assert_running_server(bc.HUMO_DIET, state=state)
    assert "restarted" in str(exc.value), (
        "the message must say what the operator has to DO -- this is not "
        "fixable at render time")


def test_reserving_MORE_than_the_contract_asks_is_still_inside_the_envelope():
    """A >= comparison, deliberately: the contract's number is the floor the
    measurement was taken at, and clamping harder stays inside it."""
    assert bc.check_running_server(bc.HUMO_DIET, state={
        "available": True, "reserve_vram_gb": 4.0,
        "disable_pinned_memory": True}) == []


def test_sage_is_checked_only_when_the_contract_names_it():
    assert bc.check_running_server(bc.HUMO_DIET, state={
        "available": True, "reserve_vram_gb": 2.921,
        "disable_pinned_memory": True, "sage_attention": True}) == []
    problems = bc.check_running_server(bc.H3, state={
        "available": True, "reserve_vram_gb": None,
        "disable_pinned_memory": True, "sage_attention": True})
    assert problems and "SageAttention" in problems[0]


# ---------------------------------------------------------------------------
# Engine compatibility -- only a lane that never shipped under `default` may
# REQUIRE a contract
# ---------------------------------------------------------------------------

def test_an_engine_that_declares_nothing_is_compatible_with_everything():
    """Every lane that shipped before this mechanism keeps working."""
    assert set(bc.compatible_contracts_for_engine(
        vreg.get_engine("wan_ti2v"))) == set(bc.BOOT_CONTRACTS)


def test_the_hero_tier_keeps_default_because_it_has_shipped_under_it(engine):
    """Requiring the diet would retire a shipping lane by side effect. What
    `default` COSTS is stated in the declaration's comment -- 14.98 GiB, over
    the gate -- rather than hidden behind a refusal."""
    assert bc.compatible_contracts_for_engine(engine) == (
        "default", "humo_diet")


def test_the_cast_is_expressed_in_the_profile_not_by_an_engine_refusal(
        engine, profile, monkeypatch):
    """The server state is now SUPPLIED, and that is the point of this edit.

    `assert_usable` reaches `assert_running_server`, which since the retro bug
    hunt refuses a constrained contract it cannot verify. On a CPU box
    `comfy.cli_args` does not import, so this test was asking the engine to
    prove a VRAM clamp on a machine with no server -- and passing only because
    the check used to answer "satisfied" to that question.

    So the fixture hands it a server that genuinely honours the diet. The
    subject is unchanged (the cast lives in the profile, not in an engine
    refusal); what changed is that the test no longer depends on the bug.
    """
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True, "reserve_vram_gb": 2.921,
        "disable_pinned_memory": True, "sage_attention": None})
    assert bc.check_engine_against_profile(engine, profile) == []
    assert engine.assert_usable(host_caps={}, profile=profile) == LANE


def test_a_contract_the_tier_is_not_proven_on_is_refused_by_name(engine,
                                                                profile):
    hostile = dict(profile)
    hostile["launch"] = dict(profile["launch"])
    hostile["launch"]["boot_contract"] = bc.H3
    from nodes._otr_video_engines.registry import EngineUnusable
    with pytest.raises(EngineUnusable) as exc:
        engine.assert_usable(host_caps={}, profile=hostile)
    assert "proven on boot contract" in str(exc.value)


# ---------------------------------------------------------------------------
# The lane itself: canvas truth (S8b-4) and receipt completeness (S8b-6)
# ---------------------------------------------------------------------------

def test_the_hero_tier_declares_its_measured_canvas(engine):
    assert tuple(engine.render_canvas) == DECLARED_CANVAS
    assert rd.declared_render_canvas(LANE) == DECLARED_CANVAS
    width, height = DECLARED_CANVAS
    assert width % 32 == 0 and height % 32 == 0


def test_the_env_overrides_can_no_longer_contradict_the_declaration(
        engine, monkeypatch):
    """S8b-4's precision. `_native_dims` honoured OTR_HUMO_WIDTH/HEIGHT, so
    832x480 was a DEFAULT and not a runtime guarantee -- the declaration would
    have said one size while the graph rendered another, invisibly."""
    from nodes._otr_video_engines.registry import EngineUnusable
    monkeypatch.setenv("OTR_HUMO_WIDTH", "1472")
    monkeypatch.setenv("OTR_HUMO_HEIGHT", "832")
    with pytest.raises(EngineUnusable) as exc:
        engine._native_dims()
    assert "declares render_canvas" in str(exc.value)
    # An override that AGREES is fine -- this refuses contradiction, not use.
    monkeypatch.setenv("OTR_HUMO_WIDTH", "832")
    monkeypatch.setenv("OTR_HUMO_HEIGHT", "480")
    assert engine._native_dims() == DECLARED_CANVAS


def test_the_override_refusal_is_SCOPED_to_tiers_that_declare(monkeypatch):
    """The refusal applies to a tier that DECLARES a canvas, and to no other.

    This started life as "an undeclared SIBLING still honours its overrides"
    and moved twice -- humo_1.7B held it until lane 3, `humo` until lane 4 --
    and then the HuMo family ran out of undeclared tiers, because closing the
    family was the point. So the invariant is asserted directly rather than
    parked on whichever tier has not been done yet: strip the declaration and
    the overrides go back to winning, exactly as they did for every tier before
    this build. A control with no occupant left is a control that has to be
    rewritten, not deleted.
    """
    monkeypatch.setenv("OTR_HUMO_WIDTH", "640")
    monkeypatch.setenv("OTR_HUMO_HEIGHT", "384")
    engine = vreg.get_engine("humo")
    monkeypatch.setattr(type(engine), "render_canvas", None, raising=False)
    assert engine._native_dims() == (640, 384)


def test_the_profile_canvas_agrees_with_the_declaration(profile):
    render = profile["render"]
    assert (int(render["canvas_w"]), int(render["canvas_h"])) == DECLARED_CANVAS


@pytest.mark.parametrize("tier", ["humo", "humo_1.7B", "humo_1.7B_169",
                                  "humo_14B_169"])
def test_every_humo_tier_can_now_produce_its_manifest_row(tier):
    """S8b-6 / lesson L4. The peak was MEASURED and LOGGED since 2026-08-06 and
    then dropped, so every HuMo clip reached the ledger with these fields null
    and the driver fell back to an instantaneous VRAM read -- a sample at an
    arbitrary moment wearing the name of a peak. S2's envelope work is built on
    these numbers."""
    engine = vreg.get_engine(tier)
    telemetry = engine._clip_telemetry(832, 480)
    assert telemetry["render_canvas"] == "832x480"
    assert isinstance(telemetry["use_lora"], bool)
    assert engine._recipe_receipt().startswith(tier.replace(".", "p"))
    clip = engine._clip_from_raw(
        {"out_path": "x.mp4", "frame_count": 97, "vram_peak_mb": 13372,
         "recipe": engine._recipe_receipt(), **telemetry}, {"shot_id": "s1"})
    for field in ("vram_peak_mb", "recipe", "quant", "use_lora",
                  "render_canvas"):
        assert field in clip, "%s drops %r on the floor of _clip_from_raw" % (
            tier, field)
    assert clip["vram_peak_mb"] == 13372


def test_the_quant_label_is_read_off_the_resolved_name_not_assumed():
    """A swapped weight must not leave a receipt describing the file it
    replaced."""
    assert vreg.get_engine(LANE)._quant_label() == "fp8_e4m3fn"


def test_the_stale_49_frame_comment_is_gone():
    """S8b-7. The cap became 97 on 2026-08-02 while the comment that EXPLAINS
    the cap still said 49 -- a stale number inside the explanation of a number
    is worse than no comment."""
    src = (REPO / "nodes" / "_otr_video_engines" / "eng_humo.py").read_text(
        encoding="utf-8")
    assert "= 49 HERE" not in src
    assert "49 = 4*12+1" not in src
    assert vreg.get_engine(LANE).safe_render_frames == 97
    assert vreg.get_engine(LANE).frame_contract.max_frames == 97


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def test_exactly_one_live_menu_option_and_the_id_states_what_the_lane_is():
    assert vd.exact_menu_option_for(LANE) == "%s (16:9)" % PUBLIC
    assert "audio_in" in PUBLIC, (
        "HuMo is audio-driven and the id must say so (operator, 2026-08-10)")
    assert "wide" in PUBLIC, (
        "the aspect belongs in the id, not only the label suffix -- a bare "
        "humo14_high_face hid which way its sibling renders")


def test_the_lane_has_an_evidence_row_and_an_admission_confession():
    manifest = json.loads(
        (REPO / "docs" / "evidence" / "video_evidence_manifest.json")
        .read_text(encoding="utf-8"))
    rows = [e for e in manifest["entries"] if e["lane"] == LANE]
    assert rows and any("f97" in r["envelope_key"] for r in rows)
    assert LANE in manifest["admission_unenforced"], (
        "a measured receipt is not a qualified cost row; nothing refuses an "
        "over-budget plan on this lane yet and the receipts must say so")


# ---------------------------------------------------------------------------
# LANE 3 -- the 1.7B pair. Same family, so lane 2's lessons applied almost
# unchanged; what was NEW here is the honesty guard that a VRAM knob was
# gating.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier,canvas", [
    ("humo_1.7B", (480, 832)),
    ("humo_1.7B_169", (832, 480)),
])
def test_the_1p7b_pair_declares_its_canvas(tier, canvas):
    engine = vreg.get_engine(tier)
    assert tuple(engine.render_canvas) == canvas
    assert rd.declared_render_canvas(tier) == canvas
    assert canvas[0] % 32 == 0 and canvas[1] % 32 == 0


def test_the_portrait_profile_stopped_claiming_landscape():
    """`otr_w45_humo_1_7b.json` said 832x480 on the tier whose whole identity
    is the pillarbox talking head. The declaration is what renders, so the
    profile was lying to whoever read it."""
    prof = load_profile("otr_w45_humo_1_7b")
    assert (prof["render"]["canvas_w"], prof["render"]["canvas_h"]) == (480, 832)


@pytest.mark.parametrize("tier", ["humo_1.7B", "humo_1.7B_169"])
def test_the_longbeat_tier_keeps_BOTH_boot_contracts(tier):
    """It is the auto-downgrade target -- the floor a heavy episode falls to --
    so requiring the diet would remove the floor as a side effect."""
    assert bc.compatible_contracts_for_engine(vreg.get_engine(tier)) == (
        "default", "humo_diet")


def test_the_exact_fit_guard_no_longer_hangs_off_a_VRAM_KNOB():
    """S8b item 3. The honesty check read `if cap is not None and
    target_fc > 0`, so an UNCAPPED tier skipped it entirely: a beat asking for
    more than the 177-frame ceiling rendered 177 and returned them stamped
    extension_mode "none" with native_frame_count == frame_count --
    indistinguishable from an honest clip. The video ran out before the audio
    and nothing said so.

    Asserted against the SOURCE because the alternative is a live over-ladder
    render, and the property is structural: the guard's condition must not
    mention the cap."""
    src = (REPO / "nodes" / "_otr_video_engines" / "eng_humo.py").read_text(
        encoding="utf-8")
    assert "if cap is not None and target_fc > 0:" not in src, (
        "the exact-fit guard is conditional on a VRAM knob again")
    assert "if target_fc > 0:" in src
    # And the refusal must be able to explain BOTH shapes of the failure.
    assert "This tier is uncapped" in src
    assert "VRAM-safe cap" in src


@pytest.mark.parametrize("tier,public", [
    ("humo_1.7B", "humo17_high_audio_in_portrait"),
    ("humo_1.7B_169", "humo17_high_audio_in_wide"),
])
def test_the_1p7b_public_ids_state_the_aspect(tier, public):
    """Same checkpoint, same VRAM class -- the aspect IS the difference between
    these two lanes, so the aspect belongs in the id rather than only in the
    label suffix."""
    assert vd.exact_menu_option_for(tier).startswith(public)
    assert "audio_in" in public


# ---------------------------------------------------------------------------
# LANE 4 -- the last HuMo tier. It held the "declares NOTHING" control until
# now, so closing it moves that control off the family entirely.
# ---------------------------------------------------------------------------

def test_the_last_humo_tier_declares_its_canvas():
    engine = vreg.get_engine("humo")
    assert tuple(engine.render_canvas) == (480, 832)
    assert rd.declared_render_canvas("humo") == (480, 832)


@pytest.mark.parametrize("pid", ["otr_w45_humo", "otr_g4_humo"])
def test_both_humo_profiles_stopped_claiming_landscape(pid):
    """BOTH of this tier's profiles said 832x480 on the pillarbox lane. The
    w45 one was found by G2.3; the g4 one only surfaced because the gate reads
    EVERY profile that selects the engine, not just the one being edited."""
    prof = load_profile(pid)
    assert (prof["render"]["canvas_w"], prof["render"]["canvas_h"]) == (480, 832)


def test_every_humo_tier_now_declares_a_canvas_and_a_contract():
    """The family is closed. Four tiers, four declarations, four contract
    lists -- and the aspect each one renders is now readable without loading
    anything."""
    expected = {
        "humo": (480, 832), "humo_1.7B": (480, 832),
        "humo_1.7B_169": (832, 480), "humo_14B_169": (832, 480),
    }
    for tier, canvas in expected.items():
        engine = vreg.get_engine(tier)
        assert tuple(engine.render_canvas) == canvas, tier
        assert bc.compatible_contracts_for_engine(engine) == (
            "default", "humo_diet"), tier


@pytest.mark.parametrize("tier", ["humo", "humo_1.7B", "humo_1.7B_169",
                                  "humo_14B_169"])
def test_the_lora_receipt_AGREES_WITH_THE_GRAPH(tier):
    """A RECEIPT THAT RECORDED A FALSEHOOD, for six lanes, under a green row.

    Found by the retro bug hunt on lanes 0-6 (2026-08-11). The graph decides
    whether the distill LoRA loads via ``_lora_is_skipped`` -- THE ONE reading
    of "this tier runs LoRA-free" -- and the 1.7B tiers switch it off by
    setting the token to the STRING ``"none"``. Both receipts instead used raw
    truthiness, and ``bool("none")`` is ``True``. So both 1.7B engines rendered
    LoRA-free while stamping ``humo_1p7B_v1_lora`` with ``use_lora=True``, and
    ``otr_credits_roll.py:238-239`` printed "lora" into PUBLISHED credits for a
    LoRA that never loaded.

    THE PREVIOUS TEST COULD NOT CATCH THIS. It asserted
    ``isinstance(use_lora, bool)`` -- shape, not truth -- and a wrong bool is
    still a bool. This asserts the two fields against the SAME authority the
    graph uses, so the receipt cannot disagree with the render again whatever
    the token happens to be spelled.
    """
    engine = vreg.get_engine(tier)
    token = engine._loader_names().get("lora")
    graph_loads_lora = bool(token) and not engine._lora_is_skipped(token)

    assert engine._clip_telemetry(832, 480)["use_lora"] is graph_loads_lora, (
        "%s: use_lora must equal what the GRAPH does with token %r" % (tier, token))
    assert engine._recipe_receipt().endswith("_lora") is graph_loads_lora, (
        "%s: the recipe receipt's _lora suffix must equal what the GRAPH does "
        "with token %r" % (tier, token))


def test_a_lora_free_tier_is_not_credited_with_a_lora():
    """The published consequence, pinned at the consumer's own rule.

    `otr_credits_roll` appends the word "lora" on truthiness of `use_lora`
    (`:238-239`). This asserts the 1.7B tiers -- the ones that genuinely run
    LoRA-free -- produce a value that makes that branch NOT fire, so the fix is
    pinned where the falsehood actually surfaced rather than only at its source.
    """
    for tier in ("humo_1.7B", "humo_1.7B_169"):
        engine = vreg.get_engine(tier)
        assert engine._lora_is_skipped(engine._loader_names().get("lora")), (
            "%s is expected to run LoRA-free; if that changed, this test is the "
            "wrong shape, not the engine" % tier)
        assert not engine._clip_telemetry(832, 480)["use_lora"]
