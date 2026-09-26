"""LANE 2 -- named boot contracts, and their first real consumer.

Spec S8. Some lanes can only run correctly on a SERVER started a particular
way. By the time a beat renders the server has been up for an hour, so that is
not fixable at render time -- it has to be declared, applied at launch, and
PROVED. Since 2026-09-26 no contract reserves VRAM (operator: unload after use,
record an out-of-memory, never pre-empt it); what a contract pins is Sage, the
device, and the 8 GB H3 lab's pinned-memory switch.

The mechanism ships WITH its consumer on purpose. Unused infrastructure is how
you get a "configured" knob that reaches nothing, which is the exact defect this
lane is cleaning up: `launch.extra_args` has been written into a markdown
documentation string for months while `--disable-pinned-memory` appeared in
ZERO non-doc files repo-wide.

CPU-safe: no CUDA, no model loads, no renders.
"""

from __future__ import annotations

import copy
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
#: A shipping matrix row the tests below borrow a complete, validated profile
#: shape from. No matrix row selects a HuMo tier, so the hero cast is layered
#: on in code rather than read from a saved rig.
BASE_ROW = "otr_16gb_video"


@pytest.fixture()
def engine():
    return vreg.get_engine(LANE)


@pytest.fixture()
def profile(tmp_path):
    """The hero lane's cast, built from a live matrix row and read back
    through ``load_profile`` so the shape validator still runs."""
    prof = copy.deepcopy(load_profile(BASE_ROW))
    prof["id"] = "humo_hero_cast"
    for role in ("announcer_visual", "music_visual", "character_visual"):
        prof["role_overrides"][role] = LANE
    prof["render"] = dict(prof.get("render") or {},
                          canvas_w=DECLARED_CANVAS[0], canvas_h=DECLARED_CANVAS[1])
    prof["launch"]["env"] = {}
    prof["launch"]["boot_contract"] = bc.DEFAULT
    (tmp_path / "humo_hero_cast.json").write_text(
        json.dumps(prof), encoding="utf-8")
    return load_profile("humo_hero_cast", profile_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# The contract table
# ---------------------------------------------------------------------------

def test_the_default_contract_constrains_nothing():
    """Every profile that shipped before this key existed means `default`, so
    `default` must be a real no-op or this mechanism would retire them all."""
    spec = bc.contract_spec(bc.DEFAULT)
    assert set(spec.values()) == {None}
    assert bc.launch_env_for(bc.DEFAULT) == {}


def test_cpu_contract_owns_the_real_comfyui_argv():
    assert bc.contract_spec(bc.CPU)["cpu"] is True
    assert bc.launch_args_for(bc.CPU) == ["--cpu"]
    assert bc.launch_env_for(bc.CPU) == {}
    assert bc.check_running_server(
        bc.CPU, {"available": True, "cpu": True}) == []
    assert "--cpu" in " ".join(bc.check_running_server(
        bc.CPU, {"available": True, "cpu": False}))
    with pytest.raises(bc.BootContractError, match="generated launch recipe"):
        bc.assert_running_server(
            bc.CPU, {"available": True, "cpu": False})


def test_cpu_contract_is_identified_from_the_running_server(monkeypatch):
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True,
        "disable_pinned_memory": False,
        "sage_attention": False,
        "cpu": True,
    })
    assert bc.contract_from_running_server() == bc.CPU


@pytest.mark.parametrize("contract", [bc.H3, bc.H3_8GB_LAB])
def test_gpu_contracts_reject_a_conflicting_cpu_only_boot(contract):
    state = {
        "available": True,
        "disable_pinned_memory": True,
        "sage_attention": False,
        "cpu": True,
    }

    assert "CPU-only mode ON" in " ".join(
        bc.check_running_server(contract, state))


def test_dont_care_is_distinct_from_required_off():
    """`None` means the contract does not constrain that knob; `False` would
    mean it REQUIRES it off. Collapsing the two would let a contract silently
    forbid an unrelated flag."""
    assert bc.BOOT_CONTRACTS[bc.DEFAULT]["sage_attention"] is None
    assert bc.BOOT_CONTRACTS[bc.H3]["disable_pinned_memory"] is None
    assert bc.BOOT_CONTRACTS[bc.H3]["sage_attention"] is False
    assert bc.BOOT_CONTRACTS[bc.H3_8GB_LAB]["sage_attention"] is False


def test_no_contract_reserves_vram():
    """The operator's rule, 2026-09-26: models unload after use, and an
    out-of-memory is recorded, never pre-empted. No contract carries a reserve
    knob, and no launch argv or env row can emit one."""
    assert "reserve_vram_gb" not in bc.CONTRACT_ENV
    assert not hasattr(bc, "HUMO_DIET")
    for name, spec in bc.BOOT_CONTRACTS.items():
        assert "reserve_vram_gb" not in spec, name
        assert "--reserve-vram" not in bc.launch_args_for(name), name
        assert not any("RESERVE" in k for k in bc.launch_env_for(name)), name


def test_the_physical_8gb_h3_lab_launch_turns_pinned_memory_off():
    spec = bc.contract_spec(bc.H3_8GB_LAB)
    assert spec["disable_pinned_memory"] is True
    assert spec["sage_attention"] is False
    assert bc.launch_args_for(bc.H3_8GB_LAB) == ["--disable-pinned-memory"]
    assert bc.launch_env_for(bc.H3_8GB_LAB) == {
        "OTR_HEADLESS_DISABLE_PINNED": "1"}


def test_an_unknown_contract_raises_rather_than_meaning_no_constraints():
    """A typo in a profile must not resolve to 'unconstrained' -- that would
    turn a misspelled clamp into a silently unclamped boot."""
    with pytest.raises(bc.BootContractError):
        bc.contract_spec("humo_deit")


def test_the_env_mapping_only_emits_knobs_a_launcher_actually_reads():
    """Lesson L6. `sage_attention` gets NO env row because no launcher passes
    an attention flag -- emitting one would be another configured knob that
    reaches nothing. Sage-sensitive lanes refuse at assert_usable instead,
    which is enforcement that runs. Both halves are asserted: the knob a
    launcher reads DOES emit, and Sage still does not.
    """
    env = bc.launch_env_for(bc.H3_8GB_LAB)
    assert env == {"OTR_HEADLESS_DISABLE_PINNED": "1"}
    assert not any("SAGE" in k.upper() for k in env)
    # H3 constrains ONLY Sage since 2026-09-26 (operator: no artificial
    # reserve) -- and Sage has no launcher row, so H3 emits nothing at all.
    assert bc.launch_env_for(bc.H3) == {}
    assert bc.launch_args_for(bc.H3) == []


# ---------------------------------------------------------------------------
# The channel: launch.env is live, launch.extra_args is documentation
# ---------------------------------------------------------------------------

def test_the_launcher_applies_the_pinned_switch_and_passes_no_reserve():
    """THE POINT OF THE WHOLE LANE (lesson L6): a knob a launcher never turns
    into argv is documentation. The pinned-memory switch must reach the command
    line; the reserve channel is gone (operator, 2026-09-26)."""
    cmd = (REPO / "scripts" / "_otr_soak_server_launch.cmd").read_text(
        encoding="utf-8", errors="replace")
    assert ("if defined OTR_HEADLESS_DISABLE_PINNED set "
            "_OTR_PINNED=--disable-pinned-memory") in cmd
    # Declared is not applied: the variable must also reach the command line.
    assembly = cmd.split("main.py", 1)[1]
    assert "%_OTR_PINNED%" in assembly
    assert "_OTR_RESERVE" not in cmd


def test_every_shipped_profile_still_validates_with_the_new_optional_key():
    """`boot_contract` is OPTIONAL because the launch key set is
    closed-validated: a required key would have broken all ~20 profiles at
    once."""
    from nodes._otr_shared.capability_profiles import known_profile_ids
    ids = sorted(known_profile_ids())
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
    prof = load_profile(BASE_ROW)
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
    problems = bc.check_running_server(bc.H3_8GB_LAB, state={"available": False})
    assert problems, "a constrained contract may not pass on an unreadable server"
    assert "UNKNOWN is not satisfied" in problems[0]
    with pytest.raises(bc.BootContractError):
        bc.assert_running_server(bc.H3_8GB_LAB, state={"available": False})
    # ...and the stock contract constrains nothing, so it is genuinely met.
    assert bc.check_running_server(bc.DEFAULT, state={"available": False}) == []


def test_a_failed_sage_probe_is_not_a_pass():
    """`running_server_boot_state` recorded `sage_probe_error` and NOTHING read
    it, so a probe that raised left `sage_attention = None` and the comparison
    skipped -- silently passing a Sage-constrained contract on the exact lanes
    Sage silently corrupts. Recording an error nobody reads is swallowing it.

    The state SATISFIES every other H3 knob on purpose, so the only thing this
    can fail on is its subject.
    """
    state = {"available": True, "disable_pinned_memory": True,
             "sage_attention": None, "sage_probe_error": "ImportError"}
    problems = bc.check_running_server(bc.H3, state=state)
    assert problems and "ImportError" in problems[0]
    assert "not a pass" in problems[0]


_LAB_OK = {"available": True, "disable_pinned_memory": True,
           "sage_attention": False, "cpu": False}


@pytest.mark.parametrize("state,needle", [
    (dict(_LAB_OK, disable_pinned_memory=False), "--disable-pinned-memory"),
    (dict(_LAB_OK, sage_attention=True), "SageAttention"),
    (dict(_LAB_OK, cpu=True), "CPU-only"),
])
def test_a_server_started_without_the_lab_boot_is_named_knob_by_knob(state, needle):
    problems = bc.check_running_server(bc.H3_8GB_LAB, state=state)
    assert problems and any(needle in p for p in problems)
    with pytest.raises(bc.BootContractError) as exc:
        bc.assert_running_server(bc.H3_8GB_LAB, state=state)
    assert "restarted" in str(exc.value), (
        "the message must say what the operator has to DO -- this is not "
        "fixable at render time")


def test_sage_is_checked_only_when_the_contract_names_it():
    assert bc.check_running_server(bc.DEFAULT, state={
        "available": True, "disable_pinned_memory": True,
        "sage_attention": True}) == []
    # Every OTHER H3 knob is satisfied here on purpose, so Sage is the only
    # thing left to complain about.
    problems = bc.check_running_server(bc.H3, state={
        "available": True, "disable_pinned_memory": True,
        "sage_attention": True})
    assert problems and "SageAttention" in problems[0]


# ---------------------------------------------------------------------------
# Engine compatibility -- only a lane that never shipped under `default` may
# REQUIRE a contract
# ---------------------------------------------------------------------------

def test_an_engine_that_declares_nothing_keeps_legacy_boots_not_cpu():
    """Legacy tuning remains compatible; a device change is never implied."""
    assert set(bc.compatible_contracts_for_engine(
        vreg.get_engine("ltx_8gb"))) == set(bc.BOOT_CONTRACTS) - {bc.CPU}


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
        "available": True, "disable_pinned_memory": False,
        "sage_attention": None})
    assert bc.check_engine_against_profile(engine, profile) == []
    assert engine.assert_usable(host_caps={}, profile=profile) == LANE


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
        (REPO / "apple" / "evidence" / "video_evidence_manifest.json")
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


def test_every_humo_lane_declares_a_canvas_and_runs_on_any_gpu_boot():
    """The family is closed: four lanes, four canvases, readable without
    loading anything. No HuMo lane declares a boot contract since 2026-09-26 -- the
    reserve diet they could also run under was retired -- so each runs on
    every GPU boot, as HuMo always shipped."""
    expected = {
        "humo": (480, 832), "humo_1.7B": (480, 832),
        "humo_1.7B_169": (832, 480), "humo_14B_169": (832, 480),
    }
    for tier, canvas in expected.items():
        engine = vreg.get_engine(tier)
        assert tuple(engine.render_canvas) == canvas, tier
        assert set(bc.compatible_contracts_for_engine(engine)) == (
            set(bc.BOOT_CONTRACTS) - {bc.CPU}), tier


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


# --------------------------------------------------------------------------- #
# PBUG: the boot contract was dropped in transport, so H3 refused ITSELF
# --------------------------------------------------------------------------- #

def _stripped_policy():
    """Exactly what every adapter receives on a REAL episode leg."""
    from nodes._otr_video_engines import render_driver as rd
    return rd.build_episode_render_policy(
        {"device_policy": "cuda", "dtype_policy": "fp8_ok"})


def test_the_policy_every_adapter_receives_carries_no_boot_contract():
    """THE TRANSPORT DEFECT ITSELF, pinned so the diagnosis cannot rot.

    `build_episode_render_policy` returns four keys and no `launch`, so
    `contract_for_profile` answers `default` on every production leg no matter
    what the profile JSON declared or how the server was actually started. This
    is the fact the fix below exists to work around; if it ever stops being
    true -- because the contract starts riding the ledger the way
    `max_render_frames` does -- the fallback becomes dead code and should go.
    """
    policy = _stripped_policy()
    assert "launch" not in policy
    assert bc.contract_for_profile(policy) == bc.DEFAULT


def test_h3_stops_refusing_itself_when_the_server_really_is_booted_for_h3(monkeypatch):
    """THE LIVE FAILURE, 2026-08-26. The soak booted MiniMax H3 correctly and
    the adapter rejected itself as INCOMPATIBLE_PROFILE 9.6 minutes in, having
    never reached H3 sampling.

    The profile is silent (see the test above), so the check asks the SERVER
    what it was really started with before refusing. Since 2026-09-26 the H3
    boot is a plain stock boot with SageAttention off (operator: no artificial
    reserve), so that is the state it must recognise.
    """
    from nodes._otr_video_engines import registry as vreg
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True, "disable_pinned_memory": False,
        "sage_attention": False})
    assert bc.contract_from_running_server() == bc.H3
    h3 = vreg.get_engine("minimax_h3_video")
    assert bc.check_engine_against_profile(h3, _stripped_policy()) == []


def test_h3_still_refuses_on_a_sage_boot(monkeypatch):
    """THE GUARD MUST STILL GUARD. Probing the server is not a licence to pass:
    SageAttention turns H3's output to noise and reports success, so H3 on a
    Sage boot is exactly the case the contract exists to catch."""
    from nodes._otr_video_engines import registry as vreg
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True, "disable_pinned_memory": False,
        "sage_attention": True})
    assert bc.contract_from_running_server() == bc.DEFAULT
    h3 = vreg.get_engine("minimax_h3_video")
    assert bc.check_engine_against_profile(h3, _stripped_policy()) != []


def test_an_explicit_declaration_always_wins_over_the_probe(monkeypatch):
    """A profile that NAMES a contract is answered from the profile, never from
    the server. Otherwise a deliberate `default` pin would be silently upgraded
    by whatever the box happened to be booted with."""
    from nodes._otr_video_engines import registry as vreg
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True, "disable_pinned_memory": True,
        "sage_attention": False})
    h3 = vreg.get_engine("minimax_h3_video")
    assert bc.check_engine_against_profile(
        h3, {"launch": {"boot_contract": "default"}}) != []


def test_a_lane_that_already_passed_on_default_is_untouched(monkeypatch):
    """THE BLAST-RADIUS FENCE. The probe runs ONLY when the engine would
    otherwise be refused, so a lane compatible with `default` keeps its exact
    behaviour and never gains a dependency on the server's boot flags -- even
    when the box is booted for something else entirely."""
    from nodes._otr_video_engines import registry as vreg
    humo = vreg.get_engine("humo")
    assert bc.check_engine_against_profile(humo, _stripped_policy()) == []
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True, "disable_pinned_memory": True,
        "sage_attention": False})
    assert bc.check_engine_against_profile(humo, _stripped_policy()) == []


def test_the_probe_is_silent_off_a_comfy_server():
    """The CPU suite has no `comfy.cli_args`. The probe must answer None rather
    than guessing, which is what keeps every other test in this file
    deterministic."""
    assert bc.contract_from_running_server() is None


def test_default_never_shadows_a_real_named_boot(monkeypatch):
    """`default` constrains nothing, so it matches EVERY server. It is checked
    last on purpose -- tried first it would claim an h3 or h3_8gb_lab boot as
    `default` and the fix would silently do nothing."""
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True, "disable_pinned_memory": True,
        "sage_attention": False})
    assert bc.contract_from_running_server() == bc.H3_8GB_LAB
