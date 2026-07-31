"""CPU unit tests for the clamped multi-arm video bench (chunk 1).

No GPU, no server, no torch. Covers the checklist the bench spec requires BEFORE
any GPU time: ArmSpec validation, class substitution INCLUDING stale-input
removal, per-arm step accounting, unique cell/repeat ids, graph pin + shape
integrity, telemetry-failure injection (no PASS may be reachable through broken
telemetry), stale-output rejection, ffprobe validation, the mandatory-cell
denominator, and repeat aggregation.

Spec: docs/2026-07-31-four-arm-clamped-video-bench-SPEC.md
UTF-8, no BOM. ASCII-only. SFW.
"""
import importlib.util
import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "run_video_arm_bakeoff.py")


def _load():
    spec = importlib.util.spec_from_file_location("otr_video_arm_bench", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load()


# --------------------------------------------------------------------------
# the shipped arm table
# --------------------------------------------------------------------------
def test_offline_preflight_passes_for_every_shipped_arm():
    """Graph pins match, the three Wan graphs differ EXACTLY by the declared
    substitutions, every model pin resolves, and every rung builds."""
    assert B.offline_preflight(B.ARMS) is True


def test_arm_c_has_no_entry_by_design():
    """Arm C is CUT: licence passes, but no ComfyUI base graph exists at any
    priority. A blocked arm must not add dead branching."""
    assert "C" not in B.ARMS_BY_LABEL
    assert [a.label for a in B.ARMS] == ["A", "B-partial", "B", "D"]


def test_arm_d_is_the_only_cross_family_arm():
    engines = {a.label: a.engine_id for a in B.ARMS}
    assert engines == {"A": "wan_ti2v", "B-partial": "wan_ti2v",
                       "B": "wan_ti2v", "D": "ltx_8gb"}


def test_campaign_ladder_is_legal_on_both_frame_contracts():
    """17 / 49 / 81 must satisfy Wan (min 17, quantum 4) AND LTX 8GB (min 9,
    quantum 8), so no arm is special-cased."""
    for n in B.CAMPAIGN_LENGTHS:
        for eng, fc in B.FRAME_CONTRACTS.items():
            assert fc["min"] <= n <= fc["max"], (eng, n)
            assert (n - fc["min"]) % fc["quantum"] == 0, (eng, n)


def test_arm_b_is_gated_on_a_named_acquisition_blocker():
    arm = B.ARMS_BY_LABEL["B"]
    assert arm.gated_reason
    assert "wan2.2_ti2v_5B_fp16.safetensors" in arm.gated_reason


def test_gated_arm_records_gated_without_booting(monkeypatch):
    """A gated cell must never reach reset/boot -- it returns a named blocker."""
    def explode(*a, **k):
        raise AssertionError("a gated arm must not touch the box")
    monkeypatch.setattr(B, "reset_box", explode)
    monkeypatch.setattr(B, "boot_server", explode)
    r = B.run_cell(B.ARMS_BY_LABEL["B"], 17, 0, {"estimator_status": "x",
                                                 "estimator_seams": {}}, 1600.0)
    assert r["status"] == "gated"
    assert "not on disk" in r["error"]


# --------------------------------------------------------------------------
# ArmSpec validation
# --------------------------------------------------------------------------
def _kwargs(**over):
    base = dict(label="X", engine_id="wan_ti2v", graph="arm_a_wan_ti2v_gguf.json",
                graph_sha256="0" * 64, steps=30, canvas=(832, 480),
                lengths=(17,), required_classes=("KSampler",),
                models=(B.ModelPin("1", "UnetLoaderGGUF", "unet_name", "u.gguf",
                                   "unet"),),
                length_node="8", save_node="12", notes="n")
    base.update(over)
    return base


@pytest.mark.parametrize("over,fragment", [
    ({"label": ""}, "filesystem-safe"),
    ({"label": "bad/label"}, "filesystem-safe"),
    ({"engine_id": "nope"}, "unknown engine_id"),
    ({"graph_sha256": "abc"}, "64 hex"),
    ({"steps": 0}, "positive int"),
    ({"canvas": (832, 432)}, "both-axes/32"),
    ({"lengths": ()}, "no lengths"),
    ({"lengths": (18,)}, "illegal for wan_ti2v"),
    ({"lengths": (9,)}, "illegal for wan_ti2v"),
    ({"required_classes": ()}, "no required node classes"),
    ({"models": ()}, "no model pins"),
])
def test_armspec_rejects_invalid_construction(over, fragment):
    with pytest.raises(B.BenchError) as ei:
        B.ArmSpec(**_kwargs(**over))
    assert fragment in str(ei.value)


def test_armspec_rejects_duplicate_model_roles():
    pin = B.ModelPin("1", "UnetLoaderGGUF", "unet_name", "u.gguf", "unet")
    with pytest.raises(B.BenchError) as ei:
        B.ArmSpec(**_kwargs(models=(pin, pin)))
    assert "duplicate model role" in str(ei.value)


def test_768x432_is_rejected_because_432_is_not_divisible_by_32():
    """The superseded research proposed 768x432 as a cheaper canvas. It is
    structurally illegal under OTR's grid rule and must not be constructible."""
    with pytest.raises(B.BenchError):
        B.ArmSpec(**_kwargs(canvas=(768, 432)))


# --------------------------------------------------------------------------
# substitution -- the delta the old mutate() could not express
# --------------------------------------------------------------------------
def test_substitution_changes_class_and_adds_required_inputs():
    p = {"1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "q.gguf"}}}
    out = B.apply_substitution(p, B.UNET_TO_NATIVE)
    assert out["1"]["class_type"] == "UNETLoader"
    assert out["1"]["inputs"]["weight_dtype"] == "default"
    assert out["1"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"


def test_substitution_removes_stale_class_specific_inputs():
    """UnetLoaderGGUF must NOT carry weight_dtype and CLIPLoaderGGUF must NOT
    carry device -- a substitution that only ADDS ships an invalid graph."""
    p = {"1": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "n.safetensors", "weight_dtype": "default"}},
         "3": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "c.safetensors", "type": "wan",
                          "device": "default"}}}
    p = B.apply_substitution(p, B.UNET_TO_GGUF)
    p = B.apply_substitution(p, B.CLIP_TO_GGUF)
    assert "weight_dtype" not in p["1"]["inputs"]
    assert "device" not in p["3"]["inputs"]
    assert p["1"]["class_type"] == "UnetLoaderGGUF"
    assert p["3"]["class_type"] == "CLIPLoaderGGUF"


def test_substitution_never_mutates_its_input():
    p = {"1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "q.gguf"}}}
    B.apply_substitution(p, B.UNET_TO_NATIVE)
    assert p["1"]["class_type"] == "UnetLoaderGGUF"
    assert "weight_dtype" not in p["1"]["inputs"]


def test_substitution_on_a_missing_node_raises():
    with pytest.raises(B.BenchError) as ei:
        B.apply_substitution({"9": {"class_type": "KSampler", "inputs": {}}},
                             B.UNET_TO_NATIVE)
    assert "not in graph" in str(ei.value)


def test_the_three_wan_graphs_check_each_other():
    """A + encoder-sub == B-partial; B-partial + unet-sub == B; and B inverts
    back to A, which is what proves stale-input removal actually removes."""
    assert B.assert_arm_graphs_consistent() is True


# --------------------------------------------------------------------------
# graph loading and pins
# --------------------------------------------------------------------------
def test_graph_pin_mismatch_fails_the_cell(tmp_path, monkeypatch):
    arm = B.ARMS_BY_LABEL["A"]
    bogus = B.ArmSpec(**dict(label="A2", engine_id="wan_ti2v", graph=arm.graph,
                             graph_sha256="1" * 64, steps=30, canvas=(832, 480),
                             lengths=(17,), required_classes=("KSampler",),
                             models=arm.models, length_node="8", save_node="12",
                             notes="n"))
    with pytest.raises(B.BenchError) as ei:
        B.load_graph(bogus)
    assert "SHA-256 mismatch" in str(ei.value)


def test_litegraph_ui_format_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "ui.json"
    p.write_text(json.dumps({"last_node_id": 3, "nodes": [], "links": []}),
                 encoding="utf-8")
    monkeypatch.setattr(B, "GRAPH_DIR", str(tmp_path))
    arm = B.ArmSpec(**_kwargs(graph="ui.json",
                              graph_sha256=B.sha256_file(str(p))))
    with pytest.raises(B.BenchError) as ei:
        B.load_graph(arm)
    assert "LiteGraph UI format" in str(ei.value)


def test_armspec_that_pins_a_class_the_graph_does_not_have_is_refused():
    arm = B.ARMS_BY_LABEL["A"]
    wrong = B.ArmSpec(label="A3", engine_id="wan_ti2v", graph=arm.graph,
                      graph_sha256=arm.graph_sha256, steps=30,
                      canvas=(832, 480), lengths=(17,),
                      required_classes=arm.required_classes,
                      models=(B.ModelPin("1", "UNETLoader", "unet_name",
                                         "x.safetensors", "unet"),),
                      length_node="8", save_node="12", notes="n")
    with pytest.raises(B.BenchError) as ei:
        B.offline_preflight([wrong])
    assert "the ArmSpec pins" in str(ei.value)


# --------------------------------------------------------------------------
# per-cell mutation, step accounting, cell ids
# --------------------------------------------------------------------------
def test_build_cell_prompt_sets_only_length_models_and_prefix():
    arm = B.ARMS_BY_LABEL["A"]
    base = B.load_graph(arm)
    resolved = {p.role: "resolved/" + p.filename for p in arm.models}
    before = base["8"]["inputs"]["length"]
    out = B.build_cell_prompt(arm, base, 81, resolved, "otr/x/y")
    assert out["8"]["inputs"]["length"] == 81
    assert out["12"]["inputs"]["filename_prefix"] == "otr/x/y"
    assert out["1"]["inputs"]["unet_name"].startswith("resolved/")
    # untouched controls stay exactly as pinned in the graph
    assert out["9"]["inputs"]["seed"] == 42
    assert out["9"]["inputs"]["steps"] == 30
    assert (out["8"]["inputs"]["width"], out["8"]["inputs"]["height"]) == (832, 480)
    assert base["8"]["inputs"]["length"] == before   # the base graph is not mutated


def test_build_cell_prompt_refuses_an_illegal_rung():
    arm = B.ARMS_BY_LABEL["A"]
    base = B.load_graph(arm)
    resolved = {p.role: p.filename for p in arm.models}
    with pytest.raises(B.BenchError) as ei:
        B.build_cell_prompt(arm, base, 18, resolved, "p")
    assert "illegal for wan_ti2v" in str(ei.value)


def test_build_cell_prompt_refuses_an_unresolved_model():
    arm = B.ARMS_BY_LABEL["A"]
    base = B.load_graph(arm)
    with pytest.raises(B.BenchError) as ei:
        B.build_cell_prompt(arm, base, 17, {"unet": "u"}, "p")
    assert "no resolved enum" in str(ei.value)


def test_step_count_is_arm_owned_not_a_global():
    """A global STEPS=30 corrupts s/it for the 8-step LTX arm."""
    assert B.ARMS_BY_LABEL["A"].steps == 30
    assert B.ARMS_BY_LABEL["D"].steps == 8
    assert not hasattr(B, "STEPS")


def test_cell_ids_are_unique_across_lengths_and_repeats():
    arm = B.ARMS_BY_LABEL["A"]
    ids = [arm.cell_id(n, r) for n in arm.lengths for r in range(3)]
    assert len(set(ids)) == len(ids)
    assert arm.cell_id(17, 0) == "A__17f"
    assert arm.cell_id(17, 2) == "A__17f__r2"


def test_port_and_output_root_have_one_owner():
    """G4: the harness used to hard-code 8000 while the launcher honoured
    OTR_HEADLESS_PORT, and honoured COMFYUI_OUTPUT while the launcher did not."""
    from urllib.parse import urlsplit
    assert B.PORT == (urlsplit(B.BASE_URL).port or 8000)
    assert os.path.isabs(B.OUTPUT_BASE)
    assert B.BENCH_DIR.startswith(B.OUTPUT_BASE)


def test_the_sole_campaign_clamp_is_8_gib():
    """CLAMP_TIERS used to default to three tiers, one of them UNCLAMPED, which
    would triple the matrix and contradict 'every cell runs clamped'."""
    assert B.CAMPAIGN_CLAMP_GIB == 8
    assert not hasattr(B, "CLAMP_TIERS")


def test_greenlight_bar_arithmetic_and_its_declared_assumption():
    assert B.DISPLAY_ALLOWANCE_MIB == 1024
    assert B.GREENLIGHT_PEAK_DELTA_MIB == 8192 - 1024 == 7168


# --------------------------------------------------------------------------
# telemetry -- NO PASS may be reachable through broken telemetry
# --------------------------------------------------------------------------
def _clean_cell(arm="A", length=17, **over):
    r = {
        "cell_id": "%s__%df" % (arm, length), "arm": arm, "length": length,
        "repeat": 0, "status": "ok", "peak_delta_mib": 5000.0,
        "sysram_delta_mib": 4000.0, "pagefile_commit_delta_mib": 1000.0,
        "spill_signatures": [],
        "telemetry": {"valid": True, "invalid_reasons": []},
        "stage_probe": {"valid": True, "torch_max_allocated_mib": 4200.0},
        "watchdog": {"ran": True, "dead": False},
        "asset": {"path": "x.mp4"},
    }
    r.update(over)
    return r


def test_a_clean_cell_blocks_nothing():
    assert B.cell_blocks_pass(_clean_cell()) == []


@pytest.mark.parametrize("over,fragment", [
    ({"status": "error", "error": "boom"}, "status=error"),
    ({"telemetry": {"valid": False, "invalid_reasons": ["no NVML peak recorded"]}},
     "telemetry invalid"),
    ({"stage_probe": {"valid": False, "reason": "expected exactly one reset"}},
     "stage probe invalid"),
    ({"watchdog": {"ran": True, "dead": True}}, "watchdog declared the leg dead"),
    ({"spill_signatures": ["cuda_oom"]}, "named spill signature"),
    ({"peak_delta_mib": 7169.0}, "exceeds the 7168 MiB bar"),
    ({"peak_delta_mib": None}, "no peak_delta_mib recorded"),
    ({"sysram_delta_mib": 999999.0}, "host thrash is a failure"),
    ({"pagefile_commit_delta_mib": 999999.0}, "pagefile commit delta"),
    ({"asset": {}}, "no validated asset"),
])
def test_every_telemetry_failure_mode_blocks_pass(over, fragment):
    why = B.cell_blocks_pass(_clean_cell(**over))
    assert why, "expected this cell to block PASS"
    assert any(fragment in w for w in why), why


def test_a_minus_one_sentinel_peak_can_never_pass():
    """The bar is `peak_delta <= 7168`, and `-1 <= 7168` is True. A sampler that
    failed must be rejected by the telemetry gate, not admitted by arithmetic."""
    s = B.PeakSampler()
    s.errors.append("nvmlInit failed")
    assert s.valid is False
    why = B.cell_blocks_pass(_clean_cell(
        peak_delta_mib=-1.0,
        telemetry=dict(s.as_dict())))
    assert any("telemetry invalid" in w for w in why)


def test_peak_sampler_validity_requires_samples_peaks_and_a_tight_gap():
    s = B.PeakSampler()
    s.samples = B.MIN_SAMPLES
    s.peak_vram_mib = 5000.0
    s.peak_sysram_mib = 9000.0
    s.max_gap_s = 0.3
    assert s.valid is True
    s.max_gap_s = B.MAX_SAMPLE_GAP_S + 0.1
    assert s.valid is False
    assert any("inter-sample gap" in r for r in s.invalid_reasons())
    s.max_gap_s = 0.3
    s.samples = B.MIN_SAMPLES - 1
    assert s.valid is False


def test_stage_probe_fails_closed_on_zero_or_duplicate_markers():
    reset = "[OTR_BakeoffVramReset] marker=aabbccdd1122 reset_peak_memory_stats OK"
    probe = ("[OTR_BakeoffVramProbe] marker=99887766aabb "
             "max_allocated_mb=4210.5 max_reserved_mb=5120.0")
    good = B.parse_stage_probe(reset + "\n" + probe)
    assert good["valid"] is True
    assert good["torch_max_allocated_mib"] == 4210.5
    assert good["torch_max_reserved_mib"] == 5120.0
    assert B.parse_stage_probe(probe)["valid"] is False          # no reset
    assert B.parse_stage_probe(reset)["valid"] is False          # no probe
    assert B.parse_stage_probe(reset + "\n" + probe + "\n" + reset
                               + "\n" + probe)["valid"] is False  # spans cells


# --------------------------------------------------------------------------
# log parsing
# --------------------------------------------------------------------------
def test_bare_offload_is_not_a_spill_signature():
    """The old harness matched a bare `offload`, which is ordinary loader
    chatter -- the hint fired on healthy runs and meant nothing."""
    healthy = ("Requested to load WAN21\nloaded partially; offload device cpu\n"
               "Prompt executed in 00:04:12\n")
    assert B.parse_spill(healthy) == []


@pytest.mark.parametrize("line,name", [
    ("torch.cuda.OutOfMemoryError: CUDA out of memory.", "cuda_oom"),
    ("CUDA out of memory. Tried to allocate 2.00 GiB", "cuda_oom"),
    ("Falling back to CPU because the device is out of memory", "sysmem_fallback"),
    ("Unloading model WAN21 to free 3.2 GB", "model_evicted_midrun"),
])
def test_named_spill_signatures_fire(line, name):
    assert name in B.parse_spill(line)


def test_parse_timing_prefers_the_log_and_reports_its_source():
    sit, source, pexec = B.parse_timing(
        "10%|#  | 3/30 [00:12<01:50,  4.10s/it]\n"
        "100%|###| 30/30 [02:03<00:00,  4.25s/it]\n"
        "Prompt executed in 00:02:19\n")
    assert sit == 4.25        # last s/it wins (the sampler bar)
    assert source == "log"
    assert pexec == "00:02:19"


def test_parse_timing_reports_unavailable_rather_than_inventing_a_number():
    sit, source, _ = B.parse_timing("nothing useful here\n")
    assert sit is None
    assert source == "unavailable"


# --------------------------------------------------------------------------
# asset validation -- from /history, never a glob
# --------------------------------------------------------------------------
def _probe_ok(**over):
    d = {"width": 832, "height": 480, "frames": 17, "fps": 24.0,
         "codec": "h264", "duration_s": 0.71}
    d.update(over)
    return d


def _stage_asset(tmp_path, monkeypatch, name="A__17f_00001.mp4", probe=None):
    out = tmp_path / "otr" / "episodes" / "_bench_4arm" / "A"
    out.mkdir(parents=True, exist_ok=True)
    f = out / name
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(B, "OUTPUT_BASE", str(tmp_path))
    monkeypatch.setattr(B, "_history_outputs", lambda pid, node: [
        {"filename": name, "subfolder": "otr/episodes/_bench_4arm/A",
         "type": "output"}])
    monkeypatch.setattr(B, "ffprobe_video", lambda p: dict(probe or _probe_ok()))
    return f


def test_validate_asset_accepts_a_fresh_history_owned_file(tmp_path, monkeypatch):
    f = _stage_asset(tmp_path, monkeypatch)
    t = os.path.getmtime(str(f))
    got = B.validate_asset("pid", B.ARMS_BY_LABEL["A"], 17, t - 1, t + 1)
    assert got["path"] == os.path.abspath(str(f))
    assert got["bytes"] == 4096


def test_validate_asset_rejects_a_stale_file_from_an_earlier_cell(tmp_path,
                                                                  monkeypatch):
    """A glob could select this file and report success for a render that never
    wrote anything."""
    f = _stage_asset(tmp_path, monkeypatch)
    old = os.path.getmtime(str(f)) - 100000
    os.utime(str(f), (old, old))
    now = os.path.getmtime(str(f)) + 100000
    with pytest.raises(B.BenchError) as ei:
        B.validate_asset("pid", B.ARMS_BY_LABEL["A"], 17, now, now + 1)
    assert "stale asset" in str(ei.value)


@pytest.mark.parametrize("probe,fragment", [
    (_probe_ok(width=512, height=288), "expected 832x480"),
    (_probe_ok(frames=16), "expected 17"),
    (_probe_ok(duration_s=0.0), "non-positive duration"),
])
def test_validate_asset_rejects_a_bad_decode(tmp_path, monkeypatch, probe,
                                             fragment):
    f = _stage_asset(tmp_path, monkeypatch, probe=probe)
    t = os.path.getmtime(str(f))
    with pytest.raises(B.BenchError) as ei:
        B.validate_asset("pid", B.ARMS_BY_LABEL["A"], 17, t - 1, t + 1)
    assert fragment in str(ei.value)


def test_validate_asset_rejects_a_path_escaping_the_output_root(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(B, "OUTPUT_BASE", str(tmp_path / "root"))
    (tmp_path / "root").mkdir()
    monkeypatch.setattr(B, "_history_outputs", lambda pid, node: [
        {"filename": "escape.mp4", "subfolder": "../..", "type": "output"}])
    with pytest.raises(B.BenchError) as ei:
        B.validate_asset("pid", B.ARMS_BY_LABEL["A"], 17, 0, 9e9)
    assert "escapes the configured output root" in str(ei.value)


def test_validate_asset_rejects_a_missing_or_multiple_history_entry(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(B, "OUTPUT_BASE", str(tmp_path))
    monkeypatch.setattr(B, "_history_outputs", lambda pid, node: [])
    with pytest.raises(B.BenchError) as ei:
        B.validate_asset("pid", B.ARMS_BY_LABEL["A"], 17, 0, 9e9)
    assert "reports 0 output files" in str(ei.value)


# --------------------------------------------------------------------------
# grading -- the mandatory denominator, and repeats
# --------------------------------------------------------------------------
def test_a_clean_full_ladder_passes():
    arm = B.ARMS_BY_LABEL["A"]
    cells = [_clean_cell("A", n) for n in arm.lengths]
    g = B.grade_arm(arm, cells)
    assert g["verdict"] == "PASS"
    assert g["blockers"] == []
    assert g["clean_lengths"] == list(arm.lengths)
    assert g["label"] == B.RESULT_LABEL


def test_a_missing_cell_blocks_pass_and_never_shrinks_the_denominator():
    arm = B.ARMS_BY_LABEL["A"]
    g = B.grade_arm(arm, [_clean_cell("A", 17), _clean_cell("A", 49)])
    assert g["verdict"] == "FAIL"
    assert g["required_lengths"] == list(arm.lengths)     # denominator intact
    assert any("length 81" in b for b in g["blockers"])


def test_passing_at_the_floor_alone_is_floor_only_not_a_tier_candidate():
    arm = B.ARMS_BY_LABEL["A"]
    cells = [_clean_cell("A", 17),
             _clean_cell("A", 49, status="error", error="OOM"),
             _clean_cell("A", 81, status="error", error="OOM")]
    g = B.grade_arm(arm, cells)
    assert g["verdict"] == "FAIL"      # blockers exist, so it cannot be PASS
    assert g["clean_lengths"] == [17]


def test_repeats_do_not_count_toward_the_required_denominator():
    """Only repeat 0 is a required cell; the winner's cold repeats must not be
    able to satisfy a rung that never passed on its own."""
    arm = B.ARMS_BY_LABEL["A"]
    cells = [_clean_cell("A", 17), _clean_cell("A", 49),
             _clean_cell("A", 81, repeat=1), _clean_cell("A", 81, repeat=2)]
    g = B.grade_arm(arm, cells)
    assert g["verdict"] == "FAIL"
    assert any("length 81" in b and "got 0" in b for b in g["blockers"])


def test_two_receipts_for_one_rung_block_pass():
    arm = B.ARMS_BY_LABEL["A"]
    cells = [_clean_cell("A", n) for n in arm.lengths] + [_clean_cell("A", 49)]
    g = B.grade_arm(arm, cells)
    assert g["verdict"] == "FAIL"
    assert any("got 2" in b for b in g["blockers"])


def test_another_arms_cells_cannot_satisfy_this_arm():
    arm = B.ARMS_BY_LABEL["A"]
    g = B.grade_arm(arm, [_clean_cell("D", n) for n in arm.lengths])
    assert g["verdict"] == "FAIL"
    assert len(g["blockers"]) == len(arm.lengths)


def test_arm_d_licence_is_unresolved_and_says_so():
    """O4: the >= $10,000,000 revenue clause admits two readings. It gates
    shipping, not measuring, so it must NOT block the arm's measurement."""
    lic = B.ARM_LICENCE["D"]
    assert lic["shippable"] is None
    assert "O4" in lic["note"]
    g = B.grade_arm(B.ARMS_BY_LABEL["D"],
                    [_clean_cell("D", n) for n in B.ARMS_BY_LABEL["D"].lengths])
    assert g["verdict"] == "PASS"
    assert g["licence"]["shippable"] is None


def test_an_unshippable_licence_blocks_pass(monkeypatch):
    monkeypatch.setitem(B.ARM_LICENCE, "A",
                        {"shippable": False, "licence": "CC BY-NC-SA"})
    arm = B.ARMS_BY_LABEL["A"]
    g = B.grade_arm(arm, [_clean_cell("A", n) for n in arm.lengths])
    assert g["verdict"] == "FAIL"
    assert any("not shippable" in b for b in g["blockers"])


# --------------------------------------------------------------------------
# the vendored helper (spec G3)
# --------------------------------------------------------------------------
def test_vendored_helper_is_tracked_in_this_repo_and_exports_the_probes():
    src = os.path.join(B.HELPER_SRC, "__init__.py")
    assert os.path.exists(src), "the bench helper must be vendored, not depended " \
                                "on in an untracked custom_nodes sibling"
    text = open(src, encoding="utf-8").read()
    for cls in ("OTR_BakeoffVramReset", "OTR_BakeoffVramProbe",
                "OTR_BakeoffReclaim"):
        assert cls in text
    assert not text.startswith("﻿"), "no BOM"


def test_helper_install_verifies_byte_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "HELPER_DST", str(tmp_path / "otr_bakeoff_helper"))
    sha = B.install_bench_helper()
    assert sha == B.sha256_file(os.path.join(B.HELPER_SRC, "__init__.py"))
    # a drifted install is repaired, not tolerated
    dst = os.path.join(B.HELPER_DST, "__init__.py")
    open(dst, "a", encoding="utf-8").write("\n# drift\n")
    assert B.sha256_file(dst) != sha
    assert B.install_bench_helper() == sha
    assert B.sha256_file(dst) == sha


def test_every_arm_graph_wires_both_probes():
    """Without the reset/probe pair the order-safe sample+decode segment cannot
    be measured, and parse_stage_probe fails the cell."""
    for arm in B.ARMS:
        classes = [nd["class_type"] for nd in B.load_graph(arm).values()]
        assert classes.count("OTR_BakeoffVramReset") == 1, arm.label
        assert classes.count("OTR_BakeoffVramProbe") == 1, arm.label


def test_watchdog_exit_two_is_the_only_failing_code():
    assert B.Watchdog.verdict(2)["dead"] is True
    assert B.Watchdog.verdict(0)["dead"] is False
    assert B.Watchdog.verdict(None)["dead"] is False
