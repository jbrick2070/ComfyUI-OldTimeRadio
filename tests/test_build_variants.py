"""S5: variant generator + semantic master_hash + validator tripwire.

docs/2026-07-09-platform-portability-final.md section 1: a platform
variant = stamped regenerated JSON + launch recipe from ONE canonical;
the SAME semantic normalizer guards both the generator (stamp time) and
OTR_WorkflowValidator._assert_stamp (verify time); ratify_before_emit
refuses emission until the operator clears it; the otr_api stale-variant
soft skip stays dead.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_variants as bv  # noqa: E402
from nodes import _otr_workflow_apply as wa  # noqa: E402


@pytest.fixture(scope="module")
def schemas():
    return wa.build_offline_schemas()


@pytest.fixture(scope="module")
def mapping():
    return wa.load_widget_mapping()


@pytest.fixture(scope="module")
def canonical():
    return bv._load_canonical()


# ---------------------------------------------------------------------------
# semantic master_hash
# ---------------------------------------------------------------------------

def test_semantic_hash_ignores_creative_flags_managed(canonical, schemas,
                                                      mapping):
    base = wa.semantic_master_hash(canonical, mapping=mapping,
                                   schemas=schemas)
    # Creative widget edit (episode_title is exempt) -> hash UNCHANGED.
    ep = copy.deepcopy(canonical)
    wa.patch_widget_by_name(ep, 1, "episode_title", "A Different Title",
                            schemas)
    assert wa.semantic_master_hash(ep, mapping=mapping,
                                   schemas=schemas) == base
    # Managed widget edit (fps) -> hash CHANGES.
    man = copy.deepcopy(canonical)
    wa.patch_widget_by_name(man, 87, "fps", 24, schemas)
    assert wa.semantic_master_hash(man, mapping=mapping,
                                   schemas=schemas) != base
    # Wiring edit -> hash CHANGES.
    lk = copy.deepcopy(canonical)
    lk["links"] = list(lk["links"]) + [[999, 63, 0, 88, 1, "STRING"]]
    assert wa.semantic_master_hash(lk, mapping=mapping,
                                   schemas=schemas) != base
    # Stamps are excluded: writing them must NOT move the hash.
    st = copy.deepcopy(canonical)
    wa.patch_widget_by_name(st, 63, "master_hash", "deadbeef", schemas)
    wa.patch_widget_by_name(st, 63, "profile_id", "cpu_floor", schemas)
    assert wa.semantic_master_hash(st, mapping=mapping,
                                   schemas=schemas) == base


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

def test_build_variant_refuses_ratify_gated(canonical, schemas, mapping):
    # Ratified 2026-07-10 morning: 16gb_full (baseline regen) + otr_mac_mps
    # (ceiling 10.0) emit now; the cloud tier stays gated on its OpenRouter
    # slot pins -- the ONE live example of the refusal contract.
    with pytest.raises(bv.EmitRefused, match="UNRATIFIED"):
        bv.build_variant("otr_cloud_lanes", schemas=schemas, mapping=mapping,
                         canonical=canonical)


def test_build_variant_cpu_floor_stamps_and_selfchecks(canonical, schemas,
                                                       mapping):
    variant, rel, recipe = bv.build_variant(
        "cpu_floor", schemas=schemas, mapping=mapping, canonical=canonical)
    assert rel == "workflows/variants/otr_cpu_floor.json"
    vnode = next(n for n in variant["nodes"]
                 if n["type"] == "OTR_WorkflowValidator")
    wv = vnode["widgets_values"]
    assert wv[0] == rel
    assert wv[3] == "cpu_floor"
    assert wv[5] == bv.GENERATED_BY
    assert wv[4] == wa.semantic_master_hash(variant, mapping=mapping,
                                            schemas=schemas)
    # Profile-managed values landed: castlock voice_device (slot 5) = cpu,
    # writer llm_device (slot 26) = cpu, gguf_quant (slot 31) = Q4_K_M.
    # (2026-08-14: target_words removal shifted these from 28/33 to 27/32.
    #  2026-08-28: refine_target_grade removal shifted them again to 26/31 --
    #  it sat at slot 20, above both, so everything below moved down one.)
    n80 = next(n for n in variant["nodes"] if n["id"] == 80)
    assert n80["widgets_values"][5] == "cpu"
    n1 = next(n for n in variant["nodes"] if n["id"] == 1)
    assert n1["widgets_values"][26] == "cpu"
    assert n1["widgets_values"][31] == "Q4_K_M"
    # Recipe carries args + env pointers + key names, never key values.
    assert "--cpu" in recipe
    assert "OTR_COMFYUI_MODELS_ROOT" in recipe
    assert "NEVER stored" in recipe


@pytest.mark.parametrize("profile_id", [
    "otr_4060_h3_nano",
    "otr_nvidia_8gb_h3",
])
def test_8gb_h3_launch_recipe_emits_no_reserve_clamp(
        profile_id, canonical, schemas, mapping):
    _variant, _rel, recipe = bv.build_variant(
        profile_id, schemas=schemas, mapping=mapping, canonical=canonical)
    args_line = next(line for line in recipe.splitlines()
                     if line.startswith("- args:"))
    assert args_line == "- args: `--disable-pinned-memory`"
    assert "--reserve-vram" not in args_line

    _control, _control_rel, control_recipe = bv.build_variant(
        "otr_w45_minimax_h3_video", schemas=schemas, mapping=mapping,
        canonical=canonical)
    control_args = next(line for line in control_recipe.splitlines()
                        if line.startswith("- args:"))
    assert control_args == \
        "- args: `--reserve-vram 12 --disable-pinned-memory`"


def test_build_variant_leaves_canonical_untouched(canonical, schemas,
                                                  mapping):
    before = json.dumps(canonical, sort_keys=True)
    bv.build_variant("cpu_floor", schemas=schemas, mapping=mapping,
                     canonical=canonical)
    assert json.dumps(canonical, sort_keys=True) == before


# ---------------------------------------------------------------------------
# validator tripwire (the SAME normalizer on the verify side)
# ---------------------------------------------------------------------------

def test_validator_asserts_master_hash(tmp_path, canonical, schemas,
                                       mapping, monkeypatch):
    from nodes._otr_workflow_validator import WorkflowValidator
    from nodes._otr_shared import boot_contracts as bc

    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True,
        "reserve_vram_gb": None,
        "disable_pinned_memory": False,
        "sage_attention": False,
        "cpu": True,
    })

    variant, rel, _recipe = bv.build_variant(
        "cpu_floor", schemas=schemas, mapping=mapping, canonical=canonical)
    vnode = next(n for n in variant["nodes"]
                 if n["type"] == "OTR_WorkflowValidator")
    stamped_hash = vnode["widgets_values"][4]

    good = tmp_path / "otr_cpu_floor.json"
    good.write_text(bv._dump(variant), encoding="utf-8")
    v = WorkflowValidator()
    msg = v._assert_stamp(str(good), "cpu_floor", stamped_hash,
                          bv.GENERATED_BY)
    assert "stamp OK" in msg

    # Tamper a MANAGED widget post-emission -> MASTER-HASH MISMATCH.
    tampered = copy.deepcopy(variant)
    wa.patch_widget_by_name(tampered, 87, "fps", 24, schemas)
    bad = tmp_path / "otr_tampered.json"
    bad.write_text(bv._dump(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="MASTER-HASH MISMATCH"):
        v._assert_stamp(str(bad), "cpu_floor", stamped_hash,
                        bv.GENERATED_BY)


def test_validator_refuses_cpu_snapshot_on_a_non_cpu_server(
        tmp_path, canonical, schemas, mapping, monkeypatch):
    from nodes._otr_workflow_validator import WorkflowValidator
    from nodes._otr_shared import boot_contracts as bc

    variant, _rel, _recipe = bv.build_variant(
        "cpu_floor", schemas=schemas, mapping=mapping, canonical=canonical)
    vnode = next(n for n in variant["nodes"]
                 if n["type"] == "OTR_WorkflowValidator")
    path = tmp_path / "otr_cpu_floor.json"
    path.write_text(bv._dump(variant), encoding="utf-8")
    monkeypatch.setattr(bc, "running_server_boot_state", lambda: {
        "available": True,
        "reserve_vram_gb": None,
        "disable_pinned_memory": False,
        "sage_attention": False,
        "cpu": False,
    })

    with pytest.raises(ValueError, match="needs --cpu ON"):
        WorkflowValidator()._assert_stamp(
            str(path), "cpu_floor", vnode["widgets_values"][4],
            bv.GENERATED_BY,
        )


# ---------------------------------------------------------------------------
# otr_api: the stale-variant soft skip stays dead
# ---------------------------------------------------------------------------

def test_otr_api_widget_vector_mismatch_hard_fails(schemas):
    import otr_api

    wf = {"nodes": [{"type": "OTR_WorkflowValidator", "id": 63,
                     "widgets_values": ["", True, True]}]}
    with pytest.raises(ValueError, match="NO soft skip"):
        otr_api.normalize_stamp_widgets_for_live_schema(wf, schemas)
    src = (REPO_ROOT / "scripts" / "otr_api.py").read_text(encoding="utf-8")
    assert "trimming the 3 EMPTY stamp slots" not in src


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------

def test_check_detects_variant_drift(tmp_path, monkeypatch, canonical,
                                     schemas, mapping):
    variant, rel, recipe = bv.build_variant(
        "cpu_floor", schemas=schemas, mapping=mapping, canonical=canonical)
    vdir = tmp_path / "variants"
    vdir.mkdir()
    (vdir / "otr_cpu_floor.json").write_text(bv._dump(variant),
                                             encoding="utf-8")
    (vdir / "otr_cpu_floor.launch.md").write_text(recipe, encoding="utf-8")
    monkeypatch.setattr(bv, "VARIANTS_DIR", vdir)
    assert bv.cmd_check() == 0

    # Hand-edit a managed widget on disk -> drift + stamp disagreement.
    tampered = copy.deepcopy(variant)
    wa.patch_widget_by_name(tampered, 87, "fps", 24, schemas)
    (vdir / "otr_cpu_floor.json").write_text(bv._dump(tampered),
                                             encoding="utf-8")
    assert bv.cmd_check() == 1
