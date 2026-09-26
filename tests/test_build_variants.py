"""S5: variant generator + semantic master_hash + validator tripwire.

docs/2026-07-09-platform-portability-final.md section 1: a platform
variant = stamped regenerated JSON + its launch-recipe section from ONE canonical;
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
from nodes._otr_shared import capability_profiles as cp  # noqa: E402
from tests._support.writer_slots import value  # noqa: E402

#: The cpu-backend row the stamp / self-check / validator tests emit: it names the
#: `cpu` boot contract, so its recipe and its validator check both exercise `--cpu`.
CPU_ROW = "otr_cloud_low"


@pytest.fixture(scope="module")
def schemas():
    return wa.build_offline_schemas()


@pytest.fixture(scope="module")
def mapping():
    return wa.load_widget_mapping()


@pytest.fixture(scope="module")
def canonical():
    return bv._load_canonical()


@pytest.fixture()
def tmp_rows(tmp_path, monkeypatch):
    """Route `build_variant` at documents this test writes into `tmp_path`.

    `load_profile` resolves only matrix rows unless it is handed a
    `profile_dir`; `build_variant` passes none, so its import is pointed at
    the test's own directory. Each document starts from a REAL matrix row and
    changes only the key under test.
    """
    def write(base_id, new_id, **changes):
        doc = copy.deepcopy(cp.load_profile(base_id))
        doc["id"] = new_id
        for key, val in changes.items():
            if isinstance(val, dict) and isinstance(doc.get(key), dict):
                doc[key].update(val)
            else:
                doc[key] = val
        (tmp_path / f"{new_id}.json").write_text(json.dumps(doc),
                                                 encoding="utf-8")
        return new_id

    monkeypatch.setattr(
        bv, "load_profile",
        lambda pid: cp.load_profile(pid, profile_dir=str(tmp_path)))
    return write


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
    wa.patch_widget_by_name(st, 63, "profile_id", CPU_ROW, schemas)
    assert wa.semantic_master_hash(st, mapping=mapping,
                                   schemas=schemas) == base


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

def test_build_variant_refuses_ratify_gated(tmp_rows, canonical, schemas,
                                            mapping):
    # No matrix row carries the gate today -- the five shipping cloud SKUs
    # (low_1act / low / low_5act / deluxe Foley / deluxe audio-in) are
    # ratified -- so the gated document is a cloud row with one decision
    # still open.
    gated = tmp_rows(CPU_ROW, "otr_gated_probe", ratify_before_emit=[
        "openrouter_model_pins: operator ratifies concrete OpenRouter slugs "
        "for slot-a/slot-b before emission"])
    with pytest.raises(bv.EmitRefused, match="UNRATIFIED"):
        bv.build_variant(gated, schemas=schemas, mapping=mapping,
                         canonical=canonical)


@pytest.mark.parametrize("profile_id", [
    "otr_cloud_low_1act",
    "otr_cloud_low",
    "otr_cloud_low_5act",
    "otr_cloud_deluxe_3act",
    "otr_cloud_deluxe_audio_in_3act",
])
def test_build_variant_emits_shipping_cloud_skus(
        profile_id, canonical, schemas, mapping):
    variant, rel, recipe = bv.build_variant(
        profile_id, schemas=schemas, mapping=mapping, canonical=canonical)
    assert rel == f"workflows/{profile_id}.json"
    vnode = next(n for n in variant["nodes"]
                 if n["type"] == "OTR_WorkflowValidator")
    assert value(vnode, "profile_id") == profile_id
    from nodes.otr_video_director import exact_menu_option_for
    director = next(n for n in variant["nodes"]
                    if n["type"] == "OTR_VideoDirector")
    if "deluxe_audio_in" in profile_id:
        engine = "cloud_ltx25_audio_in"
    elif "deluxe" in profile_id:
        engine = "cloud_ltx25_foley_plus"
    else:
        engine = "cloud_vidu_q2_pro_fast_720p"
    label = exact_menu_option_for(engine)
    assert value(director, "announcer_video_model") == label
    assert value(director, "character_video_model") == label
    assert "OTR_COMFY_API_KEY" in recipe
    writer = next(n for n in variant["nodes"]
                  if n["type"] == "OTR_LedgerScriptWriter")
    assert value(writer, "creative_writing_model") == "comfy:slot-a"
    want_a = "anthropic/claude-sonnet-5"
    assert value(writer, "comfy_slot_a_model") == want_a
    assert value(writer, "comfy_slot_b_model") == "openai/gpt-5.6-luna"
    want_cast = 4 if "deluxe" in profile_id else 3
    assert value(writer, "num_characters") == want_cast
    if "deluxe" in profile_id:
        assert "OPENROUTER_API_KEY" not in recipe


def test_build_variant_cpu_row_stamps_and_selfchecks(canonical, schemas,
                                                     mapping):
    variant, rel, recipe = bv.build_variant(
        CPU_ROW, schemas=schemas, mapping=mapping, canonical=canonical)
    assert rel == "workflows/otr_cloud_low.json"
    # An id that already carries the prefix is not doubled; a bare one gains it.
    assert bv._variant_stem("probe") == "otr_probe"
    vnode = next(n for n in variant["nodes"]
                 if n["type"] == "OTR_WorkflowValidator")
    assert value(vnode, "workflow_json_path") == rel
    assert value(vnode, "profile_id") == CPU_ROW
    assert value(vnode, "generated_by") == bv.GENERATED_BY
    assert value(vnode, "master_hash") == wa.semantic_master_hash(
        variant, mapping=mapping, schemas=schemas)
    # The profile's managed values reached the nodes that own them: the cast
    # lock renders its voices on the CPU, and the writer runs its model on the
    # CPU. Each widget is found by its own NAME, so adding or removing an
    # unrelated control above it leaves these assertions alone -- which is the
    # whole reason no position is written down here, and is why removing the
    # writer's two retired quant widgets cost this test one line rather than a
    # re-index.
    castlock = next(n for n in variant["nodes"] if n["id"] == 80)
    assert value(castlock, "voice_device") == "cpu"
    writer = next(n for n in variant["nodes"] if n["id"] == 1)
    assert value(writer, "llm_device") == "cpu"
    # Recipe carries args + env pointers + key names, never key values.
    assert "--cpu" in recipe
    assert "OTR_COMFYUI_MODELS_ROOT" in recipe
    assert "NEVER stored" in recipe


def test_8gb_h3_launch_recipe_emits_no_reserve_clamp(
        tmp_rows, canonical, schemas, mapping):
    """The recipe's argv comes from the NAMED boot contract: the 8 GB H3 lab
    shape emits no reserve clamp, while the 16 GB H3 contract reserves 12 GiB.
    No matrix row names either contract, so each is a real row with only its
    `launch.boot_contract` changed."""
    lab = tmp_rows("otr_8gb_low", "otr_8gb_h3_lab_probe",
                   launch={"boot_contract": "h3_8gb_lab"})
    _variant, _rel, recipe = bv.build_variant(
        lab, schemas=schemas, mapping=mapping, canonical=canonical)
    args_line = next(line for line in recipe.splitlines()
                     if line.startswith("- args:"))
    assert args_line == "- args: `--disable-pinned-memory`"
    assert "--reserve-vram" not in args_line

    control = tmp_rows("otr_16gb_low", "otr_16gb_h3_probe",
                       launch={"boot_contract": "h3"})
    _control, _control_rel, control_recipe = bv.build_variant(
        control, schemas=schemas, mapping=mapping, canonical=canonical)
    control_args = next(line for line in control_recipe.splitlines()
                        if line.startswith("- args:"))
    assert control_args == \
        "- args: `--reserve-vram 12 --disable-pinned-memory`"


def test_build_variant_leaves_canonical_untouched(canonical, schemas,
                                                  mapping):
    before = json.dumps(canonical, sort_keys=True)
    bv.build_variant(CPU_ROW, schemas=schemas, mapping=mapping,
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
        CPU_ROW, schemas=schemas, mapping=mapping, canonical=canonical)
    vnode = next(n for n in variant["nodes"]
                 if n["type"] == "OTR_WorkflowValidator")
    stamped_hash = value(vnode, "master_hash")

    good = tmp_path / "otr_cloud_low.json"
    good.write_text(bv._dump(variant), encoding="utf-8")
    v = WorkflowValidator()
    msg = v._assert_stamp(str(good), CPU_ROW, stamped_hash,
                          bv.GENERATED_BY)
    assert "stamp OK" in msg

    # Tamper a MANAGED widget post-emission -> MASTER-HASH MISMATCH.
    tampered = copy.deepcopy(variant)
    wa.patch_widget_by_name(tampered, 87, "fps", 24, schemas)
    bad = tmp_path / "otr_tampered.json"
    bad.write_text(bv._dump(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="MASTER-HASH MISMATCH"):
        v._assert_stamp(str(bad), CPU_ROW, stamped_hash,
                        bv.GENERATED_BY)


def test_validator_refuses_cpu_snapshot_on_a_non_cpu_server(
        tmp_path, canonical, schemas, mapping, monkeypatch):
    from nodes._otr_workflow_validator import WorkflowValidator
    from nodes._otr_shared import boot_contracts as bc

    variant, _rel, _recipe = bv.build_variant(
        CPU_ROW, schemas=schemas, mapping=mapping, canonical=canonical)
    vnode = next(n for n in variant["nodes"]
                 if n["type"] == "OTR_WorkflowValidator")
    path = tmp_path / "otr_cloud_low.json"
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
            str(path), CPU_ROW, value(vnode, "master_hash"),
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

def test_check_still_checks_the_canonical_thumbnail_with_no_variants(
        tmp_path, monkeypatch):
    """Sonnet QA on 7cf82cda: `cmd_check` returned early when no variant was
    committed, before the thumbnail check, so a missing canonical thumbnail
    passed. An empty variants folder must still fail on it."""
    empty = tmp_path / "workflows"
    empty.mkdir()
    master = tmp_path / "otr_gallery_thumb.jpg"
    master.write_bytes(b"\xff\xd8 gallery art \xff\xd9")
    monkeypatch.setattr(bv, "VARIANTS_DIR", empty)
    monkeypatch.setattr(bv, "GALLERY_THUMB", master)
    assert bv.cmd_check() == 1
    (empty / (bv.CANONICAL.stem + ".jpg")).write_bytes(master.read_bytes())
    # The advanced app (plan 0e) ships beside the canonical, with its own art.
    (empty / "otr_app.jpg").write_bytes(master.read_bytes())
    assert bv.cmd_check() == 0


def test_check_detects_variant_drift(tmp_path, monkeypatch, canonical,
                                     schemas, mapping):
    variant, rel, recipe = bv.build_variant(
        CPU_ROW, schemas=schemas, mapping=mapping, canonical=canonical)
    vdir = tmp_path / "variants"
    vdir.mkdir()
    (vdir / "otr_cloud_low.json").write_text(bv._dump(variant),
                                             encoding="utf-8")
    recipes_doc = tmp_path / "LAUNCH_RECIPES.md"
    recipes_doc.write_text(bv.render_launch_recipes([("otr_cloud_low", recipe)]),
                           encoding="utf-8", newline="\n")
    monkeypatch.setattr(bv, "VARIANTS_DIR", vdir)
    monkeypatch.setattr(bv, "LAUNCH_RECIPES", recipes_doc)
    # The advanced app (plan 0e) is generated from the canonical; a missing
    # or hand-edited one is drift like any variant.
    app = vdir / "otr_app.json"
    app.write_text(bv._dump(bv.build_app(canonical)), encoding="utf-8")
    # The gallery thumbnail: one copy beside the canonical and each variant,
    # byte-identical to the master (2026-09-25).
    master = tmp_path / "otr_gallery_thumb.jpg"
    master.write_bytes(b"\xff\xd8 gallery art \xff\xd9")
    monkeypatch.setattr(bv, "GALLERY_THUMB", master)
    for target in bv._thumbnail_targets():
        target.write_bytes(master.read_bytes())
    assert bv.cmd_check() == 0

    # A missing, stale or orphaned gallery thumbnail each fail the check.
    thumb = vdir / "otr_cloud_low.jpg"
    thumb.unlink()
    assert bv.cmd_check() == 1
    thumb.write_bytes(b"an older picture")
    assert bv.cmd_check() == 1
    thumb.write_bytes(master.read_bytes())
    orphan = vdir / "otr_retired_graph.jpg"
    orphan.write_bytes(master.read_bytes())
    assert bv.cmd_check() == 1
    orphan.unlink()
    assert bv.cmd_check() == 0

    good_app = app.read_text(encoding="utf-8")
    app.write_text(good_app.replace('"linearMode":true', '"linearMode":false'),
                   encoding="utf-8")
    assert bv.cmd_check() == 1
    app.unlink()
    assert bv.cmd_check() == 1
    app.write_text(good_app, encoding="utf-8")
    assert bv.cmd_check() == 0

    # A hand-edited recipes doc is drift.
    good_doc = recipes_doc.read_text(encoding="utf-8")
    recipes_doc.write_text(good_doc + "hand edit\n", encoding="utf-8",
                           newline="\n")
    assert bv.cmd_check() == 1
    recipes_doc.write_text(good_doc, encoding="utf-8", newline="\n")

    # A per-graph recipe file back in the graph folder fails the check: the
    # folder the template gallery reads holds graphs only.
    stray = vdir / "otr_cloud_low.launch.md"
    stray.write_text(recipe, encoding="utf-8")
    assert bv.cmd_check() == 1
    stray.unlink()
    assert bv.cmd_check() == 0

    # Hand-edit a managed widget on disk -> drift + stamp disagreement.
    tampered = copy.deepcopy(variant)
    wa.patch_widget_by_name(tampered, 87, "fps", 24, schemas)
    (vdir / "otr_cloud_low.json").write_text(bv._dump(tampered),
                                             encoding="utf-8")
    assert bv.cmd_check() == 1

def test_the_committed_variants_match_their_source(capsys):
    """THE ONE ABOVE PROVES THE MECHANISM. THIS PROVES THE SHIPPED TREE.

    `test_check_detects_variant_drift` monkeypatches VARIANTS_DIR to a tmp_path
    and emits a single profile into it, so it verifies that drift detection
    WORKS. Nothing verified that the variants in `workflows/` -- the 24 graphs a user
    actually loads -- still matches the source it is generated from. That was
    checked only when a person remembered to run the CLI by hand.

    It matters more here than a missing test usually does, because section 0's
    hardest rule is that variants are GENERATED and never hand-edited. Without
    this, a hand-edit to a shipped graph survives the entire suite: it is not a
    syntax error, the widget count still matches, the links still resolve, and
    the next person to run `--all` silently reverts the edit or bakes it in.

    Runs the real `cmd_check()` against the real directory -- no monkeypatch,
    which is the entire point.
    """
    rc = bv.cmd_check()
    out = capsys.readouterr().out
    assert rc == 0, (
        "the committed variants in workflows/ no longer match what their "
        "source regenerates. Run:\n"
        "    python scripts/build_variants.py --all\n"
        "and commit the result -- or, if the regeneration is what is wrong, "
        "fix the source rather than the graph.\n\n" + out)

    # NON-VACUITY. Every assertion above is "no failures", which is also what
    # a run that examined nothing reports -- and `cmd_check` returns 0 early
    # and by design when it finds no committed variants at all. The socket
    # audit shipped earlier today passed while reading 1 file of 25 for
    # exactly this reason, so the count is pinned rather than assumed.
    from tests._support.shipped_graphs import variant_paths
    committed = variant_paths()
    assert len(committed) >= 20, (
        "expected the full shipped set, found %d -- this test would be "
        "passing by checking almost nothing" % len(committed))
    assert "no committed variants yet" not in out
