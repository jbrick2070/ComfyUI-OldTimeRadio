"""GATE B S2 tests -- the ONE applier + offline schema adapter + coverage.

Spec: docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md
sections 4-6 + 8. All CPU; all against FIXTURE COPIES of the master workflow
(the real file is never mutated -- apply_profile deep-copies by contract).

Covers:
  * OFFLINE adapter parity: serialized_slot_names / ordered_widget_names from
    the promoted package module agree with scripts/otr_api on EVERY OTR node
    type (the offline /object_info adapter replicates _serialized_slot_names
    semantics -- forceInput consumes no slot, seed companions injected).
  * Profile gate: applying a capability profile intentionally retargets the
    lean canonical workflow, while leaving the input workflow untouched.
  * apply_profile is PURE (input workflow untouched) and never writes node-63's
    workflow_json_path (stamping is emit_snapshot's job, S3).
  * Coverage, direction 1: every managed mapping target is a REAL widget slot
    on its node type.
  * Coverage, direction 2: every COMBO widget in the master graph whose choice
    list intersects the registry engine ids is MANAGED (in the mapping) or
    explicitly EXEMPT -- detection by CHOICES, not saved values.
  * patch_creative: whitelisted names pass; managed engine widgets refuse.
  * apply_profile refusals: typo'd profile key; ambiguous node type.

UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

from nodes import _otr_workflow_apply as wa
from nodes._otr_shared import capability_profiles as cp
from nodes._otr_shared.capability_profiles import ProfileError

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MASTER = REPO_ROOT / "workflows" / "otr_canonical.json"

_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import otr_api  # noqa: E402


@pytest.fixture(scope="module")
def schemas():
    s = wa.build_offline_schemas()
    assert s, "offline schemas came back empty -- NODE_CLASS_MAPPINGS import failed"
    return s


@pytest.fixture()
def master_copy():
    """A FIXTURE COPY of the master workflow (the real file is read-only)."""
    return json.loads(MASTER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mapping():
    return cp.load_widget_mapping()


def _registry_engine_ids():
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_video_engines import (  # noqa: F401
        cheap_families, eng_character_3d, eng_cloud_video, eng_humo,
        eng_ltx_video, eng_wan_i2v,
    )
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_audio_engines import (  # noqa: F401
        eng_bark, eng_chatterbox, eng_dia, eng_indextts2, eng_kokoro,
        eng_cloud_elevenlabs, eng_cloud_sonilo, eng_musicgen,
        eng_stable_audio, eng_stable_audio_3,
    )
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_image_engines import (  # noqa: F401
        eng_cloud_image, flux2_klein, flux_gen1, hidream_i1, lumina_image,
        qwen_image, sd35_large, z_image_turbo,
    )
    return (set(vreg.all_engine_names()) | set(areg._REGISTRY)
            | set(ireg.all_engine_names()))


# ---------------------------------------------------------------------------
# Offline adapter parity vs scripts/otr_api
# ---------------------------------------------------------------------------
def test_serialized_slot_names_parity_every_otr_type(schemas):
    for ntype in sorted(schemas):
        assert wa.serialized_slot_names(ntype, schemas) == \
            otr_api._serialized_slot_names(ntype, schemas), ntype


def test_ordered_widget_names_parity_every_otr_type(schemas):
    for ntype in sorted(schemas):
        assert wa.ordered_widget_names(ntype, schemas) == \
            otr_api._ordered_widget_names(ntype, schemas), ntype


def test_api_prompt_parity_on_master(schemas, master_copy):
    """The promoted converter and the script converter agree on the master."""
    assert wa.workflow_to_api_prompt(master_copy, schemas) == \
        otr_api.workflow_to_api_prompt(master_copy, schemas)


# ---------------------------------------------------------------------------
# Profile application gate
# ---------------------------------------------------------------------------
def test_apply_cloud_profile_retargets_canonical(schemas, master_copy):
    applied = wa.apply_profile(master_copy, "cloud_all", schemas=schemas)
    assert wa.workflow_to_api_prompt(master_copy, schemas) != \
        wa.workflow_to_api_prompt(applied, schemas), (
        "cloud_all should be an explicit profile override, not the saved "
        "canonical baseline"
    )


def test_apply_profile_is_pure(schemas, master_copy):
    before = copy.deepcopy(master_copy)
    wa.apply_profile(master_copy, "8gb_lite", schemas=schemas)
    assert master_copy == before, "apply_profile mutated its input workflow"


def test_apply_profile_never_touches_node63_path(schemas, master_copy):
    applied = wa.apply_profile(master_copy, "8gb_lite", schemas=schemas)
    node63 = [n for n in applied["nodes"] if n["type"] == "OTR_WorkflowValidator"]
    orig63 = [n for n in master_copy["nodes"] if n["type"] == "OTR_WorkflowValidator"]
    assert node63 and node63[0]["widgets_values"] == orig63[0]["widgets_values"], (
        "apply_profile must never stamp / touch node-63 (emit_snapshot's job)"
    )


def test_apply_8gb_lite_lands_its_overrides(schemas, master_copy, mapping):
    applied = wa.apply_profile(master_copy, "8gb_lite", schemas=schemas)
    prof = cp.load_profile("8gb_lite")
    nodes_by_type = {n["type"]: n for n in applied["nodes"]}
    for dotted, value in (
        ("role_overrides.announcer_visual", prof["role_overrides"]["announcer_visual"]),
        ("slot_overrides.char_voice_engine", prof["slot_overrides"]["char_voice_engine"]),
        ("seed_policy.request_seed", prof["seed_policy"]["request_seed"]),
    ):
        for node_type, widget in mapping["managed"][dotted]["targets"]:
            slots = wa.serialized_slot_names(node_type, schemas)
            node = nodes_by_type[node_type]
            assert node["widgets_values"][slots.index(widget)] == value, (dotted, node_type)


def _widget_value(nodes_by_type, schemas, node_type, widget):
    slots = wa.serialized_slot_names(node_type, schemas)
    return nodes_by_type[node_type]["widgets_values"][slots.index(widget)]


def test_apply_cloud_all_lands_cloud_only_routes(schemas, master_copy):
    applied = wa.apply_profile(master_copy, "cloud_all", schemas=schemas)
    nodes_by_type = {n["type"]: n for n in applied["nodes"]}

    assert _widget_value(
        nodes_by_type, schemas, "OTR_CastLock", "voice_bank"
    ) == "elevenlabs_cloud"
    assert _widget_value(
        nodes_by_type, schemas, "OTR_CastLock", "cast_voice_policy"
    ) == "auto_registry"
    assert _widget_value(
        nodes_by_type, schemas, "OTR_CastLock", "char_voice_engine"
    ) == "elevenlabs"
    assert _widget_value(
        nodes_by_type, schemas, "OTR_CastLock", "announcer_voice_engine"
    ) == "elevenlabs"
    assert _widget_value(
        nodes_by_type, schemas, "OTR_BatchCharacterVoices", "engine"
    ) == "elevenlabs"
    assert _widget_value(
        nodes_by_type, schemas, "OTR_AnnouncerVoice", "engine"
    ) == "elevenlabs"
    assert _widget_value(
        nodes_by_type, schemas, "OTR_StableAudioTheme", "engine"
    ) == "sonilo"
    assert _widget_value(
        nodes_by_type, schemas, "OTR_VideoRenderBatch", "engine"
    ) == "cloud_wan_i2v_audio"

    for widget in (
        "announcer_video_model", "music_video_model", "character_video_model",
    ):
        assert _widget_value(
            nodes_by_type, schemas, "OTR_VideoDirector", widget
        ) == "cloud_wan_i2v_audio"
    for widget in (
        "announcer_image_model", "music_image_model", "character_image_model",
    ):
        assert _widget_value(
            nodes_by_type, schemas, "OTR_VideoDirector", widget
        ) == "cloud_nano_banana_2"


def test_apply_profile_rejects_typoed_key(schemas, master_copy):
    prof = copy.deepcopy(cp.load_profile("cloud_all"))
    prof["role_overrides"]["announcer_visualz"] = "ltx_video"
    with pytest.raises(ProfileError, match="no widget-mapping entry"):
        wa.apply_profile(master_copy, prof, schemas=schemas)


def test_apply_profile_rejects_ambiguous_node_type(schemas, master_copy):
    dup = copy.deepcopy(master_copy)
    clone = copy.deepcopy(
        [n for n in dup["nodes"] if n["type"] == "OTR_VideoDirector"][0])
    clone["id"] = 9999
    dup["nodes"].append(clone)
    with pytest.raises(ProfileError, match="occurs 2x"):
        wa.apply_profile(dup, "cloud_all", schemas=schemas)


# ---------------------------------------------------------------------------
# Coverage -- closed by test, not by list
# ---------------------------------------------------------------------------
def test_coverage_direction1_every_mapping_target_is_real(schemas, master_copy, mapping):
    """Every profile key in the mapping points at a real widget slot of a real
    node type that exists in the master graph."""
    master_types = {n["type"] for n in master_copy["nodes"]}
    for section in ("managed", "emit_only"):
        for key, entry in mapping[section].items():
            for node_type, widget in entry["targets"]:
                assert node_type in master_types, f"{key}: {node_type} not in master"
                assert node_type in schemas, f"{key}: {node_type} has no schema"
                slots = wa.serialized_slot_names(node_type, schemas)
                assert widget in slots, (
                    f"{key}: {node_type}.{widget} is not a serialized widget slot "
                    f"(slots: {slots})"
                )


def test_coverage_direction2_every_engine_combo_is_managed(schemas, master_copy, mapping):
    """Engine-widget DETECTION: any COMBO widget on a master node type whose
    choice list intersects the registry engine ids must be patched via the
    applier (i.e. appear in the mapping) or be explicitly exempt. Detection is
    by COMBO CHOICES -- never by saved values."""
    engine_ids = _registry_engine_ids()
    managed_targets = set()
    for section in ("managed", "emit_only"):
        for entry in mapping[section].values():
            for node_type, widget in entry["targets"]:
                managed_targets.add((node_type, widget))
    exempt_types = set(mapping["exempt_node_types"])
    # S2 (v2 mapping): fine-grained per-node-type widget-name exemptions.
    # The writer LEFT exempt_node_types; its 20 creative widgets are exempt
    # by NAME and its 8 model widgets are managed via llm.*.
    exempt_widgets = mapping.get("exempt_widget_names", {}) or {}

    missing = []
    for node in master_copy["nodes"]:
        ntype = node["type"]
        if ntype in exempt_types or ntype not in schemas:
            continue
        schema = schemas[ntype].get("input", {}) or {}
        for section in ("required", "optional"):
            for widget, spec in (schema.get(section, {}) or {}).items():
                if widget in (exempt_widgets.get(ntype) or []):
                    continue
                type_def = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else spec
                if not isinstance(type_def, (list, tuple)):
                    continue  # not an inline-choices COMBO
                if not (set(map(str, type_def)) & engine_ids):
                    continue
                if (ntype, widget) not in managed_targets:
                    missing.append((ntype, widget))
    assert not missing, (
        f"engine-valued COMBO widgets not managed by the applier (add to "
        f"config/profiles/widget_mapping.json or exempt them): {missing}"
    )


_ALL_COMMITTED_PROFILES = (
    "16gb_full", "8gb_lite", "cpu_floor", "cloud_all",
    "google_veo_media", "google_omni_media", "google_veo_all",
    "google_omni_all",
)


@pytest.mark.parametrize("profile_id", _ALL_COMMITTED_PROFILES)
def test_apply_every_committed_profile_patches_clean(
        schemas, master_copy, mapping, profile_id):
    """S2: every committed v2 profile applies onto the canonical master
    without an unmapped-key or widget-validation refusal -- proves the
    new llm.* / render.* managed entries end to end (incl. the GGUF
    dropdown label and the openrouter: admit path)."""
    out = wa.apply_profile(master_copy, profile_id, mapping=mapping,
                           schemas=schemas)
    assert out is not master_copy


def test_coverage_every_request_seed_widget_is_managed(schemas, master_copy, mapping):
    """Every widget literally named request_seed in the master graph is under
    seed_policy management; widgets named 'seed' are NEVER managed."""
    seed_targets = {tuple(t) for t in mapping["managed"]["seed_policy.request_seed"]["targets"]}
    for node in master_copy["nodes"]:
        ntype = node["type"]
        if ntype not in schemas:
            continue
        slots = wa.serialized_slot_names(ntype, schemas)
        if "request_seed" in slots:
            assert (ntype, "request_seed") in seed_targets, ntype
    # And no mapping entry anywhere targets a widget literally named 'seed'.
    for section in ("managed", "emit_only"):
        for entry in mapping[section].values():
            for _ntype, widget in entry["targets"]:
                assert widget not in ("seed", "noise_seed")


# ---------------------------------------------------------------------------
# patch_creative whitelist
# ---------------------------------------------------------------------------
def test_patch_creative_allows_whitelisted(schemas, master_copy):
    writer = [n for n in master_copy["nodes"] if n["type"] == "OTR_LedgerScriptWriter"][0]
    wa.patch_creative(master_copy, writer["id"], "target_words", 30, schemas=schemas)
    slots = wa.serialized_slot_names("OTR_LedgerScriptWriter", schemas)
    assert writer["widgets_values"][slots.index("target_words")] == 30


def test_patch_creative_refuses_managed_engine_widget(schemas, master_copy):
    director = [n for n in master_copy["nodes"] if n["type"] == "OTR_VideoDirector"][0]
    with pytest.raises(ProfileError, match="not on the creative whitelist"):
        wa.patch_creative(master_copy, director["id"], "announcer_video_model",
                          "still_motion", schemas=schemas)


# ---------------------------------------------------------------------------
# S2 headless conversion (the drift kill, decision doc section 8)
# ---------------------------------------------------------------------------
def test_creative_whitelist_parity_package_vs_script():
    """scripts/otr_api.CREATIVE_WHITELIST mirrors the package's -- one
    whitelist, two lanes, never diverging."""
    assert otr_api.CREATIVE_WHITELIST == wa.CREATIVE_WHITELIST


def test_script_patch_creative_refuses_managed_names(schemas, master_copy):
    with pytest.raises(ValueError, match="creative whitelist"):
        otr_api.patch_creative(master_copy, 81, "engine", "bark", schemas)


def test_apply_profile_to_workflow_headless_seam(schemas, master_copy, capsys):
    """The script-lane seam drives the SAME applier and prints the resolved
    profile so a headless mutation is visible in logs."""
    applied = otr_api.apply_profile_to_workflow(master_copy, "cloud_all", schemas)
    out = capsys.readouterr().out
    assert "RESOLVED PROFILE cloud_all" in out
    assert wa.workflow_to_api_prompt(applied, schemas) != \
        wa.workflow_to_api_prompt(master_copy, schemas)


def test_scripts_never_direct_patch_managed_widget_names(mapping):
    """REGRESSION (decision doc sec. 8): no tracked top-level script may call
    patch_widget_by_name with a MANAGED widget-name literal -- managed widgets
    reach the graph ONLY via the applier. (Creative/unmanaged names stay
    allowed; the whitelist guards those at runtime.)"""
    import re as _re
    managed_names = set()
    for section in ("managed", "emit_only"):
        for entry in mapping[section].values():
            for _ntype, widget in entry["targets"]:
                managed_names.add(widget)
    scripts_dir = REPO_ROOT / "scripts"
    offenders = []
    for py in scripts_dir.glob("*.py"):
        src = py.read_text(encoding="utf-8", errors="replace")
        for m in _re.finditer(r"patch_widget_by_name\s*\(([^)]*)\)", src):
            call = m.group(1)
            for name in managed_names:
                if _re.search(r"['\"]%s['\"]" % _re.escape(name), call):
                    offenders.append(f"{py.name}: {name}")
    assert not offenders, (
        "managed widgets patched directly in scripts (use "
        "apply_profile_to_workflow / --profile instead): %r" % offenders)
