"""GATE B S0/S1 tests -- capability profiles, widget mapping, derived enable-set.

Spec: docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md
(sections 3 + 5) sequenced as GATE B in docs/2026-06-09-3d-toolkit/
3D_TOOLKIT_PLAN.md section 0.

Covers:
  * S0 shape validator: the 3 committed tiers load + validate; unknown keys,
    missing keys and bad enums are rejected fail-closed.
  * S0 mapping: loads + validates; raw node ids banned; companion-trap widget
    names (`seed`/`noise_seed`) banned; every managed target is a REAL widget
    on a REAL node type (verified against INPUT_TYPES); managed node types are
    unique in the master graph.
  * S0 16gb extraction: the 16gb_full profile's values EQUAL the master
    workflow's saved widget values (the identity-bootstrap precondition).
  * S1 declarations: every registered engine has a CAPABILITIES row and vice
    versa, in all three namespaces; rows validate.
  * S1 enable-set: derived availability with reason codes; every profile
    override is in enabled(P) for its namespace (cross-validation), for all
    three tiers; the two-heavy-roles regression (NO static co-residency
    rejection).

UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from nodes._otr_shared import capability_profiles as cp

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MASTER = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"
TIERS = ("16gb_full", "8gb_lite", "cpu_floor")


# ---------------------------------------------------------------------------
# registries (adapters self-register on import)
# ---------------------------------------------------------------------------
def _video_registry():
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_video_engines import (  # noqa: F401  (register adapters)
        cheap_families, eng_character_3d, eng_humo,
        eng_ltx_video, eng_wan_i2v, eng_wan_ti2v,
    )
    return vreg


def _audio_registry():
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_audio_engines import (  # noqa: F401  (register adapters)
        eng_bark, eng_chatterbox, eng_dia, eng_indextts2, eng_kokoro,
        eng_musicgen, eng_stable_audio, eng_stable_audio_3,
    )
    return areg


def _image_registry():
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_image_engines import (  # noqa: F401  (register adapters)
        flux2_klein, flux_gen1, hidream_i1, lumina_image,
        qwen_image, sd35_large, z_image_turbo,
    )
    return ireg


def _declarations_by_registry():
    return {
        "video": _video_registry().CAPABILITIES,
        "audio": _audio_registry().CAPABILITIES,
        "image": _image_registry().CAPABILITIES,
    }


# ---------------------------------------------------------------------------
# S0 -- profile shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tier", TIERS)
def test_committed_tier_loads_and_validates(tier):
    profile = cp.load_profile(tier)
    assert profile["id"] == tier


def test_unknown_top_level_key_rejected():
    profile = cp.load_profile("16gb_full")
    bad = copy.deepcopy(profile)
    bad["surprise_knob"] = 1
    with pytest.raises(cp.ProfileError, match="unknown top-level key"):
        cp.validate_profile_shape(bad)


def test_missing_required_key_rejected():
    bad = copy.deepcopy(cp.load_profile("16gb_full"))
    del bad["toolchains"]
    with pytest.raises(cp.ProfileError, match="missing required key"):
        cp.validate_profile_shape(bad)


@pytest.mark.parametrize("key,value,match", [
    ("device_backend", "mps", "device_backend"),       # mps parked in v1
    ("platform", "linux", "platform"),
    ("status", "experimental", "status"),
])
def test_bad_enum_values_rejected(key, value, match):
    bad = copy.deepcopy(cp.load_profile("16gb_full"))
    bad[key] = value
    with pytest.raises(cp.ProfileError, match=match):
        cp.validate_profile_shape(bad)


def test_unknown_seed_policy_key_rejected():
    bad = copy.deepcopy(cp.load_profile("16gb_full"))
    bad["seed_policy"]["seed"] = 7  # the companion-trap name does not belong here
    with pytest.raises(cp.ProfileError, match="seed_policy"):
        cp.validate_profile_shape(bad)


def test_id_filename_agreement_enforced(tmp_path):
    profile = copy.deepcopy(cp.load_profile("16gb_full"))
    profile["id"] = "16gb_full"
    p = tmp_path / "wrong_name.json"
    p.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(cp.ProfileError, match="filename and id"):
        cp.load_profile("wrong_name", profile_dir=str(tmp_path))


def test_unknown_profile_id_names_known_profiles():
    with pytest.raises(cp.ProfileError, match="16gb_full"):
        cp.load_profile("no_such_tier")


# ---------------------------------------------------------------------------
# S0 -- widget mapping
# ---------------------------------------------------------------------------
def test_mapping_loads_and_validates():
    mapping = cp.load_widget_mapping()
    assert "managed" in mapping and mapping["managed"]


def test_mapping_raw_node_id_banned():
    mapping = copy.deepcopy(cp.load_widget_mapping())
    mapping["managed"]["role_overrides.announcer_visual"]["targets"] = [["87", "announcer_video_model"]]
    with pytest.raises(cp.ProfileError, match="raw node id"):
        cp.validate_widget_mapping_shape(mapping)


def test_mapping_companion_trap_widget_banned():
    mapping = copy.deepcopy(cp.load_widget_mapping())
    mapping["managed"]["seed_policy.request_seed"]["targets"] = [["OTR_VideoDirector", "seed"]]
    with pytest.raises(cp.ProfileError, match="forbidden widget"):
        cp.validate_widget_mapping_shape(mapping)


def test_managed_node_types_unique_in_master():
    """Each mapped node TYPE appears exactly once in the master graph --
    the precondition for find-by-type patching (raw ids stay banned)."""
    mapping = cp.load_widget_mapping()
    wf = json.loads(MASTER.read_text(encoding="utf-8"))
    type_counts: dict = {}
    for node in wf["nodes"]:
        type_counts[node["type"]] = type_counts.get(node["type"], 0) + 1
    for key, entry in mapping["managed"].items():
        for node_type, _widget in entry["targets"]:
            assert type_counts.get(node_type, 0) == 1, (
                f"{key}: node type {node_type} occurs "
                f"{type_counts.get(node_type, 0)}x in master (must be exactly 1)"
            )


def test_16gb_profile_extracted_from_master_values():
    """BOOTSTRAP precondition for the S2 identity gate: the 16gb profile IS the
    master's current values, so apply_profile(master, 16gb) is a no-op."""
    mapping = cp.load_widget_mapping()
    profile = cp.load_profile("16gb_full")
    wf = json.loads(MASTER.read_text(encoding="utf-8"))
    nodes_by_type = {n["type"]: n for n in wf["nodes"]}

    # Widget slot layout per node type, from the REAL saved arrays. We use the
    # probe-verified slot orders (these are pinned independently in
    # tests/test_workflow_apply.py against INPUT_TYPES).
    flat = {}
    for section in ("role_overrides", "slot_overrides", "features"):
        for k, v in profile[section].items():
            flat[f"{section}.{k}"] = v
    flat["seed_policy.request_seed"] = profile["seed_policy"]["request_seed"]
    flat["seed_policy.seed_mode"] = profile["seed_policy"]["seed_mode"]

    from nodes._otr_workflow_apply import build_offline_schemas, serialized_slot_names
    schemas = build_offline_schemas()
    for dotted, value in flat.items():
        entry = mapping["managed"][dotted]
        for node_type, widget in entry["targets"]:
            node = nodes_by_type[node_type]
            slots = serialized_slot_names(node_type, schemas)
            idx = slots.index(widget)
            assert node["widgets_values"][idx] == value, (
                f"{dotted}: master {node_type}.{widget} = "
                f"{node['widgets_values'][idx]!r} but 16gb_full says {value!r}"
            )


# ---------------------------------------------------------------------------
# S1 -- declarations + enable-set
# ---------------------------------------------------------------------------
def test_every_registered_engine_has_a_declaration_and_vice_versa():
    vreg = _video_registry()
    areg = _audio_registry()
    ireg = _image_registry()
    assert set(vreg.CAPABILITIES) == set(vreg.all_engine_names())
    assert set(areg.CAPABILITIES) == set(areg._REGISTRY)
    assert set(ireg.CAPABILITIES) == set(ireg.all_engine_names())


def test_all_declarations_validate():
    for ns, decls in _declarations_by_registry().items():
        for name, decl in decls.items():
            cp.validate_declaration(name, decl, source=ns)


def test_declaration_unknown_key_rejected():
    with pytest.raises(cp.ProfileError, match="unknown key"):
        cp.validate_declaration("x", {
            "required_toolchain": None,
            "requires_sidecar": False, "cpu_ok": True, "model_requirements": [],
            "speed": "fast",
        })


def test_availability_reason_codes():
    # Post-VRAM-rip: fit is CUDA / toolchain / sidecar ONLY -- no VRAM tier or
    # budget gating (the operator's tier JSON owns the OOM budget now).
    decls = {
        "gpu_heavy": {"required_toolchain": None, "requires_sidecar": False,
                      "cpu_ok": False, "model_requirements": []},
        "side": {"required_toolchain": None, "requires_sidecar": True,
                 "cpu_ok": False, "model_requirements": []},
        "compiled": {"required_toolchain": "cu128_toolkit", "requires_sidecar": False,
                     "cpu_ok": False, "model_requirements": []},
        "procgen": {"required_toolchain": None, "requires_sidecar": False,
                    "cpu_ok": True, "model_requirements": []},
    }
    lite = cp.load_profile("8gb_lite")
    avail = cp.availability(lite, decls)
    assert avail["gpu_heavy"] == cp.REASON_OK          # no VRAM cap -> fits
    assert avail["side"] == cp.REASON_SIDECARS_DISABLED
    assert avail["compiled"] == cp.REASON_MISSING_TOOLCHAIN
    assert avail["procgen"] == cp.REASON_OK

    floor = cp.load_profile("cpu_floor")
    avail = cp.availability(floor, decls)
    assert avail["gpu_heavy"] == cp.REASON_REQUIRES_CUDA
    assert avail["procgen"] == cp.REASON_OK


@pytest.mark.parametrize("tier", TIERS)
def test_cross_validation_green_for_committed_tiers(tier):
    profile = cp.load_profile(tier)
    mapping = cp.load_widget_mapping()
    cp.cross_validate_profile(profile, mapping, _declarations_by_registry())


def test_cross_validation_rejects_disabled_engine():
    profile = copy.deepcopy(cp.load_profile("cpu_floor"))
    profile["role_overrides"]["character_visual"] = "humo"  # GPU-only on a CPU floor
    with pytest.raises(cp.ProfileError, match="requires_cuda"):
        cp.cross_validate_profile(profile, cp.load_widget_mapping(), _declarations_by_registry())


def test_cross_validation_rejects_typoed_override_key():
    profile = copy.deepcopy(cp.load_profile("16gb_full"))
    profile["role_overrides"]["announcer_visualz"] = "ltx_video"
    with pytest.raises(cp.ProfileError, match="no widget-mapping entry"):
        cp.cross_validate_profile(profile, cp.load_widget_mapping(), _declarations_by_registry())


# ---------------------------------------------------------------------------
# S1 -- the cold-import gate (the dynamic VRAM ceiling was ripped 2026-07-03:
# the operator's tier JSON owns the OOM budget now, so there is no env ceiling)
# ---------------------------------------------------------------------------
def test_cold_import_gate_profile_layer_pulls_no_heavy_libs():
    """S1 BLOCKING cold-import gate: the capability layer + the VIDEO/IMAGE
    registry TABLE modules import with NO torch/transformers/diffusers/comfy
    (V-12). The AUDIO registry is deliberately EXCLUDED: its package __init__
    hard-imports torch by design (the frozen audio lane) -- that is the
    documented exception, not a drift."""
    import subprocess
    import sys as _sys
    code = (
        "import sys;"
        "import nodes._otr_shared.capability_profiles;"
        "import nodes._otr_shared.engine_registry_base;"
        "import nodes._otr_video_engines.registry;"
        "import nodes._otr_image_engines.registry;"
        "heavy=[m for m in ('torch','transformers','diffusers','comfy')"
        " if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([_sys.executable, "-c", code],
                       cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, f"cold-import gate FAILED: {r.stdout} {r.stderr}"


def test_two_heavy_roles_still_validate():
    """Regression pinned by the spec: per-engine fit ONLY -- a profile with two
    heavy roles yields a valid enable-set (single-heavy residency is
    wrapper_bridge's RUNTIME invariant, never a static profile rejection)."""
    profile = cp.load_profile("16gb_full")
    # 2026-06-28 (operator): the 16gb master defaults announcer/music video to
    # viz_green (the working/clean default -- no GPU, no FLUX since viz_green
    # ignores init_image, no gated HuMo) and pins CHARACTER to humo_14B_169
    # (2026-07-03: character is a first-class slot). humo_1.7B stays registered.
    assert profile["role_overrides"]["announcer_visual"] == "viz_green"
    assert profile["role_overrides"]["character_visual"] == "humo_14B_169"
    # Force TWO heavy roles to keep the static two-heavy regression green (single-
    # heavy residency is wrapper_bridge's RUNTIME invariant, never a static profile
    # rejection).
    profile["role_overrides"]["announcer_visual"] = "humo_14B_169"        # force heavy
    profile["role_overrides"]["music_visual"] = "humo"                   # force heavy
    decls = _declarations_by_registry()
    enabled = cp.enabled_engines(profile, decls["video"])
    assert "ltx_audio_in" in enabled and "humo" in enabled
    assert "humo_1.7B" in enabled       # the new default stays in the enable-set
    cp.cross_validate_profile(profile, cp.load_widget_mapping(), decls)
