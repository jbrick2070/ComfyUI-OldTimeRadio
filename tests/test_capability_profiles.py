"""GATE B S0/S1 tests -- capability profiles, widget mapping, derived enable-set.

Spec: docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md
(sections 3 + 5) sequenced as GATE B in docs/2026-06-09-3d-toolkit/
3D_TOOLKIT_PLAN.md section 0.

Covers:
  * S0 shape validator: every `config/workflow_matrix.json` row loads +
    validates; unknown keys, missing keys and bad enums are rejected
    fail-closed.
  * S0 mapping: loads + validates; raw node ids banned; companion-trap widget
    names (`seed`/`noise_seed`) banned; every managed target is a REAL widget
    on a REAL node type (verified against INPUT_TYPES); managed node types are
    unique in the master graph.
  * S0 profile separation: the lean canonical workflow is allowed to differ
    from the heavier otr_16gb_low row; profile application is explicit.
  * S1 declarations: every registered engine has a CAPABILITIES row and vice
    versa, in all three namespaces; rows validate.
  * S1 enable-set: derived availability with reason codes; every profile
    override is in enabled(P) for its namespace (cross-validation), for every
    matrix row; the two-heavy-roles regression (NO static co-residency
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
MASTER = REPO_ROOT / "workflows" / "otr_canonical.json"
#: Every workflow the pack runs -- the matrix is the only source `load_profile` reads.
MATRIX_IDS = cp.known_profile_ids()
#: The rows the host-shape tests stand on. Named, not picked by position, so a
#: matrix reorder never silently moves a test onto a different kind of machine.
NV16 = "otr_16gb_low"       # 16 GB NVIDIA, sidecars allowed
NV8 = "otr_8gb_still"       # 8 GB NVIDIA, sidecars off
CPU_ROW = "otr_cloud_low"   # the cloud rows are the cpu-backend documents


def _a_row_stating(section, key=None):
    """A deep copy of the first matrix row that states `section` (and `key`).

    The rows are delta documents -- each states only what it changes -- so no
    single row carries every section. A test that edits one section takes a real
    row that has it rather than inventing a document.
    """
    for rid in MATRIX_IDS:
        prof = cp.load_profile(rid)
        if section in prof and (key is None or key in prof[section]):
            return copy.deepcopy(prof)
    wanted = f"{section}.{key}" if key else section
    raise AssertionError(f"no workflow_matrix.json row states {wanted!r}")


# ---------------------------------------------------------------------------
# registries (adapters self-register on import)
# ---------------------------------------------------------------------------
def _video_registry():
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_video_engines import (  # noqa: F401  (register adapters)
        cheap_families, eng_humo, eng_ltx25,
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
        flux_gen1, lumina_image, z_image_turbo,
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
@pytest.mark.parametrize("profile_id", MATRIX_IDS)
def test_every_matrix_row_loads_and_validates(profile_id):
    profile = cp.load_profile(profile_id)
    assert profile["id"] == profile_id


# ---------------------------------------------------------------------------
# S2 -- schema v2 sections (platform-portability, 2026-07-10)
# ---------------------------------------------------------------------------
#: The sections every row must carry: each one that holds a KEY INDICATOR (the
#: matrix states those on every row, even where they equal the canonical), plus
#: `launch` and `preflight`, which default from the matrix's own block because the
#: canonical holds nothing for them. `image` and `render` are not in the set: a row
#: omits them to follow the canonical, which the delta relaxation made legal.
_ROW_SECTIONS = tuple(sorted(
    {k.split(".")[0] for k in cp.load_matrix()["key_indicators"]}
    | {"launch", "preflight"}))


@pytest.mark.parametrize("profile_id", MATRIX_IDS)
def test_v2_sections_present_on_every_matrix_row(profile_id):
    prof = cp.load_profile(profile_id)
    assert {"llm", "video", "audio", "preflight"} <= set(_ROW_SECTIONS)
    for section in _ROW_SECTIONS:
        assert isinstance(prof[section], dict), section
    assert prof["gpu_vendor"] in ("nvidia", "amd", "apple", "none")
    assert isinstance(prof["launch"]["env"], dict)


@pytest.mark.parametrize("section,key,value,match", [
    ("llm", "device", "tpu", "llm section invalid"),
    ("llm", "quant_policy", "gptq", "llm section invalid"),
    ("llm", "lane_allowlist", [], "llm section invalid"),
    ("llm", "creative_model", "", "llm.creative_model"),
    ("video", "dtype_policy", "fp4_only", "video.dtype_policy"),
    ("image", "dtype_policy", "anything_goes", "image.dtype_policy"),
    ("audio", "voice_device", "dsp", "audio.voice_device"),
    ("render", "fps", 0, "render.fps"),
    ("render", "composite_res", "1080p", "render.composite_res"),
    ("preflight", "required_keys", [1], "preflight.required_keys"),
])
def test_v2_bad_section_values_rejected(section, key, value, match):
    bad = _a_row_stating(section)
    bad[section][key] = value
    with pytest.raises(cp.ProfileError, match=match):
        cp.validate_profile_shape(bad)


def test_v2_unknown_llm_key_rejected():
    bad = copy.deepcopy(cp.load_profile(NV16))
    bad["llm"]["surprise_lane"] = "x"
    with pytest.raises(cp.ProfileError, match="unknown llm key"):
        cp.validate_profile_shape(bad)


#: `toolchains` was in this list and came out on 2026-09-24. It describes the
#: HOST, so the canonical holds no value for it to fall back to, and `_fit_reason`
#: indexes it directly -- see `test_the_host_identity_keys_stay_required`.
@pytest.mark.parametrize("section", [
    "render", "llm", "seed_policy", "features", "video", "image", "audio",
    "launch", "preflight", "role_overrides", "slot_overrides",
])
def test_an_absent_section_is_legal(section):
    """A CONFIG STATES ONLY WHAT IT CHANGES (2026-09-24).

    This test asserted the opposite until the delta relaxation, and it was
    right to fail: an omitted section now means "take the canonical's value",
    which is what the applier always did -- `_flatten_profile_values` guards
    every key with `if k in`, so a partial document was already safe to apply
    and only the validator forbade writing one.

    Parametrized across every section because the relaxation edited a table.
    Asserting on `render` alone would have passed while `llm` was still
    all-or-nothing, which is precisely what happened mid-change. Each case
    takes the section out of a real matrix row that states it.
    """
    doc = _a_row_stating(section)
    del doc[section]
    assert cp.validate_profile_shape(doc) is doc


@pytest.mark.parametrize("section,key", [
    ("render", "frame_budget"),
    ("llm", "vram_ceiling_gb"),
    ("seed_policy", "request_seed"),
])
def test_a_section_may_state_one_key_and_omit_the_rest(section, key):
    """The delta shape a workflow-matrix row actually writes.

    Deleting a whole section proves the top-level table; this proves the
    per-section one, which is a different code path and the one the llm
    section failed.
    """
    full = _a_row_stating(section, key)
    doc = _probe(**{section: {key: full[section][key]}})
    assert cp.validate_profile_shape(doc) is doc


@pytest.mark.parametrize("section", ["render", "llm", "seed_policy"])
def test_a_typo_inside_a_partial_section_is_still_fatal(section):
    """THE HALF THAT MUST NOT HAVE MOVED.

    Relaxing "every key must be present" must not relax "every key must be
    one we declared". A typo'd key silently doing nothing is the drift class
    this validator exists to kill, and a partial section is exactly where a
    typo would otherwise hide -- there is no longer a missing-key error to
    trip over it.
    """
    doc = _probe(**{section: {"ths_is_not_a_key": 1}})
    with pytest.raises(cp.ProfileError, match="unknown"):
        cp.validate_profile_shape(doc)


def test_v2_llm_section_matches_runtime_policy_enums():
    """ONE enum truth: the llm section validates by CONSTRUCTING an
    LLMRuntimePolicy, so profile enums can never drift from runtime.

    Built from exactly the runtime keys the cloud row states, which is what
    the validator itself does -- an unstated key takes the dataclass default."""
    from nodes._otr_shared.llm_policy import LLMRuntimePolicy

    prof = cp.load_profile(CPU_ROW)
    stated = {k: prof["llm"][k] for k in cp._LLM_RUNTIME_KEYS if k in prof["llm"]}
    pol = LLMRuntimePolicy(**stated)
    assert pol.device == "cpu"
    assert pol.vram_ceiling_gb == 0
    assert pol.quant_policy == "none"
    assert prof["llm"]["creative_model"] == "comfy:slot-a"
    assert prof["llm"]["technical_model"] == "comfy:slot-b"
    assert prof["llm"]["comfy_slot_a_model"] == "anthropic/claude-sonnet-5"
    assert prof["llm"]["comfy_slot_b_model"] == "openai/gpt-5.6-luna"


def test_v2_mapping_declares_exempt_widget_names():
    mapping = cp.load_widget_mapping()
    assert mapping["version"] == 2
    assert mapping["exempt_node_types"] == []
    ewn = mapping["exempt_widget_names"]
    assert "OTR_LedgerScriptWriter" in ewn
    assert "source_bank" in ewn["OTR_LedgerScriptWriter"]
    # The writer's 8 model widgets are MANAGED (not exempt).
    assert "creative_writing_model" not in ewn["OTR_LedgerScriptWriter"]
    assert "llm.creative_model" in mapping["managed"]
    assert "render.frame_budget" in mapping["managed"]


# ---------------------------------------------------------------------------
# S3 -- registry CAPABILITIES v2 (platform-portability, 2026-07-10)
# ---------------------------------------------------------------------------
def test_v1_cpu_ok_row_cannot_survive_v2_semantics():
    """The spec's structural proof: a v1 row (bare cpu_ok bool) is REJECTED
    outright by the v2 declaration schema -- unknown key AND missing v2
    keys. The old false bark row cannot come back."""
    v1_bark = {"required_toolchain": None, "requires_sidecar": False,
               "cpu_ok": True, "model_requirements": ["suno-bark"]}
    with pytest.raises(cp.ProfileError, match="unknown key"):
        cp.validate_declaration("bark", v1_bark)


def test_v2_every_registry_row_validates():
    for namespace, decls in _declarations_by_registry().items():
        for name, decl in decls.items():
            cp.validate_declaration(name, decl, source=namespace)
            assert "cpu_ok" not in decl, (namespace, name)


def test_v2_vendor_pins_gate_amd_hosts():
    """dia / indextts2 / chatterbox (cu128 sidecars) are vendor-locked
    nvidia: an AMD cuda profile must see REASON_REQUIRES_VENDOR, an nvidia
    one REASON_OK-or-sidecar-gated."""
    amd = copy.deepcopy(cp.load_profile(NV16))
    amd["gpu_vendor"] = "amd"
    decls = _declarations_by_registry()
    audio_avail = cp.availability(amd, decls["audio"])
    for eng in ("dia", "indextts2", "chatterbox"):
        assert audio_avail[eng] == cp.REASON_REQUIRES_VENDOR, eng

    nv = cp.load_profile(NV16)
    nv_audio = cp.availability(nv, decls["audio"])
    assert nv_audio["indextts2"] == cp.REASON_OK


def test_v2_bark_excluded_from_cpu_but_fine_on_cuda():
    """2026-08-25: bark's device_backends grew "cpu" once
    _generate_single_line stopped hardcoding CUDA (it now asks the loaded
    model for its real device -- see nodes/_otr_bark_lib.py and
    tests/test_platform_s0_guards.py::test_bark_registry_row_admits_cpu_but_stays_impractical_there).
    A cpu-backend document still excludes it -- a ~1B-parameter three-stage
    autoregressive stack is not a PRACTICAL cpu choice even though it now
    runs there -- so the exclusion reason moved from REASON_REQUIRES_CUDA
    (cuda was genuinely the only backend) to REASON_IMPRACTICAL_ON_CPU (cpu
    works, it is just not a cpu host's job to offer it)."""
    decls = _declarations_by_registry()["audio"]
    floor = cp.load_profile(CPU_ROW)
    assert cp.availability(floor, decls)["bark"] == cp.REASON_IMPRACTICAL_ON_CPU
    nv = cp.load_profile(NV16)
    assert cp.availability(nv, decls)["bark"] == cp.REASON_OK


def test_v2_stable_audio_3_lists_mps_but_not_cpu():
    """The otr_mac16_* rows ship stable_audio_3 as their music engine (Comfy
    core owns its device layer); musicgen stays the cpu-capable one."""
    decls = _declarations_by_registry()["audio"]
    row = decls["stable_audio_3"]
    assert "mps" in row["device_backends"]
    assert "cpu" not in row["device_backends"]
    assert "cpu" in decls["musicgen"]["device_backends"]


def test_v2_humo_fp8_dependency_is_table_visible():
    decls = _declarations_by_registry()["video"]
    assert decls["humo"]["needs_fp8_te"] is True
    assert decls["humo_14B_169"]["needs_fp8_te"] is True
    assert decls["humo_1.7B"]["needs_fp8_te"] is False


#: The keys `validate_profile_shape` requires: an identity plus the four that
#: describe the HOST. A probe document needs them before it can exercise anything
#: else, because they are checked first.
#:
#: They are required because the canonical graph holds NOTHING for them -- unlike
#: every widget-mapped key, there is no value for an omitted one to follow, and
#: seven sites index them without `.get()`, one of them on the render path outside
#: the guard that lets an unreadable profile still run.
_HOST_KEYS = {
    "platform": "any",
    "device_backend": "cuda",
    "toolchains": [],
    "allow_sidecars": False,
}


def _probe(**extra):
    """A minimal VALID document, plus whatever the caller is testing."""
    doc = {"id": "probe"}
    doc.update(_HOST_KEYS)
    doc.update(extra)
    return doc


def test_unknown_top_level_key_rejected():
    profile = cp.load_profile(NV16)
    bad = copy.deepcopy(profile)
    bad["surprise_knob"] = 1
    with pytest.raises(cp.ProfileError, match="unknown top-level key"):
        cp.validate_profile_shape(bad)


@pytest.mark.parametrize("key", ["id", "platform", "device_backend",
                                 "toolchains", "allow_sidecars"])
def test_a_required_key_cannot_be_omitted(key):
    """The complete required set, one key per case so a failure names it."""
    doc = copy.deepcopy(cp.load_profile(NV16))
    del doc[key]
    with pytest.raises(cp.ProfileError, match="missing required key"):
        cp.validate_profile_shape(doc)


def test_the_host_identity_keys_stay_required():
    """THE FOUR KEYS THE DELTA RELAXATION SHOULD NOT HAVE TOUCHED (2026-09-24).

    The relaxation's premise is that an omitted key follows the canonical. That
    holds for every widget-mapped key, because the canonical graph carries a value
    for it. `platform`, `device_backend`, `toolchains` and `allow_sidecars`
    describe the HOST -- the canonical holds nothing for them and nothing
    re-derives them -- so relaxing them handed them a fallback that does not exist.

    It was not theoretical. `availability()` -> `_fit_reason` indexes
    `device_backend`, `toolchains` and `allow_sidecars` directly, and
    `_host_reality_problems` in the validator indexes `device_backend` and
    `platform` on the render path, OUTSIDE the try/except that exists so an
    unreadable profile still lets the workflow run. A document with only an id
    validated cleanly and then raised a bare KeyError mid-render.
    """
    with pytest.raises(cp.ProfileError, match="missing required key"):
        cp.validate_profile_shape({"id": "probe"})
    assert cp.validate_profile_shape(_probe()) == _probe()


def test_availability_refuses_an_incomplete_profile_rather_than_raising_keyerror():
    """A caller handing `availability()` a raw dict gets a named refusal.

    Not a `KeyError` out of a private helper, and not a silent "does not fit",
    which is what a `.get()` default inside `_fit_reason` would have produced.
    """
    from nodes._otr_audio_engines.registry import CAPABILITIES
    with pytest.raises(cp.ProfileError, match="missing"):
        cp.availability({"id": "probe"}, CAPABILITIES)


# S2 platform-portability (2026-07-10): mps + linux are FIRST-CLASS enum
# values now (schema v2). The rejection list keeps genuinely-invalid values.
@pytest.mark.parametrize("key,value", [
    ("device_backend", "mps"),
    ("device_backend", "cpu"),
    ("platform", "linux"),
    ("platform", "mac"),
    ("gpu_vendor", "amd"),
    ("gpu_vendor", "apple"),
])
def test_v2_enum_values_accepted(key, value):
    prof = copy.deepcopy(cp.load_profile(NV16))
    prof[key] = value
    assert cp.validate_profile_shape(prof) is prof


@pytest.mark.parametrize("key,value,match", [
    ("device_backend", "rocm", "device_backend"),  # ROCm presents as cuda
    ("platform", "bsd", "platform"),
    ("status", "experimental", "status"),
    ("gpu_vendor", "intel", "gpu_vendor"),
])
def test_bad_enum_values_rejected(key, value, match):
    bad = copy.deepcopy(cp.load_profile(NV16))
    bad[key] = value
    with pytest.raises(cp.ProfileError, match=match):
        cp.validate_profile_shape(bad)


def test_unknown_seed_policy_key_rejected():
    bad = _a_row_stating("seed_policy")
    bad["seed_policy"]["seed"] = 7  # the companion-trap name does not belong here
    with pytest.raises(cp.ProfileError, match="seed_policy"):
        cp.validate_profile_shape(bad)


def test_id_filename_agreement_enforced(tmp_path):
    profile = copy.deepcopy(cp.load_profile(NV16))
    assert profile["id"] == NV16
    p = tmp_path / "wrong_name.json"
    p.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(cp.ProfileError, match="filename and id"):
        cp.load_profile("wrong_name", profile_dir=str(tmp_path))


def test_unknown_profile_id_names_known_profiles():
    with pytest.raises(cp.ProfileError, match=NV16):
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


def test_16gb_profile_is_separate_from_lean_canonical_values():
    """The saved workflow is the quick 30-word canonical. otr_16gb_low is a
    named heavier row and must not be assumed to equal the saved canvas."""
    mapping = cp.load_widget_mapping()
    profile = cp.load_profile(NV16)
    wf = json.loads(MASTER.read_text(encoding="utf-8"))
    nodes_by_type = {n["type"]: n for n in wf["nodes"]}

    # Widget slot layout per node type, from the REAL saved arrays. We use the
    # probe-verified slot orders (these are pinned independently in
    # tests/test_workflow_apply.py against INPUT_TYPES). A matrix row is a
    # delta, so only the sections and keys it states are compared.
    flat = {}
    for section in ("role_overrides", "slot_overrides", "features", "seed_policy"):
        for k, v in profile.get(section, {}).items():
            flat[f"{section}.{k}"] = v
    assert flat, "otr_16gb_low states no widget-mapped value at all"

    from nodes._otr_workflow_apply import build_offline_schemas, serialized_slot_names
    schemas = build_offline_schemas()
    differences = []
    for dotted, value in flat.items():
        entry = mapping["managed"][dotted]
        for node_type, widget in entry["targets"]:
            node = nodes_by_type[node_type]
            slots = serialized_slot_names(node_type, schemas)
            idx = slots.index(widget)
            if node["widgets_values"][idx] != value:
                differences.append((dotted, node_type, widget))
    assert differences, "otr_16gb_low unexpectedly matches the lean canonical"


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
            "requires_sidecar": False,
            "device_backends": ["cuda", "cpu", "mps"],
            "requires_vendor": None,
            "needs_fp8_te": False,
            "needs_fp4_te": False,
            "practical_without_gpu": True,
            "sidecar_conditional": False,
            "model_requirements": [],
            "speed": "fast",
        })


def test_availability_reason_codes():
    # Post-VRAM-rip: fit is CUDA / toolchain / sidecar ONLY -- no VRAM tier or
    # budget gating (the operator's tier JSON owns the OOM budget now).
    decls = {
        "gpu_heavy": {"required_toolchain": None, "requires_sidecar": False,
                      "device_backends": ["cuda"], "requires_vendor": None,
                      "needs_fp8_te": False, "needs_fp4_te": False,
                      "practical_without_gpu": False, "sidecar_conditional": False,
                      "model_requirements": []},
        "side": {"required_toolchain": None, "requires_sidecar": True,
                 "device_backends": ["cuda"], "requires_vendor": None,
                 "needs_fp8_te": False, "needs_fp4_te": False,
                 "practical_without_gpu": False, "sidecar_conditional": False,
                 "model_requirements": []},
        "compiled": {"required_toolchain": "cu128_toolkit", "requires_sidecar": False,
                     "device_backends": ["cuda"], "requires_vendor": None,
                     "needs_fp8_te": False, "needs_fp4_te": False,
                     "practical_without_gpu": False, "sidecar_conditional": False,
                     "model_requirements": []},
        "procgen": {"required_toolchain": None, "requires_sidecar": False,
                    "device_backends": ["cuda", "cpu", "mps"], "requires_vendor": None,
                    "needs_fp8_te": False, "needs_fp4_te": False,
                    "practical_without_gpu": True, "sidecar_conditional": False,
                    "model_requirements": []},
    }
    lite = cp.load_profile(NV8)
    avail = cp.availability(lite, decls)
    assert avail["gpu_heavy"] == cp.REASON_OK          # no VRAM cap -> fits
    assert avail["side"] == cp.REASON_SIDECARS_DISABLED
    assert avail["compiled"] == cp.REASON_MISSING_TOOLCHAIN
    assert avail["procgen"] == cp.REASON_OK

    floor = cp.load_profile(CPU_ROW)
    avail = cp.availability(floor, decls)
    assert avail["gpu_heavy"] == cp.REASON_REQUIRES_CUDA
    assert avail["procgen"] == cp.REASON_OK


@pytest.mark.parametrize("profile_id", MATRIX_IDS)
def test_cross_validation_green_for_every_matrix_row(profile_id):
    profile = cp.load_profile(profile_id)
    mapping = cp.load_widget_mapping()
    cp.cross_validate_profile(profile, mapping, _declarations_by_registry())


def test_cross_validation_rejects_disabled_engine():
    profile = copy.deepcopy(cp.load_profile(CPU_ROW))
    profile["role_overrides"]["character_visual"] = "humo"  # GPU-only on a cpu host
    with pytest.raises(cp.ProfileError, match="requires_cuda"):
        cp.cross_validate_profile(profile, cp.load_widget_mapping(), _declarations_by_registry())


def test_cross_validation_rejects_typoed_override_key():
    profile = copy.deepcopy(cp.load_profile(NV16))
    profile["role_overrides"]["announcer_visualz"] = "ltx25_video"
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
    profile = cp.load_profile(NV16)
    # S6 ratification (2026-07-10): the 16 GB row is REGENERATED from
    # canonical (viz lanes + z_image_turbo) -- the old humo/flux pins moved
    # out. This test's subject is the two-heavy STATIC rule, so it forces
    # heavy roles onto a copy below regardless of the committed defaults.
    assert profile["role_overrides"]["announcer_visual"] == "viz_mxc_cpu"
    assert profile["role_overrides"]["character_visual"] == "viz_camera"
    # Force TWO heavy roles to keep the static two-heavy regression green (single-
    # heavy residency is wrapper_bridge's RUNTIME invariant, never a static profile
    # rejection).
    profile["role_overrides"]["announcer_visual"] = "humo_14B_169"        # force heavy
    profile["role_overrides"]["music_visual"] = "humo"                   # force heavy
    decls = _declarations_by_registry()
    enabled = cp.enabled_engines(profile, decls["video"])
    assert "ltx25_native_audio_in_16gb" in enabled and "humo" in enabled
    assert "humo_1.7B" in enabled       # legacy selectable engine stays registered
    cp.cross_validate_profile(profile, cp.load_widget_mapping(), decls)
