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
  * S0 profile separation: the lean canonical workflow is allowed to differ
    from the heavier 16gb_full profile; profile application is explicit.
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
MASTER = REPO_ROOT / "workflows" / "otr_canonical.json"
TIERS = ("16gb_full", "8gb_lite", "cpu_floor")
GOOGLE_MEDIA_PROFILES = (
    "google_veo_media",
    "google_omni_media",
    "google_veo_all",
    "google_omni_all",
)


# ---------------------------------------------------------------------------
# registries (adapters self-register on import)
# ---------------------------------------------------------------------------
def _video_registry():
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_video_engines import (  # noqa: F401  (register adapters)
        cheap_families, eng_humo,
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
        sd35_large, z_image_turbo,
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


@pytest.mark.parametrize("profile_id", GOOGLE_MEDIA_PROFILES)
def test_google_media_profile_loads_and_validates(profile_id):
    profile = cp.load_profile(profile_id)
    assert profile["id"] == profile_id


# ---------------------------------------------------------------------------
# S2 -- schema v2 sections (platform-portability, 2026-07-10)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("profile_id", TIERS + GOOGLE_MEDIA_PROFILES + ("otr_cloud_lanes",))
def test_v2_sections_present_on_every_committed_profile(profile_id):
    prof = cp.load_profile(profile_id)
    for section in ("llm", "video", "image", "audio", "render", "preflight"):
        assert isinstance(prof[section], dict), section
    assert prof["gpu_vendor"] in ("nvidia", "amd", "apple", "none")
    assert isinstance(prof["launch"]["env"], dict)


@pytest.mark.parametrize("section,key,value,match", [
    ("llm", "device", "tpu", "llm section invalid"),
    ("llm", "quant_policy", "gptq", "llm section invalid"),
    ("llm", "gguf_quant", "Q5_K", "llm section invalid"),
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
    bad = copy.deepcopy(cp.load_profile("16gb_full"))
    bad[section][key] = value
    with pytest.raises(cp.ProfileError, match=match):
        cp.validate_profile_shape(bad)


def test_v2_unknown_llm_key_rejected():
    bad = copy.deepcopy(cp.load_profile("16gb_full"))
    bad["llm"]["surprise_lane"] = "x"
    with pytest.raises(cp.ProfileError, match="unknown llm key"):
        cp.validate_profile_shape(bad)


def test_v2_missing_section_rejected():
    bad = copy.deepcopy(cp.load_profile("16gb_full"))
    del bad["render"]
    with pytest.raises(cp.ProfileError, match="missing required key"):
        cp.validate_profile_shape(bad)


def test_v2_llm_section_matches_runtime_policy_enums():
    """ONE enum truth: the llm section validates by CONSTRUCTING an
    LLMRuntimePolicy, so profile enums can never drift from runtime."""
    from nodes._otr_shared.llm_policy import LLMRuntimePolicy

    prof = cp.load_profile("cpu_floor")
    pol = LLMRuntimePolicy(
        device=prof["llm"]["device"],
        attn_impl=prof["llm"]["attn_impl"],
        quant_policy=prof["llm"]["quant_policy"],
        vram_ceiling_gb=prof["llm"]["vram_ceiling_gb"],
        gguf_n_ctx=prof["llm"]["gguf_n_ctx"],
        gguf_quant=prof["llm"]["gguf_quant"],
        lane_allowlist=tuple(prof["llm"]["lane_allowlist"]),
    )
    assert pol.device == "cpu"
    assert pol.vram_ceiling_gb == 0
    assert "transformers" not in pol.lane_allowlist


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
    """dia / indextts2 / chatterbox (cu128 sidecars) and ltx_audio_in
    (NVML gate) are vendor-locked nvidia: an AMD cuda profile must see
    REASON_REQUIRES_VENDOR, an nvidia one REASON_OK-or-sidecar-gated."""
    amd = copy.deepcopy(cp.load_profile("16gb_full"))
    amd["gpu_vendor"] = "amd"
    decls = _declarations_by_registry()
    audio_avail = cp.availability(amd, decls["audio"])
    for eng in ("dia", "indextts2", "chatterbox"):
        assert audio_avail[eng] == cp.REASON_REQUIRES_VENDOR, eng
    video_avail = cp.availability(amd, decls["video"])
    assert video_avail["ltx_audio_in"] == cp.REASON_REQUIRES_VENDOR

    nv = cp.load_profile("16gb_full")
    nv_audio = cp.availability(nv, decls["audio"])
    assert nv_audio["indextts2"] == cp.REASON_OK
    assert cp.availability(nv, decls["video"])["ltx_audio_in"] == cp.REASON_OK


def test_v2_bark_excluded_from_cpu_but_fine_on_cuda():
    decls = _declarations_by_registry()["audio"]
    floor = cp.load_profile("cpu_floor")
    assert cp.availability(floor, decls)["bark"] == cp.REASON_REQUIRES_CUDA
    nv = cp.load_profile("16gb_full")
    assert cp.availability(nv, decls)["bark"] == cp.REASON_OK


def test_v2_stable_audio_3_lists_mps_but_not_cpu():
    """The mac_mps tier ships stable_audio_3 as its music engine (Comfy
    core owns its device layer); the cpu floor keeps musicgen."""
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
    prof = copy.deepcopy(cp.load_profile("16gb_full"))
    prof[key] = value
    assert cp.validate_profile_shape(prof) is prof


@pytest.mark.parametrize("key,value,match", [
    ("device_backend", "rocm", "device_backend"),  # ROCm presents as cuda
    ("platform", "bsd", "platform"),
    ("status", "experimental", "status"),
    ("gpu_vendor", "intel", "gpu_vendor"),
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


def test_16gb_profile_is_separate_from_lean_canonical_values():
    """The saved workflow is the quick 30-word canonical. 16gb_full remains a
    named heavier profile and must not be assumed to equal the saved canvas."""
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
    differences = []
    for dotted, value in flat.items():
        entry = mapping["managed"][dotted]
        for node_type, widget in entry["targets"]:
            node = nodes_by_type[node_type]
            slots = serialized_slot_names(node_type, schemas)
            idx = slots.index(widget)
            if node["widgets_values"][idx] != value:
                differences.append((dotted, node_type, widget))
    assert differences, "16gb_full unexpectedly matches the lean canonical"


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


@pytest.mark.parametrize("profile_id", GOOGLE_MEDIA_PROFILES)
def test_cross_validation_green_for_google_media_profiles(profile_id):
    profile = cp.load_profile(profile_id)
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
    # S6 ratification (2026-07-10): the nv50 profile is REGENERATED from
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
    assert "ltx_audio_in" in enabled and "humo" in enabled
    assert "humo_1.7B" in enabled       # legacy selectable engine stays registered
    cp.cross_validate_profile(profile, cp.load_widget_mapping(), decls)


# ---------------------------------------------------------------------------
# A6 follow-up (2026-07-27, post-push QA fan-out): a profile may only select a
# GGUF quant whose artifact is PINNED.
#
# A6 made an unpinned GGUF artifact a refusal at load, which is right -- an
# artifact with no expected size and no sha cannot be shown to be the file it
# claims to be. But two shipped DRAFT profiles selected Q6_K, which has neither
# pin, so their GENERATED variant workflows carried "Q6_K" baked into the
# writer node's widgets_values and would have raised on the first creative call
# with no in-workflow remedy: the widget value is hard-coded and neither
# profile sets the GEMMA4_12B_GGUF_PATH escape hatch.
#
# Q6_K was also the wrong quant for those two tiers on its own arithmetic: at
# roughly 9.1 GiB of weights plus a 2.8 GiB KV cache at n_ctx=4096 it does not
# fit either profile's declared 10.0 / 10.5 GB ceiling, while the Q4_K_M they
# now select does. The refusal surfaced a second, older defect rather than
# creating one -- which is the argument for keeping the refusal and fixing the
# selection, instead of relaxing the gate back to a warning.
# ---------------------------------------------------------------------------

def _pinned_quants():
    from nodes._otr_gguf_backend import GGUF_ARTIFACTS

    return {q for q, (_name, size, sha) in GGUF_ARTIFACTS.items()
            if size is not None and sha is not None}


def test_every_profile_selects_a_pinned_gguf_quant():
    """The selection is what reaches the widget, so it is the selection that
    has to be verifiable -- checking only the artifact table would leave the
    profile free to name a quant the loader will refuse."""
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    pinned = _pinned_quants()
    assert pinned, "no quant is pinned at all -- the check would be vacuous"

    offenders = {}
    for path in sorted((root / "config" / "profiles").glob("*.json")):
        quant = ((json.loads(path.read_text(encoding="utf-8")) or {})
                 .get("llm", {}) or {}).get("gguf_quant")
        if quant is not None and quant not in pinned:
            offenders[path.name] = quant
    assert not offenders, offenders


def test_no_generated_variant_bakes_in_an_unpinned_quant():
    """The profile is the source, but the VARIANT is what an operator loads,
    and the value is positional inside widgets_values -- so prove it there
    too. This is the artifact that would actually have failed."""
    import json
    import pathlib

    from nodes._otr_gguf_backend import GGUF_ARTIFACTS

    root = pathlib.Path(__file__).resolve().parents[1]
    unpinned = {q for q, (_n, size, sha) in GGUF_ARTIFACTS.items()
                if size is None or sha is None}
    assert unpinned, "nothing is unpinned -- this test would prove nothing"

    checked, offenders = 0, {}
    for profile in sorted((root / "config" / "profiles").glob("*.json")):
        variant = root / "workflows" / "variants" / profile.name
        if not variant.exists():
            continue
        raw = variant.read_text(encoding="utf-8")
        checked += 1
        hits = sorted(q for q in unpinned if '"%s"' % q in raw)
        if hits:
            offenders[variant.name] = hits
    assert checked, "found no generated variants -- the sweep is broken"
    assert not offenders, offenders
