"""Capability profiles -- GATE B S0/S1 of the switchable-workflow architecture.

Spec: docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md
(sections 3 + 5); sequencing: docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md
section 0 (GATE B).

S0 -- profile FOUNDATION:
  * the committed profile shape (``config/profiles/<id>.json``) -- capability
    POLICY, not creative presets; OVERRIDES only, registry defaults supply the
    base;
  * a fail-closed SHAPE validator (unknown keys rejected, enums enforced);
  * the checked-in widget MAPPING (``config/profiles/widget_mapping.json``)
    loader -- profile key -> ``(node_type, widget_name)`` targets; raw node
    ids are banned by construction.

S1 -- DERIVED ENABLE-SET, never hand-listed:
  * per-engine capability DECLARATIONS live in the registry table modules
    (``nodes/_otr_video_engines/registry.py`` etc., ``CAPABILITIES`` dict --
    NOT in adapter modules);
  * ``availability(profile, declarations)`` -> the shared availability object
    with one reason code per engine (reused by validator / wizard / logs);
  * ``enabled_engines`` = engines whose declarations fit the profile;
  * ``cross_validate_profile`` -- every profile override must be in the
    enable-set of its namespace (per-engine fit ONLY; NO static co-residency
    rejection -- residency is wrapper_bridge's runtime invariant).

Dependency-free: stdlib ``json``/``os``/``typing`` only. Importing this module
pulls in no torch / comfy / model framework (V-12 cold-import clean).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

__all__ = [
    "ProfileError",
    "PROFILE_DIR",
    "load_profile",
    "load_widget_mapping",
    "validate_profile_shape",
    "validate_widget_mapping_shape",
    "availability",
    "enabled_engines",
    "cross_validate_profile",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILE_DIR = os.path.join(_REPO_ROOT, "config", "profiles")
WIDGET_MAPPING_PATH = os.path.join(PROFILE_DIR, "widget_mapping.json")


class ProfileError(ValueError):
    """A profile / mapping file failed validation. FAIL CLOSED -- the caller
    must not proceed with a half-understood capability policy."""


# ---------------------------------------------------------------------------
# S0 -- profile shape (schema v2: platform-portability S2, 2026-07-10 --
# docs/2026-07-09-platform-portability-final.md section 2)
# ---------------------------------------------------------------------------
_PLATFORMS = ("any", "win", "mac", "linux")
_DEVICE_BACKENDS = ("cuda", "cpu", "mps")  # ROCm presents as "cuda"
_STATUSES = ("shipping", "draft")
_GPU_VENDORS = ("nvidia", "amd", "apple", "none")
_DTYPE_POLICIES = ("fp8_ok", "no_fp8", "no_fp8_no_fp4")
_DEVICE_POLICIES = ("cuda", "cpu", "mps")


def _is_positive_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _is_res_string(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    parts = v.split("x")
    return len(parts) == 2 and all(p.isdigit() and int(p) > 0 for p in parts)


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(s, str) and s for s in v)


# key -> (required, validator-callable, human description)
_TOP_LEVEL_KEYS: dict[str, tuple[bool, Any, str]] = {
    "id": (True, lambda v: isinstance(v, str) and bool(v), "non-empty str"),
    "display_name": (True, lambda v: isinstance(v, str) and bool(v), "non-empty str"),
    "status": (True, lambda v: v in _STATUSES, f"one of {_STATUSES}"),
    "platform": (True, lambda v: v in _PLATFORMS, f"one of {_PLATFORMS}"),
    "device_backend": (True, lambda v: v in _DEVICE_BACKENDS, f"one of {_DEVICE_BACKENDS}"),
    "gpu_vendor": (True, lambda v: v in _GPU_VENDORS, f"one of {_GPU_VENDORS}"),
    "toolchains": (True, lambda v: isinstance(v, list) and all(isinstance(t, str) for t in v), "list[str]"),
    "allow_sidecars": (True, lambda v: isinstance(v, bool), "bool"),
    "role_overrides": (True, lambda v: _is_str_dict(v), "dict[str, str]"),
    "slot_overrides": (True, lambda v: _is_str_dict(v), "dict[str, str]"),
    "features": (True, lambda v: isinstance(v, dict), "dict"),
    "seed_policy": (True, lambda v: isinstance(v, dict), "dict"),
    "launch": (True, lambda v: isinstance(v, dict), "dict"),
    "llm": (True, lambda v: isinstance(v, dict), "dict"),
    "video": (True, lambda v: isinstance(v, dict), "dict"),
    "image": (True, lambda v: isinstance(v, dict), "dict"),
    "audio": (True, lambda v: isinstance(v, dict), "dict"),
    "render": (True, lambda v: isinstance(v, dict), "dict"),
    "preflight": (True, lambda v: isinstance(v, dict), "dict"),
}

_SEED_POLICY_KEYS = {
    "request_seed": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "seed_mode": lambda v: isinstance(v, str) and bool(v),
    "cast_seed_env": lambda v: v is None or (isinstance(v, str) and bool(v)),
    "style_seed_env": lambda v: v is None or (isinstance(v, str) and bool(v)),
}

_LAUNCH_KEYS = {
    "sage_attention": lambda v: isinstance(v, bool),
    "extra_args": lambda v: isinstance(v, list) and all(isinstance(a, str) for a in v),
    # v2: launch-only env vars for the generated recipe (names -> values;
    # secrets NEVER go here -- key NAMES belong in preflight.required_keys).
    "env": lambda v: isinstance(v, dict) and all(
        isinstance(k, str) and k and isinstance(val, str) for k, val in v.items()),
}

# --- v2 sections -------------------------------------------------------
_VIDEO_KEYS = {
    "device_policy": lambda v: v in _DEVICE_POLICIES,
    "dtype_policy": lambda v: v in _DTYPE_POLICIES,
}
_IMAGE_KEYS = {
    "dtype_policy": lambda v: v in _DTYPE_POLICIES,
}
_AUDIO_KEYS = {
    "voice_device": lambda v: v in _DEVICE_POLICIES,
}
_RENDER_KEYS = {
    "fps": _is_positive_int,
    "canvas_w": _is_positive_int,
    "canvas_h": _is_positive_int,
    "composite_res": _is_res_string,
    "composite_w": _is_positive_int,
    "composite_h": _is_positive_int,
    "frame_budget": _is_positive_int,
    "beats": _is_positive_int,
}
_PREFLIGHT_KEYS = {
    "required_models": _is_str_list,
    "required_keys": _is_str_list,
}

# llm section: 8 model keys (widget-mapped 1:1 to the writer's existing
# model widgets) + the 7 runtime-policy fields. The runtime fields are
# validated by CONSTRUCTING an LLMRuntimePolicy -- ONE enum truth
# (nodes/_otr_shared/llm_policy.py), zero duplication.
_LLM_MODEL_KEYS = (
    "creative_model", "technical_model",
    "openrouter_slot_a_model", "openrouter_slot_b_model",
    "comfy_slot_a_model", "comfy_slot_b_model",
    "google_api_slot_a_model", "google_api_slot_b_model",
)
_LLM_RUNTIME_KEYS = (
    "device", "attn_impl", "quant_policy", "vram_ceiling_gb",
    "gguf_n_ctx", "gguf_quant", "lane_allowlist",
)


def _validate_llm_section(sub: Any, source: str) -> None:
    allowed = set(_LLM_MODEL_KEYS) | set(_LLM_RUNTIME_KEYS)
    unknown = set(sub) - allowed
    if unknown:
        raise ProfileError(f"profile {source}: unknown llm key(s) {sorted(unknown)!r}")
    missing = [k for k in (*_LLM_MODEL_KEYS, *_LLM_RUNTIME_KEYS) if k not in sub]
    if missing:
        raise ProfileError(f"profile {source}: llm missing key(s) {missing!r}")
    for k in _LLM_MODEL_KEYS:
        v = sub[k]
        if not isinstance(v, str):
            raise ProfileError(f"profile {source}: llm.{k} must be a str; got {v!r}")
        if k in ("creative_model", "technical_model") and not v:
            raise ProfileError(f"profile {source}: llm.{k} must be non-empty")
    from .llm_policy import LLMPolicyError, LLMRuntimePolicy
    try:
        LLMRuntimePolicy(
            device=sub["device"], attn_impl=sub["attn_impl"],
            quant_policy=sub["quant_policy"],
            vram_ceiling_gb=sub["vram_ceiling_gb"],
            gguf_n_ctx=sub["gguf_n_ctx"], gguf_quant=sub["gguf_quant"],
            lane_allowlist=tuple(sub["lane_allowlist"]),
        )
    except (LLMPolicyError, TypeError) as e:
        raise ProfileError(f"profile {source}: llm section invalid: {e}") from e


def _is_str_dict(v: Any) -> bool:
    return isinstance(v, dict) and all(
        isinstance(k, str) and isinstance(val, str) and val for k, val in v.items()
    )


def validate_profile_shape(profile: Any, source: str = "<dict>") -> dict:
    """S0 shape validator. Returns the profile on success; raises
    :class:`ProfileError` naming the first offending key otherwise.
    Unknown top-level / seed_policy / launch keys are REJECTED (a typo'd
    policy key silently doing nothing is the drift class this kills)."""
    if not isinstance(profile, dict):
        raise ProfileError(f"profile {source}: expected a JSON object, got {type(profile).__name__}")

    unknown = set(profile) - set(_TOP_LEVEL_KEYS)
    if unknown:
        raise ProfileError(f"profile {source}: unknown top-level key(s) {sorted(unknown)!r}")
    missing = [k for k, (req, _, _) in _TOP_LEVEL_KEYS.items() if req and k not in profile]
    if missing:
        raise ProfileError(f"profile {source}: missing required key(s) {missing!r}")
    for k, (_, check, desc) in _TOP_LEVEL_KEYS.items():
        if k in profile and not check(profile[k]):
            raise ProfileError(f"profile {source}: key {k!r} must be {desc}; got {profile[k]!r}")

    for sub_name, sub_spec in (
        ("seed_policy", _SEED_POLICY_KEYS),
        ("launch", _LAUNCH_KEYS),
        ("video", _VIDEO_KEYS),
        ("image", _IMAGE_KEYS),
        ("audio", _AUDIO_KEYS),
        ("render", _RENDER_KEYS),
        ("preflight", _PREFLIGHT_KEYS),
    ):
        sub = profile[sub_name]
        unknown = set(sub) - set(sub_spec)
        if unknown:
            raise ProfileError(f"profile {source}: unknown {sub_name} key(s) {sorted(unknown)!r}")
        missing = [k for k in sub_spec if k not in sub]
        if missing:
            raise ProfileError(f"profile {source}: {sub_name} missing key(s) {missing!r}")
        for k, check in sub_spec.items():
            if not check(sub[k]):
                raise ProfileError(f"profile {source}: {sub_name}.{k} has invalid value {sub[k]!r}")

    # v2: the llm section (constructor-based validation; ONE enum truth).
    _validate_llm_section(profile["llm"], source)

    # features: bool/str values only in v1 (widget-backed BOOLEANs + COMBO styles)
    for k, v in profile["features"].items():
        if not isinstance(k, str) or not isinstance(v, (bool, str)):
            raise ProfileError(
                f"profile {source}: features.{k} must be bool or str; got {v!r}"
            )
    return profile


def load_profile(profile_id: str, profile_dir: Optional[str] = None) -> dict:
    """Load + shape-validate ``config/profiles/<id>.json``. Fail closed."""
    d = profile_dir or PROFILE_DIR
    path = os.path.join(d, f"{profile_id}.json")
    if not os.path.isfile(path):
        try:
            known = sorted(
                f[:-5] for f in os.listdir(d)
                if f.endswith(".json") and f != "widget_mapping.json"
            )
        except OSError:
            known = []
        raise ProfileError(
            f"profile {profile_id!r}: no such file {path!r}; known profiles: {known!r}"
        )
    with open(path, "r", encoding="utf-8") as f:
        try:
            profile = json.load(f)
        except json.JSONDecodeError as e:
            raise ProfileError(f"profile {profile_id!r}: {path!r} failed to parse: {e}") from e
    profile = validate_profile_shape(profile, source=path)
    if profile["id"] != profile_id:
        raise ProfileError(
            f"profile {profile_id!r}: file {path!r} declares id={profile['id']!r} "
            f"(filename and id must agree)"
        )
    return profile


# ---------------------------------------------------------------------------
# S0 -- widget mapping
# ---------------------------------------------------------------------------
_MAPPING_SECTIONS = ("managed", "emit_only")
_MAPPING_KEYS = ("version", "_comment", "managed", "emit_only",
                 "exempt_node_types", "exempt_widget_names",
                 "never_patch_widget_names")
_REGISTRY_NAMES = ("video", "audio", "image")


def validate_widget_mapping_shape(mapping: Any, source: str = "<dict>") -> dict:
    """Shape-validate the checked-in widget mapping. Targets are
    ``[node_type, widget_name]`` string pairs; anything that looks like a raw
    node id (an int, or a digit-string node_type) is REJECTED."""
    if not isinstance(mapping, dict):
        raise ProfileError(f"mapping {source}: expected a JSON object")
    unknown = set(mapping) - set(_MAPPING_KEYS)
    if unknown:
        raise ProfileError(f"mapping {source}: unknown key(s) {sorted(unknown)!r}")
    for section in _MAPPING_SECTIONS:
        entries = mapping.get(section)
        if not isinstance(entries, dict):
            raise ProfileError(f"mapping {source}: section {section!r} must be a dict")
        for key, entry in entries.items():
            if not isinstance(entry, dict) or set(entry) != {"registry", "targets"}:
                raise ProfileError(
                    f"mapping {source}: entry {key!r} must have exactly "
                    f"'registry' + 'targets'"
                )
            reg = entry["registry"]
            if reg is not None and reg not in _REGISTRY_NAMES:
                raise ProfileError(
                    f"mapping {source}: entry {key!r} registry must be one of "
                    f"{_REGISTRY_NAMES} or null; got {reg!r}"
                )
            targets = entry["targets"]
            if not isinstance(targets, list) or not targets:
                raise ProfileError(f"mapping {source}: entry {key!r} targets must be a non-empty list")
            for t in targets:
                if (not isinstance(t, list) or len(t) != 2
                        or not all(isinstance(x, str) and x for x in t)):
                    raise ProfileError(
                        f"mapping {source}: entry {key!r} target {t!r} must be "
                        f"[node_type, widget_name] (two non-empty strings)"
                    )
                if t[0].isdigit():
                    raise ProfileError(
                        f"mapping {source}: entry {key!r} target {t!r} looks like a "
                        f"raw node id -- node ids are BANNED; use the node TYPE"
                    )
                never = mapping.get("never_patch_widget_names") or []
                if t[1] in never:
                    raise ProfileError(
                        f"mapping {source}: entry {key!r} targets forbidden widget "
                        f"name {t[1]!r} (companion-slot trap)"
                    )
    # v2: per-node-type widget-name exemptions (the coverage audit's
    # fine-grained sibling of exempt_node_types). Shape:
    # {node_type: [widget_name, ...]}.
    ewn = mapping.get("exempt_widget_names")
    if ewn is not None:
        if not isinstance(ewn, dict):
            raise ProfileError(
                f"mapping {source}: exempt_widget_names must be a dict of "
                f"node_type -> [widget names]")
        for ntype, names in ewn.items():
            if not isinstance(ntype, str) or not ntype or ntype.isdigit():
                raise ProfileError(
                    f"mapping {source}: exempt_widget_names key {ntype!r} "
                    f"must be a node TYPE (raw ids banned)")
            if not isinstance(names, list) or not names or not all(
                    isinstance(n, str) and n for n in names):
                raise ProfileError(
                    f"mapping {source}: exempt_widget_names[{ntype!r}] must "
                    f"be a non-empty list of widget names")
    return mapping


def load_widget_mapping(path: Optional[str] = None) -> dict:
    p = path or WIDGET_MAPPING_PATH
    with open(p, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return validate_widget_mapping_shape(mapping, source=p)


# ---------------------------------------------------------------------------
# S1 -- capability declarations + the derived enable-set
# ---------------------------------------------------------------------------
# Registry CAPABILITIES v2 (platform-portability S3, 2026-07-10): the bare
# v1 ``cpu_ok`` bool is SUPERSEDED by an explicit ``device_backends`` list +
# ``practical_without_gpu`` + vendor/dtype table-visibility. A v1 row (any
# row still carrying ``cpu_ok``) is REJECTED outright -- the old false bark
# row cannot survive v2 semantics.
_DECL_KEYS = {
    "required_toolchain": lambda v: v is None or (isinstance(v, str) and bool(v)),
    "requires_sidecar": lambda v: isinstance(v, bool),
    "device_backends": lambda v: (
        isinstance(v, (list, tuple)) and bool(v)
        and all(b in _DEVICE_BACKENDS for b in v)),
    "requires_vendor": lambda v: v is None or v in ("nvidia", "amd", "apple"),
    "needs_fp8_te": lambda v: isinstance(v, bool),
    "needs_fp4_te": lambda v: isinstance(v, bool),
    "practical_without_gpu": lambda v: isinstance(v, bool),
    "sidecar_conditional": lambda v: isinstance(v, bool),
    "model_requirements": lambda v: isinstance(v, (list, tuple)) and all(isinstance(m, str) for m in v),
}

# Availability reason codes (the shared availability object's vocabulary --
# reused by the validator, the wizard and the queue-start LOUD log).
REASON_OK = "ok"
REASON_REQUIRES_CUDA = "requires_cuda"
REASON_MISSING_TOOLCHAIN = "missing_toolchain"
REASON_SIDECARS_DISABLED = "sidecars_disabled"
# v2 additions:
REASON_REQUIRES_VENDOR = "requires_vendor"
REASON_IMPRACTICAL_ON_CPU = "impractical_on_cpu"


def validate_declaration(name: str, decl: Any, source: str = "<registry>") -> dict:
    """Validate ONE engine capability declaration (registry-table row)."""
    if not isinstance(decl, dict):
        raise ProfileError(f"{source}: declaration for {name!r} must be a dict")
    unknown = set(decl) - set(_DECL_KEYS)
    if unknown:
        raise ProfileError(f"{source}: declaration {name!r} has unknown key(s) {sorted(unknown)!r}")
    missing = [k for k in _DECL_KEYS if k not in decl]
    if missing:
        raise ProfileError(f"{source}: declaration {name!r} missing key(s) {missing!r}")
    for k, check in _DECL_KEYS.items():
        if not check(decl[k]):
            raise ProfileError(f"{source}: declaration {name!r} key {k!r} invalid: {decl[k]!r}")
    return decl


def _fit_reason(decl: dict, profile: dict) -> str:
    """Why does (or doesn't) ONE engine declaration fit ONE profile?
    Per-engine fit ONLY -- backend / vendor / toolchain / sidecar gating,
    never a VRAM tier or budget (the operator's tier JSON owns the OOM
    budget now) and never co-residency (that is a runtime invariant).

    v2 (S3): ``device_backends`` supersedes the bare ``cpu_ok`` bool; a
    profile backend the engine does not list is a mismatch (the code stays
    ``requires_cuda`` -- vocabulary continuity for the validator/wizard/
    logs, and the missing backend IS cuda for every current local row).
    ``practical_without_gpu`` keeps technically-cpu-capable-but-impractical
    engines off the cpu floor; ``requires_vendor`` makes the NVML/cu128
    vendor pins table-visible."""
    if profile["device_backend"] not in decl["device_backends"]:
        return REASON_REQUIRES_CUDA
    if profile["device_backend"] == "cpu" and not decl["practical_without_gpu"]:
        return REASON_IMPRACTICAL_ON_CPU
    if decl["requires_vendor"] and profile.get("gpu_vendor") != decl["requires_vendor"]:
        return REASON_REQUIRES_VENDOR
    if decl["required_toolchain"] and decl["required_toolchain"] not in profile["toolchains"]:
        return REASON_MISSING_TOOLCHAIN
    if decl["requires_sidecar"] and not profile["allow_sidecars"]:
        return REASON_SIDECARS_DISABLED
    return REASON_OK


def availability(profile: dict, declarations: dict) -> dict:
    """The shared availability object: ``{engine_name: reason_code}`` for every
    declared engine of ONE namespace. ``reason == "ok"`` means enabled."""
    out: dict[str, str] = {}
    for name in sorted(declarations):
        decl = validate_declaration(name, declarations[name])
        out[name] = _fit_reason(decl, profile)
    return out


def enabled_engines(profile: dict, declarations: dict) -> list:
    """``enabled(P)`` for one namespace -- DERIVED, never hand-listed."""
    return [n for n, reason in availability(profile, declarations).items() if reason == REASON_OK]


def cross_validate_profile(profile: dict, mapping: dict,
                           declarations_by_registry: dict) -> None:
    """S1 capability cross-checks: every engine-valued override in the profile
    must be in ``enabled(P)`` of its namespace (the mapping names the
    namespace). Raises :class:`ProfileError` listing every violation.

    Deliberately NO static co-residency rejection: a profile with two heavy
    roles is VALID (single-heavy residency is wrapper_bridge's runtime
    invariant, not a profile-shape concern)."""
    problems: list[str] = []
    managed = mapping["managed"]
    flat = {}
    for section in ("role_overrides", "slot_overrides"):
        for key, value in profile.get(section, {}).items():
            flat[f"{section}.{key}"] = value
    for dotted, value in sorted(flat.items()):
        entry = managed.get(dotted)
        if entry is None:
            problems.append(
                f"{dotted}={value!r}: no widget-mapping entry (typo'd override key?)"
            )
            continue
        registry = entry["registry"]
        if registry is None:
            continue
        decls = declarations_by_registry.get(registry)
        if decls is None:
            problems.append(f"{dotted}={value!r}: unknown registry namespace {registry!r}")
            continue
        avail = availability(profile, decls)
        reason = avail.get(value)
        if reason is None:
            problems.append(
                f"{dotted}={value!r}: engine not declared in the {registry} "
                f"registry CAPABILITIES table"
            )
        elif reason != REASON_OK:
            problems.append(
                f"{dotted}={value!r}: engine excluded from profile "
                f"{profile['id']!r} enable-set ({reason})"
            )
    if problems:
        raise ProfileError(
            f"profile {profile.get('id')!r} failed capability cross-validation:\n  "
            + "\n  ".join(problems)
        )
