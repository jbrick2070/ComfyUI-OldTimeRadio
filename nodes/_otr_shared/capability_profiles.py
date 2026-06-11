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
    "VRAM_CLASS_RANK",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILE_DIR = os.path.join(_REPO_ROOT, "config", "profiles")
WIDGET_MAPPING_PATH = os.path.join(PROFILE_DIR, "widget_mapping.json")


class ProfileError(ValueError):
    """A profile / mapping file failed validation. FAIL CLOSED -- the caller
    must not proceed with a half-understood capability policy."""


# ---------------------------------------------------------------------------
# S0 -- profile shape
# ---------------------------------------------------------------------------
VRAM_CLASS_RANK = {"cpu": 0, "light": 1, "medium": 2, "heavy": 3}

_PLATFORMS = ("any", "win", "mac")
_DEVICE_BACKENDS = ("cuda", "cpu")  # mps deliberately absent in v1 (parked)
_STATUSES = ("shipping", "draft")

# key -> (required, validator-callable, human description)
_TOP_LEVEL_KEYS: dict[str, tuple[bool, Any, str]] = {
    "id": (True, lambda v: isinstance(v, str) and bool(v), "non-empty str"),
    "display_name": (True, lambda v: isinstance(v, str) and bool(v), "non-empty str"),
    "status": (True, lambda v: v in _STATUSES, f"one of {_STATUSES}"),
    "platform": (True, lambda v: v in _PLATFORMS, f"one of {_PLATFORMS}"),
    "device_backend": (True, lambda v: v in _DEVICE_BACKENDS, f"one of {_DEVICE_BACKENDS}"),
    "vram_budget_mb": (True, lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0, "int >= 0"),
    "toolchains": (True, lambda v: isinstance(v, list) and all(isinstance(t, str) for t in v), "list[str]"),
    "allow_sidecars": (True, lambda v: isinstance(v, bool), "bool"),
    "max_model_class": (True, lambda v: v in VRAM_CLASS_RANK, f"one of {tuple(VRAM_CLASS_RANK)}"),
    "role_overrides": (True, lambda v: _is_str_dict(v), "dict[str, str]"),
    "slot_overrides": (True, lambda v: _is_str_dict(v), "dict[str, str]"),
    "features": (True, lambda v: isinstance(v, dict), "dict"),
    "seed_policy": (True, lambda v: isinstance(v, dict), "dict"),
    "launch": (True, lambda v: isinstance(v, dict), "dict"),
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
}


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

    for sub_name, sub_spec in (("seed_policy", _SEED_POLICY_KEYS), ("launch", _LAUNCH_KEYS)):
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
                 "exempt_node_types", "never_patch_widget_names")
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
    return mapping


def load_widget_mapping(path: Optional[str] = None) -> dict:
    p = path or WIDGET_MAPPING_PATH
    with open(p, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return validate_widget_mapping_shape(mapping, source=p)


# ---------------------------------------------------------------------------
# S1 -- capability declarations + the derived enable-set
# ---------------------------------------------------------------------------
_DECL_KEYS = {
    "vram_class": lambda v: v in VRAM_CLASS_RANK,
    "vram_estimate_mb": lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
    "required_toolchain": lambda v: v is None or (isinstance(v, str) and bool(v)),
    "requires_sidecar": lambda v: isinstance(v, bool),
    "cpu_ok": lambda v: isinstance(v, bool),
    "model_requirements": lambda v: isinstance(v, (list, tuple)) and all(isinstance(m, str) for m in v),
}

# Availability reason codes (the shared availability object's vocabulary --
# reused by the validator, the wizard and the queue-start LOUD log).
REASON_OK = "ok"
REASON_REQUIRES_CUDA = "requires_cuda"
REASON_MISSING_TOOLCHAIN = "missing_toolchain"
REASON_SIDECARS_DISABLED = "sidecars_disabled"
REASON_CLASS_OVER_CAP = "model_class_over_cap"
REASON_VRAM_OVER_BUDGET = "vram_over_budget"


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
    Per-engine fit ONLY -- never co-residency (that is a runtime invariant)."""
    if profile["device_backend"] == "cpu":
        # On a CPU floor there is no VRAM economy at all: the only question
        # is whether the engine can run on CPU. Class/budget checks are
        # GPU-residency concepts and do not apply.
        if not decl["cpu_ok"]:
            return REASON_REQUIRES_CUDA
        if decl["required_toolchain"] and decl["required_toolchain"] not in profile["toolchains"]:
            return REASON_MISSING_TOOLCHAIN
        if decl["requires_sidecar"] and not profile["allow_sidecars"]:
            return REASON_SIDECARS_DISABLED
        return REASON_OK
    if decl["required_toolchain"] and decl["required_toolchain"] not in profile["toolchains"]:
        return REASON_MISSING_TOOLCHAIN
    if decl["requires_sidecar"] and not profile["allow_sidecars"]:
        return REASON_SIDECARS_DISABLED
    if VRAM_CLASS_RANK[decl["vram_class"]] > VRAM_CLASS_RANK[profile["max_model_class"]]:
        return REASON_CLASS_OVER_CAP
    if decl["vram_estimate_mb"] > profile["vram_budget_mb"]:
        return REASON_VRAM_OVER_BUDGET
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
