#!/usr/bin/env python3
"""Resolve one OTR profile or ``machine:<key>`` into a launch contract.

The output is data, never shell code.  Pod launchers consume it through a
checked temporary file so a failed resolver cannot look like an empty/default
configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config" / "profiles"
BOOT_CONTRACTS_PATH = ROOT / "nodes" / "_otr_shared" / "boot_contracts.py"
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RUNTIME_IDENTITY_KEYS = (
    "OTR_COMFY_ROOT", "OTR_REPO_ROOT", "COMFY_PY",
    "OTR_COMFYUI_MODELS_ROOT", "HF_HOME", "OTR_INDEXTTS2_ROOT",
    "OTR_INDEXTTS2_DIR", "OTR_INDEXTTS2_WORKER",
    "OTR_VOICE_REFERENCE_BANK", "OTR_OUTPUT_ROOT", "OTR_OUTPUT_DIR",
    "OTR_OBS_DIR", "OTR_TMP", "TMP", "TEMP", "OTR_GPU_LEASE_DIR",
    "OTR_HEADLESS_PORT", "COMFYUI_URL", "OTR_PROVISION_GENERATION",
    "OTR_RUNTIME_SECRETS_FILE",
)
SECRET_IDENTITY_KEYS = (
    "HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)


class LaunchConfigError(RuntimeError):
    """A profile cannot be represented by the audited launch channel."""


def _git_head(path: str) -> str:
    """Return a checkout identity without making git availability a gate."""
    if not path:
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _runtime_identity() -> dict:
    """Non-secret server inputs plus hashes of inherited credentials."""
    values = {key: os.environ.get(key, "") for key in RUNTIME_IDENTITY_KEYS}
    secrets = {}
    for key in SECRET_IDENTITY_KEYS:
        value = os.environ.get(key, "")
        secrets[key] = (hashlib.sha256(value.encode("utf-8")).hexdigest()
                        if value else "")
    otr_root = values.get("OTR_REPO_ROOT") or str(ROOT)
    return {
        "env": values,
        "secret_hashes": secrets,
        "revisions": {
            "comfyui": _git_head(values.get("OTR_COMFY_ROOT", "")),
            "otr": _git_head(otr_root),
        },
    }


def _load_boot_contracts():
    spec = importlib.util.spec_from_file_location(
        "otr_profile_launch_boot_contracts", BOOT_CONTRACTS_PATH
    )
    if spec is None or spec.loader is None:
        raise LaunchConfigError(
            f"cannot load boot-contract owner: {BOOT_CONTRACTS_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_profile(profile_id: str) -> dict:
    if str(profile_id or "").startswith("machine:"):
        machine_key = str(profile_id).split(":", 1)[1]
        if not PROFILE_ID_RE.fullmatch(machine_key):
            raise LaunchConfigError(f"invalid machine key: {machine_key!r}")
        helper_path = ROOT / "scripts" / "otr_machine_profile.py"
        spec = importlib.util.spec_from_file_location(
            "otr_launch_machine_profile", helper_path)
        if spec is None or spec.loader is None:
            raise LaunchConfigError(
                f"cannot load machine-profile owner: {helper_path}")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        try:
            matrix = helper.load_matrix()
            profile = helper.build_profile(helper.resolve(machine_key, matrix), matrix)
        except SystemExit as exc:
            raise LaunchConfigError(str(exc)) from exc
        if str(profile.get("id") or "") != profile_id:
            raise LaunchConfigError(
                f"machine selector drift: requested {profile_id!r}, "
                f"builder returned {profile.get('id')!r}")
        return profile
    if not PROFILE_ID_RE.fullmatch(profile_id or ""):
        raise LaunchConfigError(f"invalid profile id: {profile_id!r}")
    path = PROFILES / f"{profile_id}.json"
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaunchConfigError(f"profile does not exist: {profile_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchConfigError(f"cannot read profile {profile_id}: {exc}") from exc
    if not isinstance(profile, dict):
        raise LaunchConfigError(f"profile {profile_id} is not a JSON object")
    if str(profile.get("id") or "") != profile_id:
        raise LaunchConfigError(
            f"profile filename/id drift: requested {profile_id!r}, "
            f"document says {profile.get('id')!r}"
        )
    return profile


def resolve_launch(profile: dict) -> dict:
    launch = profile.get("launch") or {}
    if not isinstance(launch, dict):
        raise LaunchConfigError("profile launch field must be an object")
    contract_name = str(launch.get("boot_contract") or "default")
    contracts = _load_boot_contracts()
    try:
        contract_spec = contracts.contract_spec(contract_name)
        expected_env = contracts.launch_env_for(contract_name)
    except Exception as exc:  # noqa: BLE001 - normalize owner errors for the CLI
        raise LaunchConfigError(str(exc)) from exc

    raw_env = launch.get("env") or {}
    if not isinstance(raw_env, dict):
        raise LaunchConfigError("profile launch.env must be an object")
    env: dict[str, str] = {}
    for raw_key, raw_value in raw_env.items():
        key = str(raw_key)
        if not ENV_KEY_RE.fullmatch(key):
            raise LaunchConfigError(f"invalid launch environment key: {key!r}")
        if isinstance(raw_value, (dict, list)) or raw_value is None:
            raise LaunchConfigError(
                f"launch environment value for {key} must be a scalar"
            )
        value = str(raw_value)
        if "\n" in value or "\r" in value or "\x00" in value:
            raise LaunchConfigError(
                f"launch environment value for {key} contains a line break"
            )
        env[key] = value

    contract_env_keys = set(contracts.CONTRACT_ENV.values())
    actual_contract_env = {
        key: value for key, value in env.items() if key in contract_env_keys
    }
    if actual_contract_env != expected_env:
        raise LaunchConfigError(
            "profile launch.env disagrees with boot contract %r: expected %r, got %r"
            % (contract_name, expected_env, actual_contract_env)
        )

    sage = launch.get("sage_attention")
    if sage is not None and not isinstance(sage, bool):
        raise LaunchConfigError("profile launch.sage_attention must be boolean")
    wanted_sage = contract_spec.get("sage_attention")
    if wanted_sage is not None and sage is not wanted_sage:
        raise LaunchConfigError(
            "profile SageAttention setting disagrees with boot contract %r: "
            "expected %r, got %r" % (contract_name, wanted_sage, sage)
        )

    argv = contracts.launch_args_for(contract_name)

    fingerprint_doc = {
        "argv": argv,
        "boot_contract": contract_name,
        "env": {key: env[key] for key in sorted(env)},
        "sage_attention": sage,
        "runtime": _runtime_identity(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_doc, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    slots = profile.get("slot_overrides") or {}
    requires_indextts2 = isinstance(slots, dict) and any(
        str(slots.get(key) or "").strip() == "indextts2"
        for key in ("char_voice_engine", "announcer_voice_engine")
    )
    return {
        "argv": argv,
        "boot_contract": contract_name,
        "env": env,
        "fingerprint": fingerprint,
        "requires_indextts2": requires_indextts2,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "profile", help="exact config/profiles id or machine:<matrix-key>")
    parser.add_argument(
        "--mode",
        choices=("args", "env", "contract", "fingerprint", "requires-indextts2"),
        default="args",
    )
    args = parser.parse_args(argv)
    try:
        resolved = resolve_launch(load_profile(args.profile))
    except LaunchConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.mode == "args":
        rows = resolved["argv"]
        sys.stdout.write("\n".join(rows) + ("\n" if rows else ""))
    elif args.mode == "env":
        for key in sorted(resolved["env"]):
            print(f"{key}={resolved['env'][key]}")
    elif args.mode == "contract":
        print(resolved["boot_contract"])
    elif args.mode == "fingerprint":
        print(resolved["fingerprint"])
    else:
        print("1" if resolved["requires_indextts2"] else "0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
