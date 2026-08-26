"""Platform-variant generator -- the never-built GATE B "S3 emit_snapshot".

docs/2026-07-09-platform-portability-final.md section 1. An OFFLINE CLI
(never a node): a platform variant = TWO artifacts generated together from
the ONE canonical graph + a committed capability profile:

  1. workflows/variants/otr_<profile_id>.json  -- apply_profile(canonical,
     profile) + stamps written into OTR_WorkflowValidator's EXISTING
     widgets (profile_id / master_hash / generated_by / its own path).
  2. workflows/variants/otr_<profile_id>.launch.md -- the launch recipe
     from the SAME validated profile object (args, sage flag, env, key
     NAMES only -- values are never stored -- and install pointers).

Self-check on every emit: apply -> hash -> stamp -> RE-apply -> the
semantic hash must not move (a managed widget the applier cannot
reproduce refuses the emit). ``--check`` regenerates every committed
variant AND recipe in memory, diffs both against disk, verifies the
stamps agree, asserts the retired otr_api stale-variant soft-skip stays
dead, and exits nonzero on ANY drift (CI rule: variants are GENERATED,
never hand-edited).

REFUSES emission while a profile carries ratify_before_emit entries --
the operator ratifies each named decision and clears the list first.

Usage:
  python scripts/build_variants.py --all
  python scripts/build_variants.py --profiles cpu_floor,8gb_lite
  python scripts/build_variants.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nodes._otr_shared.capability_profiles import (  # noqa: E402
    PROFILE_DIR, load_profile,
)
from nodes._otr_workflow_apply import (  # noqa: E402
    apply_profile, build_offline_schemas, load_widget_mapping,
    patch_widget_by_name, semantic_master_hash,
)

CANONICAL = REPO / "workflows" / "otr_canonical.json"
VARIANTS_DIR = REPO / "workflows" / "variants"
GENERATED_BY = "scripts/build_variants.py"
# Lane presets are applied onto a live canonical by the operator, not
# emitted as standalone platform variants.
LANE_PRESETS = ("google_veo_media", "google_omni_media",
                "google_veo_all", "google_omni_all",
                # SOAK INSTRUMENTS (2026-08-16), not platform targets. They
                # exist so `scripts/otr_gpu_soak_matrix.py` can rotate the
                # still/image engines through the SANCTIONED surface -- those
                # widgets are managed and `patch_creative` refuses them, so a
                # profile's role_overrides is the only legitimate lever. They
                # are excluded from emission because nobody installs OTR to
                # run a soak: shipping ten near-identical variant graphs would
                # be noise in the user-facing variant set.
                "otr_soak_still_flat_z_image_turbo",
                "otr_soak_still_flat_flux_gen1",
                "otr_soak_still_motion_lumina_image",
                "otr_soak_still_motion_flux2_klein",
                "otr_soak_still_pan_flux_gen1",
                "otr_soak_still_pan_ideo",
                "otr_soak_still_word_flux2_klein",
                "otr_soak_still_word_z_image_turbo",
                "otr_soak_word_razzle_ideo",
                "otr_soak_word_razzle_lumina_image",
                # LLM x IMAGE x UPSCALE SWEEP (2026-08-25, operator: "one act
                # test of all seven [LLMs] onboard ... varying image models
                # ... a variety of upscalers"). Each leg pairs one LLM as
                # creative with a DIFFERENT one as technical (cyclic, so all
                # 7 rows play both roles across the 7 legs), THREE distinct
                # still engines and THREE distinct local image engines (one
                # per role: announcer/music/character), and alternates the
                # only two real upscale engines. The llm.* block in each file
                # is a harmless unused default -- scripts/otr_llm_image_upscale_sweep.py
                # passes --creative-model/--technical-model on the CLI, which
                # apply_profile-then-shortcuts ordering makes authoritative.
                "otr_soak_llmsweep_01",
                "otr_soak_llmsweep_02",
                "otr_soak_llmsweep_03",
                "otr_soak_llmsweep_04",
                "otr_soak_llmsweep_05",
                "otr_soak_llmsweep_06",
                "otr_soak_llmsweep_07")


class EmitRefused(RuntimeError):
    """Emission refused (ratify gate or self-check failure)."""


def _load_canonical() -> dict:
    with open(CANONICAL, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump(workflow: dict) -> str:
    return json.dumps(workflow, ensure_ascii=True, separators=(",", ":"))


def _validator_node_id(workflow: dict) -> int:
    hits = [n for n in workflow["nodes"]
            if n.get("type") == "OTR_WorkflowValidator"]
    if len(hits) != 1:
        raise EmitRefused(
            f"expected exactly ONE OTR_WorkflowValidator node; found "
            f"{len(hits)}")
    return int(hits[0]["id"])


def _variant_stem(profile_id: str) -> str:
    """otr_<id>, without double-prefixing ids that already carry otr_."""
    return profile_id if profile_id.startswith("otr_") else f"otr_{profile_id}"


def _profile_id_from_stem(stem: str) -> str:
    """Inverse of _variant_stem against the COMMITTED profile ids."""
    committed = set(_committed_profile_ids())
    if stem in committed:
        return stem
    if stem.startswith("otr_") and stem[len("otr_"):] in committed:
        return stem[len("otr_"):]
    return stem


def _committed_profile_ids() -> list[str]:
    out = []
    for p in sorted(Path(PROFILE_DIR).glob("*.json")):
        if p.stem == "widget_mapping":
            continue
        out.append(p.stem)
    # Stem-collision guard (post-ship audit): a bare id X and a prefixed
    # otr_X would map to the SAME variant filename -- refuse loudly.
    for pid in out:
        if not pid.startswith("otr_") and f"otr_{pid}" in out:
            raise EmitRefused(
                f"profile ids '{pid}' and 'otr_{pid}' collide on the "
                f"variant filename otr_{pid}.json -- rename one.")
    return out


def build_variant(profile_id: str, *, schemas=None, mapping=None,
                  canonical=None) -> tuple[dict, str, str]:
    """Return (stamped_variant, variant_rel_path, recipe_text). Pure --
    writes nothing. Raises EmitRefused on the ratify gate or a failed
    self-check."""
    schemas = schemas or build_offline_schemas()
    mapping = mapping or load_widget_mapping()
    canonical = canonical if canonical is not None else _load_canonical()

    profile = load_profile(profile_id)
    ratify = profile.get("ratify_before_emit") or []
    if ratify:
        raise EmitRefused(
            f"profile '{profile_id}' carries UNRATIFIED field(s):\n  - "
            + "\n  - ".join(ratify)
            + "\nThe operator ratifies each decision and clears "
            "ratify_before_emit; emission stays refused until then.")

    # Queue item 8 (2026-08-08): cross-validate the profile's capability
    # overrides against every registry's enable-set BEFORE applying it. Catches
    # a bad upscale_stage.engine, a stale role/slot override, etc., before any
    # canonical mutation. Codex r4 MF-3.
    from nodes._otr_shared.capability_profiles import cross_validate_profile
    from nodes._otr_audio_engines.registry import CAPABILITIES as _AUDIO_CAPS
    from nodes._otr_video_engines.registry import CAPABILITIES as _VIDEO_CAPS
    from nodes._otr_image_engines.registry import CAPABILITIES as _IMAGE_CAPS
    from nodes._otr_upscale_engines.registry import CAPABILITIES as _UPSCALE_CAPS
    cross_validate_profile(profile, mapping, {
        "audio": _AUDIO_CAPS, "video": _VIDEO_CAPS,
        "image": _IMAGE_CAPS, "upscale": _UPSCALE_CAPS,
    })

    applied = apply_profile(canonical, profile, mapping=mapping,
                            schemas=schemas)
    master_hash = semantic_master_hash(applied, mapping=mapping,
                                       schemas=schemas)
    variant_rel = f"workflows/variants/{_variant_stem(profile_id)}.json"
    nid = _validator_node_id(applied)
    for widget, value in (
        ("workflow_json_path", variant_rel),
        ("profile_id", profile_id),
        ("master_hash", master_hash),
        ("generated_by", GENERATED_BY),
    ):
        patch_widget_by_name(applied, nid, widget, value, schemas)

    # Self-check: re-apply onto the STAMPED variant; the semantic hash
    # must not move (stamps are excluded by construction) and a managed
    # widget the applier cannot reproduce refuses the emit here, before
    # anything is written.
    recheck = apply_profile(applied, profile, mapping=mapping,
                            schemas=schemas)
    if semantic_master_hash(recheck, mapping=mapping,
                            schemas=schemas) != master_hash:
        raise EmitRefused(
            f"self-check FAILED for '{profile_id}': re-applying the "
            "profile moved the semantic hash -- the applier cannot "
            "reproduce this variant deterministically; refusing to emit.")

    recipe = _launch_recipe(profile, profile_id, variant_rel, master_hash)
    return applied, variant_rel, recipe


def _launch_recipe(profile: dict, profile_id: str, variant_rel: str,
                   master_hash: str) -> str:
    launch = profile.get("launch") or {}
    preflight = profile.get("preflight") or {}
    backend = profile.get("device_backend")
    vendor = profile.get("gpu_vendor")
    args = list(launch.get("extra_args") or [])
    env = dict(launch.get("env") or {})

    if backend == "cuda" and vendor == "amd":
        torch_note = ("torch ROCm build (https://pytorch.org rocm wheels); "
                      "presents as cuda. bnb/fp8/sage lanes are OFF.")
        llama_note = "llama-cpp-python HIP wheel"
    elif backend == "cuda":
        torch_note = "torch cu128+ build (the nv baseline is 2.10/cu130)"
        llama_note = "llama-cpp-python CUDA wheel"
    elif backend == "mps":
        torch_note = "default PyPI torch wheels (Metal/MPS included)"
        llama_note = "llama-cpp-python Metal wheel"
    else:
        torch_note = "CPU torch wheels (--index-url .../cpu)"
        llama_note = "llama-cpp-python CPU wheel"

    lines = [
        f"# Launch recipe: {profile_id}",
        "",
        f"GENERATED by {GENERATED_BY} -- do not hand-edit (regenerate "
        "instead). Paired variant: `" + variant_rel + "`.",
        "",
        f"- profile: `{profile_id}` ({profile.get('display_name')})",
        f"- status: {profile.get('status')}",
        f"- platform/backend/vendor: {profile.get('platform')}/"
        f"{backend}/{vendor}",
        f"- master_hash: `{master_hash}`",
        "",
        "## ComfyUI launch",
        "",
        f"- args: `{' '.join(args) if args else '(none)'}`",
        f"- sage_attention: {bool(launch.get('sage_attention'))}",
        "",
        "## Environment",
        "",
        "- Windows hosts: set `PYTHONUTF8=1` (cp1252 consoles crash the "
        "prestartup banner otherwise).",
        "- Models root override: `OTR_COMFYUI_MODELS_ROOT` (the "
        "`C:\\ComfyUI-Models` default in _otr_hf_env/_otr_gguf_backend "
        "is a Windows-only convenience); GGUF path escape hatch: "
        "`GEMMA4_12B_GGUF_PATH`.",
    ]
    for k, v in sorted(env.items()):
        lines.append(f"- `{k}={v}`")
    lines += [
        "",
        "## Required key NAMES (values are NEVER stored here)",
        "",
    ]
    keys = list(preflight.get("required_keys") or [])
    if keys:
        for k in keys:
            lines.append(f"- `{k}`")
        if any("GOOGLE" in k for k in keys):
            lines.append("- Google key aliases accepted by existing "
                         "clients: `GEMINI_API_KEY`, `GOOGLE_API_KEY` "
                         "(preferred name: `OTR_GOOGLE_API_KEY`).")
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Install pointers",
        "",
        f"- {torch_note}",
        f"- {llama_note}",
        "- ffmpeg on PATH (mac: ensure libx264 + aac encoders are in the "
        "build).",
        "- minimal Linux: libcairo for viz_mandala.",
        "",
        "## Preflight models",
        "",
    ]
    models = list(preflight.get("required_models") or [])
    if models:
        lines += [f"- {m}" for m in models]
    else:
        lines.append("- (registry-driven; see the engine rows for the "
                     "selected lanes)")
    return "\n".join(lines) + "\n"


def cmd_emit(profile_ids: list[str], explicit: bool) -> int:
    schemas = build_offline_schemas()
    mapping = load_widget_mapping()
    canonical = _load_canonical()
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    emitted, refused = [], []
    for pid in profile_ids:
        try:
            variant, rel, recipe = build_variant(
                pid, schemas=schemas, mapping=mapping, canonical=canonical)
        except EmitRefused as e:
            refused.append((pid, str(e)))
            continue
        (REPO / rel).write_text(_dump(variant), encoding="utf-8")
        # newline="\n": platform-safe by construction (Windows universal
        # newlines otherwise bake CRLF into locally-regenerated recipes).
        (VARIANTS_DIR / f"{_variant_stem(pid)}.launch.md").write_text(
            recipe, encoding="utf-8", newline="\n")
        emitted.append(rel)
        print(f"EMITTED {rel} (+ launch recipe)")
    for pid, why in refused:
        print(f"REFUSED {pid}:\n{why}\n")
    print(f"done: {len(emitted)} emitted, {len(refused)} refused")
    # Explicitly requesting a refused profile is an error; --all treats
    # refusals as the expected pre-ratification state.
    return 1 if (explicit and refused) else 0


def cmd_check() -> int:
    schemas = build_offline_schemas()
    mapping = load_widget_mapping()
    canonical = _load_canonical()
    failures = []

    # The retired otr_api stale-variant soft-skip must STAY dead.
    api_src = (REPO / "scripts" / "otr_api.py").read_text(encoding="utf-8")
    if "trimming the 3 EMPTY stamp slots" in api_src or \
            "[wv[0], False, wv[2]]" in api_src:
        failures.append("otr_api.py: the stale-variant soft-skip TRIM "
                        "path is back (must stay a hard fail)")

    # The paired <variant>.env.json recipe-knob files also match otr_*.json but are
    # NOT variants -- exclude them (video-tiers 2026-07-20).
    committed = sorted(p for p in VARIANTS_DIR.glob("otr_*.json")
                       if not p.name.endswith(".env.json"))
    if not committed:
        print("check: no committed variants yet (nothing to diff); "
              "soft-skip guard " +
              ("FAILED" if failures else "OK"))
        return 1 if failures else 0
    for vpath in committed:
        pid = _profile_id_from_stem(vpath.stem)
        try:
            regen, rel, recipe = build_variant(
                pid, schemas=schemas, mapping=mapping, canonical=canonical)
        except EmitRefused as e:
            failures.append(f"{vpath.name}: committed variant exists but "
                            f"regeneration is refused ({e})")
            continue
        disk = vpath.read_text(encoding="utf-8")
        if disk != _dump(regen):
            failures.append(f"{vpath.name}: DRIFT vs regeneration "
                            "(variants are generated, never hand-edited)")
        rpath = VARIANTS_DIR / f"{_variant_stem(pid)}.launch.md"
        if not rpath.is_file():
            failures.append(f"{rpath.name}: launch recipe missing")
        elif rpath.read_text(encoding="utf-8") != recipe:
            failures.append(f"{rpath.name}: recipe DRIFT vs regeneration")
        # Stamp agreement on the DISK variant -- resolved BY NAME against
        # the live validator schema (post-ship audit: positional wv[3..5]
        # was the exact drift class the widget rules exist to prevent).
        from nodes._otr_workflow_apply import serialized_slot_names
        wf = json.loads(disk)
        nid = _validator_node_id(wf)
        vnode = next(n for n in wf["nodes"] if int(n["id"]) == nid)
        wv = vnode.get("widgets_values") or []
        slots = serialized_slot_names("OTR_WorkflowValidator", schemas)
        live_hash = semantic_master_hash(wf, mapping=mapping,
                                         schemas=schemas)
        try:
            stamped = {name: wv[slots.index(name)]
                       for name in ("profile_id", "master_hash",
                                    "generated_by")}
        except (ValueError, IndexError):
            stamped = {}
        if (stamped.get("profile_id") != pid
                or stamped.get("master_hash") != live_hash
                or stamped.get("generated_by") != GENERATED_BY):
            failures.append(f"{vpath.name}: stamp disagreement "
                            f"(profile_id/master_hash/generated_by)")
    for f in failures:
        print("CHECK FAIL:", f)
    print(f"check: {len(committed)} variants, {len(failures)} failures")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="emit every committed non-lane-preset profile "
                        "(ratify-gated ones are reported + skipped)")
    g.add_argument("--profiles", help="comma-separated profile ids "
                                      "(ratify-gated => exit 1)")
    g.add_argument("--check", action="store_true",
                   help="regenerate + diff committed variants/recipes; "
                        "nonzero on drift")
    args = ap.parse_args(argv)
    if args.check:
        return cmd_check()
    if args.all:
        ids = [p for p in _committed_profile_ids() if p not in LANE_PRESETS]
        return cmd_emit(ids, explicit=False)
    return cmd_emit([s.strip() for s in args.profiles.split(",") if
                     s.strip()], explicit=True)


if __name__ == "__main__":
    sys.exit(main())
