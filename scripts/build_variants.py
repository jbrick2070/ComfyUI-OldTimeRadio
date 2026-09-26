"""Platform-variant generator -- the never-built GATE B "S3 emit_snapshot".

Per the 2026-07-09 platform-portability final spec, section 1. An OFFLINE CLI
(never a node): a platform variant = TWO artifacts generated together from
the ONE canonical graph + a committed capability profile:

  1. workflows/otr_<profile_id>.json  -- apply_profile(canonical,
     profile) + stamps written into OTR_WorkflowValidator's EXISTING
     widgets (profile_id / master_hash / generated_by / its own path).
  2. its section in apple/LAUNCH_RECIPES.md -- the launch recipe
     from the SAME validated profile object (args, sage flag, env, key
     NAMES only -- values are never stored -- and install pointers). One
     generated doc holds every recipe, so workflows/ holds only graphs.

Self-check on every emit: apply -> hash -> stamp -> RE-apply -> the
semantic hash must not move (a managed widget the applier cannot
reproduce refuses the emit). ``--check`` regenerates every committed
variant AND the recipes doc in memory, diffs both against disk, verifies the
stamps agree, asserts the retired otr_api stale-variant soft-skip stays
dead, and exits nonzero on ANY drift (CI rule: variants are GENERATED,
never hand-edited).

REFUSES emission while a profile carries ratify_before_emit entries --
the operator ratifies each named decision and clears the list first.

Usage:
  python scripts/build_variants.py --all
  python scripts/build_variants.py --profiles otr_8gb_low,otr_16gb_low
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
    load_profile, shipping_ids,
)
from nodes._otr_shared.boot_contracts import (  # noqa: E402
    contract_for_profile, launch_args_for,
)
from nodes._otr_workflow_apply import (  # noqa: E402
    apply_profile, build_offline_schemas, load_widget_mapping,
    patch_widget_by_name, semantic_master_hash,
)

CANONICAL = REPO / "workflows" / "otr_canonical.json"
#: THE VARIANTS LIVE BESIDE THE CANONICAL, IN workflows/ ITSELF (operator,
#: 2026-09-25: "we can't store the variants in a subfolder"). ComfyUI's template
#: gallery globs `*/workflows/*.json` ONE level deep, so `workflows/variants/`
#: shipped 24 graphs no user could find from the menu. One folder is still the
#: rule that matters -- the 2026-09-01 silent 404 came from a SECOND
#: template-named folder, not from how many graphs one folder holds
#: (tests/test_workflow_templates_single_folder.py).
VARIANTS_DIR = REPO / "workflows"
#: The retired subfolder. `--check` fails if it reappears with a graph in it.
RETIRED_VARIANTS_DIR = REPO / "workflows" / "variants"
GENERATED_BY = "scripts/build_variants.py"
#: EVERY LAUNCH RECIPE IN ONE GENERATED DOC (2026-09-25). They used to sit
#: beside each graph as workflows/<variant>.launch.md: 24 near-identical files in
#: the folder ComfyUI's template gallery reads, which is a folder for loadable
#: graphs. The doc lives with the rest of the docs, and `--check` fails if a
#: `.launch.md` comes back into workflows/.
LAUNCH_RECIPES = REPO / "apple" / "LAUNCH_RECIPES.md"
#: THE GALLERY THUMBNAIL (2026-09-25). ComfyUI's template gallery builds each
#: custom-pack card with `mediaSubtype: "jpg"` hardcoded and requests
#: /api/workflow_templates/<pack>/<stem>.jpg, so every shipped graph needs its
#: own same-stem JPEG or its card is blank. One image, the operator's own art
#: (assets/otr_gallery_thumb_master.webp, reduced to this 400x400 JPEG), is
#: copied beside every graph; `--check` fails on a missing, stale or orphaned
#: copy. assets/ does not ship; the copies in workflows/ do.
GALLERY_THUMB = REPO / "assets" / "otr_gallery_thumb.jpg"

#: THE SHIPPING SET, DERIVED FROM THE MATRIX (2026-09-24). The rows in
#: `config/workflow_matrix.json` that say `ships` are the only configs that emit a
#: graph into `workflows/`. Edit that file and this follows; there is no
#: second list to keep in step, which is the entire reason it moved out of here.
#:
#: Still an allow-list, deliberately, exactly as the hand-kept tuple was: a row
#: has to say `ships` to reach a user, so a new row defaults to NOT shipping --
#: the safe direction to be wrong in.
#:
#: Kept as a module attribute because two readers import it by name:
#: `otr_dropdown_matrix` and `tests/test_shipping_writer_pins`.
#: `otr_tier_matrix` was the third and was retired 2026-09-24 -- its doc
#: held per-workflow configuration, which the matrix itself now owns.
SHIPPING_SET = shipping_ids()

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
    """Inverse of _variant_stem against the matrix rows.

    A stem that names no row is returned unchanged, so `--check` pointed at a
    stray graph still names the file it came from.
    """
    known = set(shipping_ids())
    if stem in known:
        return stem
    bare = stem[len("otr_"):] if stem.startswith("otr_") else stem
    if bare != stem and bare in known:
        return bare
    return stem


def _committed_profile_ids() -> list[str]:
    """The ids `--all` emits: exactly what the matrix says ships.

    THIS USED TO GLOB `config/profiles/*.json` AND INTERSECT WITH SHIPPING_SET,
    which made the FOLDER decide what got emitted while the matrix only decided
    what was allowed to. Two silent failures came out of that: a new matrix row
    with no file twin was never emitted, and an emptied folder made `--all` emit
    nothing and still exit 0. `--check` hid it, because it enumerates the
    already-committed graphs in `workflows/` and never looks at the
    source folder at all.

    `shipping_ids()` reads the matrix, so the enumeration and the allow-list are
    now the same list rather than two that agreed by coincidence.
    """
    out = list(shipping_ids())
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
    variant_rel = f"workflows/{_variant_stem(profile_id)}.json"
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
    args = launch_args_for(contract_for_profile(profile))
    env = dict(launch.get("env") or {})

    if backend == "cuda" and vendor == "amd":
        torch_note = ("torch ROCm build (https://pytorch.org rocm wheels); "
                      "presents as cuda. bnb/fp8/sage lanes are OFF.")
    elif backend == "cuda":
        torch_note = "torch cu128+ build (the nv baseline is 2.10/cu130)"
    elif backend == "mps":
        torch_note = "default PyPI torch wheels (Metal/MPS included)"
    else:
        torch_note = "CPU torch wheels (--index-url .../cpu)"

    lines = [
        f"## {profile_id}",
        "",
        f"- graph: `{variant_rel}`",
        f"- workflow: `{profile_id}` ({profile.get('display_name')})",
        f"- status: {profile.get('status')}",
        f"- platform/backend/vendor: {profile.get('platform')}/"
        f"{backend}/{vendor}",
        f"- master_hash: `{master_hash}`",
    ]
    lines += [
        "",
        "### ComfyUI launch",
        "",
        f"- args: `{' '.join(args) if args else '(none)'}`",
        f"- sage_attention: {bool(launch.get('sage_attention'))}",
        "",
        "### Environment",
        "",
        "- Windows hosts: set `PYTHONUTF8=1` (cp1252 consoles crash the "
        "prestartup banner otherwise).",
        "- Models root override: `OTR_COMFYUI_MODELS_ROOT` (the "
        "`C:\\ComfyUI-Models` default in _otr_hf_env/_otr_models_root "
        "is a Windows-only convenience).",
    ]
    for k, v in sorted(env.items()):
        lines.append(f"- `{k}={v}`")
    # THE SHIPPED MUSIC DEFAULTS, read from the palette so this recipe cannot
    # drift from the composer (2026-09-12: four banks gained a fixed genre
    # and a one-bank widget, and no user-facing file said so).
    from nodes._otr_music_palette import bank_music_table  # noqa: E402
    lines += [
        "",
        "### Music (per bank; only My Story takes its own)",
        "",
    ]
    # A RHYTHMIC palette is the bank's fixed identity whatever the source's
    # year (`story_palette`: a declared genre beats the period band). A
    # non-rhythmic one is only the bank's default when the source carries no
    # year; otherwise the period band decides -- and for Shakespeare every
    # play predates the first cutover, so it lands on the same consort.
    for bank, idiom, rhythmic in bank_music_table():
        if rhythmic:
            lines.append(f"- `{bank}`: {idiom} (fixed)")
        else:
            lines.append(f"- `{bank}`: chosen by the source's year; "
                         f"its own default is {idiom}")
    lines += [
        "- `my_story`: the `music_style` widget on `OTR_StableAudioTheme` "
        "(blank = the house radio orchestra). Every other bank keeps its "
        "genre whatever is typed.",
    ]
    lines += [
        "",
        "### Required key NAMES (values are NEVER stored here)",
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
        if "OTR_COMFY_API_KEY" in keys:
            lines.append("- `OTR_COMFY_API_KEY` is read by the headless "
                         "SUBMITTER (`scripts/otr_api.py`) and sent as "
                         "`extra_data.api_key_comfy_org`; the ComfyUI server "
                         "never reads it. In the app, sign into Comfy instead.")
    else:
        lines.append("- (none)")
    lines += [
        "",
        "### Install pointers",
        "",
        f"- {torch_note}",
        "- ffmpeg on PATH (mac: ensure libx264 + aac encoders are in the "
        "build).",
        # Named ONLY when this profile actually selects the one engine that
        # imports cairo (2026-09-12). Every recipe used to carry the line,
        # including profiles pinned to the cairo-free viz_mxc_cpu -- an
        # install step a tester would have paid for nothing.
        *(["- minimal Linux: libcairo2-dev + pkg-config, then `pip install "
           "pycairo` (this graph selects `viz_mxc_mandala`)."]
          if "viz_mxc_mandala" in json.dumps(profile.get("role_overrides") or {})
          else []),
        "",
        "### Preflight models",
        "",
    ]
    models = list(preflight.get("required_models") or [])
    if models:
        lines += [f"- {m}" for m in models]
    else:
        lines.append("- (registry-driven; see the engine rows for the "
                     "selected lanes)")
    return "\n".join(lines) + "\n"


def render_launch_recipes(recipes: list[tuple[str, str]]) -> str:
    """The whole apple/LAUNCH_RECIPES.md from (profile_id, recipe) pairs."""
    ordered = sorted(recipes)
    lines = [
        "# Launch recipes",
        "",
        f"GENERATED by {GENERATED_BY} from `config/workflow_matrix.json` -- do "
        "not hand-edit. Run `python scripts/build_variants.py --all` to rebuild "
        "it; `--check` fails when it drifts.",
        "",
        "One section per per-machine graph in `workflows/`: the ComfyUI launch "
        "arguments, environment, music per bank, required key NAMES and install "
        "pointers for running that graph headless from the git clone. "
        "`workflows/otr_canonical.json` runs on any machine and needs none of "
        "this; `apple/MACHINES.md` says which graph fits which card.",
        "",
        "## Contents",
        "",
    ]
    lines += [f"- [{pid}](#{pid})" for pid, _ in ordered]
    body = "\n".join(lines) + "\n"
    for _pid, recipe in ordered:
        body += "\n" + recipe
    return body


def _committed_variant_paths() -> list[Path]:
    # The paired <variant>.env.json recipe-knob files also match otr_*.json but are
    # NOT variants -- exclude them (video-tiers 2026-07-20).
    # The canonical lives in the same folder and is the SOURCE, not a variant.
    return sorted(p for p in VARIANTS_DIR.glob("otr_*.json")
                  if not p.name.endswith(".env.json")
                  and p.name != CANONICAL.name)


def _committed_recipes(schemas, mapping, canonical) -> list[tuple[str, str]]:
    """A recipe for every committed variant graph. A refused row is left out
    here and reported by `--check`, so emit and check agree on the doc."""
    out = []
    for vpath in _committed_variant_paths():
        pid = _profile_id_from_stem(vpath.stem)
        try:
            _variant, _rel, recipe = build_variant(
                pid, schemas=schemas, mapping=mapping, canonical=canonical)
        except EmitRefused:
            continue
        out.append((pid, recipe))
    return out


def _thumbnail_targets() -> list[Path]:
    """`<stem>.jpg` beside the canonical and every committed variant."""
    return ([VARIANTS_DIR / (CANONICAL.stem + ".jpg")]
            + [path.with_suffix(".jpg") for path in _committed_variant_paths()])


def _write_thumbnails() -> None:
    data = GALLERY_THUMB.read_bytes()
    for target in _thumbnail_targets():
        target.write_bytes(data)
    print(f"WROTE {len(_thumbnail_targets())} gallery thumbnails")


def _thumbnail_failures() -> list[str]:
    if not GALLERY_THUMB.is_file():
        return [f"{GALLERY_THUMB.name}: the gallery thumbnail master is missing"]
    master = GALLERY_THUMB.read_bytes()
    failures = []
    targets = _thumbnail_targets()
    for target in targets:
        if not target.is_file():
            failures.append(f"{target.name}: gallery thumbnail missing (run --all)")
        elif target.read_bytes() != master:
            failures.append(f"{target.name}: gallery thumbnail differs from "
                            f"{GALLERY_THUMB.name} (run --all)")
    wanted = {t.name for t in targets}
    for stray in sorted(VARIANTS_DIR.glob("*.jpg")):
        if stray.name not in wanted:
            failures.append(f"{stray.name}: a gallery thumbnail with no graph "
                            "of that name (delete it)")
    return failures


def _write_launch_recipes(schemas, mapping, canonical) -> None:
    # newline="\n": platform-safe by construction (Windows universal newlines
    # otherwise bake CRLF into a locally regenerated doc).
    LAUNCH_RECIPES.write_text(
        render_launch_recipes(_committed_recipes(schemas, mapping, canonical)),
        encoding="utf-8", newline="\n")
    print(f"WROTE {LAUNCH_RECIPES.name}")


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
        emitted.append(rel)
        print(f"EMITTED {rel}")
    for pid, why in refused:
        print(f"REFUSED {pid}:\n{why}\n")
    if emitted:
        _write_launch_recipes(schemas, mapping, canonical)
        _write_thumbnails()
    print(f"done: {len(emitted)} emitted, {len(refused)} refused")
    # Explicitly requesting a refused profile is an error; --all treats
    # refusals as the expected pre-ratification state.
    return 1 if (explicit and refused) else 0



#: Generated docs rebuilt alongside the variants, so one edit needs one command.
#: Run as SUBPROCESSES on purpose -- see `cmd_regenerate_docs`.
DOC_GENERATORS = (
    # No README block: both generators only STRIP the old BEGIN/END markers and
    # neither re-injects, and README.md carries no marker any more. The earlier
    # label here claimed one and was written from an assumption.
    ("apple/MACHINE_MATRIX.md", "otr_machine_matrix.py"),
    ("apple/DROPDOWN_MATRIX.md + apple/MACHINES.md", "otr_dropdown_matrix.py"),
)


def cmd_regenerate_docs() -> int:
    """Rebuild every generated doc from the same matrix the variants came from.

    Operator: "when it's updated it updates the variants AND the documentation, all
    at once". Before this, a matrix edit meant remembering three commands, and a
    forgotten one left a doc disagreeing with the graphs until somebody noticed --
    which is exactly how the retired tier doc froze four workflows behind.

    SUBPROCESSES, NOT IMPORTS, for three measured reasons. The generators use
    incompatible exit codes (3 means "degraded interpreter, refuse to write" for the
    machine matrix; 2 means "engine-row conflict" for the dropdown one, neither of
    which this module uses). `otr_dropdown_matrix` already imports THIS module for
    SHIPPING_SET, so importing it back is an order-dependent cycle. And
    `otr_machine_matrix` refuses to write under a torch-less interpreter because such
    a run once replaced its voice table with a placeholder and reported success --
    a subprocess keeps that guard rather than importing its failure mode here.

    A DOC FAILURE IS REPORTED, NOT FATAL. The variants are already written and
    correct when this runs; exiting nonzero over an unhappy doc generator would mean
    a red command over a good tree. `--check` is where staleness actually blocks.
    """
    import subprocess

    print("\nregenerating the docs from the same matrix:")
    failures = []
    for label, script in DOC_GENERATORS:
        path = REPO / "scripts" / script
        if not path.exists():
            # A FAILURE, not a note. A renamed or moved generator would otherwise do
            # nothing quietly -- the same shape as the tier generator that sat dead
            # for weeks while its doc froze four workflows behind.
            failures.append((script, "not present"))
            print(f"  FAIL  {label} -- {script} is not present")
            continue
        proc = subprocess.run([sys.executable, str(path)],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"  OK    {label}")
        else:
            failures.append((script, "exit %d" % proc.returncode))
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            why = tail[-1] if tail else "(no output)"
            print(f"  FAIL  {label} -- {script} exit {proc.returncode}: {why}")
    if failures:
        print(f"{len(failures)} doc generator(s) failed; the variants are written "
              f"and correct. Fix the generator, then re-run -- `--check` is the "
              f"gate that blocks on staleness.")
    return 1 if failures else 0

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

    committed = _committed_variant_paths()
    stray = sorted(p.name for p in VARIANTS_DIR.glob("*.launch.md"))
    if stray:
        failures.append(
            f"{len(stray)} .launch.md file(s) are back in workflows/ "
            f"({stray[0]} ...); the recipes live in "
            "apple/LAUNCH_RECIPES.md since 2026-09-25 and workflows/ holds "
            "only graphs")
    if RETIRED_VARIANTS_DIR.is_dir() and any(RETIRED_VARIANTS_DIR.iterdir()):
        failures.append(
            "workflows/variants/ is back with files in it; the variants live "
            "in workflows/ since 2026-09-25, and a graph in the subfolder is "
            "invisible to the template gallery")
    if not committed:
        print("check: no committed variants yet (nothing to diff); "
              "soft-skip guard " +
              ("FAILED" if failures else "OK"))
        return 1 if failures else 0
    recipes = []
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
        recipes.append((pid, recipe))
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
    if not LAUNCH_RECIPES.is_file():
        failures.append(f"{LAUNCH_RECIPES.name}: missing (run --all)")
    elif LAUNCH_RECIPES.read_text(encoding="utf-8") != \
            render_launch_recipes(recipes):
        failures.append(f"{LAUNCH_RECIPES.name}: DRIFT vs regeneration "
                        "(generated, never hand-edited)")
    failures.extend(_thumbnail_failures())
    for f in failures:
        print("CHECK FAIL:", f)
    print(f"check: {len(committed)} variants, {len(failures)} failures")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="emit every shipping matrix row "
                        "(ratify-gated ones are reported + skipped)")
    g.add_argument("--profiles", help="comma-separated profile ids "
                                      "(ratify-gated => exit 1)")
    g.add_argument("--check", action="store_true",
                   help="regenerate + diff committed variants and the recipes doc; "
                        "nonzero on drift")
    ap.add_argument("--no-docs", action="store_true",
                    help="with --all, emit the graphs but skip the generated docs")
    args = ap.parse_args(argv)
    if args.check:
        return cmd_check()
    if args.all:
        ids = _committed_profile_ids()
        rc = cmd_emit(ids, explicit=False)
        if not args.no_docs:
            doc_rc = cmd_regenerate_docs()
            # The graphs are already written and correct, so a doc failure must not
            # turn a good tree red -- but it must not vanish either. Reported through
            # the exit code only when the EMIT itself was clean, so the louder
            # failure still wins.
            if rc == 0:
                rc = doc_rc
        return rc
    return cmd_emit([s.strip() for s in args.profiles.split(",") if
                     s.strip()], explicit=True)


if __name__ == "__main__":
    sys.exit(main())
