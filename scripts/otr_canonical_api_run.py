"""Build and optionally submit the canonical OTR workflow through ComfyUI API.

This is the small, boring headless entrypoint agents should use when the
operator asks for an API render/smoke:

* Always load workflows/otr_canonical.json from this repo.
* Apply engine/dropdown capability profiles through the single profile applier.
* Patch only creative/story widgets directly, through otr_api.patch_creative.
* Convert with scripts/otr_api.py and fail loud on schema/widget drift.

It deliberately has no --workflow argument. If a run needs a different graph,
that is a workflow change, not a headless smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_canonical.json"
DEFAULT_PROMPT_DUMP = HERE / "_otr_canonical_api_prompt.json"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from otr_api import (  # noqa: E402
    COMFYUI_URL,
    apply_profile_to_workflow,
    fetch_schemas,
    load_workflow,
    patch_creative,
    poll_history,
    queue_snapshot,
    submit_prompt,
    workflow_to_api_prompt,
)


def _parse_value(raw: str) -> Any:
    """Parse CLI values as JSON when possible; otherwise keep a string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _node_id_for(workflow: dict, node_ref: str) -> int:
    """Resolve a node id or unique node type in the canonical workflow."""
    if node_ref.isdigit():
        nid = int(node_ref)
        if any(n.get("id") == nid for n in workflow.get("nodes", [])):
            return nid
        raise SystemExit(f"node id {nid} is not in the canonical workflow")

    matches = [
        int(n["id"])
        for n in workflow.get("nodes", [])
        if n.get("type") == node_ref
    ]
    if not matches:
        raise SystemExit(
            f"node type {node_ref!r} is not in the canonical workflow"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"node type {node_ref!r} appears {len(matches)} times; use id.widget"
        )
    return matches[0]


def _apply_set(workflow: dict, schemas: dict, patch: str) -> str:
    """Apply NODE_OR_ID.widget=value through the creative whitelist."""
    if "=" not in patch:
        raise SystemExit(f"--set must look like NODE.widget=value, got {patch!r}")
    left, raw_value = patch.split("=", 1)
    if "." not in left:
        raise SystemExit(f"--set must name NODE.widget, got {patch!r}")
    node_ref, widget = left.rsplit(".", 1)
    nid = _node_id_for(workflow, node_ref)
    value = _parse_value(raw_value)
    patch_creative(workflow, nid, widget, value, schemas)
    return f"{node_ref}.{widget}={value!r}"


def _apply_writer_shortcuts(workflow: dict, schemas: dict, args) -> list[str]:
    writer_id = _node_id_for(workflow, "OTR_LedgerScriptWriter")
    shortcuts: list[tuple[str, Any]] = []
    # `--words` was removed 2026-08-14 with the target_words widget. Episode
    # shape is set with --act-count below; length is an observation.
    if args.title is not None:
        shortcuts.append(("episode_title", args.title))
    if args.premise is not None:
        shortcuts.append(("custom_premise", args.premise))
    if args.source_bank is not None:
        shortcuts.append(("source_bank", args.source_bank))
    if args.visual_style is not None:
        shortcuts.append(("visual_style", args.visual_style))
    if args.creative_model is not None:
        shortcuts.append(("creative_writing_model", args.creative_model))
    if args.technical_model is not None:
        shortcuts.append(("technical_model", args.technical_model))
    if args.google_slot_a_model is not None:
        shortcuts.append(("google_api_slot_a_model", args.google_slot_a_model))
    if args.google_slot_b_model is not None:
        shortcuts.append(("google_api_slot_b_model", args.google_slot_b_model))
    if args.num_characters is not None:
        shortcuts.append(("num_characters", int(args.num_characters)))
    if args.act_count is not None:
        shortcuts.append(("act_count", str(args.act_count)))

    applied = []
    for widget, value in shortcuts:
        patch_creative(workflow, writer_id, widget, value, schemas)
        applied.append(f"OTR_LedgerScriptWriter.{widget}={value!r}")
    return applied


def _schemas(offline: bool) -> dict:
    if not offline:
        return fetch_schemas()
    from nodes import _otr_workflow_apply as workflow_apply

    return workflow_apply.build_offline_schemas()


def build_api_prompt(args) -> tuple[dict, list[str]]:
    # Default: the canonical workflow, with the deliberate path assertion.
    # Opt-in --workflow loads an EXPLICIT graph (the story-only scoring graph),
    # which is a workflow change the caller has deliberately asked for -- not a
    # silent smoke-vs-canonical drift. When --workflow is absent the behaviour is
    # byte-identical to before.
    explicit = getattr(args, "workflow", None)
    if explicit:
        wf_path = pathlib.Path(explicit).resolve()
        if not wf_path.is_file():
            raise SystemExit(f"--workflow missing: {wf_path}")
        print(f"[canonical-api] workflow={wf_path} (explicit --workflow)", flush=True)
    else:
        wf_path = CANONICAL_WORKFLOW.resolve()
        expected = (REPO_ROOT / "workflows" / "otr_canonical.json").resolve()
        if wf_path != expected:
            raise SystemExit(f"canonical workflow path mismatch: {wf_path}")
        if not wf_path.is_file():
            raise SystemExit(f"canonical workflow missing: {wf_path}")
        print(f"[canonical-api] workflow={wf_path}", flush=True)
    schemas = _schemas(args.offline_schemas)
    workflow = load_workflow(str(wf_path))

    applied: list[str] = []
    if getattr(args, "machine", None):
        # A machine key expands to the same dict a profile file would have
        # produced. No file, no second applier -- config/machine_classes.json
        # is the only place any of these values exist.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from otr_machine_profile import build_profile, resolve
        _prof = build_profile(resolve(args.machine))
        workflow = apply_profile_to_workflow(workflow, _prof, schemas)
        applied.append("machine=%s" % args.machine)
    elif args.profile and args.profile.lower() != "none":
        workflow = apply_profile_to_workflow(workflow, args.profile, schemas)
        applied.append(f"profile={args.profile}")
        checked = _assert_profile_models_present(
            args.profile, schemas, offline=bool(args.offline_schemas))
        if checked:
            print("[canonical-api] preflight: %d required model(s) visible to "
                  "the server: %s" % (len(checked), ", ".join(checked)),
                  flush=True)

    applied.extend(_apply_writer_shortcuts(workflow, schemas, args))
    for patch in args.set:
        applied.append(_apply_set(workflow, schemas, patch))

    prompt = workflow_to_api_prompt(workflow, schemas)
    return prompt, applied


def _server_visible_model_names(schemas) -> set:
    """Every filename the SERVER offers in any node's combo, from /object_info.

    Deliberately not folder_paths: this asks the running server what it can
    actually see, which is the only authority that matters, and it needs no
    knowledge of extra_model_paths.yaml or of which category a weight lives in.
    """
    names = set()

    def _walk(node):
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            # A ComfyUI combo is a list whose FIRST element is the option list.
            for value in node:
                if isinstance(value, str):
                    names.add(value)
                else:
                    _walk(value)

    _walk(schemas)
    return names


#: Weight-file suffixes ComfyUI's folder_paths actually enumerates. A
#: `preflight.required_models` entry ending in one of these is a FILENAME and is
#: checkable against /object_info; anything else is a logical or repo id and is
#: not. Deliberately a closed list rather than "contains a dot": `wan2.2-ti2v-5b`
#: and `ltx-2.3-22b-dev` both contain dots and are ids, not files.
_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".gguf",
                    ".onnx", ".sft")


def _is_weight_filename(name: str) -> bool:
    """True for a real weight FILENAME, false for a logical or HF repo id."""
    return str(name or "").lower().endswith(_WEIGHT_SUFFIXES)


def classify_timeout(running: int, pending: int) -> str:
    """What a poll TIMEOUT actually means, given the server's queue.

    Pure so it can be tested without a server. Three outcomes, and conflating
    them is the defect this exists to prevent (2026-08-23):

      ``still_running``  the observation window closed while the render carried
                         on. `--timeout` defaults to 5400s and a full wan_ti2v
                         episode on the 16 GB box exceeds it, so this is the
                         COMMON case for the slowest lane -- and it is not a
                         failure at all. The episode still publishes.
      ``unknown``        the queue could not be read (``queue_snapshot`` returns
                         -1/-1 best-effort). Absence of evidence, reported as
                         such rather than guessed either way.
      ``not_running``    the queue is empty, so the render really has ended.

    Before this, all three printed "RESULT TIMEOUT" and exited 1, which reads as
    "the render died" -- the wrong conclusion in the most common case.
    """
    if running == -1 or pending == -1:
        return "unknown"
    if running > 0 or pending > 0:
        return "still_running"
    return "not_running"


def _assert_profile_models_present(profile_name, schemas, offline=False) -> list:
    """Refuse in SECONDS what would otherwise fail seven minutes into a render.

    On 2026-08-22 a Ghost domain-adapter leg ran the script pass, the whole
    voice pass and part of the video pass before ``assert_usable`` reported
    ``v3_sd15_adapter.ckpt`` missing -- the weight was on disk but under a root
    the headless model-paths config does not name. The engine guard did its job
    and failed closed; it simply could not do it until the first video beat.
    The bench runners already confirm every model filename in /object_info
    before submit (SPEC G1/O6); the canonical runner did not, so it does now.

    Returns the checked names. A profile with no ``preflight.required_models``
    is not an error -- most profiles do not declare any.
    """
    import json as _json

    path = REPO_ROOT / "config" / "profiles" / ("%s.json" % profile_name)
    if not path.is_file():
        return []
    try:
        profile = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- a malformed profile fails later, louder
        return []
    required = ((profile.get("preflight") or {}).get("required_models") or [])
    if not required:
        return []
    # ONLY A FILENAME IS GATEABLE HERE, and this split is the whole fix
    # (2026-08-23). `/object_info` lists what ComfyUI's folder_paths enumerates:
    # FILENAMES. It has never contained a logical model id (`real-esrgan-x2plus`,
    # `wan2.2-ti2v-5b`) or an HF repo id (`google/gemma-4-E2B-it`), and an LLM
    # weight is not in it at all. So for a non-filename entry, "absent from
    # /object_info" is not evidence of absence on disk -- it is the wrong
    # question, and answering it with SystemExit blocks a profile whose weights
    # are sitting right there.
    #
    # THIS WAS NOT THEORETICAL. The gate shipped 2026-08-22 validated against
    # the ghost_signal profiles, which are the only ones declaring filenames.
    # Every profile using the logical-id vocabulary -- both upscale profiles,
    # both 16gb_ltx, all three 8gb_*, both 4060_* -- could not pass it at any
    # time, for any state of the disk. `otr_upscale_ship` was carried in the
    # queue as "unexercised" for exactly this reason: `RealESRGAN_x2plus.pth`
    # was visible in `/object_info` the entire time.
    #
    # A filename is enforced (that is what caught the missing v3_sd15_adapter).
    # Anything else is REPORTED and allowed through, because a gate that cannot
    # verify a claim must not pretend it refuted it.
    gateable = [n for n in required if _is_weight_filename(n)]
    ungateable = [n for n in required if not _is_weight_filename(n)]
    visible = _server_visible_model_names(schemas)
    missing = [name for name in gateable if name not in visible]
    if missing and offline:
        # OFFLINE SCHEMAS ARE NOT A SERVER, so this gate cannot fire here --
        # the same rule the filename/logical-id split above is built on: "a
        # gate that cannot verify a claim must not pretend it refuted it."
        # `build_offline_schemas()` synthesizes widget choices from the node
        # classes; it never had `--extra-model-paths-config` applied, so its
        # model lists reflect THIS process's folder_paths, not the roots the
        # real server booted with. Refusing on that evidence blocks a profile
        # whose weights are on disk and visible to the server that will
        # actually run it.
        #
        # NOT THEORETICAL (2026-08-26): `otr_ghost_signal_v3_haunted` was
        # reported FAIL by a `--dry-run --offline-schemas` validation sweep --
        # all three names, including two sitting under roots the yaml DOES
        # name -- and passed preflight cleanly against the live server
        # moments later. The old message asserted "the running server cannot
        # see", a claim it had not checked and, offline, could not.
        print(
            "[canonical-api] preflight: %d filename requirement(s) NOT checked "
            "-- %s. --offline-schemas has no server and no model-paths config, "
            "so absence here is not evidence of absence on disk. Re-run without "
            "--offline-schemas to gate these against the real server."
            % (len(missing), ", ".join(missing)), flush=True)
    elif missing:
        raise SystemExit(
            "[canonical-api] PREFLIGHT FAIL: profile %r requires model file(s) "
            "the running server cannot see: %s.\nThe weight may be on disk "
            "under a root this server was not started with -- check "
            "scripts/_otr_headless_model_paths.yaml, drop the file under a "
            "named root, and RESTART the server (folder_paths caches its "
            "listing at boot). Refusing now rather than failing part-way "
            "through a render." % (profile_name, ", ".join(missing)))
    if ungateable:
        print(
            "[canonical-api] preflight: %d requirement(s) NOT checked -- %s. "
            "These are logical/repo ids, not filenames, so /object_info cannot "
            "speak to them; the engine's own assert_usable remains the gate for "
            "these. Declare a weight FILENAME to get it checked here."
            % (len(ungateable), ", ".join(ungateable)), flush=True)
    # The return value is "what this gate actually VERIFIED", and the caller
    # prints it as "visible to the server". Offline there is no server and
    # nothing was verified, so claiming otherwise would contradict the notice
    # printed above it in the same run.
    return [] if offline else list(gateable)


def main(argv: list[str] | None = None) -> int:
    global COMFYUI_URL
    parser = argparse.ArgumentParser(
        description="Run the canonical OTR workflow through the ComfyUI API."
    )
    parser.add_argument(
        "--comfyui-url", default=None,
        help="running ComfyUI base URL (Desktop normally "
             "http://127.0.0.1:8188; env/default remains available)",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--profile", default="none",
        help="explicit capability profile id, e.g. none or otr_cloud_lanes",
    )
    selector.add_argument(
        "--machine", default=None,
        help="a machine key from config/machine_classes.json "
             "(8gb, 12gb, 16gb, amd). Expands to settings IN MEMORY -- "
             "there is no per-machine profile file.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--premise", default=None)
    parser.add_argument("--source-bank", default=None)
    parser.add_argument("--visual-style", default=None)
    parser.add_argument("--creative-model", default=None)
    parser.add_argument("--technical-model", default=None)
    parser.add_argument("--google-slot-a-model", default=None)
    parser.add_argument("--google-slot-b-model", default=None)
    parser.add_argument("--num-characters", type=int, default=None)
    parser.add_argument("--act-count", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NODE_OR_ID.widget=value",
        help="direct creative patch; managed engine widgets are refused",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="build and dump the API prompt without POST /prompt")
    parser.add_argument("--offline-schemas", action="store_true",
                        help="use offline node schemas; for unit tests only")
    parser.add_argument("--workflow", default=None,
                        help="explicit workflow JSON path (opt-in; default = the "
                             "canonical graph). Used for the story-only scoring "
                             "graph (writer+freeze, no media).")
    parser.add_argument("--dump-prompt", default=str(DEFAULT_PROMPT_DUMP))
    parser.add_argument(
        "--timeout",
        type=int,
        default=5400,
        help="history observation timeout in seconds; 0 waits until terminal",
    )
    parser.add_argument("--poll-s", type=int, default=5)
    args = parser.parse_args(argv)

    if args.comfyui_url is not None:
        candidate = str(args.comfyui_url).strip().rstrip("/")
        if not candidate.startswith(("http://", "https://")):
            raise SystemExit("--comfyui-url must start with http:// or https://")
        import otr_api as _otr_api
        _otr_api.COMFYUI_URL = candidate
        COMFYUI_URL = candidate

    prompt, applied = build_api_prompt(args)
    dump_path = pathlib.Path(args.dump_prompt)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")

    print(f"[canonical-api] comfy_url={COMFYUI_URL}", flush=True)
    print(f"[canonical-api] prompt_nodes={len(prompt)}", flush=True)
    print(f"[canonical-api] prompt_dump={dump_path}", flush=True)
    if applied:
        print("[canonical-api] applied:", flush=True)
        for item in applied:
            print(f"  {item}", flush=True)

    if args.dry_run:
        print("[canonical-api] DRY_RUN complete; prompt not submitted", flush=True)
        return 0

    prompt_id = submit_prompt(prompt)
    print(f"[canonical-api] QUEUED prompt_id={prompt_id}", flush=True)
    print(
        f"[canonical-api] t=0s prompt_id={prompt_id} status=queued",
        flush=True,
    )

    def heartbeat(elapsed_s: float, status: dict) -> None:
        phase = str(status.get("status_str") or "pending")
        print(
            f"[canonical-api] t={int(elapsed_s)}s prompt_id={prompt_id} "
            f"status={phase}",
            flush=True,
        )

    status, err = poll_history(
        prompt_id, timeout_s=args.timeout, poll_s=args.poll_s,
        on_tick=heartbeat,
    )
    print(f"[canonical-api] RESULT {status} prompt_id={prompt_id}", flush=True)
    if status == "TIMEOUT":
        # A TIMEOUT HERE IS ABOUT THIS PROCESS, NOT ABOUT THE RENDER, and saying
        # so is the whole point of this branch (2026-08-23). `--timeout`
        # defaults to 5400s; a full wan_ti2v episode on the 16 GB box exceeds
        # that, so the observation window closes while the server is still at
        # 98% GPU happily rendering beat 34. The old line printed
        # "RESULT TIMEOUT" and exited 1 for BOTH that case and a genuinely dead
        # render -- two opposite situations, one indistinguishable message, and
        # the reader's natural conclusion is the wrong one.
        running, pending = queue_snapshot()
        verdict = classify_timeout(running, pending)
        if verdict == "still_running":
            print(
                "[canonical-api] ...BUT THE RENDER IS STILL ALIVE: the server "
                f"reports {running} running / {pending} pending. This process "
                "stopped WATCHING; it did not stop the render, and the episode "
                "should still publish to otr/obs on its own. Re-run with "
                "`--timeout 0` to wait for a terminal result (the documented "
                "operator mode for long lanes), or watch the server log.",
                flush=True)
        elif verdict == "unknown":
            print(
                "[canonical-api] and the queue could not be read, so whether "
                "the render survived is UNKNOWN -- check the server log before "
                "concluding anything.", flush=True)
        else:
            print(
                "[canonical-api] and the queue is EMPTY, so the render is no "
                "longer running. Check the server log for how it ended.",
                flush=True)
    if status != "SUCCESS":
        if err:
            print(f"[canonical-api] ERROR {err}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
