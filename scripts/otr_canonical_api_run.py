"""Build and optionally submit the canonical OTR workflow through ComfyUI API.

This is the small, boring headless entrypoint agents should use when the
operator asks for an API render/smoke:

* Load workflows/otr_canonical.json from this repo, or one of its generated
  variants beside it in workflows/ via --workflow (a saved-dropdown
  projection of the same graph, never an ad-hoc one).
* Apply engine/dropdown capability profiles through the single profile applier.
* Patch only creative/story widgets directly, through otr_api.patch_creative.
* Convert with scripts/otr_api.py and fail loud on schema/widget drift -- the
  conversion reads the SERVER's /object_info, so a server carrying an older
  pack than the graph refuses at the widget-count check rather than drifting.

While the leg runs, three display-only helpers report; none of them changes
or gates the render, and a runtime failure in any of them is printed and
ignored:

* LIVE VIEW. The heartbeat line carries ``live="node 7 Rendering the video
  34/96"``, read from ComfyUI's websocket (labels: config/node_phase_labels.json).
  The ``[canonical-api] LIVE`` line after RESULT counts the frames seen.
  ``OTR_PREVIEW_METHOD`` (none | auto | latent2rgb | taesd) asks ComfyUI for
  preview frames for this prompt; unset leaves the server's own setting alone,
  and an unknown value is refused before anything is queued.
  ``--preview-file PATH`` keeps the newest preview frame in PATH (replaced about
  once a second, written atomically) and, unless OTR_PREVIEW_METHOD is set, asks
  for latent2rgb frames. It is a diagnostic for a person to open: a failed
  write is reported on the live snapshot and never fails the leg.
* FAILURE HINTS. A failure line ends with ``-- hint: ...`` when its text
  matches a known platform failure (scripts/otr_known_failures.py).
* PHONE PING. ``OTR_NOTIFY_URL`` (an ntfy topic URL) sends one message when
  the leg ends -- a result, or an exception such as a refused or unreachable
  submission. The start of the run prints ``notify: on -> <host>`` or
  ``notify: off (OTR_NOTIFY_URL unset)``. The topic is a secret (anyone who
  knows an ntfy.sh topic name can read it): use a long random topic or a
  self-hosted ntfy. A failed ping never changes the exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import uuid
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_canonical.json"
DEFAULT_PROMPT_DUMP = HERE / "_otr_canonical_api_prompt.json"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import otr_env_drift  # noqa: E402
import otr_notify  # noqa: E402
from otr_api import (  # noqa: E402
    COMFYUI_URL,
    LiveView,
    apply_row_to_workflow,
    fetch_schemas,
    format_live,
    load_workflow,
    patch_creative,
    patch_widget_by_name,
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
    if widget.strip() == "direction_json":
        # Only --direction-json may set it: that path validates the direction
        # first, and a later --set could otherwise blank or replace a direction
        # the run already reported as applied (Codex QA, 2026-10-22).
        raise SystemExit("--set cannot write direction_json; use --direction-json FILE, "
                         "which validates the direction before the prompt is sent")
    nid = _node_id_for(workflow, node_ref)
    value = _parse_value(raw_value)
    patch_creative(workflow, nid, widget, value, schemas)
    return f"{node_ref}.{widget}={value!r}"


#: The three video dropdowns a person sets in the app, one per role.
VIDEO_LANE_WIDGETS = (
    "announcer_video_model", "character_video_model", "music_video_model")


def _apply_video_lane(workflow: dict, schemas: dict, lane: str) -> list[str]:
    """Set all three video dropdowns to one video lane, as the app does.

    This is ComfyUI's own headless practice: change a widget's value in the
    workflow, then POST it to /prompt -- exactly what picking the dropdown in
    the app does, so no new workflow is needed to try a lane (operator,
    2026-09-26: "hand roll the dropdown for a headless run"). It is the one
    engine widget this runner lets a caller set, and only through the
    dropdown's own label lookup, so the value is always a real menu entry;
    ``--set`` still refuses engine widgets.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nodes._otr_shared import route_freeze
    from nodes._otr_shared.public_engines import resolve_engine_id
    from nodes._otr_visual_assets import VisualAssetError, planned_downloads
    from nodes.otr_video_director import exact_menu_option_for
    engine = resolve_engine_id(str(lane))
    try:
        option = exact_menu_option_for(engine)
    except ValueError as exc:
        raise SystemExit(
            f"--video-lane {lane!r} is not a video lane in the dropdown ({exc})")
    # Say plainly whether anything checks this lane's weights before the run.
    # The queue-time preflight downloads and checks only the lanes the pack
    # fetches itself; for any other lane a missing file surfaces when the lane
    # loads -- the same as picking it in the app (Sonnet review, 9e677825).
    #
    # THE LANES THE RUN WILL USE (4060 cold drill, 2026-09-29): plan the same
    # frozen role map the preflight plans. With no role-based swap
    # (2026-10-15) this is the picked lane unless OTR_FORCE_ENGINE_MAP routes
    # a role elsewhere, and any such routed lane is named.
    effective = route_freeze.freeze_role_engines({w: engine for w in VIDEO_LANE_WIDGETS})
    lanes = {resolve_engine_id(v) for v in effective.values() if v} or {engine}
    routed = sorted(lanes - {engine})
    try:
        fetched = planned_downloads(lanes)
    except VisualAssetError as exc:
        fetched = set()
        print(f"[canonical-api] video lane {lane!r}: the queue-time preflight "
              f"will refuse it ({exc})", flush=True)
    else:
        if fetched:
            also = (" and the lane(s) its other roles are routed to (%s)"
                    % ", ".join(routed)) if routed else ""
            print(f"[canonical-api] video lane {lane!r}: the queue-time "
                  f"preflight downloads and checks {len(fetched)} weight "
                  f"file(s) for it{also}", flush=True)
        else:
            print(f"[canonical-api] WARNING video lane {lane!r}: the pack does "
                  f"not download this lane, so nothing checks its weights "
                  f"before the run; a missing file shows up when the lane "
                  f"loads", flush=True)
    director = _node_id_for(workflow, "OTR_VideoDirector")
    applied = []
    for widget in VIDEO_LANE_WIDGETS:
        patch_widget_by_name(workflow, director, widget, option, schemas)
        applied.append(f"OTR_VideoDirector.{widget}={option!r}")
    return applied


def _apply_writer_shortcuts(workflow: dict, schemas: dict, args) -> list[str]:
    writer_id = _node_id_for(workflow, "OTR_LedgerScriptWriter")
    shortcuts: list[tuple[str, Any]] = []
    # Episode shape is set with --act-count below; length is an observation.
    if args.title is not None:
        shortcuts.append(("episode_title", args.title))
    if getattr(args, "replay_from", None):
        # CANONICAL REPLAY (campaign item 0): a bundle directory, patched as a
        # plain string through the creative whitelist -- never through the
        # --set value parser, where a path that happens to parse as JSON would
        # change type.
        shortcuts.append(("replay_from", str(args.replay_from)))
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


def _apply_direction(workflow: dict, schemas: dict, path: str) -> list[str]:
    """SCENE PRODUCER (2026-10-22): Shot Lock's ``direction_json`` from a file.

    The file is validated with the pack's own schema before anything is sent,
    so a malformed direction fails here with its reason instead of at node 63.
    The text is patched verbatim as a string through the creative whitelist --
    never through the ``--set`` value parser, which would turn JSON into a dict.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nodes._otr_producer_direction import DirectionError, parse_direction
    source = pathlib.Path(path)
    if not source.is_file():
        raise SystemExit(f"--direction-json missing: {source}")
    # BYTES, DECODED ONCE (Sonnet QA, 2026-10-22). read_text turned CRLF into
    # LF, so the widget did not get the file's text verbatim; a BOM was refused
    # as bad JSON; and a UTF-16 file -- what Windows PowerShell's `>` writes --
    # died with a raw UnicodeDecodeError instead of a reason.
    try:
        text = source.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise SystemExit(f"--direction-json {source} is not UTF-8 text; save it as "
                         f"UTF-8 (Windows PowerShell's > writes UTF-16)")
    try:
        direction = parse_direction(text)
    except DirectionError as exc:
        raise SystemExit(f"--direction-json {source}: {exc}")
    if direction is None:
        raise SystemExit(f"--direction-json {source} is empty; leave the option off "
                         f"for an ordinary run")
    lock_id = _node_id_for(workflow, "OTR_ShotLock")
    patch_creative(workflow, lock_id, "direction_json", text, schemas)
    return ["OTR_ShotLock.direction_json=<stage %s, project %s, out %s>"
            % (direction["stage"], direction["project_id"], direction["out"] or "none")]


def _schemas(offline: bool) -> dict:
    if not offline:
        return fetch_schemas()
    from nodes import _otr_workflow_apply as workflow_apply

    return workflow_apply.build_offline_schemas()


def _refuse_video_lane_on_replay(args) -> None:
    """A REPLAY RENDERS ON THE ENGINE ITS BUNDLE NAMES (2026-09-26). The
    frozen plan's shots already carry their engine, and ShotLock reuses that
    plan, so --video-lane only moved dropdowns nothing read: a timing probe
    asked for LTX 2.5 and rendered H3 while this runner printed the new lane.
    The supported way to replay on another lane is a derived bundle."""
    lane = getattr(args, "video_lane", None)
    bundle = getattr(args, "replay_from", None)
    if not (lane and bundle):
        return
    engine = str(lane)
    try:
        from nodes._otr_shared.public_engines import resolve_engine_id
        engine = resolve_engine_id(str(lane)) or engine
    except Exception:  # noqa: BLE001 -- the message still names the lane
        pass
    raise SystemExit(
        "--video-lane does not apply to --replay-from: a replay renders every "
        "shot on the engine its frozen plan names. To replay on %r, derive a "
        "bundle first:\n  python scripts/otr_freeze_replay_bundle.py %s "
        "--derive-engine %s\nthen pass the derived bundle to --replay-from."
        % (str(lane), bundle, engine))


def build_api_prompt(args) -> tuple[dict, list[str]]:
    _refuse_video_lane_on_replay(args)
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
    # The run label is NOT a graph value -- it never reaches a widget. It is
    # echoed here so a proof or soak run is identifiable in the console and the
    # receipt without renaming the operator's episode.
    if getattr(args, "run_label", None):
        print("[canonical-api] run_label=%s (label only -- the writer names the "
              "episode)" % args.run_label, flush=True)
    schemas = _schemas(args.offline_schemas)
    workflow = load_workflow(str(wf_path))

    applied: list[str] = []
    if args.workflow_id and args.workflow_id.lower() != "none":
        workflow = apply_row_to_workflow(workflow, args.workflow_id, schemas)
        applied.append(f"workflow={args.workflow_id}")
        if getattr(args, "video_lane", None):
            # The row's required models belong to the video lane it names,
            # which --video-lane replaces; _apply_video_lane says what, if
            # anything, checks the chosen lane's weights instead.
            print("[canonical-api] preflight: row model check skipped -- "
                  "--video-lane replaces the row's video lane", flush=True)
        else:
            checked = _assert_profile_models_present(
                args.workflow_id, schemas, offline=bool(args.offline_schemas))
            if checked:
                print("[canonical-api] preflight: %d required model(s) "
                      "visible to the server: %s"
                      % (len(checked), ", ".join(checked)), flush=True)

    if getattr(args, "video_lane", None):
        applied.extend(_apply_video_lane(workflow, schemas, args.video_lane))
    applied.extend(_apply_writer_shortcuts(workflow, schemas, args))
    # `is not None`: `--direction-json ""` names no file and must be refused as
    # one, not read as "option off" and run as an ordinary episode.
    if getattr(args, "direction_json", None) is not None:
        applied.extend(_apply_direction(workflow, schemas, args.direction_json))
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


def classify_timeout(running: int, pending: int) -> str:
    """What a poll TIMEOUT actually means, given the server's queue.

    Pure so it can be tested without a server. Three outcomes, and conflating
    them is the defect this exists to prevent (2026-08-23):

      ``still_running``  the observation window closed while the render carried
                         on. `--timeout` defaults to 5400s and a full episode
                         on the slowest video lane can exceed it, so this is
                         the COMMON case for that lane -- and it is not a
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

    The engine guard (``assert_usable``) fails closed but cannot do it until
    the first video beat, after the script pass, the whole voice pass and part
    of the video pass -- e.g. a weight on disk under a root the headless
    model-paths config does not name. Like the bench runners (SPEC G1/O6),
    this confirms every model filename in /object_info before submit.

    Returns the checked names. A profile with no ``preflight.required_models``
    is not an error -- most profiles do not declare any.
    """
    # THE PROFILE IS A MATRIX ROW, not a `config/profiles/<id>.json` file: such
    # a read finds no file for any id and silently checks nothing.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nodes._otr_shared.workflow_rows import WorkflowError, load_workflow_row
    from nodes._otr_shared.public_engines import resolve_engine_id
    from nodes._otr_visual_assets import VisualAssetError, planned_downloads
    try:
        profile = load_workflow_row(profile_name)
    except WorkflowError:
        return []                  # "none", or no such row: nothing declared
    required = ((profile.get("preflight") or {}).get("required_models") or [])
    # A FILE THE PACK FETCHES FOR THIS ROW IS NOT A REFUSAL. The validator's
    # queue-time preflight downloads the allowlisted weights the row's own
    # engines ask for -- before the writer runs, so the failure this gate
    # exists to prevent cannot happen for them -- and refusing here because
    # such a file is not on disk YET would block exactly the download that
    # fixes it. The skip is computed from THIS row's engines through the same
    # adapters the preflight asks; a file that merely appears somewhere in the
    # allowlist is still checked, because another lane fetching it proves
    # nothing about this one (found by review 2026-09-25: an engine-blind skip
    # let the AnimateDiff rows through with their checkpoint unchecked).
    # PLANNED ONE ENGINE AT A TIME. A selection the preflight would refuse
    # skips nothing -- but only for ITS OWN files. Planning the row as one set
    # let any single refusal empty the whole skip-list: on otr_16gb_animatediff
    # the z_image_turbo image slot (nvfp4 on Blackwell, no allowlisted
    # download) raised, and the gate then refused the SD 1.5 checkpoint that
    # the AnimateDiff lane downloads for itself (measured live 2026-09-24).
    selected = set((profile.get("role_overrides") or {}).values())
    slots = profile.get("slot_overrides") or {}
    selected.add(slots.get("music_engine"))
    self_fetched = set()
    for engine in {resolve_engine_id(str(e)) for e in selected if e}:
        try:
            self_fetched.update(name for (_category, name)
                                in planned_downloads({engine}))
        except VisualAssetError:
            continue
    required = [n for n in required if os.path.basename(str(n)) not in self_fetched]
    if not required:
        return []
    # Every entry is a weight FILENAME -- the profile schema refuses anything
    # else -- so each one is checkable against /object_info.
    visible = _server_visible_model_names(schemas)
    missing = [name for name in required if name not in visible]
    if missing and offline:
        # OFFLINE SCHEMAS ARE NOT A SERVER, so this gate cannot fire here: a
        # gate that cannot verify a claim must not pretend it refuted it.
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
        # moments later.
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
    # The return value is "what this gate actually VERIFIED", and the caller
    # prints it as "visible to the server". Offline there is no server and
    # nothing was verified, so claiming otherwise would contradict the notice
    # printed above it in the same run.
    return [] if offline else list(required)


def episode_of_prompt(prompt_id: str) -> str | None:
    """The episode id this prompt wrote, read back from its own history.

    Best-effort and never raises: a leg that cannot name its episode is
    still a leg, and this runs after the render is already terminal. The
    scan walks every string in the decoded history rather than a known output
    slot on purpose -- which node reports the path has changed twice, the id
    format has not. Decoding first matters for multilingual titles: raw JSON
    may spell a native path as ``\\uXXXX`` escapes, which an ASCII-only
    expression cannot name (a successfully published Hindi or CJK episode).
    """
    try:
        import requests
        response = requests.get(
            f"{COMFYUI_URL}/history/{prompt_id}", timeout=20)
        response.raise_for_status()
        history = response.json()
    except Exception:  # noqa: BLE001 -- naming the episode is never fatal
        return None

    def _strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from _strings(key)
                yield from _strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _strings(item)

    names = []
    # A replay id ends in one or more replay stamps (`_rp<8 hex>`, or the
    # older `_replay_<date>_<time>_<us>`); the same grammar as
    # nodes/_otr_shared/obs_name.REPLAY_STAMP (plan item 19, 2026-10-14). The
    # history lists the prompt's inputs before its outputs, so the LAST match
    # -- this run's own output -- names the episode, never the replay source.
    pattern = re.compile(
        r"(?:^|[\\/])episodes[\\/]+([^\\/]+_\d{8}_\d{6}"
        r"(?:_replay_\d{8}_\d{6}_\d{6}|_rp[0-9a-f]{8})*)(?=[\\/]|$)")
    for text in _strings(history):
        for match in pattern.finditer(text):
            if match.group(1) not in names:
                names.append(match.group(1))
    return names[-1] if names else None


def _format_elapsed(seconds: float) -> str:
    """``3725`` -> ``1h02m``; ``725`` -> ``12m05s``; ``45`` -> ``45s``."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def obs_file_from_history(
    prompt: dict, history: dict, prompt_id: str
) -> tuple[str | None, str]:
    """The obs file the mux node reported publishing, confirmed on disk.

    Returns ``(path, "")`` when the file is there and not empty, else
    ``(None, why)``. ``OTR_MasterAudioMux`` returns the published path as the
    ``text`` of its history output (``_canvas_preview`` in
    nodes/otr_master_audio_mux.py), so this reads the answer the node gave
    instead of working out where ``otr/obs`` is from a process that is not
    running inside ComfyUI. Only reads: nothing here touches ``otr/obs/``.
    """
    mux_ids = [str(node_id) for node_id, spec in prompt.items()
               if isinstance(spec, dict)
               and spec.get("class_type") == "OTR_MasterAudioMux"]
    if not mux_ids:
        return None, "this graph has no OTR_MasterAudioMux node, so nothing is published"
    entry = history.get(prompt_id) if isinstance(history, dict) else None
    outputs = (entry or {}).get("outputs") or {}
    for node_id in mux_ids:
        for text in (outputs.get(node_id) or {}).get("text") or []:
            reported = str(text)
            if reported.startswith("not published"):
                # "not published -- archived at <path>": keep the verdict, drop the path.
                return None, reported.split(" at ", 1)[0]
            if not os.path.isfile(reported):
                return None, "the mux node reported an obs file this machine cannot see"
            if os.path.getsize(reported) <= 0:
                return None, "the obs file is zero bytes"
            return reported, ""
    return None, "the mux node reported no obs path"


def published_obs_file(prompt: dict, prompt_id: str) -> tuple[str | None, str]:
    """``obs_file_from_history`` for a finished prompt, reading its history."""
    try:
        import requests
        response = requests.get(
            f"{COMFYUI_URL}/history/{prompt_id}", timeout=20)
        response.raise_for_status()
        history = response.json()
    except Exception as exc:  # noqa: BLE001 -- reported in the ping
        return None, f"history could not be read ({type(exc).__name__})"
    return obs_file_from_history(prompt, history, prompt_id)


def compose_result_ping(
    status: str,
    err: str,
    episode: str | None,
    obs_path: str | None,
    obs_why: str,
    elapsed_s: float,
) -> tuple[str, str, str | None]:
    """``(title, body, priority)`` for the phone. Pure.

    Messages carry the episode id and the obs FILE NAME, no other local path.
    SUCCESS says "published" only when ``obs_path`` is a confirmed file: a leg
    whose obs file is missing is the case the operator most needs to hear about
    ("obs IS the success signal"), so it pings at high priority instead of
    staying quiet.
    """
    took = _format_elapsed(elapsed_s)
    name = episode or "episode id not read"
    if status == "SUCCESS":
        if obs_path:
            return ("OTR episode published",
                    f"{name}\n{os.path.basename(obs_path)}\ntook {took}", None)
        return ("OTR leg finished, nothing in obs",
                f"{name}\n{obs_why}\ntook {took}", "high")
    if status == "TIMEOUT":
        return ("OTR watch timed out",
                f"stopped watching after {took}; the render may still be "
                "running", None)
    # The hint is on its own line in a parsed execution error and at the END of
    # a line in a refusal, a SERVER-GONE message or a raw fallback: find it
    # wherever it is, report it once under the headline, and keep it out of the
    # headline itself.
    text = err or ""
    hint = None
    found = re.search(r"-- hint:[ \t]*([^\r\n]*)", text)
    if found:
        hint = "-- hint: " + found.group(1).strip()
        text = text[:found.start()] + text[found.end():]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parts = [lines[0][:300] if lines else "no error text"]
    if hint:
        parts.append(hint[:300])
    parts.append(f"after {took}")
    return ("OTR leg FAILED", "\n".join(parts), "high")


def announce_result(
    prompt: dict,
    prompt_id: str,
    status: str,
    err: str,
    episode: str | None,
    elapsed_s: float,
) -> None:
    """The phone ping for a finished leg (PLAN row 1.3). Reports only: whatever
    happens in here, the leg's result and exit code are unchanged."""
    if not otr_notify.notify_url():
        return
    try:
        obs_path, obs_why = (None, "")
        if status == "SUCCESS":
            obs_path, obs_why = published_obs_file(prompt, prompt_id)
        title, body, priority = compose_result_ping(
            status, err, episode, obs_path, obs_why, elapsed_s)
        sent = otr_notify.notify(title, body, priority=priority)
        print(f"[canonical-api] notify {'sent' if sent else 'NOT sent'}: {title}",
              flush=True)
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        print(f"[canonical-api] notify skipped: {type(exc).__name__}: {exc}",
              flush=True)


def record_known_good_after_success(prompt: dict, prompt_id: str,
                                    episode: str | None) -> str:
    """A SUCCESS whose obs file is on disk makes that episode's environment the
    known-good snapshot (PLAN row 2.2). One line for the leg log; never raises,
    and nothing it finds changes the leg's result or exit code."""
    if not episode:
        return "known-good not recorded: the episode id was not read"
    try:
        obs_path, obs_why = published_obs_file(prompt, prompt_id)
        if not obs_path:
            return f"known-good not recorded: {obs_why}"
        return otr_env_drift.record_known_good(episode)
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        return f"known-good not recorded: {type(exc).__name__}: {exc}"


def preview_plan(preview_file: str | None,
                 env_method: str | None) -> tuple[str | None, str | None]:
    """``(method for submit_prompt, start line)`` for ``--preview-file``. Pure.

    ComfyUI sends preview frames only while a preview method is on, so asking
    for the file also asks for ``latent2rgb`` -- unless OTR_PREVIEW_METHOD is
    set: the operator's choice stands, and the line says when it means that no
    file can appear. Without ``--preview-file`` there is nothing to say.
    """
    if not preview_file:
        return None, None
    stem = os.path.splitext(os.fspath(preview_file))[0]
    target = f"{stem}.jpg (or {stem}.png)"
    chosen = (env_method or "").strip().lower()
    if not chosen:
        return "latent2rgb", (
            f"preview file: on -> {target}, the newest frame, replaced as the "
            "render runs; asking ComfyUI for latent2rgb")
    if chosen == "none":
        return None, (
            f"preview file: on -> {target}, but OTR_PREVIEW_METHOD=none asks "
            "ComfyUI for no frames, so no file will appear")
    return None, (
        f"preview file: on -> {target}, the newest frame, replaced as the "
        f"render runs; OTR_PREVIEW_METHOD={chosen} decides the frames")


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
    parser.add_argument(
        "--workflow-id", default="none",
        help="a workflow id -- a row of config/workflow_matrix.json, e.g. "
             "otr_8gb_low or otr_16gb_video -- or none",
    )
    parser.add_argument("--title", default=None,
                        help="NAME THE EPISODE. This fills the episode_title "
                             "widget, so the writer generates no title of its "
                             "own and the TITLE CARD shows exactly this. For "
                             "labelling a proof or soak run, use --run-label.")
    parser.add_argument("--run-label", default=None, dest="run_label",
                        help="a name for THIS RUN, echoed to the CONSOLE "
                             "only -- this script writes no receipt; a "
                             "wrapping harness can capture the echo and "
                             "build its own. Deliberately does NOT touch "
                             "episode_title, so the writer still names the "
                             "episode and the title card shows the story's "
                             "name rather than your harness label.")
    parser.add_argument("--replay-from", default=None, dest="replay_from",
                        help="CANONICAL REPLAY: a frozen replay bundle directory "
                             "(scripts/otr_freeze_replay_bundle.py). The writer, "
                             "cast, TTS, music and stills pass through frozen; "
                             "only the video phase and the publish tail run.")
    parser.add_argument("--direction-json", default=None, dest="direction_json",
                        help="SCENE PRODUCER: a file holding Shot Lock's direction_json "
                             "(planning/scene-producer/PLAN.md). Validated before the "
                             "prompt is sent; pair it with --replay-from for every "
                             "stage after the first.")
    parser.add_argument("--video-lane", default=None, dest="video_lane",
                        help="set the announcer, character and music video "
                             "dropdowns to this video lane (e.g. "
                             "h3_low_video), as picking it in the app does; "
                             "applied after --workflow-id. Not with "
                             "--replay-from: a replay keeps its bundle's "
                             "engine (derive a bundle with "
                             "otr_freeze_replay_bundle.py --derive-engine)")
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
    parser.add_argument("--preview-file", default=None, dest="preview_file",
                        metavar="PATH",
                        help="keep the newest preview frame of the running render "
                             "in PATH (.jpg, or .png when ComfyUI sends a PNG), "
                             "replaced about once a second, so a person can open "
                             "it and see the render. A diagnostic, not a "
                             "deliverable: keep it out of otr/episodes and "
                             "otr/obs. Asks ComfyUI for latent2rgb frames unless "
                             "OTR_PREVIEW_METHOD is set.")
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

    # Never a hidden switch: say at the start whether the phone ping is on.
    print(f"[canonical-api] {otr_notify.state_line()}", flush=True)
    preview_method, preview_line = preview_plan(
        args.preview_file, os.environ.get("OTR_PREVIEW_METHOD"))
    if preview_line:
        print(f"[canonical-api] {preview_line}", flush=True)
    # The preview arguments are passed only when --preview-file asked for them,
    # so a run without the flag builds and submits exactly what it always did.
    submit_extra = {"preview_method": preview_method} if preview_method else {}
    live_extra = {"preview_path": args.preview_file} if args.preview_file else {}
    # The live view listens on the websocket of the client that submits, so
    # the SAME id goes to submit_prompt. It starts first so no frame is missed.
    client_id = str(uuid.uuid4())
    live = LiveView(client_id, prompt, **live_extra).start()
    started = time.monotonic()
    prompt_id = ""
    try:
        prompt_id = submit_prompt(prompt, client_id=client_id, **submit_extra)
        live.set_prompt_id(prompt_id)
        print(f"[canonical-api] QUEUED prompt_id={prompt_id}", flush=True)
        print(
            f"[canonical-api] t=0s prompt_id={prompt_id} status=queued",
            flush=True,
        )

        def heartbeat(elapsed_s: float, status: dict) -> None:
            phase = str(status.get("status_str") or "pending")
            # During an outage poll_history hands over the strike count; print
            # it, or the number exists nowhere a person can read it.
            # `unreachable (3/12)` is the whole point of counting.
            strikes = status.get("unreachable_strikes")
            if strikes:
                phase = f"{phase} ({strikes}/{status.get('unreachable_limit', '?')})"
            # Which node is running and how far along (PLAN row 1.1), appended
            # AFTER status= so a log reader keyed on `t=<N>s ... status=`
            # (scripts/otr_render_watchdog.ps1) is unaffected.
            live_snapshot = status.get("live")
            live_text = (f' live="{format_live(live_snapshot)}"'
                         if live_snapshot else "")
            print(
                f"[canonical-api] t={int(elapsed_s)}s prompt_id={prompt_id} "
                f"status={phase}{live_text}",
                flush=True,
            )

        status, err = poll_history(
            prompt_id, timeout_s=args.timeout, poll_s=args.poll_s,
            on_tick=live.wrap(heartbeat),
        )
    except Exception as exc:
        # The leg ended on an EXCEPTION, not on a result: the submission was
        # refused or unreachable (the commonest unattended failure -- a bad
        # dropdown value, a server that is not up), or watching crashed. Say so
        # on the phone, then let the exception go on exactly as before: same
        # traceback, same exit code. Ctrl-C is a KeyboardInterrupt, not an
        # Exception, so it sends no ping on purpose: an operator stopping the
        # watch is not a failure to report.
        where =("while watching the render" if prompt_id
                 else "before the prompt was queued")
        announce_result(prompt, prompt_id, "FAIL",
                        f"{type(exc).__name__} {where}: {exc}", None,
                        time.monotonic() - started)
        raise
    finally:
        live.stop()
    elapsed_s = time.monotonic() - started
    print(f"[canonical-api] RESULT {status} prompt_id={prompt_id}", flush=True)
    # Frame counts: the preview measurement (OTR_PREVIEW_METHOD) reads them here.
    print(f"[canonical-api] LIVE {live.summary()}", flush=True)
    episode = episode_of_prompt(prompt_id)
    if episode:
        # WHICH EPISODE DID THIS LEG MAKE? Until 2026-09-12 this process
        # never said, and every reader -- a person reading a leg log, a
        # harness attributing a measurement -- had to guess from directory
        # timestamps. On a box where two renders can overlap that guess is
        # wrong exactly when it matters (cursor r3).
        print(f"[canonical-api] EPISODE episodes/{episode}", flush=True)
    if status == "TIMEOUT":
        # A TIMEOUT HERE IS ABOUT THIS PROCESS, NOT ABOUT THE RENDER, and saying
        # so is the whole point of this branch (2026-08-23). `--timeout`
        # defaults to 5400s; a full episode on the slowest video lane can
        # exceed that, so the observation window closes while the server is still at
        # 98% GPU happily rendering beat 34. A bare "RESULT TIMEOUT" with exit 1
        # for BOTH that case and a genuinely dead render would be two opposite
        # situations behind one indistinguishable message, and the reader's
        # natural conclusion is the wrong one.
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
        announce_result(prompt, prompt_id, status, err, episode, elapsed_s)
        return 1
    print(f"[canonical-api] {record_known_good_after_success(prompt, prompt_id, episode)}",
          flush=True)
    announce_result(prompt, prompt_id, status, err, episode, elapsed_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
