"""Command line: serve the app, check the setup, run one episode, or dump the graph.

    python -m radio_play_studio doctor
    python -m radio_play_studio serve
    python -m radio_play_studio run --bank shakespeare --acts 1
    python -m radio_play_studio export-graph graph_api.json
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .config import DEFAULT_API_URL, DEFAULT_COMFYUI_URL, Settings, WorkflowNotFound, load_settings
from .form import FIELDS, describe_form, locate_nodes
from .graph import NOTE_NODE_TYPES, GraphError, ServerUnreachable, fetch_object_info, load_workflow
from .runner import Studio
from .server import make_server


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--comfyui", help=f"your ComfyUI (default {DEFAULT_COMFYUI_URL})")
    p.add_argument("--api", help=f"comfy-api-proxy, the bridge the Comfy SDK uses (default {DEFAULT_API_URL})")
    p.add_argument("--workflow", help="path to otr_app.json (default: found from OTR_PACK_DIR / COMFYUI_DIR)")
    p.add_argument("--out", help="folder finished episodes are saved to (default ./episodes)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="radio_play_studio", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the web app")
    _add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    doctor = sub.add_parser("doctor", help="check ComfyUI, the pack, the bridge and the SDK")
    _add_common(doctor)

    run = sub.add_parser("run", help="make one episode from the command line")
    _add_common(run)
    for f in FIELDS:
        run.add_argument("--" + f.key.replace("_", "-"), dest=f.key, help=f"{f.label}: {f.help}")

    export = sub.add_parser("export-graph", help="write the API graph the app would submit")
    _add_common(export)
    export.add_argument("path", help="where to write the JSON")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    return load_settings(comfyui_url=args.comfyui, api_url=args.api,
                         workflow=args.workflow, out_dir=args.out)


def _choices(args: argparse.Namespace) -> dict:
    return {f.key: getattr(args, f.key) for f in FIELDS if getattr(args, f.key, None) is not None}


LOOPBACK_BINDS = ("127.0.0.1", "localhost", "::1")


def cmd_serve(args: argparse.Namespace) -> int:
    if args.host not in LOOPBACK_BINDS:
        raise ValueError(f"--host {args.host!r}: this app has no login, so it only listens on this "
                         f"machine ({', '.join(LOOPBACK_BINDS)})")
    studio = Studio(_settings(args))
    server = make_server(studio, args.host, args.port)
    print(f"Radio Play Studio on http://{args.host}:{args.port}  (ComfyUI {studio.settings.comfyui_url}, "
          f"bridge {studio.settings.api_url}, episodes -> {studio.settings.out_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    studio = Studio(_settings(args))
    code = 1
    for event in studio.stream(_choices(args)):
        if event["type"] == "progress":
            print(f"{event['value']:.0%} {event['message']}".rstrip(), flush=True)
        elif event["type"] == "done":
            print(f"DONE -> {event['path']}")
            code = 0
        elif event["type"] == "error":
            print("ERROR: " + event["message"], file=sys.stderr)
        elif event["type"] == "status":
            print("status: " + event["status"], flush=True)
        elif event["type"] == "submitted":
            print("submitted", flush=True)
    return code


def cmd_export_graph(args: argparse.Namespace) -> int:
    graph, _ = Studio(_settings(args)).build_graph({})
    Path(args.path).write_text(json.dumps(graph, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(graph)} nodes -> {args.path}")
    return 0


def _bridge_reachable(url: str, timeout: float = 3.0) -> bool:
    parts = urlsplit(url)
    try:
        with socket.create_connection((parts.hostname or "127.0.0.1", parts.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def doctor_report(settings: Settings) -> list:
    """`[(ok, what, detail)]` for each thing the app needs, in the order a fix
    would go. Reports every check; one failure does not hide the rest."""
    rows: list = []

    def add(ok: bool, what: str, detail: str = "") -> None:
        rows.append((ok, what, detail))

    add(True, "workflow file", str(settings.workflow_path))
    try:
        schemas = fetch_object_info(settings.comfyui_url)
        add(True, "ComfyUI answers", f"{settings.comfyui_url} ({len(schemas)} node types)")
    except ServerUnreachable as exc:
        add(False, "ComfyUI answers", str(exc))
        schemas = None
    if schemas is not None:
        workflow = load_workflow(settings.workflow_path)
        wanted = sorted({n["type"] for n in workflow["nodes"]} - NOTE_NODE_TYPES)
        missing = [t for t in wanted if t not in schemas]
        add(not missing, "Old-Time Radio pack installed",
            ("missing: " + ", ".join(missing)) if missing else f"all {len(wanted)} node types present")
        try:
            locate_nodes(workflow)
            describe_form(schemas)
            add(True, "form fields resolve", f"{len(FIELDS)} fields")
        except GraphError as exc:
            add(False, "form fields resolve", str(exc))
    try:
        import comfy_sdk  # noqa: F401
        add(True, "Comfy SDK installed", getattr(comfy_sdk, "__version__", ""))
    except ImportError:
        add(False, "Comfy SDK installed", "pip install -r requirements.txt")
    add(_bridge_reachable(settings.api_url), "comfy-api-proxy answers",
        settings.api_url + "  (start it with: comfy-api-proxy)")
    return rows


def cmd_doctor(args: argparse.Namespace) -> int:
    rows = doctor_report(_settings(args))
    for ok, what, detail in rows:
        print(("ok    " if ok else "FAIL  ") + what + (" -- " + detail if detail else ""))
    return 0 if all(ok for ok, _, _ in rows) else 1


COMMANDS = {"serve": cmd_serve, "doctor": cmd_doctor, "run": cmd_run, "export-graph": cmd_export_graph}


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (WorkflowNotFound, ServerUnreachable, GraphError, ValueError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
