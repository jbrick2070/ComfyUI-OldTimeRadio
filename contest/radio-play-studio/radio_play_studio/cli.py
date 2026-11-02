"""Command line: serve the app, check the setup, run one episode, or dump the graph.

    python -m radio_play_studio start      # the one-click launcher's command
    python -m radio_play_studio doctor
    python -m radio_play_studio serve
    python -m radio_play_studio run --bank shakespeare --acts 1
    python -m radio_play_studio export-graph graph_api.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .config import (DEFAULT_API_URL, DEFAULT_COMFYUI_URL, DESKTOP_COMFYUI_URL, Settings, WorkflowNotFound,
                     load_settings)
from .form import FIELDS, describe_form, locate_nodes
from .graph import NOTE_NODE_TYPES, GraphError, ServerUnreachable, fetch_object_info, load_workflow
from .runner import Studio
from .server import make_server


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--comfyui", help=f"your ComfyUI (default: {DEFAULT_COMFYUI_URL}, or Desktop's {DESKTOP_COMFYUI_URL} if only that answers)")
    p.add_argument("--api", help=f"comfy-api-proxy, the bridge the Comfy SDK uses (default {DEFAULT_API_URL})")
    p.add_argument("--workflow", help="path to a shipped workflow (default: otr_16gb_foley_open.json, found from OTR_PACK_DIR / COMFYUI_DIR)")
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

    start = sub.add_parser("start", help="one click: start the bridge if it is down, check the "
                                         "setup, serve the app and open it in the browser")
    _add_common(start)
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8765)

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


def _comfy_key() -> str | None:
    """The Comfy API key a headless submitter uses (OTR_COMFY_API_KEY), so a
    Comfy Cloud run started here is paid for without a page typing it in."""
    return os.environ.get("OTR_COMFY_API_KEY", "").strip() or None


def _check_host(args: argparse.Namespace) -> None:
    if args.host not in LOOPBACK_BINDS:
        raise ValueError(f"--host {args.host!r}: this app has no login, so it only listens on this "
                         f"machine ({', '.join(LOOPBACK_BINDS)})")


def cmd_serve(args: argparse.Namespace, settings: Settings | None = None,
              open_browser: bool = False) -> int:
    _check_host(args)
    studio = Studio(settings or _settings(args), comfy_key=_comfy_key())
    server = make_server(studio, args.host, args.port)
    host = f"[{args.host}]" if ":" in args.host else args.host
    url = f"http://{host}:{args.port}/"
    print(f"Radio Play Studio on {url}  (ComfyUI {studio.settings.comfyui_url}, "
          f"bridge {studio.settings.api_url}, episodes -> {studio.settings.out_dir})")
    if open_browser:
        # The socket is already listening, so the browser's first request waits for serve_forever.
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    studio = Studio(_settings(args), comfy_key=_comfy_key())
    code = 1
    for event in studio.stream(_choices(args)):
        if event["type"] == "progress":
            if event.get("minutes_left") is not None:
                stages = f" (about {event['minutes_left']} min left)"
            elif event.get("nodes_total"):
                stages = f" ({event['nodes_done']} of {event['nodes_total']} stages)"
            else:
                stages = ""
            print(f"{event['value']:.0%}{stages} {event['message']}".rstrip(), flush=True)
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
        settings.api_url + "  (start it with: comfy-api-proxy start, or the Start Radio Play Studio launcher)")
    return rows


def _print_report(rows: list) -> None:
    for ok, what, detail in rows:
        print(("ok    " if ok else "FAIL  ") + what + (" -- " + detail if detail else ""))


def cmd_doctor(args: argparse.Namespace) -> int:
    rows = doctor_report(_settings(args))
    _print_report(rows)
    return 0 if all(ok for ok, _, _ in rows) else 1


def bridge_command() -> list | None:
    """The `comfy-api-proxy` this Python installed (beside its own executable,
    where pip puts it in a venv), else the one on PATH; None when there is none."""
    scripts = Path(sys.executable).parent
    for name in ("comfy-api-proxy.exe", "comfy-api-proxy"):
        for folder in (scripts, scripts / "Scripts", scripts / "bin"):
            if (folder / name).is_file():
                return [str(folder / name)]
    found = shutil.which("comfy-api-proxy")
    return [found] if found else None


BRIDGE_WAIT_S = 20.0


def start_bridge(settings: Settings, *, run=subprocess.run, reachable=_bridge_reachable,
                 sleep=time.sleep, command=bridge_command) -> str:
    """Start comfy-api-proxy in the background for this ComfyUI when nothing
    answers on the bridge's address, and wait for it to answer. Returns what
    happened, for the console. Only a loopback bridge is started: one on
    another machine is that machine's to run. It keeps running after the app
    stops (`comfy-api-proxy stop` ends it), as the README's setup always had it."""
    if reachable(settings.api_url, timeout=1.0):
        return f"comfy-api-proxy already answers on {settings.api_url}"
    parts = urlsplit(settings.api_url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        return f"comfy-api-proxy is not answering on {settings.api_url}, and it is not on this machine to start"
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        return f"comfy-api-proxy is not answering on {settings.api_url}; name its port to start it here"
    exe = command()
    if exe is None:
        return "comfy-api-proxy is not installed: pip install -r requirements.txt"
    # Bound to the address the app will call, so a [::1] bridge answers on [::1].
    cmd = exe + ["start", "--comfyui", settings.comfyui_url, "--host", parts.hostname,
                 "--port", str(port)]
    try:
        result = run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"comfy-api-proxy did not start: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return "comfy-api-proxy did not start" + (f": {detail[-1]}" if detail else "")
    waited = 0.0
    while not reachable(settings.api_url, timeout=1.0):
        if waited >= BRIDGE_WAIT_S:
            return f"comfy-api-proxy started but is not answering on {settings.api_url} yet"
        sleep(0.5)
        waited += 0.5
    return f"started comfy-api-proxy on {settings.api_url} for ComfyUI {settings.comfyui_url}"


def cmd_start(args: argparse.Namespace) -> int:
    """What the Start Radio Play Studio launcher runs: the bridge, the setup
    check, then the app in the browser. A failed check is printed, not fatal:
    the page says what is wrong too, and ComfyUI may come up a minute later."""
    _check_host(args)               # refused before anything is started for it
    settings = _settings(args)
    print(start_bridge(settings))
    rows = doctor_report(settings)
    _print_report(rows)
    if not all(ok for ok, _, _ in rows):
        print("Something above is not ready; the app starts anyway and will work once it is.")
    return cmd_serve(args, settings, open_browser=True)


COMMANDS = {"serve": cmd_serve, "start": cmd_start, "doctor": cmd_doctor, "run": cmd_run,
            "export-graph": cmd_export_graph}


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (WorkflowNotFound, ServerUnreachable, GraphError, ValueError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
