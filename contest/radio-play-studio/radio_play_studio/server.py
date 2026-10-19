"""The local web app: a form, a live progress stream and a player.

Standard library only. It listens on loopback, refuses requests whose Host is
not loopback (a web page cannot reach it by DNS rebinding) and only takes JSON
posts (a cross-site form cannot send those without a preflight).
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .graph import GraphError, ServerUnreachable
from .runner import Studio

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
HEARTBEAT_S = 15.0
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class Run:
    """One episode: its events so far, and whether it has ended."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.events: list = []
        self.finished = False
        self.cond = threading.Condition()

    def add(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            if event.get("type") in ("done", "error"):
                self.finished = True
            self.cond.notify_all()


class Runs:
    def __init__(self, studio: Studio) -> None:
        self.studio = studio
        self._runs: dict = {}
        self._lock = threading.Lock()

    def start(self, choices: dict) -> Run:
        """Check the choices against the server, then run the episode on a thread.
        A bad choice raises here, before any thread starts."""
        self.studio.build_graph(choices)
        run = Run(uuid.uuid4().hex[:12])
        with self._lock:
            self._runs[run.id] = run
        threading.Thread(target=self._drive, args=(run, choices), daemon=True).start()
        return run

    def _drive(self, run: Run, choices: dict) -> None:
        for event in self.studio.stream(choices):
            run.add(event)

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)


class Handler(BaseHTTPRequestHandler):
    server_version = "RadioPlayStudio/0.1"

    # -- plumbing -----------------------------------------------------------
    @property
    def runs(self) -> Runs:
        return self.server.runs  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # quiet: progress is the log
        return

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):  # [::1] or [::1]:8765
            name = host[:host.find("]") + 1]
        else:
            name = host.split(":", 1)[0]
        return name in LOOPBACK_HOSTS

    def _send_json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _guard(self) -> bool:
        if self._host_ok():
            return True
        self._send_json(HTTPStatus.FORBIDDEN, {"error": "this app only answers on localhost"})
        return False

    # -- routes -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        if not self._guard():
            return
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/api/options":
            return self._options()
        match = re.fullmatch(r"/api/episodes/([0-9a-f]{12})/events", path)
        if match:
            return self._events(match.group(1))
        if path.startswith("/episodes/"):
            return self._episode_file(path[len("/episodes/"):])
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        if urlsplit(self.path).path != "/api/episodes":
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            choices = body.get("choices", {})
            if not isinstance(choices, dict):
                raise ValueError("choices must be an object")
        except (ValueError, TypeError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {exc}"})
        try:
            run = self.runs.start(choices)
        except ServerUnreachable as exc:
            return self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        except (ValueError, GraphError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        self._send_json(HTTPStatus.ACCEPTED, {"id": run.id})

    def _options(self) -> None:
        try:
            self._send_json(HTTPStatus.OK, {"fields": self.runs.studio.options()})
        except (ServerUnreachable, GraphError) as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})

    def _events(self, run_id: str) -> None:
        run = self.runs.get(run_id)
        if run is None:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such episode"})
        try:
            sent = int(self.headers.get("Last-Event-ID", "-1")) + 1
        except ValueError:
            sent = 0
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                with run.cond:
                    if sent >= len(run.events) and not run.finished:
                        run.cond.wait(HEARTBEAT_S)
                    batch = run.events[sent:]
                    finished = run.finished
                if batch:
                    for event in batch:
                        self.wfile.write(
                            f"id: {sent}\ndata: {json.dumps(event)}\n\n".encode("utf-8"))
                        sent += 1
                else:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                if finished and sent >= len(run.events):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def _episode_file(self, rel: str) -> None:
        root = self.runs.studio.settings.out_dir.resolve()
        target = (root / rel).resolve()
        if root != target and root not in target.parents:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        self._send_file(target, mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    def _send_file(self, target: Path, content_type: str) -> None:
        if not target.is_file():
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        size = target.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        header = self.headers.get("Range")
        if header:
            match = _RANGE.match(header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                return self._send_range_error(size)
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:  # "-N": the last N bytes
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            if start > end or start >= size:
                return self._send_range_error(size)
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        remaining = end - start + 1
        try:
            with target.open("rb") as handle:
                handle.seek(start)
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_range_error(self, size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()


def make_server(studio: Studio, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.runs = Runs(studio)  # type: ignore[attr-defined]
    return server
