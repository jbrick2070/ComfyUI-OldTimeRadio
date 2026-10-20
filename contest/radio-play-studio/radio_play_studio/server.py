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
import time
import traceback
import urllib.request
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
#: A page says it is open this often; with none heard for GPU_IDLE_S, and no
#: episode being made, the app asks ComfyUI to drop the models it loaded.
PRESENCE_S = 20.0
#: Browsers slow a hidden tab's timers to about once a minute, so the idle
#: window is well above that: a background tab still counts as open.
GPU_IDLE_S = 150.0
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class Run:
    """One episode: its events so far, and whether it has ended."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.events: list = []
        self.finished = False
        self.started = time.time()
        self.cond = threading.Condition()

    def add(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            if event.get("type") in ("done", "error"):
                self.finished = True
            self.cond.notify_all()


class Busy(RuntimeError):
    """An episode is already being made; one GPU makes one at a time."""


_BUSY = "an episode is already being made; the page shows its progress"


class Runs:
    def __init__(self, studio: Studio, *, now=time.monotonic) -> None:
        self.studio = studio
        self._runs: dict = {}
        self._lock = threading.Lock()
        self._now = now
        self._pages: dict = {}          # page id -> when it last said it is open
        self._gpu_dirty = False         # this app has rendered since its last free

    def _busy(self) -> bool:
        return any(not r.finished for r in self._runs.values())

    def start(self, choices: dict) -> Run:
        """Check the choices against the server, then run the episode on a thread.
        A bad choice raises here, before any thread starts. A second episode while
        one is being made is refused: it would wait an hour behind it on the GPU."""
        with self._lock:
            if self._busy():
                raise Busy(_BUSY)
        self.studio.build_graph(choices)
        run = Run(uuid.uuid4().hex[:12])
        with self._lock:
            if self._busy():
                raise Busy(_BUSY)
            self._runs[run.id] = run
            self._gpu_dirty = True
        threading.Thread(target=self._drive, args=(run, choices), daemon=True).start()
        return run

    def listing(self) -> list:
        """Every episode this app has started since it was launched, newest first."""
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.started, reverse=True)
        return [{"id": r.id, "finished": r.finished, "started": r.started} for r in runs]

    # -- the GPU goes back when the last page closes -----------------------
    def page_open(self, page_id: str) -> None:
        with self._lock:
            self._pages[page_id] = self._now()

    def page_closed(self, page_id: str) -> None:
        with self._lock:
            self._pages.pop(page_id, None)

    def should_free_gpu(self) -> bool:
        """True once: no page heard from for GPU_IDLE_S, nothing being made, and
        this app has rendered since it last freed -- so it only ever releases
        models its own episodes loaded, never someone else's idle canvas."""
        with self._lock:
            now = self._now()
            self._pages = {k: t for k, t in self._pages.items() if now - t < GPU_IDLE_S}
            if self._pages or self._busy() or not self._gpu_dirty:
                return False
            self._gpu_dirty = False
            return True

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

    def _serve(self, route) -> None:
        """Run a route so that no failure inside it ends as an empty reply: a
        person gets a JSON error, the console gets the traceback, and a browser
        that closed the tab mid-reply is not an error at all."""
        if not self._guard():
            return
        try:
            route()
        except ConnectionError:
            return
        except Exception:  # noqa: BLE001 - last line of defence for the handler thread
            traceback.print_exc()
            try:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                {"error": "internal error; the app's console has the details"})
            except Exception:  # noqa: BLE001 - the reply was already under way
                pass

    # -- routes -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        self._serve(self._get)

    def do_POST(self) -> None:  # noqa: N802
        self._serve(self._post)

    def _get(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/api/options":
            return self._options()
        if path == "/api/episodes":
            return self._send_json(HTTPStatus.OK, {"episodes": self.runs.listing()})
        match = re.fullmatch(r"/api/episodes/([0-9a-f]{12})/events", path)
        if match:
            return self._events(match.group(1))
        if path.startswith("/episodes/"):
            return self._episode_file(path[len("/episodes/"):])
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _post(self) -> None:
        path = urlsplit(self.path).path
        if path not in ("/api/episodes", "/api/presence"):
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
        if path == "/api/presence":
            return self._presence()
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
        except Busy as exc:
            return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except ServerUnreachable as exc:
            return self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        except (ValueError, GraphError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        self._send_json(HTTPStatus.ACCEPTED, {"id": run.id})

    def _presence(self) -> None:
        """A page saying it is open, or closing: {"page": id, "closing": bool}."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            page = str(body["page"])[:64]
        except (ValueError, TypeError, KeyError):
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "send {'page': id}"})
        if body.get("closing"):
            self.runs.page_closed(page)
        else:
            self.runs.page_open(page)
        self._send_json(HTTPStatus.OK, {"ok": True})

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
            sent = max(0, int(self.headers.get("Last-Event-ID", "-1")) + 1)
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
        except ConnectionError:
            return

    def _episode_file(self, rel: str) -> None:
        root = self.runs.studio.settings.out_dir.resolve()
        try:
            target = (root / rel).resolve()
        except (ValueError, OSError):  # an embedded NUL, an unusable name
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
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
        except ConnectionError:
            return

    def _send_range_error(self, size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()


def free_comfyui_models(comfyui_url: str) -> None:
    """Ask ComfyUI to unload its models and free memory (POST /free), so the
    card is free for whatever the person does next. ComfyUI applies it between
    prompts; it never stops a render. Best effort: a failure is only printed."""
    req = urllib.request.Request(
        comfyui_url.rstrip("/") + "/free", method="POST",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
        print("Radio Play Studio: every page is closed, so ComfyUI was asked to free its models.")
    except OSError as exc:
        print(f"Radio Play Studio: could not ask ComfyUI to free its models ({exc}).")


def _watch_gpu(runs: Runs, free=free_comfyui_models, sleep=time.sleep) -> None:
    while True:
        sleep(PRESENCE_S / 2)
        if runs.should_free_gpu():
            free(runs.studio.settings.comfyui_url)


def make_server(studio: Studio, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.runs = Runs(studio)  # type: ignore[attr-defined]
    threading.Thread(target=_watch_gpu, args=(server.runs,), daemon=True).start()
    return server
