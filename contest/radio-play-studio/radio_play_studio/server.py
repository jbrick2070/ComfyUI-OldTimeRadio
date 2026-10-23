"""The local web app: a form, a live progress stream and a player -- and the
Scene Producer's projects (``producer.py``): read them, save a draft, start a
stage. Only a stage start takes the GPU; reads and drafts are answered while
anything is being made.

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

from . import producer
from .graph import GraphError, ServerUnreachable
from .photos import MAX_PHOTO_BYTES, PhotoError, upload_photo
from .runner import Studio

STATIC_DIR = Path(__file__).resolve().parent / "static"
#: One JSON file per language the page can speak: `{"_name": "Espanol", "key": "text", ...}`.
#: The name is a language code (`es`) or a code and script (`zh-hant`), all lower case.
#: Adding a file adds a language; the page reads the list from `/api/languages`.
I18N_DIR = STATIC_DIR / "i18n"
_I18N_FILE = re.compile(r"([a-z]{2,3}(?:-[a-z]{4})?)\.json")
#: The author's photos that ship in the pack for its built-in My Story (the
#: example's cast cards show them). The pack reads them by " [bundled]" names.
BUNDLED_PHOTO_DIR = (Path(__file__).resolve().parents[3]
                     / "nodes" / "story_packs" / "my_story" / "photos")
_BUNDLED_NAME = re.compile(r"[a-z0-9_]{1,64}\.jpg")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
HEARTBEAT_S = 15.0
#: A page says it is open this often; with none heard for GPU_IDLE_S, and no
#: episode being made, the app asks ComfyUI to drop the models it loaded.
PRESENCE_S = 20.0
#: Browsers slow a hidden tab's timers to about once a minute, so the idle
#: window is well above that: a background tab still counts as open.
GPU_IDLE_S = 150.0
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
#: The largest body any route takes: a photo's data URL, with room to spare.
MAX_BODY_BYTES = 32 * 1024 * 1024
_PROJECT = r"([a-z0-9][a-z0-9-]{3,31})"


def available_languages() -> list:
    """`[{"code", "name"}]` for each language file, English first. A file with no
    `_name` (or that does not parse) is not offered: the page would have nothing
    to call it."""
    found = []
    for path in sorted(I18N_DIR.glob("*.json")):
        if not _I18N_FILE.fullmatch(path.name):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = data.get("_name") if isinstance(data, dict) else None
        if isinstance(name, str) and name:
            found.append({"code": path.stem, "name": name})
    found.sort(key=lambda lang: (lang["code"] != "en", lang["code"]))
    return found


class Run:
    """One episode, or one Scene Producer stage: its events so far, and
    whether it has ended."""

    def __init__(self, run_id: str, label: dict | None = None) -> None:
        self.id = run_id
        #: What a stage run makes -- its project and stage -- so a reloaded
        #: page finds the board it belongs to; empty for an episode.
        self.label = dict(label or {})
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


def _body_length(headers) -> int:
    """The request body's declared length. A negative one is refused: reading
    -1 bytes would block until the client hung up. So is one past
    MAX_BODY_BYTES, before a byte of it is read."""
    length = int(headers.get("Content-Length") or 0)
    if length < 0:
        raise ValueError("a negative Content-Length")
    if length > MAX_BODY_BYTES:
        raise ValueError(f"a body over {MAX_BODY_BYTES // (1024 * 1024)} MB")
    return length


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

    def _busy(self, settling: str = "") -> bool:
        """Anything being made: a run of this app, or a project's attempt
        ComfyUI may still be working on that no run is following -- after a
        restart, or once Studio lost touch with it. ``settling`` is the
        project a reconciliation is about to look at."""
        if any(not r.finished for r in self._runs.values()):
            return True
        return bool(self._held(settling))

    def _held(self, settling: str = "") -> list:
        return [pid for pid, pending in producer.held_projects(self.studio.settings.out_dir)
                if pid != settling and pending.get("state") in producer.MAY_BE_RUNNING]

    def start(self, choices: dict) -> Run:
        """Check the choices against the server, then run the episode on a thread.
        A bad choice raises here, before any thread starts. A second episode while
        one is being made is refused: it would wait an hour behind it on the GPU."""
        return self._launch(lambda: self.studio.build_graph(choices),
                            lambda _graph: self.studio.stream(choices))

    def start_stage(self, project_id: str, stage: str) -> Run:
        """One Scene Producer stage of a project, the same way: checked here,
        run on a thread, refused while anything else is being made."""
        # The project's own pending attempt is prepare_stage's to refuse, by name.
        return self._launch(lambda: self.studio.prepare_stage(project_id, stage),
                            self.studio.stream_stage,
                            {"project_id": project_id, "stage": stage}, settling=project_id)

    def start_project(self, choices: dict) -> tuple:
        """A new Scene Producer project for these choices, with its first
        Stills run started: ``(project, run)``. The choices are checked
        against the server before the project is made."""
        with self._lock:
            self._refuse_if_busy()
        self.studio.build_graph(choices)
        project = producer.new_project(self.studio.settings.out_dir,
                                       workflow=str(self.studio.workflow_for(choices)),
                                       choices=choices)
        return project, self.start_stage(project["project_id"], "stills")

    def start_reconcile(self, project_id: str) -> Run:
        """Look again at a project's pending attempt (Studio.reconcile_stage),
        as a run the page follows like a stage. Refused while anything else
        is being made, and for a project with nothing pending."""
        def check() -> str:
            pending = producer.load_project(self.studio.settings.out_dir,
                                            project_id)["pending"]
            if pending is None:
                raise producer.ProjectError("this project has no run to look at again")
            label["stage"] = pending["stage"]
            return project_id

        label = {"project_id": project_id, "stage": "", "reconcile": True}
        return self._launch(check, self.studio.reconcile_stage, label, settling=project_id)

    def resume_pending(self, wait: float = 3600.0) -> None:
        """At start-up, look again at every project left pending, one after
        another: a run that was going when Studio stopped is followed, a lost
        submit found, a finished result saved."""
        for project_id, _pending in producer.held_projects(self.studio.settings.out_dir):
            try:
                run = self.start_reconcile(project_id)
            except Exception:  # noqa: BLE001 - it stays pending, and the page can look again
                traceback.print_exc()
                continue
            with run.cond:
                run.cond.wait_for(lambda: run.finished, timeout=wait)

    def forget(self, project_id: str, attempt_id: str) -> dict:
        """The person gives up on a pending attempt (producer.forget_attempt);
        refused while a run of this app is following it."""
        with self._lock:
            if any(not r.finished and r.label.get("project_id") == project_id
                   for r in self._runs.values()):
                raise Busy("this project's run is being followed right now; wait for it")
            return producer.forget_attempt(self.studio.settings.out_dir, project_id, attempt_id)

    def _launch(self, check, events, label: dict | None = None, settling: str = "") -> Run:
        """``check()`` raises for a request that cannot run, before any thread
        starts; ``events(checked)`` then yields the run's events on one."""
        with self._lock:
            self._refuse_if_busy(settling)
        checked = check()
        run = Run(uuid.uuid4().hex[:12], label)
        with self._lock:
            self._refuse_if_busy(settling)
            self._runs[run.id] = run
            self._gpu_dirty = True
        try:
            threading.Thread(target=self._drive, args=(run, events, checked), daemon=True).start()
        except BaseException:
            with self._lock:
                self._runs.pop(run.id, None)
            raise
        return run

    def _refuse_if_busy(self, settling: str = "") -> None:
        if any(not r.finished for r in self._runs.values()):
            raise Busy(_BUSY)
        held = self._held(settling)
        if held:
            raise Busy(f"project {held[0]} is waiting to hear how its last run ended; open it "
                       "and look again, or forget that run")

    def listing(self) -> list:
        """Every episode and stage this app has started since it was launched,
        newest first; a stage with its project and stage."""
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.started, reverse=True)
        return [dict(r.label, id=r.id, finished=r.finished, started=r.started) for r in runs]

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

    def _drive(self, run: Run, events, checked) -> None:
        """Feed a run its events. A run always ends -- a failure inside it is
        its last event -- or the GPU guard would stay held for good."""
        try:
            for event in events(checked):
                run.add(event)
        except Exception as exc:  # noqa: BLE001 - reported on the run, and in the console
            traceback.print_exc()
            run.add({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            if not run.finished:
                run.add({"type": "error", "message": "the run stopped without saying why"})

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)


class Handler(BaseHTTPRequestHandler):
    server_version = "RadioPlayStudio/0.1"
    #: Seconds a socket may sit with nothing sent before the connection is
    #: closed (socketserver applies it to each connection). Without it, a
    #: client that declares a body and sends only part of it holds a handler
    #: thread for good (Codex QA, 2026-10-23). It bounds a silence, not a
    #: request: an upload that keeps moving, or an event stream that writes
    #: a keep-alive every HEARTBEAT_S, never meets it.
    timeout = 60

    # -- plumbing -----------------------------------------------------------
    @property
    def runs(self) -> Runs:
        return self.server.runs  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # quiet: progress is the log
        return

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):  # [::1] or [::1]:8765, and nothing else after it
            name, _, rest = host.partition("]")
            name += "]"
            if rest and not re.fullmatch(r":\d{1,5}", rest):
                return False
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
        except TimeoutError:            # the client went quiet mid-request
            self.close_connection = True
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
        if path == "/api/languages":
            return self._send_json(HTTPStatus.OK, {"languages": available_languages()})
        if path.startswith("/i18n/"):
            name = path[len("/i18n/"):]
            if _I18N_FILE.fullmatch(name):
                return self._send_file(I18N_DIR / name, "application/json; charset=utf-8")
        if path == "/api/episodes":
            return self._send_json(HTTPStatus.OK, {"episodes": self.runs.listing()})
        if path == "/api/projects":
            return self._send_json(HTTPStatus.OK, {"projects": producer.list_projects(
                self.runs.studio.settings.out_dir)})
        match = re.fullmatch(r"/api/projects/" + _PROJECT, path)
        if match:
            return self._project(match.group(1))
        match = re.fullmatch(r"/api/episodes/([0-9a-f]{12})/events", path)
        if match:
            return self._events(match.group(1))
        if path.startswith("/episodes/"):
            return self._episode_file(path[len("/episodes/"):])
        if path.startswith("/bundled-photos/"):
            name = path[len("/bundled-photos/"):]
            if _BUNDLED_NAME.fullmatch(name):
                return self._send_file(BUNDLED_PHOTO_DIR / name, "image/jpeg")
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _post(self) -> None:
        path = urlsplit(self.path).path
        project = re.fullmatch(r"/api/projects/" + _PROJECT + r"/(draft|stages|reconcile|forget)",
                               path)
        if path not in ("/api/episodes", "/api/presence", "/api/photos",
                        "/api/projects") and not project:
            self._discard_body()
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            self._discard_body()
            return self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
        if path == "/api/presence":
            return self._presence()
        if path == "/api/photos":
            return self._photo()
        if path == "/api/projects":
            return self._new_project()
        if project:
            return {"draft": self._draft, "stages": self._stage, "reconcile": self._reconcile,
                    "forget": self._forget}[project.group(2)](project.group(1))
        try:
            length = _body_length(self.headers)
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

    def _discard_body(self) -> None:
        """Read the body of a request refused before it is read, so the client
        gets the refusal: answered with its body unread, a connection is reset
        under load, and the client sees that instead of the status. One past
        the size limit, or with no readable length, is not read: the
        connection closes after the answer."""
        try:
            length = _body_length(self.headers)
        except ValueError:
            self.close_connection = True
            return
        while length > 0:
            chunk = self.rfile.read(min(length, 1 << 16))
            if not chunk:
                break
            length -= len(chunk)

    def _body(self) -> dict:
        """The posted JSON object; ValueError for anything else."""
        length = _body_length(self.headers)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("the body must be a JSON object")
        return body

    def _project(self, project_id: str) -> None:
        try:
            project = producer.load_project(self.runs.studio.settings.out_dir, project_id)
        except producer.NoProject as exc:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except producer.ProjectError as exc:
            return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        self._send_json(HTTPStatus.OK, {"project": project})

    def _new_project(self) -> None:
        """{"choices": {...}} -> a new project, its first Stills run started."""
        try:
            choices = self._body().get("choices", {})
            if not isinstance(choices, dict):
                raise ValueError("choices must be an object")
        except (ValueError, TypeError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {exc}"})
        self._start(lambda: self.runs.start_project(choices),
                    lambda made: {"project_id": made[0]["project_id"], "id": made[1].id})

    def _stage(self, project_id: str) -> None:
        """{"stage": "stills" | "video" | "publish"} -> the stage's run."""
        try:
            stage = self._body().get("stage")
        except (ValueError, TypeError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {exc}"})
        self._start(lambda: self.runs.start_stage(project_id, stage),
                    lambda run: {"project_id": project_id, "id": run.id})

    def _reconcile(self, project_id: str) -> None:
        """{} -> a run that looks again at the project's pending attempt."""
        self._start(lambda: self.runs.start_reconcile(project_id),
                    lambda run: {"project_id": project_id, "id": run.id,
                                 "stage": run.label.get("stage", "")})

    def _forget(self, project_id: str) -> None:
        """{"attempt_id": id} -> the project, its pending attempt given up."""
        try:
            attempt_id = str(self._body().get("attempt_id") or "")
            project = self.runs.forget(project_id, attempt_id)
        except Busy as exc:
            return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except producer.NoProject as exc:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (producer.ProjectError, ValueError, TypeError) as exc:
            return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        self._send_json(HTTPStatus.OK, {"project": project})

    def _start(self, start, reply) -> None:
        """Start a run and answer for it: 202 with ``reply(started)``, or why
        not -- 409 while the GPU or the project is busy, 404 for no project,
        503 when ComfyUI cannot be reached, 400 for anything else wrong."""
        try:
            started = start()
        except (Busy, producer.StageInFlight) as exc:
            return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except producer.NoProject as exc:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except ServerUnreachable as exc:
            return self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        except (ValueError, GraphError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        self._send_json(HTTPStatus.ACCEPTED, reply(started))

    def _draft(self, project_id: str) -> None:
        """{"draft": {...}, "revision": n} -> the new revision. Answered while
        anything is being made: only stage starts take the GPU."""
        try:
            body = self._body()
            project = producer.save_draft(self.runs.studio.settings.out_dir, project_id,
                                          body.get("draft"), body.get("revision"))
        except producer.NoProject as exc:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except producer.StaleDraft as exc:
            return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {exc}"})
        self._send_json(HTTPStatus.OK, {"revision": project["draft"]["revision"]})

    def _presence(self) -> None:
        """A page saying it is open, or closing: {"page": id, "closing": bool}."""
        try:
            length = _body_length(self.headers)
            body = json.loads(self.rfile.read(length) or b"{}")
            page = str(body["page"])[:64]
        except (ValueError, TypeError, KeyError):
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "send {'page': id}"})
        if body.get("closing"):
            self.runs.page_closed(page)
        else:
            self.runs.page_open(page)
        self._send_json(HTTPStatus.OK, {"ok": True})

    def _photo(self) -> None:
        """A character's photo: {"data": data URL} -> {"file": stored name}."""
        try:
            length = _body_length(self.headers)
            if length > MAX_PHOTO_BYTES * 4 // 3 + 4096:
                raise PhotoError(f"the picture is over {MAX_PHOTO_BYTES // (1024 * 1024)} MB")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise PhotoError("send {'data': picture}")
            stored = upload_photo(self.runs.studio.settings.comfyui_url, body.get("data"))
        except (PhotoError, ValueError, TypeError) as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        self._send_json(HTTPStatus.OK, {"file": stored})

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
    # A stage left pending when Studio last stopped is settled before anything new is sent.
    threading.Thread(target=server.runs.resume_pending, daemon=True).start()
    return server
