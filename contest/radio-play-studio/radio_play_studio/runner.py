"""Drive one episode through the Comfy SDK.

The SDK talks to whatever `COMFY_BASE_URL` names, and with it unset it talks to
Comfy Cloud. This app is for your own machine, so `make_client` always sets the
bridge address first: a missing setting must never turn into a cloud run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from . import producer
from .config import (CLOUD_MACHINES, MACHINE_CHOICES, MACHINES, MATRIX_COLUMN, NOT_FOR_THIS_MACHINE,
                     REFERENCE_TWIN_SUFFIX, UNPROVEN, Settings, detect_machine, menu_lookup,
                     reference_twin, validate_url)
from .form import (CLOUD_WRITERS, EPISODE_NODE_TYPE, FIELDS_BY_KEY, MY_STORY_BANK, WRITER, apply_choices,
                   describe_form, locate_nodes)
from .graph import fetch_object_info, load_workflow, workflow_to_api_prompt

#: The still-engine dropdown the Still pictures field sets for the characters.
DIRECTOR = FIELDS_BY_KEY["stills"].node_type
STILLS_INPUT = FIELDS_BY_KEY["stills"].input
#: What a way card names, and where each is read in its workflow: (label,
#: node type, input). Read from the graph, so a card can never drift from it.
USES = (("video", "OTR_VideoDirector", "character_video_model"),
        ("stills", "OTR_VideoDirector", "character_image_model"),
        ("voices", "OTR_CastLock", "char_voice_engine"),
        ("music", "OTR_StableAudioTheme", "engine"),
        ("writer", WRITER, "creative_writing_model"))
#: The choice that names which menu workflow a run uses: routing, not a field.
WORKFLOW_CHOICE = "workflow"
#: A still engine that paints a character from that character's picture is named
#: after its base engine with this suffix (flux3_klein -> flux3_klein_ref).
REFERENCE_SUFFIX = "_ref"

VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv")

#: How long one poll waits for the job before looking again, how many failed
#: polls in a row are forgiven (a restart of the bridge, a dropped connection),
#: and the longest pause between them. About ten minutes of outage in all.
POLL_WINDOW_S = 60.0
#: The bridge's status for a job it took and can no longer find upstream.
EXPIRED = "expired"
#: A submit whose answer was lost is looked for in ComfyUI's own queue and
#: history this many times, this far apart, before an empty answer is
#: believed: one empty look never lets the project send the run again.
LOST_SUBMIT_LOOKS = 3
LOST_SUBMIT_GAP_S = 20.0
#: How much of ComfyUI's history a lost submit is looked for in: it was sent
#: moments before Studio lost its answer, so it is among the newest.
HISTORY_LOOKBACK = 256
MAX_TRANSIENT = 20
BACKOFF_CAP_S = 30.0
#: A live stream the SDK gives up on is opened again while ComfyUI's own queue
#: still holds the job: on 2026-10-23 a 16 GB foley episode's bar sat at 78 %
#: for three hours while ComfyUI filmed shot after shot, because the SDK's status
#: check timed out against a busy server and nothing listened after that. Only
#: this many losses in a row, with no frame between them, end the watch.
LIVE_RETRY_LIMIT = 30
LIVE_RETRY_PAUSE_S = 20.0


def relevant_choices(fields: list, machine: str) -> list:
    """The form for one machine: the still models and writers its column of
    machine_choices.json rules out are left off the dropdowns, never the one a
    row starts on. A choice the file does not name stays (unmeasured is not
    "no"), and a run can still send any choice the pack accepts -- this is what
    the page offers, not what the server allows."""
    column = MACHINE_CHOICES.get("machines", {}).get(MATRIX_COLUMN.get(machine, ""), {})
    kinds = {"stills": column.get("image") or {}, "writer": column.get("writer") or {}}
    out = []
    for row in fields:
        cells = kinds.get(row["key"])
        if cells and row.get("choices"):
            row = dict(row, choices=[c for c in row["choices"]
                                     if c == row.get("default")
                                     or cells.get(str(c)) not in NOT_FOR_THIS_MACHINE])
        if row["key"] == "writer" and machine in CLOUD_MACHINES and row.get("choices"):
            # A machine with no GPU writes in the cloud: the table has no
            # CPU verdict for a local writer, and on a CPU one would crawl.
            row = dict(row, choices=[c for c in row["choices"]
                                     if c == row.get("default") or c in CLOUD_WRITERS])
        out.append(row)
    return out


class StreamGap:
    """A lost live stream that is being opened again; the frames in between are
    gone, so a run with a gap is never a timing measurement."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class EpisodeMissing(RuntimeError):
    """The job finished but no video came back. A finished job is not a finished
    episode until the file is on disk."""


def make_client(api_url: str) -> Any:
    """A Comfy SDK client aimed at the local bridge, never at Comfy Cloud."""
    os.environ["COMFY_BASE_URL"] = validate_url("the bridge address", api_url)
    try:
        from comfy_sdk import Comfy
    except ImportError as exc:  # pragma: no cover - exercised by `doctor`
        raise RuntimeError("the Comfy SDK is not installed: pip install -r requirements.txt") from exc
    return Comfy(client_info="radio-play-studio")


def normalize_event(event: Any, graph_size: int = 0) -> dict | None:
    """One SDK event as a plain dict for the page, or None for kinds the page
    does not show (previews). Matched by class name so the app does not depend
    on where the SDK keeps its event classes.

    `graph_size` is the number of nodes submitted. comfy-api-proxy's own
    fraction cannot be the episode's: its per-node frames (nodes_total=0) are
    one sampler's steps, and its graph-wide frames count only the nodes ComfyUI
    has STARTED (core drops pending nodes from progress_state), so either read
    100% at the first finished still. The page's bar only moves forward, and on
    the first live 16 GB run (2026-10-19) it sat at 100% for the hour of video
    that followed. The share of the episode is finished nodes out of the graph."""
    kind = type(event).__name__
    if kind == "Progress":
        value = float(getattr(event, "value", 0.0) or 0.0)
        nodes_total = getattr(event, "nodes_total", None)
        step, steps = getattr(event, "step", None), getattr(event, "steps", None)
        if nodes_total == 0:
            return {"type": "step", "value": value, "step": step, "steps": steps}
        note = {"type": "progress", "value": value,
                "message": str(getattr(event, "message", "") or "")}
        if nodes_total and graph_size:
            done = int(getattr(event, "nodes_done", 0) or 0)
            running = (step / steps) if step is not None and steps else 0.0
            note["value"] = max(0.0, min(1.0, (done + min(running, 1.0)) / graph_size))
            note["nodes_done"] = min(done, graph_size)
            note["nodes_total"] = graph_size
        return note
    if kind == "StatusChange":
        return {"type": "status", "status": str(getattr(event, "status", ""))}
    if kind == "OutputReady":
        out = getattr(event, "output", None)
        return {"type": "output", "name": str(getattr(out, "name", ""))}
    return None


def stage_note(event: Any, graph_size: int) -> dict | None:
    """One SDK event as a Scene Producer stage shows it: as
    :func:`normalize_event` has it, with the node each progress frame belongs
    to. A stage's bar is drawn from the raw node, step and total, never from
    the episode's calibrated share: each stage runs its own part of the
    graph, and an episode's timings would misread every one of them."""
    note = normalize_event(event, graph_size)
    if note is not None and note["type"] in ("progress", "step"):
        node = getattr(event, "current_node", None)
        if node is not None:
            note["node"] = str(node)
    return note


def pick_episode_output(job: Any, node_id: str) -> Any:
    """The finished episode's video, from the mux node and nowhere else.

    The silent cut (`OTR_SignalLostVideo`) also publishes a video, so "any video
    in the job" would hand back the wrong file as the finished episode; if the
    mux node offers none, the run fails rather than guess."""
    for out in job.get_outputs(node_id) or []:
        if str(getattr(out, "name", "")).lower().endswith(VIDEO_SUFFIXES):
            return out
    raise EpisodeMissing(
        "the run finished but the final mux returned no video to download. The episode may "
        "still have been published: look in your ComfyUI output folder under otr/obs/ (or "
        "your OTR_OBS_DIR). If it is not there, the ComfyUI console says why "
        "('MasterAudioMux FAILED')."
    )


def save_episode_output(job: Any, node_id: str, out_dir: Path) -> Path:
    """Download the finished episode's video (:func:`pick_episode_output`)
    into a fresh folder under ``out_dir`` and return its path."""
    out = pick_episode_output(job, node_id)
    dest_dir = Path(out_dir) / uuid.uuid4().hex[:12]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(str(out.name)).name
    out.to_file(str(dest))
    if not dest.is_file() or dest.stat().st_size == 0:
        raise EpisodeMissing(f"the episode was not saved to {dest}")
    return dest


def submit_was_refused(exc: BaseException) -> bool:
    """True when a submit that raised certainly queued nothing: the server
    answered with a refusal (a 4xx -- node 63 refusing a direction is one),
    or no connection was ever made. Anything else -- a read timeout, a
    connection dropped mid-request, a 5xx from the bridge -- may have
    queued the run."""
    status = getattr(exc, "http_status", None)
    if isinstance(status, int):
        return 400 <= status < 500
    return type(exc).__name__ in ("ConnectError", "ConnectTimeout")


def is_transient(exc: BaseException) -> bool:
    """A failure of the connection or the bridge, not a verdict on the job."""
    if isinstance(exc, OSError):
        return True
    status = getattr(exc, "http_status", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return type(exc).__module__.split(".")[0] == "httpx"


def settle(job: Any, sleep=time.sleep) -> Any:
    """Wait for the job to end and return it, raising `JobFailed` if it failed.

    The live event stream is only a view; this poll is the truth. It rides out a
    bridge restart or a dropped connection (the render is still running on the
    GPU) instead of reporting a healthy multi-hour render as stopped."""
    failures = 0
    while True:
        try:
            job.wait(timeout=POLL_WINDOW_S)
            return job.result()
        except TimeoutError:
            failures = 0  # the server answered; the job is simply still running
        except Exception as exc:  # noqa: BLE001 - decided below
            if not is_transient(exc):
                raise
            failures += 1
            if failures > MAX_TRANSIENT:
                raise
            sleep(min(BACKOFF_CAP_S, 2.0 ** failures))


class Studio:
    """One configured connection to a ComfyUI + the Old-Time Radio pack."""

    def __init__(self, settings: Settings, *, client_factory: Callable[[str], Any] = make_client,
                 fetch: Callable[[str], dict] = fetch_object_info,
                 now: Callable[[], float] = time.monotonic,
                 read_output: Callable[[str, str], bytes] | None = None,
                 comfy_json: Callable[[str], Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 comfy_key: str | None = None) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._fetch = fetch
        self._now = now
        #: ``read_output(subfolder, name)``: a file from ComfyUI's output folder.
        self._read_output = read_output or (
            lambda subfolder, name: producer.read_output(settings.comfyui_url, subfolder, name))
        #: ``comfy_json(path)``: ComfyUI's own answer for ``/queue``, ``/history``.
        self._comfy_json = comfy_json or (
            lambda path: producer.read_json(settings.comfyui_url, path))
        self._sleep = sleep
        #: The Comfy API key runs are submitted with: from the page, or the
        #: OTR_COMFY_API_KEY the command line passes in. Held in memory only --
        #: never written, never logged, never sent back to a page.
        self.comfy_key: str | None = (comfy_key or "").strip() or None
        #: The machine ComfyUI reported, once it has ("" for none on the menu).
        self._machine: str | None = None
        #: What each menu workflow uses, by (path, modification time).
        self._uses: dict = {}

    def key_status(self) -> dict:
        """Which keys the cloud picks can find, never the keys themselves. The
        Comfy key is the one this server holds. Google and OpenRouter are read
        by ComfyUI itself, from its environment or from a key file in the pack
        folder (nodes/_otr_shared/api_key_files.py); the page can only see the
        files, so it says where to put a key and whether a file is there."""
        root = Path(self.settings.workflow_path).resolve().parent.parent
        status = {"comfy": {"set": bool(self.comfy_key), "env": ["OTR_COMFY_API_KEY"]}}
        for lane, env in (("google", ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OTR_GOOGLE_API_KEY"]),
                          ("openrouter", ["OPENROUTER_API_KEY"])):
            files = (root / f"{lane}.secret", root / f"{lane}_api_key.location")
            status[lane] = {"file": any(f.is_file() and f.stat().st_size > 0 for f in files),
                            "path": str(files[0]), "env": env}
        return status

    def _live(self, job: Any) -> Iterator[Any]:
        """The job's live events, opened again whenever the stream is lost while
        ComfyUI's queue still holds the job; each loss is a StreamGap. Raises
        the last loss once the job has left the queue, or after
        LIVE_RETRY_LIMIT losses with no progress between them -- the status
        snapshot every connection opens with is not progress (Sonnet QA)."""
        losses = 0
        while True:
            try:
                for event in job.events():
                    if type(event).__name__ != "StatusChange":
                        losses = 0
                    yield event
                return
            except Exception as exc:  # noqa: BLE001 - judged below, raised if final
                losses += 1
                if losses > LIVE_RETRY_LIMIT or not self._still_queued(getattr(job, "id", None)):
                    raise
                yield StreamGap(exc)
                self._sleep(LIVE_RETRY_PAUSE_S)

    def _still_queued(self, job_id: Any) -> bool:
        """Whether ComfyUI's queue says it holds this job, running or waiting. A
        queue that cannot be read says nothing, and the watch ends there: the
        run is still waited on to its end (settle), only the bar stops."""
        if not job_id:
            return False
        try:
            queue = self._comfy_json("/queue")
        except Exception:  # noqa: BLE001 - no answer, no reason to listen again
            return False
        for key in ("queue_running", "queue_pending"):
            for entry in (queue or {}).get(key) or ():
                if isinstance(entry, (list, tuple)) and len(entry) > 1 and entry[1] == job_id:
                    return True
        return False

    def _submit(self, client, workflow):
        """Submit with the Comfy key when there is one: the SDK sends it as
        extra_data.api_key_comfy_org, which pays for Comfy partner nodes."""
        if self.comfy_key:
            return client.submit(workflow, api_key=self.comfy_key)
        return client.submit(workflow)

    def options(self, schemas: dict | None = None, workflow_path: Path | None = None) -> list:
        workflow = load_workflow(workflow_path or self.settings.workflow_path)
        if schemas is None:
            schemas = self._fetch(self.settings.comfyui_url)
        graph = workflow_to_api_prompt(workflow, schemas)
        return describe_form(schemas, graph, locate_nodes(workflow))

    def menu(self, schemas: dict | None = None) -> list | None:
        """The machines and ways of filming this server can offer, as
        `[{"key", "ways": [{"key", "workflow", "uses"}]}]`, or None when it was
        started on a workflow that is not on the menu (it then runs that one, as
        before). A way whose workflow is not beside the server's own is left
        off. Given the node definitions, each way also says what its workflow
        uses (``uses``: video, stills, voices, music, writer)."""
        base = Path(self.settings.workflow_path)
        if menu_lookup(base.name) is None:
            return None
        machines = []
        for machine, ways in MACHINES:
            present = []
            for way, name in ways:
                path = base.parent / name
                if not path.is_file():
                    continue
                entry = {"key": way, "workflow": name, "proven": name not in UNPROVEN}
                if schemas is not None:
                    entry["uses"] = self.uses(path, schemas)
                present.append(entry)
            if present:
                machines.append({"key": machine, "ways": present})
        return machines

    def uses(self, path: Path, schemas: dict) -> dict:
        """`{label: value}` for what this workflow runs (USES), as it saved it;
        a label its graph cannot answer is left out. Remembered per file and
        modification time: the menu asks for every way on every page load."""
        # The node definitions are part of the key: a pack update can change
        # what an unchanged file converts to (Sonnet QA of edfe9fde).
        defs = tuple(json.dumps(schemas.get(node_type, {}), sort_keys=True, default=str)
                     for node_type in sorted({t for _, t, _ in USES}))
        stamp = (str(path), path.stat().st_mtime_ns, hashlib.sha256("".join(defs).encode()).hexdigest())
        if stamp in self._uses:
            return self._uses[stamp]
        try:
            workflow = load_workflow(path)
            graph = workflow_to_api_prompt(workflow, schemas)
        except Exception:  # noqa: BLE001 -- a card without its models still works,
            return {}        # and a failed read is never remembered
        out = {}
        for label, node_type, name in USES:
            ids = [str(n["id"]) for n in workflow.get("nodes") or [] if n.get("type") == node_type]
            value = graph.get(ids[0], {}).get("inputs", {}).get(name) if len(ids) == 1 else None
            if isinstance(value, str) and value:
                out[label] = value
        self._uses[stamp] = out
        return out

    def detected_machine(self) -> str | None:
        """The menu machine ComfyUI says it runs on, asked until it answers and
        then remembered: the card does not change while this server runs, and a
        ComfyUI busy rendering can be slow to answer /system_stats."""
        if self._machine is None:
            try:
                self._machine = detect_machine(self._comfy_json("/system_stats")) or ""
            except Exception:  # noqa: BLE001 -- a guess the page can live without
                return None
        return self._machine or None

    def resolve_workflow(self, name: str | None) -> Path:
        """The workflow file a run or a form asks for by name: the server's own
        when none is named, otherwise a menu entry beside it. Anything else is
        refused -- the page can only pick among the graphs the pack ships."""
        base = Path(self.settings.workflow_path)
        if not name or name == base.name:
            return base
        if self.menu() is None or menu_lookup(str(name)) is None:
            raise ValueError(f"{name!r} is not one of the workflows this page offers")
        path = base.parent / str(name)
        if not path.is_file() or path.resolve().parent != base.resolve().parent:
            raise ValueError(f"{name} is not in {base.parent}")
        return path

    def form(self, workflow: str | None = None) -> dict:
        """What the page draws for one workflow: its form, the still picks that
        use photos, and the machine menu with the machine ComfyUI reports. One
        read of the node definitions serves it all; on a ComfyUI that is
        rendering, that read is the slow part of opening the page."""
        path = self.resolve_workflow(workflow)
        schemas = self._fetch(self.settings.comfyui_url)
        fields = self.options(schemas, path)
        on_menu = menu_lookup(path.name)
        if on_menu is not None:
            fields = relevant_choices(fields, on_menu[0])
        body = {"fields": fields, "photo_stills": self.photo_stills(fields, schemas, path.name)}
        machines = self.menu(schemas)
        if machines is not None:
            machine, way = menu_lookup(path.name)
            body["menu"] = {"machines": machines, "detected": self.detected_machine(),
                            "current": {"machine": machine, "way": way, "workflow": path.name}}
        return body

    def photo_stills(self, fields: list, schemas: dict | None = None,
                     workflow: str | None = None) -> list:
        """The Still pictures choices whose My Story run paints the characters
        from their photos, so the page can say when a photo will go unused.

        Read from the graph a My Story run actually submits (its reference twin
        when one ships), by the rule build_graph_from applies: a pick keeps the
        graph's `<pick>_ref` character stills, and a `_ref` engine picked by name
        is a reference engine itself. Empty when that graph paints no character
        still from a picture, which makes every photo unused."""
        stills = next((f for f in fields if f.get("key") == "stills"), None)
        if stills is None:
            return []
        choices = [str(c) for c in stills.get("choices") or []]
        path = self.workflow_for({"bank": MY_STORY_BANK, WORKFLOW_CHOICE: workflow})
        workflow = load_workflow(path)
        if schemas is None:
            schemas = self._fetch(self.settings.comfyui_url)
        graph = workflow_to_api_prompt(workflow, schemas)
        director = graph.get(locate_nodes(workflow).get(DIRECTOR, ""), {}).get("inputs", {})
        saved = str(director.get(STILLS_INPUT) or "")
        out = [c for c in choices if c.endswith(REFERENCE_SUFFIX)]
        if saved.endswith(REFERENCE_SUFFIX) and saved[:-len(REFERENCE_SUFFIX)] in choices:
            out.insert(0, saved[:-len(REFERENCE_SUFFIX)])
        return out

    def workflow_for(self, choices: dict) -> Path:
        """The workflow these choices run: the menu workflow they name (the
        server's own by default), and for a My Story run that workflow's
        reference twin when one ships beside it, so its characters are painted
        from their pictures; anything else runs the workflow as named."""
        named = self.resolve_workflow((choices or {}).get(WORKFLOW_CHOICE))
        if str((choices or {}).get("bank") or "") == MY_STORY_BANK:
            twin = reference_twin(named)
            if twin.is_file() and not named.stem.endswith(REFERENCE_TWIN_SUFFIX):
                return twin
        return named

    def build_graph(self, choices: dict) -> tuple[dict, dict]:
        """`(api_graph, node_ids)` for these choices, from the workflow on disk
        and the server's live node definitions."""
        return self.build_graph_from(self.workflow_for(choices), choices)

    def build_graph_from(self, workflow_path: Path, choices: dict) -> tuple[dict, dict]:
        """`(api_graph, node_ids)` for these choices on a named workflow -- the
        one a Scene Producer project recorded on its first run."""
        workflow = load_workflow(workflow_path)
        schemas = self._fetch(self.settings.comfyui_url)
        graph = workflow_to_api_prompt(workflow, schemas)
        node_ids = locate_nodes(workflow)
        director = graph.get(node_ids.get(DIRECTOR, ""), {}).get("inputs", {})
        saved_character = director.get(STILLS_INPUT)
        apply_choices(graph, schemas, node_ids,
                      {k: copy.deepcopy(v) for k, v in (choices or {}).items() if k != WORKFLOW_CHOICE})
        # The page's Still pictures box shows the base model (flux3_klein), so a
        # run sends it back; on a graph whose characters use that model WITH
        # references (the My Story twin), they keep them -- never a silent loss.
        chosen = str((choices or {}).get("stills") or "")
        if isinstance(saved_character, str) and chosen and saved_character == chosen + REFERENCE_SUFFIX:
            director[STILLS_INPUT] = saved_character
        return graph, node_ids

    def stream(self, choices: dict) -> Iterator[dict]:
        """Run one episode and report it as events: `submitted`, then `progress`,
        `status` and `output` as the server works, then exactly one of `done`
        (with the saved video's path) or `error`. Never raises: a failure is the
        last event, so the page and the command line show the same ending."""
        try:
            graph, node_ids = self.build_graph(choices)
            acts = graph.get(node_ids.get(WRITER, ""), {}).get("inputs", {}).get("act_count", "default")
            timing_file = timing_path(self.settings.out_dir, self.workflow_for(choices), acts, graph)
            clock = ProgressClock(load_timing(timing_file), now=self._now)
            client = self._client_factory(self.settings.api_url)
            job = self._submit(client, client.workflows.from_json(graph))
            yield {"type": "submitted"}
            stream_whole, last_status, most_done, settled = True, None, 0, False
            try:
                for event in self._live(job):
                    if isinstance(event, StreamGap):
                        stream_whole = False
                        yield {"type": "note", "message": f"live progress dropped ({event.exc}); "
                                                          "listening again"}
                        continue
                    if type(event).__name__ == "StatusChange":
                        status = str(getattr(event, "status", ""))
                        # comfy-api-proxy opens every connection with the job's
                        # status, so a repeat means the SDK reconnected and the
                        # frames in between are gone.
                        if status == last_status:
                            stream_whole = False
                        last_status = status
                    note = normalize_event(event, len(graph))
                    if note and note["type"] == "progress" and getattr(event, "nodes_total", None):
                        done = int(getattr(event, "nodes_done", 0) or 0)
                        most_done = max(most_done, done)
                        # Every node ComfyUI started has finished and none is running.
                        settled = (done == int(event.nodes_total)
                                   and getattr(event, "current_node", None) is None)
                    if note and note["type"] in ("progress", "step"):
                        node = getattr(event, "current_node", None)
                        # Only a graph-wide frame can say "between nodes"; a
                        # per-node frame without a node id says nothing.
                        if node is not None or note["type"] == "progress":
                            clock.observe(node)
                        clock.bar(node, getattr(event, "step", None), getattr(event, "steps", None))
                        if clock.calibrated:
                            note = {"type": "progress", "value": clock.share(),
                                    "message": note.get("message", ""),
                                    "minutes_left": round(clock.seconds_left() / 60.0)}
                    if note:
                        yield note
            except Exception as exc:  # noqa: BLE001 - the stream is a view, settle() is the truth
                stream_whole = False
                yield {"type": "note", "message": f"live progress was lost ({exc}); still waiting on the job"}
            clock.observe(None)  # the stream is over: no node is running any more
            settle(job)
            dest = save_episode_output(job, node_ids[EPISODE_NODE_TYPE], self.settings.out_dir)
            # A lost or reconnected stream bills its gap to whichever node was
            # seen last, and those times would mislead every later run. So does a
            # starved one: ComfyUI keeps ONE socket per client id (server.py,
            # sockets[sid] = ws), so a second listener on the same job takes its
            # messages and this stream goes quiet with no error. On 2026-10-19 a
            # recorder did that 30 s in; the app saw 2 of 22 nodes finish and
            # billed the whole 76 minutes to the writer. A stream that never saw
            # most of the graph finish is not a measurement, and neither is one
            # whose last graph-wide frame still had a node running (starved late,
            # mid-video: that node's hour would be billed to whatever came last).
            if (stream_whole and settled and most_done * 2 >= len(graph)
                    and clock.timings()):
                save_timing(timing_file, clock.timings())
        except Exception as exc:  # noqa: BLE001 - reported to the person, not swallowed
            yield {"type": "error", "message": describe_failure(exc)}
            return
        yield {"type": "done", "path": str(dest), "run": dest.parent.name, "file": dest.name}

    # -- the Scene Producer -------------------------------------------------
    def prepare_stage(self, project_id: str, stage: str) -> dict:
        """Everything one stage run of a project sends, checked before any
        thread starts: its direction, and the project's recorded workflow and
        first-run choices with the stage's values set. Raises ProjectError,
        GraphError or ServerUnreachable."""
        project = producer.load_project(self.settings.out_dir, project_id)
        if project["pending"] is not None:
            raise producer.StageInFlight("this project already has a stage in flight")
        direction = producer.build_direction(project, stage)
        workflow = Path(project["workflow"])
        if not workflow.is_file():
            # A pack update can rename or retire the workflow a project started on.
            raise producer.ProjectError(f"this project's workflow is not at {workflow} any "
                                        "more; start a new project")
        graph, node_ids = self.build_graph_from(workflow, project["choices"])
        latest = producer.latest_bundle(project)
        replay_from = producer.bundle_subfolder(project_id, latest["out"]) if latest else ""
        producer.patch_graph(graph, node_ids, direction, replay_from)
        # The direction that is SENT is built again when the attempt is saved
        # (producer.begin_attempt), from the project as it stands then.
        return {"project_id": project_id, "stage": stage, "graph": graph,
                "node_ids": node_ids}

    def stream_stage(self, plan: dict) -> Iterator[dict]:
        """Run one prepared stage (:meth:`prepare_stage`) and report it as
        events, as :meth:`stream` does: ``submitted``, the server's progress
        -- each frame naming its node, for the stage's own bar -- then exactly
        one of ``done`` (with the stage and the project, and for Publish the
        saved video) or ``error``. Never raises.

        The attempt is saved before anything is sent, and its job id the
        moment the server answers. A run the server may or may not have
        taken, a job whose end could not be confirmed, and a finished job
        whose result is not yet saved here all stay pending: nothing more is
        sent for the project until they are settled. Only a run refused
        outright, or a job that ended failed with no complete bundle, lets
        the project try again."""
        out_dir = self.settings.out_dir
        pid, stage = plan["project_id"], plan["stage"]
        base = {"stage": stage, "project_id": pid}

        def error(message: str) -> dict:
            return dict(base, type="error", message=message)

        try:
            direction = producer.begin_attempt(out_dir, pid, stage, plan["graph"],
                                               plan["node_ids"])
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            yield error(describe_failure(exc))
            return
        plan = dict(plan, direction=direction)
        attempt = direction["attempt_id"]
        try:
            client = self._client_factory(self.settings.api_url)
            workflow = client.workflows.from_json(plan["graph"])
        except Exception as exc:  # noqa: BLE001 - nothing was sent
            producer.abandon_attempt(out_dir, pid, attempt, describe_failure(exc))
            yield error(describe_failure(exc))
            return
        try:
            job = self._submit(client, workflow)
        except Exception as exc:  # noqa: BLE001 - decided below
            if submit_was_refused(exc):
                producer.abandon_attempt(out_dir, pid, attempt, describe_failure(exc))
                yield error(describe_failure(exc))
            else:
                producer.hold_attempt(out_dir, pid, attempt, "uncertain", describe_failure(exc))
                yield error(f"{describe_failure(exc)}. The server may have taken this run, "
                            "so Studio sends this project nothing more until it has looked.")
            return
        try:
            producer.record_job(out_dir, pid, attempt, job.id)
        except Exception as exc:  # noqa: BLE001 - the job runs; only the note of it failed
            yield dict(base, type="note", message=f"the job's id was not saved ({exc})")
        yield dict(base, type="submitted")
        yield from self._follow(plan, job, error)

    def _follow(self, plan: dict, job: Any, error: Callable[[str], dict]) -> Iterator[dict]:
        """A taken job, to its end: its progress, then its result saved here."""
        out_dir = self.settings.out_dir
        pid, stage = plan["project_id"], plan["stage"]
        attempt = plan["direction"]["attempt_id"]
        base = {"stage": stage, "project_id": pid}
        try:
            for event in self._live(job):
                if isinstance(event, StreamGap):
                    yield dict(base, type="note", message=f"live progress dropped ({event.exc}); "
                                                          "listening again")
                    continue
                note = stage_note(event, len(plan["graph"]))
                if note:
                    yield note
        except Exception as exc:  # noqa: BLE001 - the stream is a view, settle() is the truth
            yield dict(base, type="note",
                       message=f"live progress was lost ({exc}); still waiting on the job")
        failure = None
        try:
            settle(job)
        except Exception as exc:  # noqa: BLE001 - decided below
            if type(exc).__name__ != "JobFailed":
                producer.hold_attempt(out_dir, pid, attempt, "running", describe_failure(exc))
                yield error(f"lost touch with the run ({describe_failure(exc)}); it may "
                            "still be going, so Studio sends this project nothing more "
                            "until it has looked again.")
                return
            if str(getattr(job, "status", "")) == EXPIRED:
                # "Expired" is the bridge's word for a job it took that ComfyUI
                # no longer has -- a ComfyUI restarted under a run, say. It is
                # settled the way a job the bridge has forgotten is: ComfyUI's
                # own queue and history say how it ended, a complete bundle is
                # still adopted, and a project ComfyUI cannot be asked about
                # stays pending instead of being freed to make it again.
                #
                # NOT the "expired" seen live on 2026-10-23 (Sonnet QA): that
                # one was the bridge's event STREAM, whose closing status comes
                # from one snapshot taken the moment ComfyUI reports the run
                # done -- it can read the history before the run is written
                # there and the queue after the run has left it. The SDK's
                # result() polls again before it raises, and ComfyUI's history
                # holds that run as a success. (What settle() itself saw that
                # night was not logged; the bundle was adopted either way.)
                yield self._settle_forgotten(plan, str(job.id), error)
                return
            failure = exc
        if stage == "publish":
            yield self._finish_publish(plan, job, failure, error)
        else:
            yield self._finish_bundle(plan, failure, error)

    # -- settling an attempt Studio lost track of ------------------------------
    def reconcile_stage(self, project_id: str) -> Iterator[dict]:
        """Settle a project's pending attempt without ever sending it again --
        after Studio restarts, or when a run's end could not be confirmed.
        Events as :meth:`stream_stage`; never raises.

        A job whose id was saved is followed to its end through the bridge,
        or, when the bridge has forgotten it, read from ComfyUI's own queue
        and history: a Stills or Video run's complete bundle is adopted
        however its job ended, and a Publish run's video is fetched from
        where ComfyUI saved it. A submit whose answer was lost is looked for
        by the attempt its direction carries: one match is its job; several,
        or no answer, keep it pending; only an empty look, repeated, lets the
        project try again (LOST_SUBMIT_LOOKS)."""
        out_dir = self.settings.out_dir
        try:
            project = producer.load_project(out_dir, project_id)
            pending = project["pending"]
            if pending is None:
                yield {"type": "done", "stage": "", "project_id": project_id, "settled": True}
                return
            sent = producer.load_attempt(out_dir, project_id, pending["attempt_id"])
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            yield {"type": "error", "stage": "", "project_id": project_id,
                   "message": describe_failure(exc)}
            return
        stage, attempt, state = pending["stage"], pending["attempt_id"], pending["state"]
        base = {"stage": stage, "project_id": project_id}

        def error(message: str) -> dict:
            return dict(base, type="error", message=message)

        plan = {"project_id": project_id, "stage": stage, "graph": sent["graph"],
                "node_ids": sent["node_ids"], "direction": sent["direction"]}
        job_id = pending.get("job_id")
        if not job_id:
            found, detail = self._find_lost_submit(project_id, attempt)
            if found == "none":
                producer.abandon_attempt(out_dir, project_id, attempt,
                                         "ComfyUI never took this run")
                yield error("ComfyUI never took this run, so nothing was made; the project "
                            "can run this stage again.")
                return
            if found != "one":
                producer.hold_attempt(out_dir, project_id, attempt, "uncertain", detail)
                yield error(detail)
                return
            job_id = detail
            producer.record_job(out_dir, project_id, attempt, job_id)
            state = "running"
        try:
            client = self._client_factory(self.settings.api_url)
            job = client.jobs.get(job_id)
        except Exception as exc:  # noqa: BLE001 - decided below
            if type(exc).__name__ != "NotFound":
                producer.hold_attempt(out_dir, project_id, attempt, state, describe_failure(exc))
                yield error(f"could not ask about this run ({describe_failure(exc)}); it "
                            "stays pending, and Studio sends this project nothing more "
                            "until it has looked again.")
                return
            yield self._settle_forgotten(plan, job_id, error)
            return
        yield dict(base, type="submitted")
        yield from self._follow(plan, job, error)

    def _find_lost_submit(self, project_id: str, attempt_id: str) -> tuple:
        """Look for an attempt in ComfyUI's queue and recent history:
        ``("one", job id)``, ``("many" | "unknown", why)``, or ``("none", "")``
        once every look came back empty."""
        for look in range(LOST_SUBMIT_LOOKS):
            if look:
                self._sleep(LOST_SUBMIT_GAP_S)
            try:
                queue = self._comfy_json("/queue")
                history = self._comfy_json(f"/history?max_items={HISTORY_LOOKBACK}")
            except Exception as exc:  # noqa: BLE001 - no answer is not an empty answer
                return "unknown", (f"ComfyUI could not be asked whether it took this run "
                                   f"({describe_failure(exc)}); it stays pending until it can.")
            rows = [row for key in ("queue_running", "queue_pending")
                    for row in (queue.get(key) or [] if isinstance(queue, dict) else [])]
            prompts = [(row[1], row[2]) for row in rows
                       if isinstance(row, (list, tuple)) and len(row) > 2]
            for job_id, entry in (history.items() if isinstance(history, dict) else ()):
                prompt = entry.get("prompt") if isinstance(entry, dict) else None
                if isinstance(prompt, (list, tuple)) and len(prompt) > 2:
                    prompts.append((job_id, prompt[2]))
            ids = sorted({str(job_id) for job_id, prompt in prompts
                          if producer.carries_attempt(prompt, project_id, attempt_id)})
            if len(ids) == 1:
                return "one", ids[0]
            if ids:
                return "many", (f"{len(ids)} runs in ComfyUI carry this attempt "
                                f"({', '.join(ids[:3])}); Studio sends this project nothing "
                                "more until the extra ones are gone.")
        return "none", ""

    def _comfy_knows(self, job_id: str) -> tuple:
        """Where ComfyUI itself has a job: ``("queued", None)``, ``("done" |
        "failed", its history entry)`` or ``("gone", None)``."""
        queue = self._comfy_json("/queue")
        for key in ("queue_running", "queue_pending"):
            for row in (queue.get(key) or [] if isinstance(queue, dict) else []):
                if isinstance(row, (list, tuple)) and len(row) > 1 and str(row[1]) == job_id:
                    return "queued", None
        history = self._comfy_json("/history/" + urllib.parse.quote(job_id, safe=""))
        entry = history.get(job_id) if isinstance(history, dict) else None
        if not isinstance(entry, dict):
            return "gone", None
        status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
        ok = status.get("status_str") == "success" and status.get("completed", True)
        return ("done" if ok else "failed"), entry

    def _settle_forgotten(self, plan: dict, job_id: str,
                          error: Callable[[str], dict]) -> dict:
        """A job the bridge has no record of (it restarted, or let it go):
        ComfyUI's own queue and history say how it ended."""
        out_dir = self.settings.out_dir
        pid, stage = plan["project_id"], plan["stage"]
        attempt = plan["direction"]["attempt_id"]
        try:
            where, entry = self._comfy_knows(job_id)
        except Exception as exc:  # noqa: BLE001 - no answer settles nothing
            producer.hold_attempt(out_dir, pid, attempt, "running", describe_failure(exc))
            return error(f"ComfyUI could not be asked how this run ended "
                         f"({describe_failure(exc)}); it stays pending.")
        if where == "queued":
            producer.hold_attempt(out_dir, pid, attempt, "running",
                                  "ComfyUI is still making this run")
            return error("ComfyUI is still making this run, but the bridge has lost it; look "
                         "again once it ends.")
        if stage != "publish":
            # A complete bundle recovers the result however the job ended.
            failure = None if where == "done" else RuntimeError(
                "ComfyUI no longer has this run" if where == "gone" else "the run failed")
            return self._finish_bundle(plan, failure, error)
        if where != "done":
            message = ("ComfyUI no longer has this run" if where == "gone"
                       else "the Publish run failed")
            producer.abandon_attempt(out_dir, pid, attempt, message)
            return error(message + "; publish again.")
        try:
            dest = self._save_history_video(entry, plan["node_ids"][EPISODE_NODE_TYPE])
        except EpisodeMissing as exc:
            producer.abandon_attempt(out_dir, pid, attempt, str(exc))
            return error(str(exc))
        except Exception as exc:  # noqa: BLE001 - made, not yet saved: never made again
            producer.hold_attempt(out_dir, pid, attempt, "finishing", describe_failure(exc))
            return error(f"the episode is made, but saving it here failed "
                         f"({describe_failure(exc)}); Studio saves it without making it again.")
        producer.finish_publish(out_dir, pid, attempt, dest.parent.name, dest.name)
        return {"type": "done", "stage": "publish", "project_id": pid, "path": str(dest),
                "run": dest.parent.name, "file": dest.name}

    def _save_history_video(self, entry: dict, node_id: str) -> Path:
        """A finished Publish run's video, from the mux node's own outputs in
        ComfyUI's history, read through ``/view`` into a fresh folder here."""
        outputs = (entry.get("outputs") or {}).get(str(node_id)) or {}
        files = [item for items in outputs.values() if isinstance(items, list)
                 for item in items if isinstance(item, dict)]
        found = next((item for item in files
                      if str(item.get("filename", "")).lower().endswith(VIDEO_SUFFIXES)
                      and item.get("type", "output") == "output"), None)
        if found is None:
            raise EpisodeMissing("the Publish run finished, but ComfyUI's history names no "
                                 "video from the final mux; publish again.")
        name = Path(str(found["filename"])).name
        data = self._read_output(str(found.get("subfolder") or ""), name)
        dest_dir = Path(self.settings.out_dir) / uuid.uuid4().hex[:12]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        dest.write_bytes(data)
        if dest.stat().st_size == 0:
            raise EpisodeMissing(f"the episode was not saved to {dest}")
        return dest

    def _finish_publish(self, plan: dict, job: Any, failure: BaseException | None,
                        error: Callable[[str], dict]) -> dict:
        """A Publish job's end: its video saved here, or why not."""
        out_dir = self.settings.out_dir
        pid, attempt = plan["project_id"], plan["direction"]["attempt_id"]
        if failure is not None:
            producer.abandon_attempt(out_dir, pid, attempt, describe_failure(failure))
            return error(describe_failure(failure))
        try:
            dest = save_episode_output(job, plan["node_ids"][EPISODE_NODE_TYPE], out_dir)
        except EpisodeMissing as exc:
            producer.abandon_attempt(out_dir, pid, attempt, str(exc))
            return error(str(exc))
        except Exception as exc:  # noqa: BLE001 - made, not yet saved: never made again
            producer.hold_attempt(out_dir, pid, attempt, "finishing", describe_failure(exc))
            return error(f"the episode is made, but saving it here failed "
                         f"({describe_failure(exc)}); Studio saves it without making it again.")
        producer.finish_publish(out_dir, pid, attempt, dest.parent.name, dest.name)
        return {"type": "done", "stage": "publish", "project_id": pid, "path": str(dest),
                "run": dest.parent.name, "file": dest.name}

    def _finish_bundle(self, plan: dict, failure: BaseException | None,
                       error: Callable[[str], dict]) -> dict:
        """A Stills or Video job's end: its bundle adopted, or why not. A job
        that reports failure after its bundle was written still hands that
        bundle over; one with no complete bundle lets the project try again."""
        out_dir = self.settings.out_dir
        direction = plan["direction"]
        pid, attempt, stage = plan["project_id"], direction["attempt_id"], direction["stage"]
        try:
            project = producer.load_project(out_dir, pid)
            view = producer.adopt_bundle(out_dir, project, direction["out"], self._read_output,
                                         attempt_id=attempt, source=direction.get("source"),
                                         stage=stage)
        except producer.BundleMissing as exc:
            if failure is not None:
                producer.abandon_attempt(out_dir, pid, attempt, describe_failure(failure))
                return error(describe_failure(failure))
            producer.hold_attempt(out_dir, pid, attempt, "finishing", str(exc))
            return error(f"the run finished, but its bundle is not complete yet ({exc}); "
                         "Studio looks for it again before sending this project anything more.")
        except Exception as exc:  # noqa: BLE001 - finished, not yet adopted: never run again
            producer.hold_attempt(out_dir, pid, attempt, "finishing", describe_failure(exc))
            return error(f"the run finished, but its bundle could not be read here "
                         f"({describe_failure(exc)}); Studio tries again without running it.")
        producer.finish_bundle(out_dir, pid, attempt, view)
        return {"type": "done", "stage": stage, "project_id": pid, "bundle": direction["out"]}


class ProgressClock:
    """The episode's share done, by this machine's own clock.

    Counting nodes cannot track time: on the first live 16 GB run one node of 22
    (the video render) took most of the hour, so a node-count bar sat near 80%
    for it. The clock times each node while comfy-api-proxy names it as running,
    and once a run has succeeded those times are the weights: the bar is the
    finished nodes' last-run seconds plus the running node's elapsed seconds
    (held under its last-run length) over the last-run total. Without a history
    it has nothing to say and the caller keeps the node count.

    Nodes too quick to be named between two throttled frames are never timed;
    they took about no time, so they weigh about nothing.

    A NODE'S OWN BAR, WHEN IT CAN BE TRUSTED. Run length follows the story
    (one-act foley took 57 to 76 minutes on the 5080), so last-run seconds
    alone left the bar up to 17 points behind the clock. Since ec55add4 the
    video node -- most of the run -- draws "render frames done of the episode"
    on its own bar. A node's bar is trusted while it moves forward under one
    total; ComfyUI's start frame (0 of 1) is ignored, and a bar that ever
    resets (an image node's per-picture sampler) is distrusted for the run.
    A trusted node's share comes from its bar, and once it is a tenth of the
    way through and has run a minute, its minutes left come from its own pace."""

    #: A node running longer than last time stops short of its full share.
    RUNNING_CAP = 0.95
    #: How far into a node, and how long, before its pace says how long it has left.
    PACE_MIN_FRACTION = 0.10
    PACE_MIN_SECONDS = 60.0

    def __init__(self, history: dict | None = None, now: Callable[[], float] = time.monotonic) -> None:
        self.history = {}
        for node, seconds in (history or {}).items():
            if isinstance(seconds, (int, float)) and seconds > 0:
                self.history[str(node)] = float(seconds)
        self.total = sum(self.history.values())
        self._now = now
        self.current: str | None = None
        self._since = 0.0
        self.spent: dict = {}
        self.finished: set = set()
        self._bars: dict = {}           # node -> (step, steps) last seen
        self._distrusted: set = set()
        self._best: dict = {}           # node -> the most of its share shown so far

    @property
    def calibrated(self) -> bool:
        return self.total > 0

    def observe(self, node_id: Any) -> None:
        """The node the server says is running now (None between nodes)."""
        node = None if node_id is None else str(node_id)
        if node == self.current:
            return
        now = self._now()
        if self.current is not None:
            self.spent[self.current] = self.spent.get(self.current, 0.0) + now - self._since
            self.finished.add(self.current)
        self.current, self._since = node, now
        if node is not None:
            self.finished.discard(node)

    def bar(self, node_id: Any, step: Any, steps: Any) -> None:
        """A node's own progress bar, as the server last reported it."""
        if node_id is None or not isinstance(step, (int, float)) or not isinstance(steps, (int, float)):
            return
        if steps <= 1:
            return                      # ComfyUI's start frame for every node
        node = str(node_id)
        seen = self._bars.get(node)
        if seen is not None and (steps != seen[1] or step < seen[0]):
            self._distrusted.add(node)
        self._bars[node] = (step, steps)

    def fraction(self) -> float | None:
        """How far the running node is by its own bar, or None if it has none
        worth trusting."""
        node = self.current
        if node is None or node in self._distrusted or node not in self._bars:
            return None
        step, steps = self._bars[node]
        return max(0.0, min(1.0, step / steps))

    def _elapsed(self) -> float:
        return self.spent.get(self.current, 0.0) + self._now() - self._since

    def share(self) -> float:
        done = sum(self.history.get(n, 0.0) for n in self.finished)
        running = 0.0
        if self.current is not None:
            expected = self.history.get(self.current, 0.0)
            f = self.fraction()
            if f is not None:
                running = min(f, self.RUNNING_CAP) * expected
            else:
                running = min(self._elapsed(), self.RUNNING_CAP * expected)
            # A bar that was trusted and then reset falls back to time, which can
            # be lower: a node's share never goes back down (Cursor QA, e53c6cdf).
            running = max(running, self._best.get(self.current, 0.0))
            self._best[self.current] = running
        return max(0.0, min(0.99, (done + running) / self.total))

    def seconds_left(self) -> float:
        if self.current is None:
            return max(0.0, self.total * (1.0 - self.share()))
        later = sum(v for n, v in self.history.items()
                    if n not in self.finished and n != self.current)
        elapsed = self._elapsed()
        f = self.fraction()
        if f is not None and f >= self.PACE_MIN_FRACTION and elapsed >= self.PACE_MIN_SECONDS:
            node_left = elapsed * (1.0 - f) / f
        else:
            node_left = max(0.0, self.history.get(self.current, 0.0) - elapsed)
        return node_left + later

    def timings(self) -> dict:
        """Seconds each named node ran, the running one up to now."""
        out = dict(self.spent)
        if self.current is not None:
            out[self.current] = out.get(self.current, 0.0) + self._now() - self._since
        return out


def timing_path(out_dir: Path, workflow_path: Path, acts: Any, graph: dict | None = None) -> Path:
    """Where this machine's node times for one workflow and act count are kept.

    The act count is in the key because a three-act play renders about three
    times the beats of a one-act one. So is the graph's shape (each node id and
    its type): a re-saved workflow can renumber its nodes, and times kept under
    old ids would weigh nodes that no longer exist."""
    shape = ""
    if graph:
        pairs = sorted((str(k), str(v.get("class_type", ""))) for k, v in graph.items())
        shape = "." + hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()[:8]
    return Path(out_dir) / "timing" / f"{Path(workflow_path).stem}.acts-{acts}{shape}.json"


def load_timing(path: Path) -> dict:
    try:
        nodes = json.loads(Path(path).read_text(encoding="utf-8")).get("nodes", {})
    except (OSError, ValueError, AttributeError):
        return {}
    return nodes if isinstance(nodes, dict) else {}


#: How much one run moves the saved time of a node it timed. One-act foley runs
#: on the 5080 took 64 to 76 minutes on 2026-10-19 (the beat count follows the
#: story), so the last run alone is a noisy guess: the proof run came in 13%
#: under the one before and the bar trailed the clock by up to 15 points.
#: Averaging halves the pull of any one long or short story.
TIMING_WEIGHT = 0.5


def save_timing(path: Path, nodes: dict) -> None:
    """Fold this run's node times into the saved ones, as a running average. A
    node this run did not time (ComfyUI served it from cache) keeps its saved
    time, so a mostly cached run cannot shrink the history the next full run is
    read against. Written to a side file and swapped in, so a reader never sees
    half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {k: v for k, v in load_timing(path).items() if isinstance(v, (int, float))}
    for node, seconds in nodes.items():
        old = merged.get(node)
        blended = seconds if old is None else (1 - TIMING_WEIGHT) * old + TIMING_WEIGHT * seconds
        merged[node] = round(blended, 1)
    body = {"nodes": merged, "total_s": round(sum(merged.values()), 1)}
    side = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    side.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(side, path)


def describe_failure(exc: BaseException) -> str:
    """A failure as one readable line, with the SDK's node-level detail if any."""
    detail = getattr(exc, "error", None)
    text = f"{type(exc).__name__}: {exc}"
    return f"{text} ({detail})" if detail else text
