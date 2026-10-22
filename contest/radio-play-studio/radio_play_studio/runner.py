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
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import REFERENCE_TWIN_SUFFIX, Settings, reference_twin, validate_url
from .form import (EPISODE_NODE_TYPE, FIELDS_BY_KEY, MY_STORY_BANK, WRITER, apply_choices,
                   describe_form, locate_nodes)
from .graph import fetch_object_info, load_workflow, workflow_to_api_prompt

#: The still-engine dropdown the Still pictures field sets for the characters.
DIRECTOR = FIELDS_BY_KEY["stills"].node_type
STILLS_INPUT = FIELDS_BY_KEY["stills"].input

VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv")

#: How long one poll waits for the job before looking again, how many failed
#: polls in a row are forgiven (a restart of the bridge, a dropped connection),
#: and the longest pause between them. About ten minutes of outage in all.
POLL_WINDOW_S = 60.0
MAX_TRANSIENT = 20
BACKOFF_CAP_S = 30.0


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
                 now: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._fetch = fetch
        self._now = now

    def options(self) -> list:
        workflow = load_workflow(self.settings.workflow_path)
        schemas = self._fetch(self.settings.comfyui_url)
        graph = workflow_to_api_prompt(workflow, schemas)
        return describe_form(schemas, graph, locate_nodes(workflow))

    def workflow_for(self, choices: dict) -> Path:
        """The workflow these choices run: a My Story run uses the workflow's
        reference twin when one ships beside it, so its characters are painted
        from their pictures; anything else runs the workflow as named."""
        named = Path(self.settings.workflow_path)
        if str((choices or {}).get("bank") or "") == MY_STORY_BANK:
            twin = reference_twin(named)
            if twin.is_file() and not named.stem.endswith(REFERENCE_TWIN_SUFFIX):
                return twin
        return named

    def build_graph(self, choices: dict) -> tuple[dict, dict]:
        """`(api_graph, node_ids)` for these choices, from the workflow on disk
        and the server's live node definitions."""
        workflow = load_workflow(self.workflow_for(choices))
        schemas = self._fetch(self.settings.comfyui_url)
        graph = workflow_to_api_prompt(workflow, schemas)
        node_ids = locate_nodes(workflow)
        director = graph.get(node_ids.get(DIRECTOR, ""), {}).get("inputs", {})
        saved_character = director.get(STILLS_INPUT)
        apply_choices(graph, schemas, node_ids, copy.deepcopy(choices))
        # The page's Still pictures box shows the base model (flux3_klein), so a
        # run sends it back; on a graph whose characters use that model WITH
        # references (the My Story twin), they keep them -- never a silent loss.
        chosen = str((choices or {}).get("stills") or "")
        if isinstance(saved_character, str) and chosen and saved_character == chosen + "_ref":
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
            job = client.submit(client.workflows.from_json(graph))
            yield {"type": "submitted"}
            stream_whole, last_status, most_done, settled = True, None, 0, False
            try:
                for event in job.events():
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
            out = pick_episode_output(job, node_ids[EPISODE_NODE_TYPE])
            dest_dir = self.settings.out_dir / uuid.uuid4().hex[:12]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(str(out.name)).name
            out.to_file(str(dest))
            if not dest.is_file() or dest.stat().st_size == 0:
                raise EpisodeMissing(f"the episode was not saved to {dest}")
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
        yield {"type": "done", "path": str(dest), "run": dest_dir.name, "file": dest.name}


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
