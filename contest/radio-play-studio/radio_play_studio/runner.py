"""Drive one episode through the Comfy SDK.

The SDK talks to whatever `COMFY_BASE_URL` names, and with it unset it talks to
Comfy Cloud. This app is for your own machine, so `make_client` always sets the
bridge address first: a missing setting must never turn into a cloud run.
"""
from __future__ import annotations

import copy
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import Settings, validate_url
from .form import EPISODE_NODE_TYPE, apply_choices, describe_form, locate_nodes
from .graph import fetch_object_info, load_workflow, workflow_to_api_prompt

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


def normalize_event(event: Any) -> dict | None:
    """One SDK event as a plain dict for the page, or None for kinds the page
    does not show (previews). Matched by class name so the app does not depend
    on where the SDK keeps its event classes."""
    kind = type(event).__name__
    if kind == "Progress":
        value = float(getattr(event, "value", 0.0) or 0.0)
        nodes_total = getattr(event, "nodes_total", None)
        if nodes_total == 0:
            # comfy-api-proxy sends ComfyUI's per-node sampler progress with
            # nodes_total=0: the share of ONE node's steps, not of the episode.
            # On the bar it read 100% at the first still and stayed there for
            # the hour of video that followed (first live 16 GB run, 2026-10-19).
            return {"type": "step", "value": value,
                    "step": getattr(event, "step", None), "steps": getattr(event, "steps", None)}
        note = {"type": "progress", "value": value,
                "message": str(getattr(event, "message", "") or "")}
        if nodes_total:
            note["nodes_done"] = int(getattr(event, "nodes_done", 0) or 0)
            note["nodes_total"] = int(nodes_total)
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
                 fetch: Callable[[str], dict] = fetch_object_info) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._fetch = fetch

    def options(self) -> list:
        workflow = load_workflow(self.settings.workflow_path)
        schemas = self._fetch(self.settings.comfyui_url)
        graph = workflow_to_api_prompt(workflow, schemas)
        return describe_form(schemas, graph, locate_nodes(workflow))

    def build_graph(self, choices: dict) -> tuple[dict, dict]:
        """`(api_graph, node_ids)` for these choices, from the workflow on disk
        and the server's live node definitions."""
        workflow = load_workflow(self.settings.workflow_path)
        schemas = self._fetch(self.settings.comfyui_url)
        graph = workflow_to_api_prompt(workflow, schemas)
        node_ids = locate_nodes(workflow)
        apply_choices(graph, schemas, node_ids, copy.deepcopy(choices))
        return graph, node_ids

    def stream(self, choices: dict) -> Iterator[dict]:
        """Run one episode and report it as events: `submitted`, then `progress`,
        `status` and `output` as the server works, then exactly one of `done`
        (with the saved video's path) or `error`. Never raises: a failure is the
        last event, so the page and the command line show the same ending."""
        try:
            graph, node_ids = self.build_graph(choices)
            client = self._client_factory(self.settings.api_url)
            job = client.submit(client.workflows.from_json(graph))
            yield {"type": "submitted"}
            try:
                for event in job.events():
                    note = normalize_event(event)
                    if note:
                        yield note
            except Exception as exc:  # noqa: BLE001 - the stream is a view, settle() is the truth
                yield {"type": "note", "message": f"live progress was lost ({exc}); still waiting on the job"}
            settle(job)
            out = pick_episode_output(job, node_ids[EPISODE_NODE_TYPE])
            dest_dir = self.settings.out_dir / uuid.uuid4().hex[:12]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(str(out.name)).name
            out.to_file(str(dest))
            if not dest.is_file() or dest.stat().st_size == 0:
                raise EpisodeMissing(f"the episode was not saved to {dest}")
        except Exception as exc:  # noqa: BLE001 - reported to the person, not swallowed
            yield {"type": "error", "message": describe_failure(exc)}
            return
        yield {"type": "done", "path": str(dest), "run": dest_dir.name, "file": dest.name}


def describe_failure(exc: BaseException) -> str:
    """A failure as one readable line, with the SDK's node-level detail if any."""
    detail = getattr(exc, "error", None)
    text = f"{type(exc).__name__}: {exc}"
    return f"{text} ({detail})" if detail else text
