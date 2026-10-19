"""Drive one episode through the Comfy SDK.

The SDK talks to whatever `COMFY_BASE_URL` names, and with it unset it talks to
Comfy Cloud. This app is for your own machine, so `make_client` always sets the
bridge address first: a missing setting must never turn into a cloud run.
"""
from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import Settings
from .form import EPISODE_NODE_TYPE, apply_choices, describe_form, locate_nodes
from .graph import fetch_object_info, load_workflow, workflow_to_api_prompt

VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv")


class EpisodeMissing(RuntimeError):
    """The job finished but no video came back. A finished job is not a finished
    episode until the file is on disk."""


def make_client(api_url: str) -> Any:
    """A Comfy SDK client aimed at the local bridge, never at Comfy Cloud."""
    os.environ["COMFY_BASE_URL"] = api_url
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
        return {"type": "progress", "value": float(getattr(event, "value", 0.0) or 0.0),
                "message": str(getattr(event, "message", "") or "")}
    if kind == "StatusChange":
        return {"type": "status", "status": str(getattr(event, "status", ""))}
    if kind == "OutputReady":
        out = getattr(event, "output", None)
        return {"type": "output", "name": str(getattr(out, "name", ""))}
    return None


def pick_episode_output(job: Any, node_id: str) -> Any:
    """The finished episode's video among a job's outputs: the mux node's first,
    then any video in the job."""
    for pool in (job.get_outputs(node_id), job.outputs):
        for out in pool or []:
            if str(getattr(out, "name", "")).lower().endswith(VIDEO_SUFFIXES):
                return out
    raise EpisodeMissing(
        "the run finished but returned no video. Look in your ComfyUI output folder "
        "under otr/obs/ and in the ComfyUI console for 'MasterAudioMux FAILED'."
    )


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
            for event in job.events():
                note = normalize_event(event)
                if note:
                    yield note
            job.result()  # authoritative: polls the job and raises JobFailed
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
