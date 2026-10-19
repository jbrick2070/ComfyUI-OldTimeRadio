"""Where everything lives: the ComfyUI server, the Comfy API v2 bridge, the
workflow file and the folder finished episodes are saved to.

Two URLs on purpose. `comfyui_url` is ComfyUI itself (its `/object_info` is the
authority on what a node accepts). `api_url` is `comfy-api-proxy`, the bridge
that lets the Comfy SDK drive a self-hosted ComfyUI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_API_URL = "http://127.0.0.1:8189"
WORKFLOW_NAME = "otr_app.json"
PACK_DIR_NAME = "ComfyUI-OldTimeRadio"


class WorkflowNotFound(FileNotFoundError):
    """No `otr_app.json` where the caller said it would be, or anywhere usual."""


@dataclass(frozen=True)
class Settings:
    comfyui_url: str
    api_url: str
    workflow_path: Path
    out_dir: Path


def _checkout_workflow() -> Path:
    # contest/radio-play-studio/radio_play_studio/config.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "workflows" / WORKFLOW_NAME


def workflow_candidates(explicit: str | None = None) -> list[Path]:
    """Every place the workflow may be, in the order they are tried."""
    found: list[Path] = []
    if explicit:
        found.append(Path(explicit).expanduser())
    env_file = os.environ.get("OTR_APP_WORKFLOW", "").strip()
    if env_file:
        found.append(Path(env_file).expanduser())
    pack_dir = os.environ.get("OTR_PACK_DIR", "").strip()
    if pack_dir:
        found.append(Path(pack_dir).expanduser() / "workflows" / WORKFLOW_NAME)
    comfy_dir = os.environ.get("COMFYUI_DIR", "").strip()
    if comfy_dir:
        found.append(Path(comfy_dir).expanduser() / "custom_nodes" / PACK_DIR_NAME
                     / "workflows" / WORKFLOW_NAME)
    found.append(_checkout_workflow())
    return found


def find_workflow(explicit: str | None = None) -> Path:
    """The first existing candidate. A path given on the command line that does
    not exist is an error, never a quiet fall-through to some other file."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise WorkflowNotFound(f"--workflow {path} does not exist")
        return path
    tried = workflow_candidates()
    for path in tried:
        if path.is_file():
            return path
    raise WorkflowNotFound(
        "could not find %s. Set OTR_PACK_DIR to the Old-Time Radio pack folder "
        "(custom_nodes/%s), or COMFYUI_DIR to your ComfyUI folder, or pass "
        "--workflow. Looked in: %s"
        % (WORKFLOW_NAME, PACK_DIR_NAME, ", ".join(str(p) for p in tried))
    )


def load_settings(*, comfyui_url: str | None = None, api_url: str | None = None,
                  workflow: str | None = None, out_dir: str | None = None) -> Settings:
    return Settings(
        comfyui_url=(comfyui_url or os.environ.get("COMFYUI_URL")
                     or DEFAULT_COMFYUI_URL).rstrip("/"),
        api_url=(api_url or os.environ.get("COMFY_BASE_URL")
                 or DEFAULT_API_URL).rstrip("/"),
        workflow_path=find_workflow(workflow),
        out_dir=Path(out_dir or os.environ.get("RPS_OUT_DIR") or "episodes").expanduser(),
    )
