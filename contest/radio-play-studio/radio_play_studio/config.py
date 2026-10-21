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
from urllib.parse import urlsplit

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_API_URL = "http://127.0.0.1:8189"
#: The contest default: the 16 GB NVIDIA foley lane, where LTX 2.5 films each beat
#: with its own sound under the voices (operator, 2026-10-19: "i like foley
#: better"), in its all-open-weights form -- FLUX 3 Klein stills and the open
#: ffmpeg Lanczos 1080p step (no model) instead of NVIDIA RTX VSR (operator, 2026-10-20: "we skip
#: VSR, we stay open weight"). Any other shipped graph runs with --workflow.
WORKFLOW_NAME = "otr_16gb_foley_open.json"
#: A workflow's reference twin sits beside it: the same graph with every
#: character still painted from that character's reference picture
#: (flux3_klein_ref). My Story runs use it when it exists -- the one story whose
#: characters have pictures (operator, 2026-10-20: "16gb foley flux3 ref in my
#: story"); every other story keeps the workflow as named.
REFERENCE_TWIN_SUFFIX = "_flux3_ref"


def reference_twin(workflow_path: Path) -> Path:
    """``workflows/x.json`` -> ``workflows/x_flux3_ref.json`` (it may not exist)."""
    path = Path(workflow_path)
    return path.with_name(path.stem + REFERENCE_TWIN_SUFFIX + path.suffix)
PACK_DIR_NAME = "ComfyUI-OldTimeRadio"


class WorkflowNotFound(FileNotFoundError):
    """No workflow where the caller said it would be, or anywhere usual."""


@dataclass(frozen=True)
class Settings:
    comfyui_url: str
    api_url: str
    workflow_path: Path
    out_dir: Path


def _checkout_workflow() -> Path:
    # contest/radio-play-studio/radio_play_studio/config.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "workflows" / WORKFLOW_NAME


def validate_url(label: str, value: str) -> str:
    """`value` as a clean http(s) base URL, or ValueError. A blank or hostless
    address must never reach the Comfy SDK: it reads a blank `COMFY_BASE_URL` as
    "use Comfy Cloud"."""
    url = (value or "").strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{label} must be an http(s) address with a host, got {value!r}")
    return url


def _configured_workflow_sources() -> list:
    """`[(where, path)]` for each place the person pointed at, most specific
    first. These are promises: if one is set, the file must be there."""
    sources: list = []
    env_file = os.environ.get("OTR_APP_WORKFLOW", "").strip()
    if env_file:
        sources.append(("OTR_APP_WORKFLOW", Path(env_file).expanduser()))
    pack_dir = os.environ.get("OTR_PACK_DIR", "").strip()
    if pack_dir:
        sources.append(("OTR_PACK_DIR", Path(pack_dir).expanduser() / "workflows" / WORKFLOW_NAME))
    comfy_dir = os.environ.get("COMFYUI_DIR", "").strip()
    if comfy_dir:
        sources.append(("COMFYUI_DIR", Path(comfy_dir).expanduser() / "custom_nodes" / PACK_DIR_NAME
                        / "workflows" / WORKFLOW_NAME))
    return sources


def find_workflow(explicit: str | None = None) -> Path:
    """The workflow to run. Whatever the person named (the flag, or an
    environment variable) must exist: a missing path is an error, never a quiet
    fall-through to some other graph, which on a small card would run the wrong
    machine's workflow. Only when nothing was named is a checkout's own
    `workflows/<WORKFLOW_NAME>` used."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise WorkflowNotFound(f"--workflow {path} does not exist")
        return path
    for where, path in _configured_workflow_sources():
        if path.is_file():
            return path
        raise WorkflowNotFound(f"{where} points at {path}, which does not exist")
    checkout = _checkout_workflow()
    if checkout.is_file():
        return checkout
    raise WorkflowNotFound(
        "could not find %s. Set OTR_PACK_DIR to the Old-Time Radio pack folder "
        "(custom_nodes/%s), or COMFYUI_DIR to your ComfyUI folder, or pass --workflow."
        % (WORKFLOW_NAME, PACK_DIR_NAME)
    )


def _first_set(*candidates: str | None) -> str | None:
    """The first candidate that is not blank. A blank is "not given", a
    non-blank is validated by the caller."""
    for value in candidates:
        if value is not None and value.strip():
            return value
    return None


def load_settings(*, comfyui_url: str | None = None, api_url: str | None = None,
                  workflow: str | None = None, out_dir: str | None = None) -> Settings:
    return Settings(
        comfyui_url=validate_url("--comfyui / COMFYUI_URL", _first_set(
            comfyui_url, os.environ.get("COMFYUI_URL")) or DEFAULT_COMFYUI_URL),
        api_url=validate_url("--api / COMFY_BASE_URL", _first_set(
            api_url, os.environ.get("COMFY_BASE_URL")) or DEFAULT_API_URL),
        workflow_path=find_workflow(workflow),
        out_dir=Path(_first_set(out_dir, os.environ.get("RPS_OUT_DIR")) or "episodes").expanduser(),
    )
