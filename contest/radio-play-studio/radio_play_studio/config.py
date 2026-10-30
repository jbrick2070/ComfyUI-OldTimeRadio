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

#: The machines the page offers first, in the order it shows them, each with the
#: ways it can film; a machine's first way is its default. Every entry is a
#: workflow the pack ships and its regression rounds run, so the page chooses
#: among real graphs and never builds one of its own (operator, 2026-10-23:
#: "first you choose ... AMD / Mac / NVIDIA 8/16/24 ... then it prepopulates
#: those and from there you edit, so we can serve everyone via one page"). The
#: three ways are his: TTS + Foley video, TTS w/ video, TTS w/ stills.
#: The proof for each way not in UNPROVEN, for a reader who wants it: 8 GB --
#: all three published on the RTX 4060 in both regression rounds of 2026-10-20/21
#: and 2026-10-23 (planning/regression/ROUNDS.md); 16 GB video and stills -- the
#: 5080 round of 2026-10-23 (same file); 16 GB open foley -- the contest
#: episodes the Studio published on the 5080 (n16f + fk3i in otr/obs/, and its
#: reference twin's fk3r episodes); Mac -- its engines proven in the Mac 16 GB
#: column of apple/MACHINES.md; AMD -- the R9700 receipt in
#: config/workflow_matrix.json.
#: After the three come the rest of each machine's video engines (operator,
#: 2026-10-29: "all video engine models should be available", AnimateDiff named):
#: AnimateDiff, the mime and audio-in LTX lanes, and the visualiser that loads no
#: video weights. The 8 GB four published in both 4060 rounds (ROUNDS.md); the
#: 16 GB AnimateDiff and visualiser in apple/MACHINES.md and the 5080 round; the
#: Mac's AnimateDiff Lightning and visualiser in the Mac 16 GB column.
WAYS: tuple = ("foley", "video", "stills", "animatediff", "mime", "audio_in", "visualiser")
MACHINES: tuple = (
    ("nvidia8", (("foley", "otr_8gb_ltx25_foley.json"), ("video", "otr_8gb_video.json"),
                 ("stills", "otr_8gb_still.json"), ("animatediff", "otr_8gb_animatediff.json"),
                 ("mime", "otr_8gb_ltx25_mime.json"), ("audio_in", "otr_8gb_ltx25_audio_in.json"),
                 ("visualiser", "otr_8gb_low.json"))),
    # 16 GB and up: the 24 GB cards that have rendered episodes (rented RTX PRO
    # 4000, A4500) ran these same graphs; otr_24gb_foley is a draft with no
    # receipt, so it is not offered. No 16 GB audio-in graph ships.
    ("nvidia16", (("foley", WORKFLOW_NAME), ("video", "otr_16gb_video.json"),
                  ("stills", "otr_16gb_still.json"), ("animatediff", "otr_16gb_animatediff.json"),
                  ("mime", "otr_16gb_mime.json"), ("visualiser", "otr_16gb_low.json"))),
    # No foley or mime on a Mac: LTX 0.9.8 video, stills with motion, AnimateDiff
    # Lightning and the visualiser are the engines proven there (apple/MACHINES.md,
    # Mac 16 GB column).
    ("mac16", (("video", "otr_mac16_video.json"), ("stills", "otr_mac16_still.json"),
               ("animatediff", "otr_mac16_animatediff.json"), ("visualiser", "otr_mac16_low.json"))),
    ("amd", (("stills", "otr_amd_still.json"),)),
    # No GPU at all: the cloud graphs. Comfy Cloud bills Comfy credits through the
    # key the page holds; Google bills a Google key ComfyUI reads itself.
    ("comfy_cloud", (("foley", "otr_cloud_deluxe_3act.json"), ("video", "otr_cloud_low.json"))),
    ("google_cloud", (("stills", "otr_google_still.json"),)),
)
def _machine_choices() -> dict:
    """The generated cells (machine_choices.json beside this file), or nothing:
    without them every machine keeps every choice."""
    path = Path(__file__).with_name("machine_choices.json")
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


MACHINE_CHOICES: dict = _machine_choices()
#: Each menu machine's column in machine_choices.json (generated by
#: scripts/otr_dropdown_matrix.py from the cells of apple/MACHINES.md); the cloud
#: machines read the no-GPU column.
MATRIX_COLUMN = {"nvidia8": "nv8", "nvidia16": "nv16", "mac16": "mac16", "amd": "amd",
                 "comfy_cloud": "cpu", "google_cloud": "cpu"}
#: A choice whose cell says one of these is left off that machine's dropdown
#: (operator, 2026-10-23: "only show the relevant options for that system").
#: Unmeasured ("?"), fits, measured, proven and paid cloud ("key") stay.
NOT_FOR_THIS_MACHINE = frozenset({"not offered", "oom", "no", "too slow"})
#: The machines that bill a cloud account rather than run on this machine.
CLOUD_MACHINES = ("comfy_cloud", "google_cloud")
#: Menu workflows with no published episode yet, and why. Offered (the operator
#: asked for cloud, 2026-10-23) but marked "not yet proven" on the page; every
#: other way has a hardware receipt (config/workflow_matrix.json) or proven
#: engines on its machine (apple/MACHINES.md). Take a name off when an episode
#: from it reaches otr/obs/.
UNPROVEN: dict = {
    "otr_cloud_deluxe_3act.json": "no published episode yet",
    "otr_cloud_low.json": "two live runs died at assembly (PBUG-20260915-01, -03); "
                          "fixed, not yet published since",
    "otr_google_still.json": "no published episode yet",
    "otr_16gb_mime.json": "not run in the 5080 round of 2026-10-23; no published episode yet",
}


def menu_lookup(name: str) -> tuple | None:
    """`(machine, way)` for a workflow file name on the menu, else None."""
    for machine, ways in MACHINES:
        for way, workflow in ways:
            if workflow == name:
                return machine, way
    return None


def detect_machine(stats: dict) -> str | None:
    """The menu machine ComfyUI's `/system_stats` describes, or None (a CPU, or
    nothing reported). A guess for the page to start on, never a gate: the
    person can pick another."""
    devices = (stats or {}).get("devices") or []
    if not devices:
        return None
    device = devices[0]
    kind = str(device.get("type") or "").lower()
    name = str(device.get("name") or "").lower()
    if kind == "mps":
        return "mac16"
    if kind != "cuda":
        return None
    if "amd" in name or "radeon" in name:
        return "amd"
    gib = float(device.get("vram_total") or 0) / 2 ** 30
    if gib >= 12:
        return "nvidia16"
    return "nvidia8" if gib > 0 else None


#: The pack's folder under custom_nodes/: a git clone takes the repository's name,
#: a registry install (ComfyUI Manager, comfy-cli) takes the node id.
PACK_DIR_NAMES = ("ComfyUI-OldTimeRadio", "comfyui-old-time-radio")


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
    """`[(where, [paths])]` for each place the person pointed at, most specific
    first. These are promises: if one is set, one of its paths must be there."""
    sources: list = []
    env_file = os.environ.get("OTR_APP_WORKFLOW", "").strip()
    if env_file:
        sources.append(("OTR_APP_WORKFLOW", [Path(env_file).expanduser()]))
    pack_dir = os.environ.get("OTR_PACK_DIR", "").strip()
    if pack_dir:
        sources.append(("OTR_PACK_DIR", [Path(pack_dir).expanduser() / "workflows" / WORKFLOW_NAME]))
    comfy_dir = os.environ.get("COMFYUI_DIR", "").strip()
    if comfy_dir:
        sources.append(("COMFYUI_DIR", [Path(comfy_dir).expanduser() / "custom_nodes" / name / "workflows"
                                        / WORKFLOW_NAME for name in PACK_DIR_NAMES]))
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
    for where, paths in _configured_workflow_sources():
        found = next((path for path in paths if path.is_file()), None)
        if found:
            return found
        if len(paths) == 1:
            raise WorkflowNotFound(f"{where} points at {paths[0]}, which does not exist")
        raise WorkflowNotFound(f"{where} is set, but none of these exists: {', '.join(map(str, paths))}")
    checkout = _checkout_workflow()
    if checkout.is_file():
        return checkout
    raise WorkflowNotFound(
        "could not find %s. Set OTR_PACK_DIR to the Old-Time Radio pack folder "
        "(custom_nodes/%s), or COMFYUI_DIR to your ComfyUI folder, or pass --workflow."
        % (WORKFLOW_NAME, " or custom_nodes/".join(PACK_DIR_NAMES))
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
