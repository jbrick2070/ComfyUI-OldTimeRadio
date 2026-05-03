"""
_otr_paths.py  --  OTR canonical filesystem paths
==================================================

Single source of truth for ComfyUI output / OTR subdirectory
locations. All node code and CLI scripts should import from here
instead of hardcoding ``Path(r"C:\\Users\\...\\ComfyUI\\output")``.

Resolution order for ``comfy_output_dir()``:
  1. ``OTR_OUTPUT_DIR`` environment variable, if non-empty. Highest
     priority -- lets cloud deployments (RunPod, Linux/Mac, Docker)
     and CI override the output root without code edits.
  2. ``folder_paths.get_output_directory()`` -- ComfyUI's own canonical
     output dir API. Available when running inside the ComfyUI
     process; honors ``--output-directory`` and friends automatically.
  3. Walk up from this module's path: ``<repo>/../../../output``. With
     the standard ``custom_nodes/ComfyUI-OldTimeRadio/nodes/`` layout
     this yields ``ComfyUI/output`` -- the location every Jeffrey-on-
     Windows install has been hardcoding for the last six months.
  4. ``Path.cwd() / "output"`` as a final fallback. Last resort for
     ad-hoc CLI invocations from an unusual working dir.

The function is a plain dispatcher with no caching: a fresh resolution
on every call keeps pytest fixtures honest (they can monkey-patch
``OTR_OUTPUT_DIR`` per-test without leaking across cases).

Subdir helpers (``otr_audio_dir``, ``otr_stills_dir``, etc.) are thin
wrappers that compose ``comfy_output_dir()`` with the canonical layout
locked 2026-04-27. None of them ``mkdir`` on access -- callers do that
explicitly when they know they're about to write -- because read-only
consumers (e.g. ledger auto-discover) shouldn't accidentally create
empty dirs as a side effect.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# Walk-up math: this file lives at
# ``ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_paths.py``.
# parents[0] = nodes/, parents[1] = ComfyUI-OldTimeRadio/,
# parents[2] = custom_nodes/, parents[3] = ComfyUI/. The trailing
# ``output`` is what every BUG-LOCAL-* fix has been pinning.
_REPO_WALKUP_OUTPUT = Path(__file__).resolve().parents[3] / "output"


def comfy_output_dir() -> Path:
    """Return the ComfyUI output root.

    See module docstring for resolution order. Caller is responsible
    for any ``mkdir(parents=True, exist_ok=True)`` before writing --
    this function is read-only and never creates directories.
    """
    # Tier 1: explicit env override
    env_override = os.environ.get("OTR_OUTPUT_DIR")
    if env_override:
        return Path(env_override).expanduser()

    # Tier 2: ComfyUI's own canonical API
    try:
        import folder_paths  # type: ignore

        api_dir = folder_paths.get_output_directory()
        if api_dir:
            return Path(api_dir)
    except ImportError:
        # Running outside the ComfyUI process (CLI scripts, pytest);
        # drop to walk-up.
        pass
    except Exception:
        # Defensive: never let a folder_paths surprise break callers.
        pass

    # Tier 3: walk up from this module's path
    if _REPO_WALKUP_OUTPUT.parent.exists():
        return _REPO_WALKUP_OUTPUT

    # Tier 4: cwd fallback
    return Path.cwd() / "output"


def comfy_models_dir() -> Path:
    """Return the ComfyUI models root.

    Used by visual backends (FLUX anchor, PuLID portrait, future
    LTX/Wan2.1) to find pre-quantized weights without hardcoding
    Jeffrey's personal path. Resolution order:

      1. ``OTR_MODELS_DIR`` env var (explicit override; cloud / 8GB
         tier deployments can park weights anywhere).
      2. ``folder_paths.models_dir`` (ComfyUI's API; respects
         ``--extra-model-paths-config``).
      3. Walk up: ``<repo>/../../../models``.
      4. ``Path.cwd() / "models"``.
    """
    env_override = os.environ.get("OTR_MODELS_DIR")
    if env_override:
        return Path(env_override).expanduser()

    try:
        import folder_paths  # type: ignore

        api_dir = getattr(folder_paths, "models_dir", None)
        if api_dir:
            return Path(api_dir)
    except ImportError:
        pass
    except Exception:
        pass

    walkup = Path(__file__).resolve().parents[3] / "models"
    if walkup.parent.exists():
        return walkup

    return Path.cwd() / "models"


def comfy_input_dir() -> Path:
    """Return the ComfyUI input root.

    Mirror of ``comfy_output_dir`` but for the input side. Used by
    nodes that drop temporary files (per-line portraits for HuMo
    smoke tests, etc.) where ComfyUI expects to read them. Same
    resolution order:

      1. ``OTR_INPUT_DIR`` env var
      2. ``folder_paths.get_input_directory()`` (ComfyUI's API)
      3. Walk up: ``<repo>/../../../input``
      4. ``Path.cwd() / "input"``
    """
    env_override = os.environ.get("OTR_INPUT_DIR")
    if env_override:
        return Path(env_override).expanduser()

    try:
        import folder_paths  # type: ignore

        api_dir = folder_paths.get_input_directory()
        if api_dir:
            return Path(api_dir)
    except ImportError:
        pass
    except Exception:
        pass

    walkup = Path(__file__).resolve().parents[3] / "input"
    if walkup.parent.exists():
        return walkup

    return Path.cwd() / "input"


def otr_audio_dir() -> Path:
    """Canonical ledger / master-audio dir: ``<output>/otr/audio/``."""
    return comfy_output_dir() / "otr" / "audio"


def otr_legacy_audio_dir() -> Path:
    """Pre-BUG-079 legacy audio dir: ``<output>/old_time_radio/``.

    Kept as a back-compat search root for ledger auto-discover so we
    don't lose track of pre-cutover episode files. Do not write here.
    """
    return comfy_output_dir() / "old_time_radio"


def otr_stills_dir() -> Path:
    """FLUX cast / environment stills: ``<output>/otr/stills/``."""
    return comfy_output_dir() / "otr" / "stills"


def otr_portraits_dir() -> Path:
    """PASS1 character portraits: ``<output>/otr/portraits/``."""
    return comfy_output_dir() / "otr" / "portraits"


def otr_videos_dir(episode_id: str) -> Path:
    """Per-episode HuMo clip dir: ``<output>/otr/videos/<episode_id>/``."""
    return comfy_output_dir() / "otr" / "videos" / episode_id


def episodes_for_obs_dir(episode_id: str) -> Path:
    """Final-episode deliverable dir, nested INSIDE the otr/ tree.

    Path: ``<output>/otr/episodes/<episode_id>/``

    Holds ONLY the user-facing final mp4(s) -- the composited 832x480
    output from VideoComposite and the upscaled 1080p output from
    OTR_RTXUpscale. The piece-level per-line clips stay under
    ``<output>/otr/videos/<episode_id>/`` (see ``otr_videos_dir``).

    History (kept here as the canonical change-log for this path):
      - Originally lived at ``<output>/episodes_for_obs/<episode_id>/``,
        a sibling of ``otr/`` (BUG-LOCAL-084 era), so OBS's
        directory_sorter could point at one root and only see finished
        episodes.
      - 2026-05-02 EVENING (Jeffrey directive after first end-to-end
        smoke succeeded): consolidate every project output under the
        ``otr/`` umbrella so the workspace has one tidy root. OBS
        config should now point at ``output/otr/episodes/`` instead.
    """
    return comfy_output_dir() / "otr" / "episodes" / episode_id


def director_raw_dump_dir() -> Path:
    """Where ``LLMDirector`` parks raw output dumps for BUG-090.

    Lives alongside the ledger files (``otr/audio/``) so a single
    ``ls`` of the audio dir surfaces both the ledger and any
    failed-parse raw dumps for the same run.
    """
    return otr_audio_dir()


def comfyui_log_path() -> Optional[str]:
    """Locate the active ComfyUI core log file.

    ComfyUI writes its stdout / stderr stream to a log file whose
    location varies by install type. There is NO authoritative API
    inside ComfyUI to query this -- ``folder_paths.get_user_directory()``
    returns the user-state directory which historically held the log
    but recent ComfyUI Desktop builds (Electron-based) write logs to a
    platform-specific app-data location instead.

    Known layouts:

    - **ComfyUI Desktop (Electron) on Windows**
      ``%APPDATA%/ComfyUI/logs/comfyui.log``
      (i.e. ``~/AppData/Roaming/ComfyUI/logs/comfyui.log``)

    - **ComfyUI Desktop (Electron) on macOS**
      ``~/Library/Logs/ComfyUI/comfyui.log``

    - **ComfyUI Desktop (Electron) on Linux**
      ``${XDG_CONFIG_HOME:-~/.config}/ComfyUI/logs/comfyui.log``

    - **ComfyUI portable / pip-installed / standalone**
      ``<user_directory>/comfyui.log`` -- where ``user_directory`` is
      the value reported by ``folder_paths.get_user_directory()`` (the
      same directory that holds ``comfyui.db``, workflow autosaves,
      ``__manager`` cache, etc.). Older builds.

    Strategy: enumerate every candidate path, keep the ones that exist
    on disk, then pick the one with the **most-recent mtime**. The
    active log is being written to RIGHT NOW by the running ComfyUI
    process, so its mtime is the freshest. If multiple log layouts
    coexist on a machine (e.g. user previously ran the portable build
    and now runs the Desktop build), this picks the live one
    automatically.

    Returns the absolute path on success, ``None`` when no candidate
    exists. Callers should treat ``None`` as "log location unknown,
    don't stamp it" rather than raising.

    The probe is cheap (a handful of ``os.path.exists`` + ``getmtime``
    calls) so it's safe to call once per render at workflow phase 0.
    """
    candidates: list[str] = []
    home = os.path.expanduser("~")

    # ComfyUI Desktop Electron build, Windows
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(os.path.join(appdata, "ComfyUI", "logs", "comfyui.log"))

    # ComfyUI Desktop Electron build, macOS
    candidates.append(os.path.join(home, "Library", "Logs", "ComfyUI", "comfyui.log"))

    # ComfyUI Desktop Electron build, Linux
    xdg_config = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    candidates.append(os.path.join(xdg_config, "ComfyUI", "logs", "comfyui.log"))

    # ComfyUI portable / pip / standalone (legacy layout). folder_paths
    # is the ComfyUI runtime module; absent in CLI / test environments.
    try:
        import folder_paths as _fp  # noqa: PLC0415 -- ComfyUI runtime import
        _user_dir = _fp.get_user_directory()
        candidates.append(os.path.join(_user_dir, "comfyui.log"))
        # Some builds also write directly into <user_dir>/.. (one level up)
        candidates.append(os.path.join(_user_dir, "..", "comfyui.log"))
    except Exception:  # noqa: BLE001 -- folder_paths unavailable outside ComfyUI
        pass

    # Probe: keep only existing files, pick most-recent mtime.
    found: list[tuple[float, str]] = []
    for c in candidates:
        try:
            c_norm = os.path.normpath(c)
            if os.path.isfile(c_norm):
                mtime = os.path.getmtime(c_norm)
                found.append((mtime, c_norm))
        except OSError:
            continue

    if not found:
        return None

    found.sort(reverse=True)
    return found[0][1]


def resolve_hf_model_path(repo_id: str) -> str:
    """Resolve a HuggingFace ``repo_id`` to a local cache path.

    Returns the parent directory of ``models--{org}--{name}`` if a
    cached copy is found, otherwise returns the original ``repo_id``
    string. Both forms are accepted by ``transformers.from_pretrained``
    and ``huggingface_hub.snapshot_download`` -- a directory triggers
    offline load, the ``repo_id`` triggers a download.

    Resolution order (each candidate must exist AND be non-empty):

    1. ``OTR_MODELS_DIR`` env var -- explicit user override.
       ``$OTR_MODELS_DIR/huggingface/hub/models--{...}``.
    2. ``HF_HOME`` env var -- HF standard.
       ``$HF_HOME/hub/models--{...}``.
    3. ``comfy_models_dir() / "huggingface" / "hub" / models--{...}`` --
       ComfyUI project convention. Falls through silently if
       ``folder_paths`` import fails (i.e. running outside ComfyUI).
    4. ``~/.cache/huggingface/hub/models--{...}`` -- HF default cache.
    5. Fall back to the bare ``repo_id`` string -- caller's
       ``from_pretrained()`` resolves online.

    Designed for shipped node distribution where the user's HF_HOME and
    ComfyUI install layout are unknown. Local OTR development with
    HF_HOME set in HKCU\\Environment usually hits candidate #2 first.

    Cache-dir naming convention is HuggingFace's: ``org/name`` becomes
    ``models--org--name``. A folder must exist AND contain at least one
    entry (snapshots/, blobs/, refs/, etc.) to be considered valid --
    skips empty stub directories that some HF tools create to hold a
    locks file before any actual download.
    """
    cache_dirname = "models--" + repo_id.replace("/", "--")
    candidates = []

    otr_dir = os.environ.get("OTR_MODELS_DIR", "").strip()
    if otr_dir:
        candidates.append(Path(otr_dir) / "huggingface" / "hub" / cache_dirname)

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        candidates.append(Path(hf_home) / "hub" / cache_dirname)

    try:
        candidates.append(comfy_models_dir() / "huggingface" / "hub" / cache_dirname)
    except Exception:  # noqa: BLE001 -- best-effort; folder_paths may be missing in CLI/test envs
        pass

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / cache_dirname)

    for c in candidates:
        try:
            if c.exists() and any(c.iterdir()):
                # Return the cache root (parent of "hub"), not the model dir.
                # transformers.from_pretrained also accepts the model dir
                # directly via its snapshots/ subdir, but returning the
                # cache root keeps the caller's loader code symmetric with
                # the bare-repo_id fallback case.
                return str(c.parent.parent)
        except OSError:  # noqa: PERF203 -- per-candidate isolation
            continue

    return repo_id


__all__ = [
    "comfy_output_dir",
    "comfy_input_dir",
    "comfy_models_dir",
    "otr_audio_dir",
    "otr_legacy_audio_dir",
    "otr_stills_dir",
    "otr_portraits_dir",
    "otr_videos_dir",
    "episodes_for_obs_dir",
    "director_raw_dump_dir",
    "resolve_hf_model_path",
    "comfyui_log_path",
]
