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
    """OBS broadcast tree, sibling of ``otr/`` (BUG-LOCAL-084).

    Final composited episode mp4 + .vtt sidecar live here so OBS's
    directory_sorter can point at ``output/episodes_for_obs/`` and
    only see finished episodes -- never the per-line HuMo clip pieces.
    """
    return comfy_output_dir() / "episodes_for_obs" / episode_id


def director_raw_dump_dir() -> Path:
    """Where ``LLMDirector`` parks raw output dumps for BUG-090.

    Lives alongside the ledger files (``otr/audio/``) so a single
    ``ls`` of the audio dir surfaces both the ledger and any
    failed-parse raw dumps for the same run.
    """
    return otr_audio_dir()


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
]
