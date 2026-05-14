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


def otr_episodes_root() -> Path:
    """Root of the per-episode workspace tree: ``<output>/otr/episodes/``.

    Each episode has its own subdir under here:
    ``<output>/otr/episodes/<episode_id>/{audio,stills,portraits,videos,
    composited}/``. The ledger auto-pick logic walks this root looking
    for ``*/audio/*_ledger.json`` files across all episodes.

    Added 2026-05-02 EVENING per Jeffrey directive: every per-episode
    working file nests inside that episode's own folder so the
    workspace is organized by episode, not by file type.
    """
    return comfy_output_dir() / "otr" / "episodes"


def otr_audio_dir(episode_id: str = "") -> Path:
    """Per-episode audio + ledger dir: ``<output>/otr/episodes/<episode_id>/audio/``.

    Holds the procgen mp4, the canonical ``<episode_id>_ledger.json``,
    Bark / Kokoro output wavs, and MusicGen + AudioGen cache files
    for this episode. (LLMDirector raw-output dumps were removed in
    voice-path-cleanbreak S23.1 along with the helper that owned
    them.)

    The ``episode_id`` argument is REQUIRED for new code. Calls without
    it fall back to a degraded per-episode-less path under
    ``<output>/otr/_legacy_audio/`` -- this exists ONLY so legacy code
    paths that auto-pick across episodes don't crash; production
    write-sites must pass a real ``episode_id``.

    For ledger auto-pick across all episodes, use
    ``otr_episodes_root()`` and walk ``*/audio/*_ledger.json``.

    Cache trade-off (Jeffrey acknowledged 2026-05-02 EVENING):
    MusicGen + AudioGen used to write SHA-keyed cache files into a
    single shared ``otr/audio/`` dir so the same prompt across two
    episodes hit the cache. Per-episode audio dirs lose that
    cross-episode cache hit -- each episode now re-renders music + sfx
    even when prompts are identical. Acceptable cost for the cleaner
    organization; revisit only if cache loss becomes a wallclock pain.
    """
    if episode_id:
        return otr_episodes_root() / episode_id / "audio"
    return comfy_output_dir() / "otr" / "_legacy_audio"


# S28 cleanbreak: otr_legacy_audio_dir() removed. The pre-BUG-079
# fallback root (`<output>/old_time_radio/`) was extinct in production
# — per-episode workspace is the only contract. All callers were
# extinguished in s28-p1-1 through s28-p1-8 before this deletion.


def otr_stills_dir(episode_id: str = "") -> Path:
    """Per-episode FLUX cast / environment stills + radio bookend dir:
    ``<output>/otr/episodes/<episode_id>/stills/``.

    Holds ``full_env_NNNNN_.png`` cast environment portraits and the
    ``radio_bookend_<episode_id>.png`` LTX I2V reference still.
    Without ``episode_id`` falls back to ``<output>/otr/_legacy_stills/``.
    """
    if episode_id:
        return otr_episodes_root() / episode_id / "stills"
    return comfy_output_dir() / "otr" / "_legacy_stills"


def otr_portraits_dir(episode_id: str = "") -> Path:
    """Per-episode PASS1 character portrait dir:
    ``<output>/otr/episodes/<episode_id>/portraits/``.
    Without ``episode_id`` falls back to ``<output>/otr/_legacy_portraits/``.
    """
    if episode_id:
        return otr_episodes_root() / episode_id / "portraits"
    return comfy_output_dir() / "otr" / "_legacy_portraits"


def otr_videos_dir(episode_id: str) -> Path:
    """Per-episode per-line HuMo + LTX clip dir:
    ``<output>/otr/episodes/<episode_id>/videos/``.

    Holds the per-line piece clips: ``l002.mp4`` (HuMo character),
    ``music_opening_001.mp4`` (LTX music), ``l001.mp4`` (LTX announcer),
    etc. VideoComposite reads these and assembles the final composite.
    """
    return otr_episodes_root() / episode_id / "videos"


def otr_composited_dir(episode_id: str) -> Path:
    """Per-episode VideoComposite intermediate dir:
    ``<output>/otr/episodes/<episode_id>/composited/``.

    Holds the 832x480 composited mp4 written by VideoComposite as
    ``<episode_id>.mp4``. Downstream OTR_RTXUpscale reads from here
    and writes its 1080p output to ``otr_upscaled_dir(episode_id)``
    (a sibling per-episode dir; pre-2026-05-05 the upscale lived in
    ``otr/obs/`` directly but that broke the "one mp4 per episode"
    contract once OTR_PostUpscaleProcgenBlend started writing the
    real final there too).
    """
    return otr_episodes_root() / episode_id / "composited"


def otr_upscaled_dir(episode_id: str) -> Path:
    """Per-episode 1080p upscaled-but-pre-blend dir:
    ``<output>/otr/episodes/<episode_id>/upscaled/``.

    Holds the 1920x1080 mp4 written by OTR_RTXUpscale as
    ``<episode_id>.mp4``. Downstream OTR_PostUpscaleProcgenBlend reads
    from here and writes the final blended deliverable to
    ``otr_obs_dir()/<episode_id>_procgen_blended.mp4``.

    Added 2026-05-05 (Jeffrey directive): obs/ is the broadcast folder
    and must hold exactly ONE mp4 per episode -- the post-blend
    deliverable. Pre-blend intermediates (832x480 native composite,
    1080p upscale) live under per-episode subdirs so the broadcast
    library stays clean. Mirror of ``otr_composited_dir`` but for the
    upscale stage of the chain.
    """
    return otr_episodes_root() / episode_id / "upscaled"


def otr_state_dir() -> Path:
    """Per-machine OTR runtime-state dir: ``<output>/otr/state/``.

    BUG-LOCAL-090 (2026-05-04): added so persistent runtime state
    (news_history.json, future per-machine cursors) lives under the
    user's ComfyUI output tree -- the natural per-machine state tier --
    instead of polluting the source repo's ``config/`` folder. State
    that's per-episode goes under ``otr/episodes/<episode_id>/``;
    state that's per-machine and not tied to any episode goes here.

    Caller is responsible for ``mkdir(parents=True, exist_ok=True)``.
    """
    return comfy_output_dir() / "otr" / "state"


def otr_obs_dir() -> Path:
    """OBS-watched final-deliverable dir: ``<output>/otr/obs/``.

    Holds EXACTLY ONE mp4 per episode -- the final user-facing
    deliverable. As of 2026-05-05 the canonical filename is
    ``<episode_id>_procgen_blended.mp4`` (post BUG-LOCAL-106
    + Jeffrey "broadcast folder" directive). OBS's directory_sorter
    watches this dir and queues each finished episode in turn.

    Render chain (one final mp4 per episode lands here):

      1. VideoComposite -> otr/episodes/<ep>/composited/<ep>.mp4
         (832x480 native composite intermediate)
      2. OTR_RTXUpscale -> otr/episodes/<ep>/upscaled/<ep>.mp4
         (1920x1080 upscale intermediate -- see otr_upscaled_dir)
      3. OTR_PostUpscaleProcgenBlend -> otr/obs/<ep>_procgen_blended.mp4
         (final broadcast cut with green-CRT overlay -- this dir)

    Pre-2026-05-05, OTR_RTXUpscale wrote step 2's output directly
    into otr/obs/ which was correct when that node was the final
    stage. After OTR_PostUpscaleProcgenBlend joined the chain (BUG-099
    onward) two mp4s ended up in obs/ per episode, breaking the
    "broadcast folder" contract. Step 2's output now lands in
    ``otr_upscaled_dir(episode_id)`` instead -- intermediates live
    under their episode, only the broadcast cut lives here.
    """
    return comfy_output_dir() / "otr" / "obs"


def episodes_for_obs_dir(episode_id: str = "") -> Path:
    """VideoComposite intermediate-mp4 dir -- per-episode workspace.

    Path: ``<output>/otr/episodes/<episode_id>/composited/`` when
    ``episode_id`` is given. Returns ``otr_episodes_root()`` (the
    parent of all per-episode subdirs) when called without an
    episode_id, for legacy callers that scan the tree.

    Holds the 832x480 native composite mp4 written by VideoComposite
    as ``<episode_id>.mp4`` (one per episode). This is the
    INTERMEDIATE -- the downstream OTR_RTXUpscale stage reads from
    here and writes the final mp4 to the OBS-watched dir at
    ``<output>/otr/obs/<episode_id>.mp4`` (see ``otr_obs_dir``).

    Despite the historical name ``episodes_for_obs_dir``, this dir
    is NOT what OBS watches anymore -- that role moved to ``otr/obs/``
    on 2026-05-02 EVENING. Function name kept for back-compat with
    existing imports; canonical OBS-watched dir is now ``otr_obs_dir``;
    canonical intermediate dir helper is ``otr_composited_dir``.

    History (canonical change-log for this path):
      - Originally ``<output>/episodes_for_obs/<episode_id>/``,
        sibling of ``otr/`` (BUG-LOCAL-084 era).
      - 2026-05-02 EVENING (Jeffrey directive 1): consolidate under
        ``otr/`` -> ``<output>/otr/episodes/<episode_id>/``.
      - 2026-05-02 EVENING (Jeffrey directive 2): flatten ->
        ``<output>/otr/episodes/`` (no per-episode subfolder).
      - 2026-05-02 EVENING (Jeffrey directive 3): split intermediate
        vs final -- new ``otr_obs_dir()`` for the single final mp4
        per episode; this dir keeps the intermediate.
      - 2026-05-02 EVENING (Jeffrey directive 4): per-episode workspace.
        Intermediate moves to per-episode subdir
        ``<output>/otr/episodes/<episode_id>/composited/<episode_id>.mp4``.
    """
    if episode_id:
        return otr_composited_dir(episode_id)
    return otr_episodes_root()


# director_raw_dump_dir was deleted in voice-path-cleanbreak S23.1
# (2026-05-13) along with the LLMDirector class itself (S2,
# commit 249bc06). No replacement is needed -- the L3 ledger and
# atomic save_ledger_safe path replace the raw-dump scheme.


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
    # S28 cleanbreak: dropped "otr_legacy_audio_dir".
    "otr_stills_dir",
    "otr_portraits_dir",
    "otr_videos_dir",
    "otr_composited_dir",
    "otr_upscaled_dir",
    "otr_obs_dir",
    "otr_state_dir",
    "episodes_for_obs_dir",
    # director_raw_dump_dir entry removed in voice-path-cleanbreak S23.1
    "resolve_hf_model_path",
    "comfyui_log_path",
]
